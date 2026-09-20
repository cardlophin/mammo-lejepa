from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest

from mammo_lejepa.models import BoxSource, CajaMamaria
from mammo_lejepa.vindr.catalog import (
    consolidate,
    parse_jsonl_records,
    validate_catalog_coherence,
    verify_no_study_leakage,
)
from mammo_lejepa.vindr.models import ImageRecord

_EMPTY_FINDINGS_SCHEMA = {
    "image_id": pl.Utf8,
    "study_id": pl.Utf8,
    "finding_categories": pl.Utf8,
    "finding_birads": pl.Utf8,
    "xmin": pl.Float64,
    "ymin": pl.Float64,
    "xmax": pl.Float64,
    "ymax": pl.Float64,
}


def _crop(**overrides: object) -> CajaMamaria:
    base = {
        "x0": 10,
        "y0": 10,
        "x1": 90,
        "y1": 190,
        "margin_px": 10,
        "source": BoxSource.OTSU,
        "threshold": 100.0,
        "image_height": 200,
        "image_width": 100,
        "area_ratio": 0.72,
    }
    base.update(overrides)
    return CajaMamaria(**base)


def _record(
    *,
    image_id: str,
    study_id: str = "study_a",
    split: str = "training",
    processed_at: str = "2026-01-01T00:00:00+00:00",
    crop: CajaMamaria | None = None,
) -> ImageRecord:
    return ImageRecord(
        study_id=study_id,
        series_id=f"{study_id}-series",
        image_id=image_id,
        status="ok",
        split=split,
        laterality="L",
        view_position="CC",
        breast_birads="BI-RADS 1",
        breast_density="DENSITY B",
        run_id="run_x",
        pipeline_version="0.1.0",
        processed_at=processed_at,
        download_seconds=1.0,
        process_seconds=1.0,
        dicom_bytes=1000,
        breast_crop=crop if crop is not None else _crop(),
        png_path="processed/study_a/img.png",
        png_bytes=1234,
    )


def _empty_findings() -> pl.DataFrame:
    return pl.DataFrame(schema=_EMPTY_FINDINGS_SCHEMA)


# --- parse_jsonl_records ---------------------------------------------------


def test_parse_jsonl_records_roundtrips_via_consolidate() -> None:
    import json
    from dataclasses import asdict
    from enum import Enum

    def default(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        raise TypeError(type(value))

    record = _record(image_id="img_1")
    line = json.dumps(asdict(record), default=default)

    parsed = parse_jsonl_records([line])

    assert len(parsed) == 1
    assert parsed[0].image_id == "img_1"
    assert parsed[0].breast_crop is not None
    assert parsed[0].breast_crop.source is BoxSource.OTSU


def test_parse_jsonl_records_discards_truncated_last_line() -> None:
    import json
    from dataclasses import asdict
    from enum import Enum

    def default(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        raise TypeError(type(value))

    complete = json.dumps(asdict(_record(image_id="img_1")), default=default)
    truncated = '{"image_id": "img_2", "stat'

    parsed = parse_jsonl_records([complete, truncated])

    assert len(parsed) == 1
    assert parsed[0].image_id == "img_1"


# --- consolidate -------------------------------------------------------


def test_consolidate_flattens_crop_fields() -> None:
    images_df, _findings_df = consolidate(
        [_record(image_id="img_1")], _empty_findings()
    )

    assert images_df.height == 1
    row = images_df.row(0, named=True)
    assert row["crop_x0"] == 10
    assert row["crop_width"] == 80
    assert row["crop_height"] == 180
    assert row["crop_otsu_threshold"] == 100.0
    assert row["source_height"] == 200


def test_consolidate_keeps_most_recent_by_processed_at() -> None:
    older = _record(image_id="img_1", processed_at="2026-01-01T00:00:00+00:00")
    newer = _record(
        image_id="img_1",
        processed_at="2026-01-02T00:00:00+00:00",
        crop=_crop(x0=0),
    )

    images_df, _ = consolidate([older, newer], _empty_findings())

    assert images_df.height == 1
    assert images_df.row(0, named=True)["crop_x0"] == 0


# --- verify_no_study_leakage ---------------------------------------------


def test_verify_no_study_leakage_passes_when_split_consistent_per_study() -> None:
    images_df = pl.DataFrame(
        {
            "study_id": ["study_a", "study_a", "study_b"],
            "split": ["training", "training", "test"],
        }
    )

    verify_no_study_leakage(images_df)  # no debe lanzar


def test_verify_no_study_leakage_raises_naming_the_study() -> None:
    images_df = pl.DataFrame(
        {
            "study_id": ["study_a", "study_a"],
            "split": ["training", "test"],
        }
    )

    with pytest.raises(ValueError, match="study_a"):
        verify_no_study_leakage(images_df)


# --- validate_catalog_coherence -------------------------------------------


def test_validate_catalog_coherence_passes_for_real_png(tmp_path: Path) -> None:
    png_path = tmp_path / "img_1.png"
    pixels = np.full((50, 30), 128, dtype=np.uint8)
    cv2.imwrite(str(png_path), pixels)

    images_df = pl.DataFrame(
        [
            {
                "image_id": "img_1",
                "status": "ok",
                "png_path": str(png_path),
                "png_bytes": png_path.stat().st_size,
                "crop_height": 50,
                "crop_width": 30,
            }
        ]
    )

    assert validate_catalog_coherence(images_df, _empty_findings()) == []


def test_validate_catalog_coherence_flags_missing_png() -> None:
    images_df = pl.DataFrame(
        [
            {
                "image_id": "img_1",
                "status": "ok",
                "png_path": "/nonexistent/img_1.png",
                "png_bytes": 10,
                "crop_height": 50,
                "crop_width": 30,
            }
        ]
    )

    problems = validate_catalog_coherence(images_df, _empty_findings())

    assert len(problems) == 1
    assert "img_1" in problems[0]


def test_validate_catalog_coherence_flags_finding_outside_crop_bounds(
    tmp_path: Path,
) -> None:
    png_path = tmp_path / "img_1.png"
    pixels = np.full((50, 30), 128, dtype=np.uint8)
    cv2.imwrite(str(png_path), pixels)

    images_df = pl.DataFrame(
        [
            {
                "image_id": "img_1",
                "status": "ok",
                "png_path": str(png_path),
                "png_bytes": png_path.stat().st_size,
                "crop_height": 50,
                "crop_width": 30,
            }
        ]
    )
    findings_df = pl.DataFrame(
        [
            {
                "finding_id": "img_1_1",
                "image_id": "img_1",
                "xmin_crop": 10.0,
                "ymin_crop": 10.0,
                "xmax_crop": 999.0,
                "ymax_crop": 20.0,
            }
        ]
    )

    problems = validate_catalog_coherence(images_df, findings_df)

    assert len(problems) == 1
    assert "img_1_1" in problems[0]
