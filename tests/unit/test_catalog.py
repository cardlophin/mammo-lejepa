from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import polars as pl

from mammo_lejepa.catalog import consolidate, validate_catalog_coherence
from mammo_lejepa.models import BoxSource, CajaMamaria, RegistroDeRecorte


def _box(**overrides: object) -> CajaMamaria:
    base = {
        "x0": 10,
        "y0": 10,
        "x1": 90,
        "y1": 190,
        "margin_px": 10,
        "source": BoxSource.MASK,
        "threshold": 128.0,
        "image_height": 200,
        "image_width": 100,
        "area_ratio": 0.72,
    }
    base.update(overrides)
    return CajaMamaria(**base)


def _record(
    *, image_id: str, processed_at: str, box: CajaMamaria | None = None
) -> RegistroDeRecorte:
    return RegistroDeRecorte(
        image_id=image_id,
        source_dataset="inbreast",
        source_subject_id="p1",
        patient_key="inbreast:p1",
        laterality="L",
        view="CC",
        status="ok",
        process_seconds=1.0,
        run_id="run_x",
        code_version="0.1.0",
        processed_at=processed_at,
        box=box if box is not None else _box(),
        crop_path="crops/inbreast/img_1.png",
        crop_bytes=1234,
        classification="Normal",
    )


def test_consolidate_flattens_box_fields() -> None:
    records = [_record(image_id="img_1", processed_at="2026-01-01T00:00:00+00:00")]

    catalog = consolidate(records)

    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert row["crop_x0"] == 10
    assert row["crop_width"] == 80
    assert row["crop_height"] == 180
    assert row["box_source"] == "MASK"
    assert "split" not in row


def test_consolidate_keeps_most_recent_by_processed_at() -> None:
    older = _record(image_id="img_1", processed_at="2026-01-01T00:00:00+00:00")
    newer = _record(
        image_id="img_1",
        processed_at="2026-01-02T00:00:00+00:00",
        box=_box(x0=0),
    )

    catalog = consolidate([older, newer])

    assert catalog.height == 1
    assert catalog.row(0, named=True)["crop_x0"] == 0


def test_consolidate_empty_input_returns_empty_frame() -> None:
    catalog = consolidate([])

    assert catalog.height == 0


def test_failed_record_has_null_box_fields() -> None:
    record = RegistroDeRecorte(
        image_id="img_1",
        source_dataset="inbreast",
        source_subject_id="p1",
        patient_key="inbreast:p1",
        laterality="L",
        view="CC",
        status="failed",
        process_seconds=1.0,
        run_id="run_x",
        code_version="0.1.0",
        processed_at="2026-01-01T00:00:00+00:00",
    )

    catalog = consolidate([record])

    row = catalog.row(0, named=True)
    assert row["crop_x0"] is None
    assert row["box_source"] is None


def test_validate_catalog_coherence_passes_for_real_crop(tmp_path: Path) -> None:
    crop_path = tmp_path / "img_1.png"
    pixels = np.full((50, 30), 128, dtype=np.uint8)
    cv2.imwrite(str(crop_path), pixels)

    catalog = pl.DataFrame(
        [
            {
                "image_id": "img_1",
                "status": "ok",
                "crop_path": str(crop_path),
                "crop_bytes": crop_path.stat().st_size,
                "crop_height": 50,
                "crop_width": 30,
            }
        ]
    )

    assert validate_catalog_coherence(catalog) == []


def test_validate_catalog_coherence_flags_missing_file() -> None:
    catalog = pl.DataFrame(
        [
            {
                "image_id": "img_1",
                "status": "ok",
                "crop_path": "/nonexistent/img_1.png",
                "crop_bytes": 10,
                "crop_height": 50,
                "crop_width": 30,
            }
        ]
    )

    problems = validate_catalog_coherence(catalog)

    assert len(problems) == 1
    assert "img_1" in problems[0]


def test_validate_catalog_coherence_flags_dimension_mismatch(tmp_path: Path) -> None:
    crop_path = tmp_path / "img_1.png"
    pixels = np.full((50, 30), 128, dtype=np.uint8)
    cv2.imwrite(str(crop_path), pixels)

    catalog = pl.DataFrame(
        [
            {
                "image_id": "img_1",
                "status": "ok",
                "crop_path": str(crop_path),
                "crop_bytes": crop_path.stat().st_size,
                "crop_height": 999,
                "crop_width": 999,
            }
        ]
    )

    problems = validate_catalog_coherence(catalog)

    assert len(problems) == 1
    assert "dimensiones" in problems[0]
