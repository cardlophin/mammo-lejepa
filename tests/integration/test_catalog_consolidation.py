from __future__ import annotations

import polars as pl

from mammo_lejepa.catalog import consolidate
from mammo_lejepa.models import BreastCrop, ImageRecord


def _crop() -> BreastCrop:
    return BreastCrop(
        x0_orig=10,
        y0_orig=10,
        x1_orig=110,
        y1_orig=110,
        margin_px=25,
        otsu_threshold=100.0,
        source_height=200,
        source_width=200,
        crop_height=100,
        crop_width=100,
        area_ratio=0.25,
    )


def _record(
    image_id: str,
    *,
    status: str = "ok",
    processed_at: str = "2026-01-01T00:00:00+00:00",
    crop: BreastCrop | None = None,
) -> ImageRecord:
    return ImageRecord(
        study_id="study_a",
        series_id="study_a-series",
        image_id=image_id,
        status=status,
        split="training",
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
        breast_crop=crop if crop is not None else (_crop() if status == "ok" else None),
        png_path=f"processed/{image_id}.png" if status == "ok" else None,
        png_bytes=1234 if status == "ok" else None,
    )


def _findings(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "image_id": pl.Utf8,
        "study_id": pl.Utf8,
        "finding_categories": pl.Utf8,
        "finding_birads": pl.Utf8,
        "xmin": pl.Float64,
        "ymin": pl.Float64,
        "xmax": pl.Float64,
        "ymax": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def test_consolidation_is_idempotent() -> None:
    records = [_record("img_1"), _record("img_2")]
    findings = _findings([])

    images_a, findings_a = consolidate(records, findings)
    images_b, findings_b = consolidate(records, findings)

    assert images_a.sort("image_id").equals(images_b.sort("image_id"))
    assert findings_a.equals(findings_b)


def test_duplicate_records_keep_the_most_recent() -> None:
    older = _record("img_1", status="failed", processed_at="2026-01-01T00:00:00+00:00")
    newer = _record("img_1", status="ok", processed_at="2026-01-02T00:00:00+00:00")

    images_df, _ = consolidate([older, newer], _findings([]))

    assert images_df.height == 1
    assert images_df.row(0, named=True)["status"] == "ok"


def test_images_without_findings_are_absent_from_findings_parquet() -> None:
    records = [_record("img_1")]
    # finding_annotations.csv real: una fila por imagen aunque no tenga lesión.
    findings = _findings(
        [
            {
                "image_id": "img_1",
                "study_id": "study_a",
                "finding_categories": None,
                "finding_birads": None,
                "xmin": None,
                "ymin": None,
                "xmax": None,
                "ymax": None,
            }
        ]
    )

    images_df, findings_df = consolidate(records, findings)

    assert images_df.height == 1
    assert findings_df.height == 0


def test_images_parquet_flattens_breast_crop_with_prefix() -> None:
    images_df, _ = consolidate([_record("img_1")], _findings([]))

    row = images_df.row(0, named=True)
    assert row["crop_x0"] == 10
    assert row["crop_width"] == 100
    assert row["crop_otsu_threshold"] == 100.0


def test_failed_images_have_null_crop_fields() -> None:
    images_df, _ = consolidate([_record("img_1", status="failed")], _findings([]))

    row = images_df.row(0, named=True)
    assert row["crop_x0"] is None
    assert row["status"] == "failed"
