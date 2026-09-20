from __future__ import annotations

import polars as pl

from mammo_lejepa.models import BoxSource, CajaMamaria
from mammo_lejepa.vindr.findings import parse_finding_categories, remap_findings

CROP = CajaMamaria(
    x0=100,
    y0=100,
    x1=300,
    y1=300,
    margin_px=25,
    source=BoxSource.OTSU,
    threshold=100.0,
    image_height=500,
    image_width=500,
    area_ratio=0.16,
)


def test_parse_finding_categories_single() -> None:
    assert parse_finding_categories("['Mass']") == ["Mass"]


def test_parse_finding_categories_multiple() -> None:
    assert parse_finding_categories("['Mass', 'Suspicious Calcification']") == [
        "Mass",
        "Suspicious Calcification",
    ]


def test_parse_finding_categories_none_is_empty() -> None:
    assert parse_finding_categories(None) == []


def _findings_df(rows: list[dict[str, object]]) -> pl.DataFrame:
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


def test_rows_without_coordinates_are_dropped_not_nulled() -> None:
    findings = _findings_df(
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

    result = remap_findings(findings, crops={"img_1": CROP})

    assert result.height == 0


def test_remap_shifts_box_into_crop_space() -> None:
    findings = _findings_df(
        [
            {
                "image_id": "img_1",
                "study_id": "study_a",
                "finding_categories": "['Mass']",
                "finding_birads": "BI-RADS 4",
                "xmin": 150.0,
                "ymin": 160.0,
                "xmax": 180.0,
                "ymax": 190.0,
            }
        ]
    )

    result = remap_findings(findings, crops={"img_1": CROP})

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["xmin_crop"] == 50.0  # 150 - 100
    assert row["ymin_crop"] == 60.0  # 160 - 100
    assert row["xmax_crop"] == 80.0
    assert row["ymax_crop"] == 90.0
    assert row["fully_contained"] is True
    assert row["clipped"] is False
    assert row["finding_categories"] == ["Mass"]


def test_partially_outside_box_is_marked_clipped() -> None:
    # crop_width=200; una caja que cruza x=200 en espacio de recorte queda parcial.
    findings = _findings_df(
        [
            {
                "image_id": "img_1",
                "study_id": "study_a",
                "finding_categories": "['Mass']",
                "finding_birads": "BI-RADS 4",
                "xmin": 250.0,
                "ymin": 150.0,
                "xmax": 350.0,
                "ymax": 200.0,
            }
        ]
    )

    result = remap_findings(findings, crops={"img_1": CROP})

    row = result.row(0, named=True)
    assert row["clipped"] is True
    assert row["fully_contained"] is False
    assert row["xmax_crop"] == 200.0  # recortado al límite del crop


def test_fully_outside_box_keeps_fully_contained_false() -> None:
    findings = _findings_df(
        [
            {
                "image_id": "img_1",
                "study_id": "study_a",
                "finding_categories": "['Mass']",
                "finding_birads": "BI-RADS 4",
                "xmin": 1000.0,
                "ymin": 1000.0,
                "xmax": 1050.0,
                "ymax": 1050.0,
            }
        ]
    )

    result = remap_findings(findings, crops={"img_1": CROP})

    row = result.row(0, named=True)
    assert row["fully_contained"] is False
    assert row["clipped"] is True
    assert row["area_crop_space"] == 0.0


def test_findings_without_a_known_crop_are_skipped() -> None:
    findings = _findings_df(
        [
            {
                "image_id": "img_missing_crop",
                "study_id": "study_a",
                "finding_categories": "['Mass']",
                "finding_birads": "BI-RADS 4",
                "xmin": 10.0,
                "ymin": 10.0,
                "xmax": 20.0,
                "ymax": 20.0,
            }
        ]
    )

    result = remap_findings(findings, crops={})

    assert result.height == 0
