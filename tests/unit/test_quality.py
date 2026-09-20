from __future__ import annotations

import polars as pl

from mammo_lejepa.models import (
    BoxSource,
    CajaMamaria,
    QualityParams,
    RegistroDeRecorte,
)
from mammo_lejepa.quality import is_suspect, summarize_by_source

_PARAMS = QualityParams()


def _box(
    *,
    x0: int = 20,
    y0: int = 20,
    x1: int = 180,
    y1: int = 180,
    image_width: int = 200,
    image_height: int = 200,
    area_ratio: float = 0.5,
) -> CajaMamaria:
    return CajaMamaria(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        margin_px=0,
        source=BoxSource.MASK,
        threshold=128.0,
        image_height=image_height,
        image_width=image_width,
        area_ratio=area_ratio,
    )


def _record(
    *,
    source_dataset: str = "inbreast",
    box: CajaMamaria | None = None,
    mask_otsu_iou: float | None = 0.9,
    status: str = "ok",
) -> RegistroDeRecorte:
    return RegistroDeRecorte(
        image_id="img_1",
        source_dataset=source_dataset,
        source_subject_id="subject_a",
        patient_key=f"{source_dataset}:subject_a",
        laterality="L",
        view="CC",
        status=status,
        process_seconds=1.0,
        run_id="run_x",
        code_version="0.1.0",
        processed_at="2026-01-01T00:00:00+00:00",
        box=box if box is not None else _box(),
        mask_otsu_iou=mask_otsu_iou,
    )


def test_well_formed_box_is_not_suspect() -> None:
    record = _record()

    suspect, reason = is_suspect(record, params=_PARAMS)

    assert suspect is False
    assert reason is None


def test_area_ratio_out_of_range_is_suspect() -> None:
    record = _record(box=_box(x0=0, y0=0, x1=5, y1=5, area_ratio=0.05))

    suspect, reason = is_suspect(record, params=_PARAMS)

    assert suspect is True
    assert reason == "area_ratio"


def test_impossible_aspect_ratio_is_suspect() -> None:
    # 190x10 en una imagen de 200x200: relación de aspecto 19.0, fuera de
    # [0.2, 5.0], pero con área dentro de rango para aislar la causa.
    record = _record(
        box=_box(x0=5, y0=90, x1=195, y1=100, area_ratio=0.475),
    )

    suspect, reason = is_suspect(record, params=_PARAMS)

    assert suspect is True
    assert reason == "aspect_ratio"


def test_low_mask_otsu_iou_is_suspect() -> None:
    record = _record(mask_otsu_iou=0.05)

    suspect, reason = is_suspect(record, params=_PARAMS)

    assert suspect is True
    assert reason == "mask_otsu_iou"


def test_box_touching_all_four_edges_is_suspect() -> None:
    # `area_ratio` se fija dentro de rango a propósito, para aislar la causa
    # de sospecha en el borde en vez de en el área (aunque geométricamente
    # una caja que toca los cuatro bordes tendría area_ratio=1.0).
    record = _record(
        box=_box(x0=0, y0=0, x1=200, y1=200, area_ratio=0.5),
        mask_otsu_iou=0.9,
    )

    suspect, reason = is_suspect(record, params=_PARAMS)

    assert suspect is True
    assert reason == "touches_all_edges"


def test_failed_record_without_box_is_not_suspect() -> None:
    record = RegistroDeRecorte(
        image_id="img_1",
        source_dataset="inbreast",
        source_subject_id="subject_a",
        patient_key="inbreast:subject_a",
        laterality="L",
        view="CC",
        status="failed",
        process_seconds=1.0,
        run_id="run_x",
        code_version="0.1.0",
        processed_at="2026-01-01T00:00:00+00:00",
        box=None,
    )

    suspect, reason = is_suspect(record, params=_PARAMS)

    assert suspect is False
    assert reason is None


def test_summarize_by_source_computes_percentiles_per_source() -> None:
    records = [
        _record(source_dataset="inbreast", box=_box(area_ratio=0.2), mask_otsu_iou=0.5),
        _record(source_dataset="inbreast", box=_box(area_ratio=0.4), mask_otsu_iou=0.7),
        _record(source_dataset="inbreast", box=_box(area_ratio=0.6), mask_otsu_iou=0.9),
        _record(source_dataset="ddsm", box=_box(area_ratio=0.3), mask_otsu_iou=0.6),
        _record(source_dataset="ddsm", box=_box(area_ratio=0.5), mask_otsu_iou=0.8),
    ]

    summary = summarize_by_source(records)

    assert isinstance(summary, pl.DataFrame)
    assert set(summary["source_dataset"]) == {"inbreast", "ddsm"}

    inbreast_row = summary.filter(pl.col("source_dataset") == "inbreast")
    assert inbreast_row["n"].item() == 3
    assert abs(inbreast_row["area_ratio_p50"].item() - 0.4) < 1e-9

    ddsm_row = summary.filter(pl.col("source_dataset") == "ddsm")
    assert ddsm_row["n"].item() == 2


def test_summarize_by_source_excludes_failed_records() -> None:
    records = [
        _record(source_dataset="inbreast", status="ok"),
        RegistroDeRecorte(
            image_id="img_2",
            source_dataset="inbreast",
            source_subject_id="subject_b",
            patient_key="inbreast:subject_b",
            laterality="R",
            view="CC",
            status="failed",
            process_seconds=1.0,
            run_id="run_x",
            code_version="0.1.0",
            processed_at="2026-01-01T00:00:00+00:00",
            box=None,
        ),
    ]

    summary = summarize_by_source(records)

    assert summary["n"].item() == 1
