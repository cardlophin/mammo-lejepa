from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from mammo_lejepa.models import BoxSource, QualityParams, RegistroDeRecorte

_PERCENTILES = (0.10, 0.50, 0.90)


def is_suspect(
    record: RegistroDeRecorte, *, params: QualityParams
) -> tuple[bool, str | None]:
    """FR de calidad (spec.md): marca un recorte como sospechoso sin decidir
    qué hacer con él —eso es cosa del entrenamiento—, sólo señala por qué."""
    if record.status != "ok" or record.box is None:
        return False, None

    box = record.box
    if not (
        params.suspect_area_ratio_min <= box.area_ratio <= params.suspect_area_ratio_max
    ):
        return True, "area_ratio"

    width = box.x1 - box.x0
    height = box.y1 - box.y0
    aspect_ratio = width / height
    if not (
        params.suspect_aspect_ratio_min
        <= aspect_ratio
        <= params.suspect_aspect_ratio_max
    ):
        return True, "aspect_ratio"

    if (
        record.mask_otsu_iou is not None
        and record.mask_otsu_iou < params.suspect_iou_threshold
    ):
        return True, "mask_otsu_iou"

    touches_all_edges = (
        box.x0 == 0
        and box.y0 == 0
        and box.x1 == box.image_width
        and box.y1 == box.image_height
    )
    if touches_all_edges:
        return True, "touches_all_edges"

    return False, None


def summarize_by_source(records: Sequence[RegistroDeRecorte]) -> pl.DataFrame:
    """Percentiles de `area_ratio` y `mask_otsu_iou` por fuente, sobre los
    registros correctos únicamente, más el recuento de respaldos a Otsu y de
    sospechosos (T029: alimenta `quality_report.md`/`quality_by_source.parquet`)."""
    rows = []
    for record in records:
        if record.status != "ok" or record.box is None:
            continue
        rows.append(
            {
                "source_dataset": record.source_dataset,
                "area_ratio": record.box.area_ratio,
                "mask_otsu_iou": record.mask_otsu_iou,
                "is_fallback_otsu": record.box.source is BoxSource.OTSU,
                "suspect": record.suspect,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "source_dataset": pl.Utf8,
                "n": pl.UInt32,
                "area_ratio_p10": pl.Float64,
                "area_ratio_p50": pl.Float64,
                "area_ratio_p90": pl.Float64,
                "mask_otsu_iou_p10": pl.Float64,
                "mask_otsu_iou_p50": pl.Float64,
                "mask_otsu_iou_p90": pl.Float64,
                "n_fallback_otsu": pl.UInt32,
                "n_suspect": pl.UInt32,
            }
        )

    frame = pl.DataFrame(rows)

    aggregations = [pl.len().alias("n")]
    for pct in _PERCENTILES:
        pct_label = int(pct * 100)
        aggregations.append(
            pl.col("area_ratio").quantile(pct).alias(f"area_ratio_p{pct_label}")
        )
        aggregations.append(
            pl.col("mask_otsu_iou").quantile(pct).alias(f"mask_otsu_iou_p{pct_label}")
        )
    aggregations.append(pl.col("is_fallback_otsu").sum().alias("n_fallback_otsu"))
    aggregations.append(pl.col("suspect").sum().alias("n_suspect"))

    return frame.group_by("source_dataset").agg(aggregations).sort("source_dataset")
