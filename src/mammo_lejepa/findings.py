from __future__ import annotations

import ast
from collections.abc import Mapping

import polars as pl

from mammo_lejepa.geometry import clip_to_crop, to_crop_space
from mammo_lejepa.models import BoundingBox, BreastCrop

_FINDINGS_SCHEMA: dict[str, pl.DataType] = {  # pyright: ignore[reportAssignmentType]
    "finding_id": pl.Utf8,
    "image_id": pl.Utf8,
    "study_id": pl.Utf8,
    "finding_categories": pl.List(pl.Utf8),
    "finding_birads": pl.Utf8,
    "xmin_orig": pl.Float64,
    "ymin_orig": pl.Float64,
    "xmax_orig": pl.Float64,
    "ymax_orig": pl.Float64,
    "xmin_crop": pl.Float64,
    "ymin_crop": pl.Float64,
    "xmax_crop": pl.Float64,
    "ymax_crop": pl.Float64,
    "fully_contained": pl.Boolean,
    "clipped": pl.Boolean,
    "area_orig": pl.Float64,
    "area_crop_space": pl.Float64,
}


def parse_finding_categories(raw: str | None) -> list[str]:
    """Interpreta el literal de lista de Python de `finding_categories`
    (p.ej. `"['Mass']"`) sin usar `eval`."""
    if raw is None:
        return []

    parsed = ast.literal_eval(raw)
    if isinstance(parsed, str):
        return [parsed]
    return list(parsed)


def remap_findings(
    findings: pl.DataFrame,
    crops: Mapping[str, BreastCrop],
) -> pl.DataFrame:
    """Traslada las cajas de hallazgos del espacio del DICOM original al
    espacio del recorte (FR-023). Sólo las filas con coordenadas y cuya
    imagen tiene un `BreastCrop` conocido (procesada con éxito) generan una
    fila; las filas sin caja son imágenes sin lesión anotada y se descartan
    aquí, contabilizadas por el llamador comparando alturas de entrada y
    salida. Un hallazgo cuya intersección con el recorte sea parcial o vacía
    se conserva marcado (`clipped`/`fully_contained`), nunca se elimina
    (FR-024)."""
    with_coords = findings.filter(
        pl.col("xmin").is_not_null()
        & pl.col("ymin").is_not_null()
        & pl.col("xmax").is_not_null()
        & pl.col("ymax").is_not_null()
    )

    counters: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for row in with_coords.iter_rows(named=True):
        image_id = row["image_id"]
        crop = crops.get(image_id)
        if crop is None:
            continue

        counters[image_id] = counters.get(image_id, 0) + 1
        finding_id = f"{image_id}_{counters[image_id]}"

        xmin, ymin, xmax, ymax = (
            float(row["xmin"]),
            float(row["ymin"]),
            float(row["xmax"]),
            float(row["ymax"]),
        )
        original_box = BoundingBox(x0=xmin, y0=ymin, x1=xmax, y1=ymax, space="orig")
        crop_box = to_crop_space(original_box, crop)
        clipped_box, was_clipped = clip_to_crop(crop_box, crop)

        area_orig = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        area_crop_space = max(0.0, clipped_box.x1 - clipped_box.x0) * max(
            0.0, clipped_box.y1 - clipped_box.y0
        )

        rows.append(
            {
                "finding_id": finding_id,
                "image_id": image_id,
                "study_id": row.get("study_id"),
                "finding_categories": parse_finding_categories(
                    row.get("finding_categories")
                ),
                "finding_birads": row.get("finding_birads"),
                "xmin_orig": xmin,
                "ymin_orig": ymin,
                "xmax_orig": xmax,
                "ymax_orig": ymax,
                "xmin_crop": clipped_box.x0,
                "ymin_crop": clipped_box.y0,
                "xmax_crop": clipped_box.x1,
                "ymax_crop": clipped_box.y1,
                "fully_contained": not was_clipped,
                "clipped": was_clipped,
                "area_orig": area_orig,
                "area_crop_space": area_crop_space,
            }
        )

    if not rows:
        return pl.DataFrame(schema=_FINDINGS_SCHEMA)

    return pl.DataFrame(rows, schema=_FINDINGS_SCHEMA)
