from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from mammo_lejepa.image_io import read_grayscale
from mammo_lejepa.models import RegistroDeRecorte
from mammo_lejepa.resume import latest_records_by_image

_CATALOG_SCHEMA: dict[str, pl.DataType] = {
    "image_id": pl.Utf8,
    "source_dataset": pl.Utf8,
    "source_subject_id": pl.Utf8,
    "patient_key": pl.Utf8,
    "laterality": pl.Utf8,
    "view": pl.Utf8,
    "status": pl.Utf8,
    "failure_category": pl.Utf8,
    "error_message": pl.Utf8,
    "crop_path": pl.Utf8,
    "crop_bytes": pl.Int64,
    "image_height": pl.Int64,
    "image_width": pl.Int64,
    "crop_x0": pl.Int64,
    "crop_y0": pl.Int64,
    "crop_x1": pl.Int64,
    "crop_y1": pl.Int64,
    "crop_height": pl.Int64,
    "crop_width": pl.Int64,
    "crop_margin_px": pl.Int64,
    "crop_area_ratio": pl.Float64,
    "box_source": pl.Utf8,
    "fallback_reason": pl.Utf8,
    "threshold": pl.Float64,
    "mask_area_ratio": pl.Float64,
    "mask_otsu_iou": pl.Float64,
    "suspect": pl.Boolean,
    "suspect_reason": pl.Utf8,
    "classification": pl.Utf8,
    "density": pl.Utf8,
    "birads": pl.Utf8,
    "abnormality": pl.Utf8,
    "molecular_subtype": pl.Utf8,
    "subject_age": pl.Utf8,
    "run_id": pl.Utf8,
    "code_version": pl.Utf8,
    "processed_at": pl.Utf8,
    "process_seconds": pl.Float64,
}


def consolidate(records: Iterable[RegistroDeRecorte]) -> pl.DataFrame:
    """Consolida `records.jsonl` en el esquema plano de `catalog.parquet`
    (data-model.md): deduplica por `image_id` quedándose con el `processed_at`
    más reciente (FR-016, idempotente) y aplana `CajaMamaria` a columnas
    `crop_*`. No incluye `split`: se añade al hacer `split` (T035). El
    esquema se fija explícitamente (`_CATALOG_SCHEMA`): con filas `ok` y
    `failed` mezcladas, columnas como `subject_age` alternan entre `None` y
    cadena en filas arbitrarias, y la inferencia automática de polars sobre
    una muestra parcial puede elegir un tipo que no acepta las filas
    siguientes."""
    latest = latest_records_by_image(records)
    if not latest:
        return pl.DataFrame(schema=_CATALOG_SCHEMA)

    rows = [_flatten_record(record) for record in latest.values()]
    return pl.DataFrame(rows, schema=_CATALOG_SCHEMA)


def _flatten_record(record: RegistroDeRecorte) -> dict[str, object]:
    box = record.box
    return {
        "image_id": record.image_id,
        "source_dataset": record.source_dataset,
        "source_subject_id": record.source_subject_id,
        "patient_key": record.patient_key,
        "laterality": record.laterality,
        "view": record.view,
        "status": record.status,
        "failure_category": (
            record.failure_category.value if record.failure_category else None
        ),
        "error_message": record.error_message,
        "crop_path": record.crop_path,
        "crop_bytes": record.crop_bytes,
        "image_height": box.image_height if box else None,
        "image_width": box.image_width if box else None,
        "crop_x0": box.x0 if box else None,
        "crop_y0": box.y0 if box else None,
        "crop_x1": box.x1 if box else None,
        "crop_y1": box.y1 if box else None,
        "crop_height": (box.y1 - box.y0) if box else None,
        "crop_width": (box.x1 - box.x0) if box else None,
        "crop_margin_px": box.margin_px if box else None,
        "crop_area_ratio": box.area_ratio if box else None,
        "box_source": box.source.value if box else None,
        "fallback_reason": (
            record.fallback_reason.value if record.fallback_reason else None
        ),
        "threshold": box.threshold if box else None,
        "mask_area_ratio": record.mask_area_ratio,
        "mask_otsu_iou": record.mask_otsu_iou,
        "suspect": record.suspect,
        "suspect_reason": record.suspect_reason,
        "classification": record.classification,
        "density": record.density,
        "birads": record.birads,
        "abnormality": record.abnormality,
        "molecular_subtype": record.molecular_subtype,
        "subject_age": record.subject_age,
        "run_id": record.run_id,
        "code_version": record.code_version,
        "processed_at": record.processed_at,
        "process_seconds": record.process_seconds,
    }


def validate_catalog_coherence(catalog: pl.DataFrame) -> list[str]:
    """Valida que toda fila `ok` tenga su PNG en disco con el tamaño
    registrado y dimensiones que coincidan con la caja (T037). Devuelve la
    lista de problemas encontrados (vacía si el catálogo es coherente)."""
    problems: list[str] = []
    if catalog.height == 0:
        return problems

    ok_rows = catalog.filter(pl.col("status") == "ok")
    for row in ok_rows.iter_rows(named=True):
        image_id = row["image_id"]
        crop_path = row["crop_path"]
        if not crop_path:
            problems.append(f"{image_id}: sin crop_path registrado.")
            continue

        path = Path(crop_path)
        if not path.exists():
            problems.append(f"{image_id}: falta el PNG en disco ({crop_path}).")
            continue

        actual_bytes = path.stat().st_size
        if actual_bytes != row["crop_bytes"]:
            problems.append(
                f"{image_id}: tamaño de PNG distinto del registrado "
                f"({actual_bytes} vs {row['crop_bytes']})."
            )

        actual_height, actual_width = read_grayscale(path).shape
        if (actual_height, actual_width) != (row["crop_height"], row["crop_width"]):
            problems.append(
                f"{image_id}: dimensiones del PNG "
                f"({actual_width}x{actual_height}) no coinciden con la caja "
                f"({row['crop_width']}x{row['crop_height']})."
            )

    return problems
