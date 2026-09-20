from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from mammo_lejepa.models import BoxSource, CajaMamaria
from mammo_lejepa.resume import latest_records_by_image
from mammo_lejepa.vindr.findings import remap_findings
from mammo_lejepa.vindr.models import FailureCategory, ImageRecord


def parse_jsonl_records(lines: Iterable[str]) -> list[ImageRecord]:
    """Parsea líneas JSONL a `ImageRecord` (propio de `vindr/`). El
    `parse_jsonl_records` de `mammo_lejepa.resume` no sirve aquí: reconstruye
    `RegistroDeRecorte`/`CajaMamaria` de Mammo-Bench de forma hardcodeada — es
    la única función de ese módulo que no es genérica por duck-typing
    (`pending_tasks`/`latest_records_by_image`/`summarize_failures` sí lo son
    y se reutilizan sin cambios). Una última línea truncada (p.ej. tras una
    interrupción a mitad de escritura) se descarta sin lanzar excepción;
    cualquier otra línea malformada sí propaga el error."""
    lines = list(lines)
    records: list[ImageRecord] = []

    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise

        records.append(_record_from_dict(data))

    return records


def _record_from_dict(data: dict) -> ImageRecord:
    kwargs = dict(data)

    breast_crop_data = kwargs.get("breast_crop")
    if breast_crop_data is not None:
        breast_crop_data = dict(breast_crop_data)
        breast_crop_data["source"] = BoxSource(breast_crop_data["source"])
        kwargs["breast_crop"] = CajaMamaria(**breast_crop_data)

    failure_category = kwargs.get("failure_category")
    kwargs["failure_category"] = (
        FailureCategory(failure_category) if failure_category is not None else None
    )

    pixel_spacing = kwargs.get("pixel_spacing")
    kwargs["pixel_spacing"] = (
        tuple(pixel_spacing) if pixel_spacing is not None else None
    )

    return ImageRecord(**kwargs)


def consolidate(
    records: Iterable[ImageRecord],
    finding_rows: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Consolida los registros incrementales en `images.parquet` (esquema
    plano de data-model.md, deduplicado por `image_id` quedándose con el
    `processed_at` más reciente — FR-018, idempotente) y `findings.parquet`
    (hallazgos remapeados a partir de las `CajaMamaria` de las imágenes
    resueltas con éxito — FR-016)."""
    latest = latest_records_by_image(records)

    images_df = (
        pl.DataFrame([_flatten_image_record(record) for record in latest.values()])
        if latest
        else pl.DataFrame()
    )

    crops: dict[str, CajaMamaria] = {
        image_id: record.breast_crop
        for image_id, record in latest.items()
        if record.breast_crop is not None
    }
    findings_df = remap_findings(finding_rows, crops)

    return images_df, findings_df


def verify_no_study_leakage(images_df: pl.DataFrame) -> None:
    """Lanza `ValueError` nombrando el `study_id` si alguna imagen de un mismo
    estudio declara una partición (`split`) distinta a la de las demás
    imágenes del mismo estudio (FR-021, SC-003). Comprobación independiente de
    la construcción: el split oficial ya viene a nivel de estudio, pero esto
    lo verifica en vez de asumirlo."""
    if images_df.height == 0:
        return

    per_study_splits = images_df.group_by("study_id").agg(
        pl.col("split").n_unique().alias("n_splits")
    )
    leaking = per_study_splits.filter(pl.col("n_splits") > 1)

    if leaking.height > 0:
        study_ids = sorted(leaking.get_column("study_id").to_list())
        raise ValueError(
            f"Fuga de partición: {len(study_ids)} estudio(s) con más de un "
            f"split entre sus imágenes: {study_ids}"
        )


def validate_catalog_coherence(
    images_df: pl.DataFrame, findings_df: pl.DataFrame
) -> list[str]:
    """Valida la coherencia del catálogo consolidado (FR-019): toda fila `ok`
    tiene su PNG en disco con el tamaño registrado, y toda caja de hallazgo
    remapeada cae dentro de las dimensiones del recorte. Devuelve la lista de
    problemas encontrados (vacía si el catálogo es coherente)."""
    problems: list[str] = []

    if images_df.height == 0:
        return problems

    ok_rows = images_df.filter(pl.col("status") == "ok")
    crop_dims: dict[str, tuple[int, int]] = {}

    for row in ok_rows.iter_rows(named=True):
        image_id = row["image_id"]
        crop_dims[image_id] = (row["crop_width"], row["crop_height"])

        png_path = Path(row["png_path"]) if row["png_path"] else None
        if png_path is None or not png_path.exists():
            problems.append(f"{image_id}: falta el PNG en disco ({png_path}).")
            continue

        actual_size = png_path.stat().st_size
        if actual_size != row["png_bytes"]:
            problems.append(
                f"{image_id}: tamaño de PNG distinto del registrado "
                f"({actual_size} vs {row['png_bytes']})."
            )

    if findings_df.height == 0:
        return problems

    for row in findings_df.iter_rows(named=True):
        dims = crop_dims.get(row["image_id"])
        if dims is None:
            continue

        crop_width, crop_height = dims
        in_bounds = (
            0 <= row["xmin_crop"] <= row["xmax_crop"] <= crop_width
            and 0 <= row["ymin_crop"] <= row["ymax_crop"] <= crop_height
        )
        if not in_bounds:
            problems.append(
                f"{row['finding_id']}: caja remapeada fuera de las dimensiones "
                f"del recorte ({crop_width}x{crop_height})."
            )

    return problems


def _flatten_image_record(record: ImageRecord) -> dict[str, object]:
    crop = record.breast_crop
    return {
        "study_id": record.study_id,
        "series_id": record.series_id,
        "image_id": record.image_id,
        "status": record.status,
        "split": record.split,
        "laterality": record.laterality,
        "view_position": record.view_position,
        "breast_birads": record.breast_birads,
        "breast_density": record.breast_density,
        "png_path": record.png_path,
        "png_bytes": record.png_bytes,
        "png_sha256": record.png_sha256,
        "source_height": crop.image_height if crop else None,
        "source_width": crop.image_width if crop else None,
        "crop_x0": crop.x0 if crop else None,
        "crop_y0": crop.y0 if crop else None,
        "crop_x1": crop.x1 if crop else None,
        "crop_y1": crop.y1 if crop else None,
        "crop_height": (crop.y1 - crop.y0) if crop else None,
        "crop_width": (crop.x1 - crop.x0) if crop else None,
        "crop_margin_px": crop.margin_px if crop else None,
        "crop_area_ratio": crop.area_ratio if crop else None,
        "crop_otsu_threshold": crop.threshold if crop else None,
        "photometric_interpretation": record.photometric_interpretation,
        "transfer_syntax_uid": record.transfer_syntax_uid,
        "window_center": record.window_center,
        "window_width": record.window_width,
        "pixel_spacing": (
            list(record.pixel_spacing) if record.pixel_spacing is not None else None
        ),
        "manufacturer": record.manufacturer,
        "model_name": record.model_name,
        "normalize_low": record.normalize_low,
        "normalize_high": record.normalize_high,
        "inverted_monochrome1": record.inverted_monochrome1,
        "run_id": record.run_id,
        "pipeline_version": record.pipeline_version,
        "processed_at": record.processed_at,
        "download_seconds": record.download_seconds,
        "process_seconds": record.process_seconds,
        "dicom_bytes": record.dicom_bytes,
        "failure_category": (
            record.failure_category.value if record.failure_category else None
        ),
        "error_message": record.error_message,
    }
