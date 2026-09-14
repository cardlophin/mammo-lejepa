from __future__ import annotations

import json
from collections.abc import Container, Iterable, Mapping, Sequence

from mammo_lejepa.models import BreastCrop, FailureCategory, ImageRecord, ImageTask


def parse_jsonl_records(lines: Iterable[str]) -> list[ImageRecord]:
    """Parsea líneas JSONL a `ImageRecord`. Una última línea truncada (p.ej.
    tras una interrupción a mitad de escritura) se descarta sin lanzar
    excepción; cualquier otra línea malformada sí propaga el error, porque no
    se explica por una interrupción normal."""
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
    kwargs["breast_crop"] = (
        BreastCrop(**breast_crop_data) if breast_crop_data is not None else None
    )

    failure_category = kwargs.get("failure_category")
    kwargs["failure_category"] = (
        FailureCategory(failure_category) if failure_category is not None else None
    )

    pixel_spacing = kwargs.get("pixel_spacing")
    kwargs["pixel_spacing"] = (
        tuple(pixel_spacing) if pixel_spacing is not None else None
    )

    return ImageRecord(**kwargs)


def latest_records_by_image(
    records: Iterable[ImageRecord],
) -> dict[str, ImageRecord]:
    """Resuelve registros repetidos de una misma imagen quedándose con el más
    reciente por `processed_at` (FR-026, FR-027)."""
    latest: dict[str, ImageRecord] = {}

    for record in records:
        current = latest.get(record.image_id)
        if current is None or record.processed_at > current.processed_at:
            latest[record.image_id] = record

    return latest


def pending_tasks(
    manifest: Sequence[ImageTask],
    completed: Mapping[str, ImageRecord],
    existing_pngs: Container[str],
) -> list[ImageTask]:
    """Determina el trabajo pendiente descontando del manifiesto las imágenes
    ya resueltas con éxito (FR-027). Una imagen sólo se considera resuelta si
    su registro más reciente es `ok` **y** su `image_id` figura en
    `existing_pngs` (el conjunto de imágenes cuyo PNG existe físicamente,
    verificado por el llamador). Las imágenes fallidas vuelven a la lista de
    pendientes salvo en modo reintento explícito (ver `cli.py retry`)."""
    pending: list[ImageTask] = []

    for task in manifest:
        record = completed.get(task.image_id)
        already_done = (
            record is not None
            and record.status == "ok"
            and task.image_id in existing_pngs
        )
        if not already_done:
            pending.append(task)

    return pending


def summarize_failures(
    records: Iterable[ImageRecord],
) -> dict[FailureCategory, int]:
    """Recuentos por categoría de error, para el resumen final (FR-029)."""
    summary: dict[FailureCategory, int] = {}

    for record in records:
        if record.status == "failed":
            category = record.failure_category or FailureCategory.UNKNOWN
            summary[category] = summary.get(category, 0) + 1

    return summary
