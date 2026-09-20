from __future__ import annotations

import json
from collections.abc import Container, Iterable, Mapping, Sequence

from mammo_lejepa.models import (
    BoxSource,
    CajaMamaria,
    FailureCategory,
    FallbackReason,
    ImagenMammoBench,
    RegistroDeRecorte,
)


def parse_jsonl_records(lines: Iterable[str]) -> list[RegistroDeRecorte]:
    """Parsea líneas JSONL a `RegistroDeRecorte`. Una última línea truncada
    (p.ej. tras una interrupción a mitad de escritura) se descarta sin lanzar
    excepción; cualquier otra línea malformada sí propaga el error, porque no
    se explica por una interrupción normal."""
    lines = list(lines)
    records: list[RegistroDeRecorte] = []

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


def _record_from_dict(data: dict) -> RegistroDeRecorte:
    kwargs = dict(data)

    box_data = kwargs.get("box")
    if box_data is not None:
        box_data = dict(box_data)
        box_data["source"] = BoxSource(box_data["source"])
        kwargs["box"] = CajaMamaria(**box_data)

    fallback_reason = kwargs.get("fallback_reason")
    kwargs["fallback_reason"] = (
        FallbackReason(fallback_reason) if fallback_reason is not None else None
    )

    failure_category = kwargs.get("failure_category")
    kwargs["failure_category"] = (
        FailureCategory(failure_category) if failure_category is not None else None
    )

    return RegistroDeRecorte(**kwargs)


def latest_records_by_image(
    records: Iterable[RegistroDeRecorte],
) -> dict[str, RegistroDeRecorte]:
    """Resuelve registros repetidos de una misma imagen quedándose con el más
    reciente por `processed_at` (FR-016)."""
    latest: dict[str, RegistroDeRecorte] = {}

    for record in records:
        current = latest.get(record.image_id)
        if current is None or record.processed_at > current.processed_at:
            latest[record.image_id] = record

    return latest


def pending_tasks(
    manifest: Sequence[ImagenMammoBench],
    completed: Mapping[str, RegistroDeRecorte],
    existing_crops: Container[str],
) -> list[ImagenMammoBench]:
    """Determina el trabajo pendiente descontando del manifiesto las imágenes
    ya resueltas con éxito (FR-024). Una imagen sólo se considera resuelta si
    su registro más reciente es `ok` **y** su `image_id` figura en
    `existing_crops` (el conjunto de imágenes cuyo recorte existe
    físicamente, verificado por el llamador). Las imágenes fallidas vuelven a
    la lista de pendientes salvo en modo reintento explícito."""
    pending: list[ImagenMammoBench] = []

    for task in manifest:
        record = completed.get(task.image_id)
        already_done = (
            record is not None
            and record.status == "ok"
            and task.image_id in existing_crops
        )
        if not already_done:
            pending.append(task)

    return pending


def summarize_failures(
    records: Iterable[RegistroDeRecorte],
) -> dict[FailureCategory, int]:
    """Recuentos por categoría de error, para el resumen final (FR-026)."""
    summary: dict[FailureCategory, int] = {}

    for record in records:
        if record.status == "failed":
            category = record.failure_category or FailureCategory.UNKNOWN
            summary[category] = summary.get(category, 0) + 1

    return summary
