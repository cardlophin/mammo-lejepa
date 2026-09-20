from __future__ import annotations

import contextlib
import signal
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime

from mammo_lejepa import CORPUS_VERSION
from mammo_lejepa.config import CorpusConfig
from mammo_lejepa.models import (
    BoxSource,
    FailureCategory,
    ImagenMammoBench,
    RegistroDeRecorte,
    RunSummary,
)
from mammo_lejepa.resume import (
    latest_records_by_image,
    parse_jsonl_records,
    pending_tasks,
)
from mammo_lejepa.storage import append_record
from mammo_lejepa.worker import process_image


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _pending_manifest(
    manifest: Sequence[ImagenMammoBench], config: CorpusConfig
) -> Sequence[ImagenMammoBench]:
    """FR-024 (T023): descuenta del manifiesto las imágenes ya resueltas con
    éxito cuyo recorte existe físicamente, para que relanzar `build` sobre el
    mismo directorio de salida continúe en vez de repetir desde cero."""
    if not config.records_jsonl_path.exists():
        return manifest

    lines = config.records_jsonl_path.read_text(encoding="utf-8").splitlines()
    completed = latest_records_by_image(parse_jsonl_records(lines))
    existing_crops = {
        image_id
        for image_id, record in completed.items()
        if record.status == "ok"
        and config.crop_path(record.source_dataset, image_id).exists()
    }
    return pending_tasks(manifest, completed, existing_crops)


def run_build(
    manifest: Sequence[ImagenMammoBench],
    *,
    config: CorpusConfig,
    run_id: str,
    command: str = "build",
    on_result: Callable[[RegistroDeRecorte], None] | None = None,
) -> RunSummary:
    """Reparte lo pendiente de `manifest` (tras descontar la reanudación,
    FR-024) entre `config.workers` procesos (FR-025), recoge los
    `RegistroDeRecorte` conforme llegan y los escribe en el JSONL con vaciado
    — escritor único, el proceso principal (FR-013) — manteniendo el
    progreso. Atiende `SIGINT`/`SIGTERM`: deja de encolar trabajo nuevo, deja
    terminar lo que ya está en marcha y sale con el JSONL consistente (nada a
    medio escribir); nunca hace falta reprocesar desde cero tras un corte."""
    manifest = _pending_manifest(manifest, config)

    started_at = _now_iso()
    shutdown_requested = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    previous_handlers: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        # p.ej. no es el hilo principal, o la plataforma no lo soporta.
        with contextlib.suppress(ValueError, OSError):
            previous_handlers[sig] = signal.signal(sig, _handle_signal)

    n_ok = 0
    n_failed = 0
    n_fallback_otsu = 0
    failures_by_category: dict[FailureCategory, int] = {}

    try:
        with ProcessPoolExecutor(max_workers=max(1, config.workers)) as pool:
            futures = {
                pool.submit(process_image, image, config): image for image in manifest
            }

            for future in as_completed(futures):
                record = future.result()
                record = replace(
                    record,
                    run_id=run_id,
                    code_version=CORPUS_VERSION,
                    processed_at=_now_iso(),
                )
                append_record(record, config.records_jsonl_path)

                if record.status == "ok":
                    n_ok += 1
                    if record.box is not None and record.box.source is BoxSource.OTSU:
                        n_fallback_otsu += 1
                else:
                    n_failed += 1
                    category = record.failure_category or FailureCategory.UNKNOWN
                    failures_by_category[category] = (
                        failures_by_category.get(category, 0) + 1
                    )

                if on_result is not None:
                    on_result(record)

                if shutdown_requested:
                    for pending in futures:
                        pending.cancel()
                    break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]

    exit_reason = "interrupted" if shutdown_requested else "completed"

    return RunSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=_now_iso(),
        command=command,
        code_version=CORPUS_VERSION,
        seed=config.seed,
        n_total=len(manifest),
        n_ok=n_ok,
        n_failed=n_failed,
        failures_by_category=failures_by_category,
        n_fallback_otsu=n_fallback_otsu,
        exit_reason=exit_reason,
    )
