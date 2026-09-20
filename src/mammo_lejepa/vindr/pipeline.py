from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from mammo_lejepa import PIPELINE_VERSION
from mammo_lejepa.storage import append_record
from mammo_lejepa.vindr.config import PhysioNetCredentials, PipelineConfig
from mammo_lejepa.vindr.download import (
    build_client_session,
    download_dicom,
    validate_access,
)
from mammo_lejepa.vindr.errors import AuthError, NetworkError, redact_secrets
from mammo_lejepa.vindr.models import (
    FailureCategory,
    ImageRecord,
    ImageTask,
    ProcessOutcome,
    RunSummary,
)
from mammo_lejepa.vindr.storage import quarantine
from mammo_lejepa.vindr.worker import process_dicom


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class _ReadyItem:
    task: ImageTask
    dicom_path: Path | None
    download_seconds: float
    dicom_bytes: int
    download_error: str | None


@dataclass(slots=True)
class _RunStats:
    images_ok: int = 0
    images_failed: int = 0
    bytes_downloaded: int = 0
    bytes_written: int = 0
    failures_by_category: dict[FailureCategory, int] | None = None

    def __post_init__(self) -> None:
        if self.failures_by_category is None:
            self.failures_by_category = {}


async def run_pipeline(
    tasks: Sequence[ImageTask],
    *,
    config: PipelineConfig,
    credentials: PhysioNetCredentials,
    run_id: str,
    command: str,
    on_result: Callable[[ProcessOutcome], None] | None = None,
    shutdown_event: asyncio.Event | None = None,
    split: str | None = None,
    n_studies: int | None = None,
) -> RunSummary:
    """Orquesta la descarga y el procesado de `tasks`: es el único punto que
    conoce a la vez la red, el pool de procesos y el disco. Garantiza el orden
    PNG → registro → borrado del DICOM, impone la contrapresión de la cola
    acotada (research.md D-06) y devuelve el resumen de la ejecución.
    `on_result`, si se da, se invoca de forma síncrona tras cada imagen
    resuelta (FR-027: progreso en consola). `shutdown_event`, si se marca
    (p.ej. por un manejador de `SIGINT`/`SIGTERM` en `cli.py`), detiene a los
    productores sin cancelar el trabajo ya en vuelo: los consumidores siguen
    drenando la cola hasta vaciarla, escriben sus registros y sólo entonces
    termina la ejecución con `exit_reason="interrupted"`."""
    started_at = _now_iso()
    shutdown_event = shutdown_event or asyncio.Event()

    pending: asyncio.Queue[ImageTask] = asyncio.Queue()
    for task in tasks:
        pending.put_nowait(task)

    ready: asyncio.Queue[_ReadyItem | None] = asyncio.Queue(
        maxsize=max(1, config.queue_size)
    )
    stats = _RunStats()
    auth_errors: list[AuthError] = []

    async with build_client_session(credentials, config) as session:
        await validate_access(session, config)

        with ProcessPoolExecutor(max_workers=max(1, config.workers)) as pool:
            producers = [
                asyncio.create_task(
                    _produce(
                        session, pending, ready, config, auth_errors, shutdown_event
                    )
                )
                for _ in range(max(1, config.downloads))
            ]
            consumers = [
                asyncio.create_task(
                    _consume(ready, pool, config, run_id, stats, on_result, credentials)
                )
                for _ in range(max(1, config.workers))
            ]

            await asyncio.gather(*producers)
            for _ in consumers:
                await ready.put(None)
            await asyncio.gather(*consumers)

    if auth_errors:
        raise auth_errors[0]

    finished_at = _now_iso()

    exit_reason = "interrupted" if shutdown_event.is_set() else "completed"

    return RunSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        command=command,
        pipeline_version=PIPELINE_VERSION,
        split=split,
        n_studies=n_studies,
        downloads=config.downloads,
        workers=config.workers,
        queue_size=config.queue_size,
        margin_px=config.margin_px,
        images_total=len(tasks),
        images_ok=stats.images_ok,
        images_failed=stats.images_failed,
        failures_by_category=dict(stats.failures_by_category or {}),
        bytes_downloaded=stats.bytes_downloaded,
        bytes_written=stats.bytes_written,
        exit_reason=exit_reason,
    )


async def _produce(
    session: object,
    pending: asyncio.Queue[ImageTask],
    ready: asyncio.Queue[_ReadyItem | None],
    config: PipelineConfig,
    auth_errors: list[AuthError],
    shutdown_event: asyncio.Event,
) -> None:
    while True:
        if auth_errors or shutdown_event.is_set():
            return
        try:
            task = pending.get_nowait()
        except asyncio.QueueEmpty:
            return

        destination = config.dicom_dir / task.study_id / f"{task.image_id}.dicom"

        try:
            result = await download_dicom(session, task, destination, config)  # type: ignore[arg-type]
        except AuthError as error:
            auth_errors.append(error)
            return
        except NetworkError as error:
            await ready.put(_ReadyItem(task, None, 0.0, 0, str(error)))
            continue

        await ready.put(
            _ReadyItem(
                task,
                result.dicom_path,
                result.download_seconds,
                result.bytes_downloaded,
                None,
            )
        )


async def _consume(
    ready: asyncio.Queue[_ReadyItem | None],
    pool: ProcessPoolExecutor,
    config: PipelineConfig,
    run_id: str,
    stats: _RunStats,
    on_result: Callable[[ProcessOutcome], None] | None,
    credentials: PhysioNetCredentials,
) -> None:
    loop = asyncio.get_running_loop()

    while True:
        item = await ready.get()
        if item is None:
            return

        if item.download_error is not None:
            outcome = ProcessOutcome(
                status="failed",
                process_seconds=0.0,
                failure_category=FailureCategory.NETWORK,
                error_message=item.download_error,
            )
        else:
            outcome = await loop.run_in_executor(  # pyright: ignore[reportArgumentType]
                pool, process_dicom, item.task, item.dicom_path, config
            )

        outcome = _redact_outcome(outcome, credentials)
        record = _to_image_record(item, outcome, run_id)
        append_record(record, config.images_jsonl_path)  # pyright: ignore[reportArgumentType]

        stats.bytes_downloaded += item.dicom_bytes
        if outcome.status == "ok":
            stats.images_ok += 1
            stats.bytes_written += outcome.png_bytes or 0
        else:
            stats.images_failed += 1
            category = outcome.failure_category or FailureCategory.UNKNOWN
            failures = stats.failures_by_category
            assert failures is not None
            failures[category] = failures.get(category, 0) + 1

        if item.dicom_path is not None:
            if outcome.status == "ok":
                item.dicom_path.unlink(missing_ok=True)
            else:
                quarantine(item.dicom_path, config.quarantine_dir)

        if on_result is not None:
            on_result(outcome)


def _redact_outcome(
    outcome: ProcessOutcome, credentials: PhysioNetCredentials
) -> ProcessOutcome:
    """Sanea `error_message` antes de que llegue a `append_record` (FR-026).
    `worker.py` nunca ve las credenciales (procesa DICOM ya descargados, sin
    red), así que esto es una segunda capa de defensa por si una librería
    externa llegase a incluir texto inesperado en una excepción."""
    if outcome.error_message is None:
        return outcome

    sanitized = redact_secrets(
        outcome.error_message,
        credentials.username,
        credentials.password,
        credentials.session_id,
    )
    if sanitized == outcome.error_message:
        return outcome

    return replace(outcome, error_message=sanitized)


def _to_image_record(
    item: _ReadyItem, outcome: ProcessOutcome, run_id: str
) -> ImageRecord:
    task = item.task
    return ImageRecord(
        study_id=task.study_id,
        series_id=task.series_id,
        image_id=task.image_id,
        status=outcome.status,
        split=task.split,
        laterality=task.laterality,
        view_position=task.view_position,
        breast_birads=task.breast_birads,
        breast_density=task.breast_density,
        run_id=run_id,
        pipeline_version=PIPELINE_VERSION,
        processed_at=_now_iso(),
        download_seconds=item.download_seconds,
        process_seconds=outcome.process_seconds,
        dicom_bytes=item.dicom_bytes,
        breast_crop=outcome.breast_crop,
        photometric_interpretation=outcome.photometric_interpretation,
        transfer_syntax_uid=outcome.transfer_syntax_uid,
        window_center=outcome.window_center,
        window_width=outcome.window_width,
        pixel_spacing=outcome.pixel_spacing,
        manufacturer=outcome.manufacturer,
        model_name=outcome.model_name,
        normalize_low=outcome.normalize_low,
        normalize_high=outcome.normalize_high,
        inverted_monochrome1=outcome.inverted_monochrome1,
        png_path=outcome.png_path,
        png_bytes=outcome.png_bytes,
        png_sha256=outcome.png_sha256,
        failure_category=outcome.failure_category,
        error_message=outcome.error_message,
    )
