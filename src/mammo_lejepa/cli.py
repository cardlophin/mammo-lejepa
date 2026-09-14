from __future__ import annotations

import asyncio
import json
import shutil
import signal
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import pydicom
import typer
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, generate_uid
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mammo_lejepa import PIPELINE_VERSION
from mammo_lejepa.catalog import consolidate as consolidate_catalog
from mammo_lejepa.catalog import validate_catalog_coherence
from mammo_lejepa.config import PhysioNetCredentials, PipelineConfig, load_credentials
from mammo_lejepa.download import build_client_session, validate_access
from mammo_lejepa.errors import AuthError
from mammo_lejepa.manifest import (
    batch_by_study,
    build_manifest,
    count_metadata_discrepancies,
)
from mammo_lejepa.models import (
    FailureCategory,
    ImageRecord,
    ImageTask,
    ProcessOutcome,
    RunSummary,
)
from mammo_lejepa.pipeline import run_pipeline
from mammo_lejepa.resume import (
    latest_records_by_image,
    parse_jsonl_records,
    pending_tasks,
)
from mammo_lejepa.storage import (
    acquire_output_lock,
    append_run_summary,
    clean_orphan_part_files,
)

app = typer.Typer(
    help="mammo-etl: ETL en streaming de VinDr-Mammo a recortes mamarios catalogados.",
    no_args_is_help=True,
)
console = Console()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@app.command()
def doctor() -> None:
    """Comprueba el entorno sin descargar ninguna imagen (D-01): CSV,
    decodificador JPEG 2000, permisos de escritura, espacio libre frente al
    techo de disco calculado y validez de la cookie de PhysioNet."""
    config = PipelineConfig()
    healthy = True

    for label, path in (
        ("breast-level_annotations.csv", config.breast_level_annotations_path),
        ("finding_annotations.csv", config.finding_annotations_path),
        ("metadata.csv", config.metadata_path),
    ):
        if path.exists():
            console.print(f"[green]OK[/green]    {label} presente en {path}")
        else:
            console.print(f"[red]FALTA[/red]  {label} ausente en {path}")
            healthy = False

    try:
        _check_jpeg2000_decoder()
        console.print(
            "[green]OK[/green]    Decodificador JPEG 2000 (pylibjpeg-openjpeg) "
            "funcional."
        )
    except Exception as error:
        console.print(f"[red]FALTA[/red]  Decodificador JPEG 2000: {error}")
        healthy = False

    try:
        _check_writable(config)
        console.print(
            f"[green]OK[/green]    Permisos de escritura en {config.data_dir}"
        )
    except OSError as error:
        console.print(f"[red]FALTA[/red]  Escritura en {config.data_dir}: {error}")
        healthy = False

    probe_dir = config.data_dir if config.data_dir.exists() else Path.cwd()
    free_bytes = shutil.disk_usage(probe_dir).free
    ceiling = config.disk_ceiling_bytes
    if free_bytes >= ceiling:
        console.print(
            f"[green]OK[/green]    Espacio libre {free_bytes / 1024**3:.1f} GiB "
            f">= techo de disco calculado {ceiling / 1024**2:.0f} MiB."
        )
    else:
        console.print(
            f"[red]FALTA[/red]  Espacio libre {free_bytes / 1024**3:.1f} GiB "
            f"< techo de disco calculado {ceiling / 1024**2:.0f} MiB."
        )
        healthy = False

    credentials: PhysioNetCredentials | None
    try:
        credentials = load_credentials()
        # FR-031 / constitución: ni usuario, ni contraseña, ni cookie se
        # imprimen jamás, sólo la confirmación de que se cargaron.
        console.print("[green]OK[/green]    Credenciales cargadas desde el entorno.")
    except RuntimeError as error:
        console.print(f"[red]FALTA[/red]  {error}")
        healthy = False
        credentials = None

    if credentials is not None:
        try:
            asyncio.run(_check_cookie_validity(credentials, config))
            console.print(
                "[green]OK[/green]    Cookie sessionid válida (acceso confirmado)."
            )
        except AuthError as error:
            console.print(f"[red]FALTA[/red]  Cookie inválida: {error}")
            healthy = False
        except Exception as error:
            console.print(
                f"[yellow]AVISO[/yellow]  No se pudo validar la cookie: {error}"
            )

    if not healthy:
        raise typer.Exit(code=1)

    console.print("\n[bold green]Entorno listo.[/bold green]")


def _check_jpeg2000_decoder() -> None:
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage  # pyright: ignore[reportAttributeAccessIssue]
    dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.SOPClassUID = dataset.file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.Rows = 64
    dataset.Columns = 64
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    pixels = np.random.default_rng(0).integers(0, 4000, size=(64, 64)).astype(np.uint16)
    dataset.PixelData = pixels.tobytes()

    dataset.compress(JPEG2000Lossless)
    _ = dataset.pixel_array


def _check_writable(config: PipelineConfig) -> None:
    for directory in (
        config.dicom_dir,
        config.processed_dir,
        config.catalog_dir,
        config.quarantine_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("ok")
        probe.unlink()


async def _check_cookie_validity(
    credentials: PhysioNetCredentials, config: PipelineConfig
) -> None:
    async with build_client_session(credentials, config) as session:
        await validate_access(session, config)


@app.command()
def run(
    split: str | None = typer.Option(None, help="training o test"),
    n_studies: int | None = typer.Option(None, "--n-studies"),
    study_id: list[str] = typer.Option(
        [], "--study-id", help="Repetible: acota a estos study_id"
    ),
    downloads: int = typer.Option(6, help="Descargas simultáneas"),
    workers: int = typer.Option(4, help="Workers de procesado de imagen"),
    queue_size: int = typer.Option(12, "--queue-size"),
    margin_px: int = typer.Option(25, "--margin-px"),
) -> None:
    """Descarga y procesa el dataset (o un subconjunto acotado) en streaming."""
    config = PipelineConfig(
        downloads=downloads,
        workers=workers,
        queue_size=queue_size,
        margin_px=margin_px,
    )

    credentials = _require_credentials()
    clean_orphan_part_files(config.dicom_dir)
    _print_disk_ceiling(config)

    if not config.breast_level_annotations_path.exists():
        console.print(
            f"[red]No existe {config.breast_level_annotations_path}.[/red] "
            "Descarga primero los CSV de anotaciones."
        )
        raise typer.Exit(code=1)

    annotations = pl.read_csv(config.breast_level_annotations_path)

    try:
        manifest = build_manifest(
            annotations,
            split=split,
            study_ids=study_id or None,
            n_studies=n_studies,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    _warn_metadata_discrepancies(config, annotations)

    tasks = [task for batch in batch_by_study(manifest) for task in batch]

    if not tasks:
        console.print(
            "[yellow]No hay ninguna imagen que procesar con estos filtros.[/yellow]"
        )
        raise typer.Exit(code=0)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    console.print(f"run_id: {run_id} — {len(tasks)} imagen(es) a procesar.")

    config.catalog_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(config.catalog_dir / f"manifest_{run_id}.parquet")

    try:
        with acquire_output_lock(config.catalog_dir):
            summary = _execute_pipeline(
                tasks, config, credentials, run_id, "run", split, n_studies
            )
    except RuntimeError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if summary.images_failed:
        raise typer.Exit(code=1)


@app.command()
def resume() -> None:
    """Reanuda la última ejecución sobre el mismo directorio de salida:
    descuenta lo ya resuelto con éxito (verificando que su PNG existe
    realmente) y descarga únicamente lo pendiente (FR-027)."""
    config = PipelineConfig()
    credentials = _require_credentials()

    last_run_id = _find_last_run_id(config)
    if last_run_id is None:
        console.print(
            "[red]No hay ninguna ejecución previa que reanudar "
            f"({config.runs_jsonl_path} no existe o está vacío).[/red]"
        )
        raise typer.Exit(code=1)

    manifest_path = config.catalog_dir / f"manifest_{last_run_id}.parquet"
    if not manifest_path.exists():
        console.print(
            f"[red]No se encuentra el manifiesto de la última ejecución: "
            f"{manifest_path}.[/red]"
        )
        raise typer.Exit(code=1)

    clean_orphan_part_files(config.dicom_dir)
    _print_disk_ceiling(config)

    manifest = pl.read_parquet(manifest_path)
    all_tasks = [task for batch in batch_by_study(manifest) for task in batch]

    completed = _load_completed_records(config)
    existing_pngs = _existing_png_ids(config)
    tasks = pending_tasks(all_tasks, completed, existing_pngs)

    console.print(
        f"Reanudando {last_run_id}: {len(all_tasks) - len(tasks)} imagen(es) ya "
        f"resueltas, {len(tasks)} pendiente(s)."
    )

    if not tasks:
        console.print("[green]No queda ningún trabajo pendiente.[/green]")
        raise typer.Exit(code=0)

    run_id = f"run_{uuid.uuid4().hex[:12]}"

    try:
        with acquire_output_lock(config.catalog_dir):
            summary = _execute_pipeline(
                tasks, config, credentials, run_id, "resume", None, None
            )
    except RuntimeError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if summary.images_failed:
        raise typer.Exit(code=1)


def _require_credentials() -> PhysioNetCredentials:
    try:
        return load_credentials()
    except RuntimeError as error:
        console.print(f"[red]Error de credenciales:[/red] {error}")
        raise typer.Exit(code=1) from error


def _print_disk_ceiling(config: PipelineConfig) -> None:
    console.print(
        f"Techo de disco intermedio calculado: "
        f"{config.disk_ceiling_bytes / 1024**2:.0f} MiB "
        f"(downloads={config.downloads}, queue_size={config.queue_size}, "
        f"workers={config.workers})."
    )


def _warn_metadata_discrepancies(
    config: PipelineConfig, annotations: pl.DataFrame
) -> None:
    if not config.metadata_path.exists():
        return

    # infer_schema_length=0: sólo necesitamos "SOP Instance UID" como texto;
    # otras columnas de metadata.csv (p.ej. "Window Center") mezclan enteros
    # y listas y rompen la inferencia automática de tipos de Polars.
    metadata = pl.read_csv(config.metadata_path, infer_schema_length=0)
    only_in_metadata, only_in_annotations = count_metadata_discrepancies(
        metadata, annotations
    )
    if only_in_metadata or only_in_annotations:
        console.print(
            f"[yellow]Aviso (FR-032):[/yellow] {only_in_metadata} imagen(es) en "
            f"metadata.csv sin fila en breast-level_annotations.csv; "
            f"{only_in_annotations} en sentido contrario."
        )


def _find_last_run_id(config: PipelineConfig) -> str | None:
    if not config.runs_jsonl_path.exists():
        return None

    lines = config.runs_jsonl_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        run_id = data.get("run_id")
        if run_id:
            return str(run_id)

    return None


def _load_completed_records(config: PipelineConfig) -> dict[str, ImageRecord]:
    if not config.images_jsonl_path.exists():
        return {}
    lines = config.images_jsonl_path.read_text(encoding="utf-8").splitlines()
    return latest_records_by_image(parse_jsonl_records(lines))


def _existing_png_ids(config: PipelineConfig) -> set[str]:
    if not config.processed_dir.exists():
        return set()
    return {png_path.stem for png_path in config.processed_dir.rglob("*.png")}


def _execute_pipeline(
    tasks: list[ImageTask],
    config: PipelineConfig,
    credentials: PhysioNetCredentials,
    run_id: str,
    command: str,
    split: str | None,
    n_studies: int | None,
) -> RunSummary:
    """Núcleo compartido por `run` y `resume`: escribe `runs.jsonl` al inicio
    y al final (T039), lanza el pipeline con barra de progreso y cierre
    ordenado ante `SIGINT`/`SIGTERM` (T037), y traduce `AuthError` en un
    mensaje que explica cómo renovar la cookie sin perder el trabajo ya hecho
    (T038, FR-009)."""
    start_marker = RunSummary(
        run_id=run_id,
        started_at=_now_iso(),
        finished_at="",
        command=command,
        pipeline_version=PIPELINE_VERSION,
        split=split,
        n_studies=n_studies,
        downloads=config.downloads,
        workers=config.workers,
        queue_size=config.queue_size,
        margin_px=config.margin_px,
        images_total=len(tasks),
        images_ok=0,
        images_failed=0,
        failures_by_category={},
        bytes_downloaded=0,
        bytes_written=0,
        exit_reason="started",
    )
    append_run_summary(start_marker, config.runs_jsonl_path)

    try:
        summary = _run_with_progress(
            tasks, config, credentials, run_id, command, split, n_studies
        )
    except AuthError as error:
        console.print(
            f"[red]Sesión de PhysioNet inválida:[/red] {error}\n"
            "[yellow]El trabajo ya completado está a salvo.[/yellow] Renueva la "
            "cookie sessionid en .env y vuelve a lanzar `mammo-etl resume`."
        )
        raise typer.Exit(code=1) from error

    append_run_summary(summary, config.runs_jsonl_path)
    _print_summary(summary, config)
    return summary


def _run_with_progress(
    tasks: list[ImageTask],
    config: PipelineConfig,
    credentials: PhysioNetCredentials,
    run_id: str,
    command: str,
    split: str | None,
    n_studies: int | None,
) -> RunSummary:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    start_time = perf_counter()
    totals = {"bytes": 0, "failures": 0}

    with progress:
        progress_task = progress.add_task("Procesando", total=len(tasks))

        def on_result(outcome: ProcessOutcome) -> None:
            totals["bytes"] += outcome.png_bytes or 0
            if outcome.status != "ok":
                totals["failures"] += 1
            elapsed = max(perf_counter() - start_time, 1e-6)
            rate_mib_s = (totals["bytes"] / 1024**2) / elapsed
            progress.update(
                progress_task,
                advance=1,
                description=(
                    f"Procesando ({rate_mib_s:.1f} MiB/s, fallos: {totals['failures']})"
                ),
            )

        async def _go() -> RunSummary:
            loop = asyncio.get_running_loop()
            shutdown_event = asyncio.Event()
            registered: list[signal.Signals] = []

            def _on_signal() -> None:
                if not shutdown_event.is_set():
                    console.print(
                        "\n[yellow]Interrupción recibida: dejando de encolar "
                        "nuevas descargas, drenando lo que ya está en vuelo…"
                        "[/yellow]"
                    )
                shutdown_event.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _on_signal)
                    registered.append(sig)
                except NotImplementedError:
                    pass  # plataformas sin add_signal_handler (p.ej. Windows)

            try:
                return await run_pipeline(
                    tasks,
                    config=config,
                    credentials=credentials,
                    run_id=run_id,
                    command=command,
                    on_result=on_result,
                    shutdown_event=shutdown_event,
                    split=split,
                    n_studies=n_studies,
                )
            finally:
                for sig in registered:
                    loop.remove_signal_handler(sig)

        return asyncio.run(_go())


def _print_summary(summary: RunSummary, config: PipelineConfig) -> None:
    console.print(
        f"\n[bold]Resumen[/bold] ({summary.exit_reason}): {summary.images_ok} ok, "
        f"{summary.images_failed} fallidas, de {summary.images_total} totales."
    )
    for category, count in summary.failures_by_category.items():
        console.print(f"  {category.value}: {count}")

    if summary.images_failed:
        quarantined = sum(1 for _ in config.quarantine_dir.rglob("*.dicom"))
        console.print(
            f"Registro de fallos: {config.images_jsonl_path} "
            f'(status="failed"). DICOM en cuarentena: {quarantined}.'
        )


@app.command()
def retry(
    failed: bool = typer.Option(False, "--failed", help="Reintenta sólo lo fallido"),
    category: str | None = typer.Option(
        None,
        "--category",
        help="Acota a una categoría de fallo (NETWORK, AUTH, DECODE, "
        "SEGMENTATION, WRITE, UNKNOWN)",
    ),
) -> None:
    """Reintenta selectivamente las imágenes marcadas como fallidas (FR-030).
    Sus registros previos se sustituyen por el resultado nuevo."""
    if not failed:
        console.print("[red]Usa `retry --failed`: es el único modo soportado.[/red]")
        raise typer.Exit(code=1)

    category_filter: FailureCategory | None = None
    if category is not None:
        try:
            category_filter = FailureCategory(category.upper())
        except ValueError as error:
            valid = ", ".join(c.value for c in FailureCategory)
            console.print(
                f"[red]Categoría {category!r} desconocida. Válidas: {valid}.[/red]"
            )
            raise typer.Exit(code=1) from error

    config = PipelineConfig()
    credentials = _require_credentials()

    last_run_id = _find_last_run_id(config)
    if last_run_id is None:
        console.print(
            f"[red]No hay ninguna ejecución previa "
            f"({config.runs_jsonl_path} no existe o está vacío).[/red]"
        )
        raise typer.Exit(code=1)

    manifest_path = config.catalog_dir / f"manifest_{last_run_id}.parquet"
    if not manifest_path.exists():
        console.print(f"[red]No se encuentra el manifiesto: {manifest_path}.[/red]")
        raise typer.Exit(code=1)

    manifest = pl.read_parquet(manifest_path)
    all_tasks = [task for batch in batch_by_study(manifest) for task in batch]

    completed = _load_completed_records(config)
    failed_ids = {
        image_id
        for image_id, record in completed.items()
        if record.status == "failed"
        and (category_filter is None or record.failure_category == category_filter)
    }
    tasks = [task for task in all_tasks if task.image_id in failed_ids]

    if not tasks:
        console.print("[green]No hay ninguna imagen fallida que reintentar.[/green]")
        raise typer.Exit(code=0)

    console.print(f"Reintentando {len(tasks)} imagen(es) fallida(s).")
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    try:
        with acquire_output_lock(config.catalog_dir):
            summary = _execute_pipeline(
                tasks, config, credentials, run_id, "retry", None, None
            )
    except RuntimeError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if summary.images_failed:
        raise typer.Exit(code=1)


@app.command()
def consolidate() -> None:
    """Compacta los registros incrementales de `images.jsonl` en
    `images.parquet` y `findings.parquet` (FR-025, FR-026)."""
    config = PipelineConfig()

    if not config.images_jsonl_path.exists():
        console.print(
            f"[red]No existe {config.images_jsonl_path}.[/red] Ejecuta `run` primero."
        )
        raise typer.Exit(code=1)

    if not config.finding_annotations_path.exists():
        console.print(f"[red]No existe {config.finding_annotations_path}.[/red]")
        raise typer.Exit(code=1)

    lines = config.images_jsonl_path.read_text(encoding="utf-8").splitlines()
    records = parse_jsonl_records(lines)
    finding_rows = pl.read_csv(config.finding_annotations_path)

    images_df, findings_df = consolidate_catalog(records, finding_rows)

    config.catalog_dir.mkdir(parents=True, exist_ok=True)
    images_df.write_parquet(config.images_parquet_path)
    findings_df.write_parquet(config.findings_parquet_path)

    total_finding_rows = finding_rows.height
    with_coords = finding_rows.filter(pl.col("xmin").is_not_null()).height
    without_coords = total_finding_rows - with_coords
    without_known_crop = with_coords - findings_df.height

    console.print(f"[green]images.parquet[/green]: {images_df.height} fila(s).")
    console.print(f"[green]findings.parquet[/green]: {findings_df.height} fila(s).")
    console.print(
        f"Discrepancias: {without_coords} fila(s) de "
        "finding_annotations.csv sin coordenadas (imagen sin lesión anotada, "
        f"no es un error) y {without_known_crop} con coordenadas pero sin "
        "imagen procesada con éxito conocida."
    )

    problems = validate_catalog_coherence(images_df, findings_df)
    if problems:
        console.print(f"[red]{len(problems)} problema(s) de coherencia:[/red]")
        for problem in problems[:20]:
            console.print(f"  - {problem}")
        if len(problems) > 20:
            console.print(f"  ... y {len(problems) - 20} más.")
        raise typer.Exit(code=1)

    console.print("[green]Catálogo coherente.[/green]")


@app.command()
def inspect(
    n: int = typer.Option(8, help="Número de imágenes a incluir en la lámina"),
    output: Path = typer.Option(
        Path("data/vindr-mammo/catalog/qc_grid.png"), "--output"
    ),
) -> None:
    """Genera una lámina de control de calidad: DICOM original, caja
    detectada y recorte con los hallazgos remapeados superpuestos (SC-001,
    SC-006). El DICOM de cada imagen inspeccionada ya se borró tras
    procesarse (FR-017); se vuelve a descargar temporalmente sólo para esta
    inspección y se elimina de nuevo al terminar."""
    config = PipelineConfig()

    if not config.images_parquet_path.exists():
        console.print(
            f"[red]No existe {config.images_parquet_path}.[/red] Ejecuta "
            "`consolidate` primero."
        )
        raise typer.Exit(code=1)

    images_df = pl.read_parquet(config.images_parquet_path)
    ok_images = images_df.filter(pl.col("status") == "ok").head(n)

    if ok_images.height == 0:
        console.print(
            "[yellow]No hay ninguna imagen con status=ok en el catálogo.[/yellow]"
        )
        raise typer.Exit(code=1)

    findings_df = (
        pl.read_parquet(config.findings_parquet_path)
        if config.findings_parquet_path.exists()
        else None
    )

    credentials = _require_credentials()

    try:
        panels = asyncio.run(
            _collect_inspection_panels(ok_images, findings_df, credentials, config)
        )
    except AuthError as error:
        console.print(f"[red]Sesión de PhysioNet inválida:[/red] {error}")
        raise typer.Exit(code=1) from error

    _render_qc_grid(panels, output)
    console.print(f"[green]Lámina de control generada en {output}.[/green]")


async def _collect_inspection_panels(
    ok_images: pl.DataFrame,
    findings_df: pl.DataFrame | None,
    credentials: PhysioNetCredentials,
    config: PipelineConfig,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    import cv2

    from mammo_lejepa.dicom_io import read_dicom
    from mammo_lejepa.download import download_dicom
    from mammo_lejepa.geometry import crop_image
    from mammo_lejepa.models import BreastCrop
    from mammo_lejepa.windowing import apply_display_window, normalize_to_uint8

    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    async with build_client_session(credentials, config) as session:
        await validate_access(session, config)

        for row in ok_images.iter_rows(named=True):
            task = ImageTask(
                study_id=row["study_id"],
                series_id=row["series_id"],
                image_id=row["image_id"],
                split=row["split"],
                laterality=row["laterality"],
                view_position=row["view_position"],
                breast_birads=row["breast_birads"],
                breast_density=row["breast_density"],
                expected_height=row["source_height"],
                expected_width=row["source_width"],
            )
            temp_path = (
                config.dicom_dir / "_inspect" / task.study_id / f"{task.image_id}.dicom"
            )
            try:
                await download_dicom(session, task, temp_path, config)
                payload = read_dicom(temp_path)
                windowed = apply_display_window(
                    payload.pixels,
                    window_center=payload.window_center,
                    window_width=payload.window_width,
                    photometric_interpretation=payload.photometric_interpretation,
                )
                normalized, _, _ = normalize_to_uint8(
                    windowed,
                    low_percentile=config.low_percentile,
                    high_percentile=config.high_percentile,
                )

                crop = BreastCrop(
                    x0_orig=row["crop_x0"],
                    y0_orig=row["crop_y0"],
                    x1_orig=row["crop_x1"],
                    y1_orig=row["crop_y1"],
                    margin_px=row["crop_margin_px"],
                    otsu_threshold=row["crop_otsu_threshold"],
                    source_height=row["source_height"],
                    source_width=row["source_width"],
                    crop_height=row["crop_height"],
                    crop_width=row["crop_width"],
                    area_ratio=row["crop_area_ratio"],
                    scale=row["crop_scale"],
                )
                cropped = crop_image(normalized, crop)

                thickness = max(3, normalized.shape[0] // 400)
                boxed = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                cv2.rectangle(
                    boxed,
                    (crop.x0_orig, crop.y0_orig),
                    (crop.x1_orig, crop.y1_orig),
                    (255, 0, 0),
                    thickness,
                )

                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
                if findings_df is not None:
                    image_findings = findings_df.filter(
                        pl.col("image_id") == task.image_id
                    )
                    for finding_row in image_findings.iter_rows(named=True):
                        pt1 = (
                            int(finding_row["xmin_crop"]),
                            int(finding_row["ymin_crop"]),
                        )
                        pt2 = (
                            int(finding_row["xmax_crop"]),
                            int(finding_row["ymax_crop"]),
                        )
                        cv2.rectangle(
                            cropped_rgb,
                            pt1,
                            pt2,
                            (0, 255, 0),
                            max(2, cropped.shape[0] // 300),
                        )

                panels.append((task.image_id, normalized, boxed, cropped_rgb))
            finally:
                temp_path.unlink(missing_ok=True)

    return panels


def _render_qc_grid(
    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(panels)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False)

    for row_axes, (image_id, original, boxed, cropped) in zip(
        axes, panels, strict=True
    ):
        row_axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
        row_axes[0].set_title(f"{image_id}\noriginal")
        row_axes[1].imshow(boxed)
        row_axes[1].set_title("caja detectada")
        row_axes[2].imshow(cropped)
        row_axes[2].set_title("recorte + hallazgos")
        for axis in row_axes:
            axis.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    app()
