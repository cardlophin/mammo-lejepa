from __future__ import annotations

import random
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import typer
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

from mammo_lejepa import CORPUS_VERSION
from mammo_lejepa.catalog import consolidate as consolidate_records
from mammo_lejepa.catalog import validate_catalog_coherence
from mammo_lejepa.config import CorpusConfig
from mammo_lejepa.image_io import read_grayscale, read_mask
from mammo_lejepa.manifest import build_manifest, missing_files
from mammo_lejepa.models import (
    BoxingParams,
    ImagenMammoBench,
    RegistroDeRecorte,
    RunSummary,
)
from mammo_lejepa.quality import is_suspect, summarize_by_source
from mammo_lejepa.resume import latest_records_by_image, parse_jsonl_records
from mammo_lejepa.runner import run_build
from mammo_lejepa.splits import assign_splits, verify_no_patient_leakage
from mammo_lejepa.storage import (
    acquire_output_lock,
    append_run_summary,
    clean_orphan_part_files,
)

_GRID_THUMB_PX = 220

app = typer.Typer(
    help="mammo-corpus: recortes de la región mamaria desde Mammo-Bench.",
    no_args_is_help=True,
)
console = Console()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_latest_records(config: CorpusConfig) -> list[RegistroDeRecorte]:
    lines = config.records_jsonl_path.read_text(encoding="utf-8").splitlines()
    return list(latest_records_by_image(parse_jsonl_records(lines)).values())


def _dataframe_to_markdown(frame: pl.DataFrame) -> str:
    columns = frame.columns
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in frame.iter_rows():
        cells = [
            f"{value:.4f}" if isinstance(value, float) else str(value) for value in row
        ]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body_lines])


@app.command()
def build(
    source: list[str] = typer.Option(
        [], "--source", help="Repetible: acota a estas fuentes"
    ),
    limit: int | None = typer.Option(None, "--limit", help="Número de imágenes"),
    image_id: list[str] = typer.Option(
        [], "--image-id", help="Repetible: acota a estos image_id"
    ),
    workers: int = typer.Option(8, help="Procesos de preprocesado"),
    mask_threshold: int = typer.Option(128, "--mask-threshold"),
    margin_px: int = typer.Option(25, "--margin-px"),
) -> None:
    """Preprocesa Mammo-Bench: recorta la región mamaria de cada imagen
    (máscara u Otsu de respaldo) y escribe el catálogo incremental."""
    config = CorpusConfig(
        workers=workers,
        boxing=BoxingParams(mask_threshold=mask_threshold, margin_px=margin_px),
    )

    if not config.csv_path.exists():
        console.print(f"[red]No existe {config.csv_path}.[/red]")
        raise typer.Exit(code=1)

    clean_orphan_part_files(config.crops_dir)

    catalog_csv = pl.read_csv(config.csv_path, infer_schema_length=0)

    try:
        manifest = build_manifest(
            catalog_csv,
            sources=source or None,
            limit=limit,
            image_ids=image_id or None,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    existing_preprocessed = {
        image.preprocessed_path
        for image in manifest
        if (config.mammobench_root / image.preprocessed_path).exists()
    }
    problems = missing_files(manifest, existing_preprocessed)
    if problems:
        console.print(
            f"[yellow]Aviso:[/yellow] {len(problems)} imagen(es) sin fichero "
            "preprocesado en disco (FR-006)."
        )
    manifest = [
        image for image in manifest if image.preprocessed_path in existing_preprocessed
    ]

    if not manifest:
        console.print(
            "[yellow]No hay ninguna imagen que procesar con estos filtros.[/yellow]"
        )
        raise typer.Exit(code=0)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    console.print(
        f"run_id: {run_id} — {len(manifest)} imagen(es) candidatas "
        f"({len(problems)} sin fichero)."
    )

    try:
        with acquire_output_lock(config.output_dir):
            summary = _run_with_progress(manifest, config, run_id)
    except RuntimeError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    _print_summary(summary)
    if summary.n_failed:
        raise typer.Exit(code=1)


def _run_with_progress(manifest: list, config: CorpusConfig, run_id: str) -> RunSummary:
    """Escribe `runs.jsonl` al inicio y al final (T025) y muestra el progreso
    con `rich` mientras `run_build` reparte el trabajo entre procesos."""
    start_marker = RunSummary(
        run_id=run_id,
        started_at=_now_iso(),
        finished_at="",
        command="build",
        code_version=CORPUS_VERSION,
        seed=config.seed,
        n_total=len(manifest),
        n_ok=0,
        n_failed=0,
        failures_by_category={},
        n_fallback_otsu=0,
        exit_reason="started",
    )
    append_run_summary(start_marker, config.runs_jsonl_path)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    totals = {"failures": 0}

    with progress:
        progress_task = progress.add_task("Procesando", total=len(manifest))

        def on_result(record: RegistroDeRecorte) -> None:
            if record.status != "ok":
                totals["failures"] += 1
            progress.update(
                progress_task,
                advance=1,
                description=f"Procesando (fallos: {totals['failures']})",
            )

        summary = run_build(manifest, config=config, run_id=run_id, on_result=on_result)

    append_run_summary(summary, config.runs_jsonl_path)
    return summary


def _print_summary(summary: RunSummary) -> None:
    console.print(
        f"\n[bold]Resumen[/bold] ({summary.exit_reason}): {summary.n_ok} ok, "
        f"{summary.n_failed} fallidas, de {summary.n_total} totales. "
        f"Respaldo a Otsu: {summary.n_fallback_otsu}."
    )
    for category, count in summary.failures_by_category.items():
        console.print(f"  {category.value}: {count}")


@app.command()
def consolidate() -> None:
    """Compacta `records.jsonl` en `catalog.parquet` (FR-016, idempotente)."""
    config = CorpusConfig()
    if not config.records_jsonl_path.exists():
        console.print(
            f"[red]No existe {config.records_jsonl_path}. Ejecuta antes "
            "'mammo-corpus build'.[/red]"
        )
        raise typer.Exit(code=1)

    lines = config.records_jsonl_path.read_text(encoding="utf-8").splitlines()
    records = parse_jsonl_records(lines)
    catalog_df = consolidate_records(records)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_df.write_parquet(config.catalog_parquet_path)

    console.print(
        f"Catálogo escrito en {config.catalog_parquet_path} "
        f"({catalog_df.height} fila(s))."
    )

    problems = validate_catalog_coherence(catalog_df)
    if problems:
        console.print(
            f"[yellow]Aviso:[/yellow] {len(problems)} problema(s) de coherencia (T037):"
        )
        for problem in problems[:20]:
            console.print(f"  {problem}")


@app.command()
def split(
    seed: int = typer.Option(0, help="Semilla de las particiones"),
    ratios: tuple[float, float, float] = typer.Option(
        (0.8, 0.1, 0.1), help="train val test"
    ),
) -> None:
    """Asigna train/val/test por `patient_key` (FR-017/FR-018), incorpora la
    columna `split` a `catalog.parquet` y escribe `splits.parquet`
    (data-model.md). Verifica FR-019 antes de escribir nada."""
    config = CorpusConfig(seed=seed, ratios=ratios)
    if not config.records_jsonl_path.exists():
        console.print(
            f"[red]No existe {config.records_jsonl_path}. Ejecuta antes "
            "'mammo-corpus build'.[/red]"
        )
        raise typer.Exit(code=1)
    if not config.catalog_parquet_path.exists():
        console.print(
            f"[red]No existe {config.catalog_parquet_path}. Ejecuta antes "
            "'mammo-corpus consolidate'.[/red]"
        )
        raise typer.Exit(code=1)

    records = _load_latest_records(config)
    assignment = assign_splits(records, ratios=ratios, seed=seed)

    catalog_df = pl.read_parquet(config.catalog_parquet_path)
    split_column = catalog_df["patient_key"].replace_strict(assignment, default=None)
    catalog_df = catalog_df.with_columns(split_column.alias("split"))

    pairs = list(
        zip(
            catalog_df["patient_key"].to_list(),
            catalog_df["split"].to_list(),
            strict=True,
        )
    )
    try:
        verify_no_patient_leakage(
            (patient_key, split_name)
            for patient_key, split_name in pairs
            if split_name is not None
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    catalog_df.write_parquet(config.catalog_parquet_path)

    splits_df = (
        catalog_df.filter(pl.col("split").is_not_null())
        .group_by(["patient_key", "source_dataset", "split"])
        .agg(pl.len().alias("n_images"))
        .with_columns(pl.lit(seed).alias("seed"))
        .sort(["source_dataset", "patient_key"])
    )
    splits_df.write_parquet(config.splits_parquet_path)

    counts = splits_df.group_by("split").agg(pl.col("n_images").sum())
    console.print(f"Particiones escritas en {config.splits_parquet_path}.")
    for row in counts.sort("split").iter_rows(named=True):
        console.print(f"  {row['split']}: {row['n_images']} imagen(es)")


@app.command()
def qc() -> None:
    """Genera el informe de calidad por fuente (FR-021): percentiles de
    `crop_area_ratio`/`mask_otsu_iou`, recuento de respaldos a Otsu y lista de
    casos sospechosos. No modifica `records.jsonl`: el indicador `suspect` se
    persiste en el catálogo al consolidar (FR-023, T034)."""
    config = CorpusConfig()
    if not config.records_jsonl_path.exists():
        console.print(
            f"[red]No existe {config.records_jsonl_path}. Ejecuta antes "
            "'mammo-corpus build'.[/red]"
        )
        raise typer.Exit(code=1)

    records = _load_latest_records(config)
    annotated = []
    for record in records:
        suspect, reason = is_suspect(record, params=config.quality)
        annotated.append(replace(record, suspect=suspect, suspect_reason=reason))

    summary = summarize_by_source(annotated)
    config.qc_dir.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(config.quality_by_source_path)

    suspects = sorted(
        (r for r in annotated if r.suspect),
        key=lambda r: (r.source_dataset, r.image_id),
    )

    report = [
        "# Informe de calidad — Mammo-Bench\n",
        "## Percentiles por fuente\n",
        _dataframe_to_markdown(summary),
        f"\n## Casos sospechosos ({len(suspects)})\n",
    ]
    report.extend(
        f"- `{record.crop_path}` — {record.suspect_reason}" for record in suspects
    )
    config.quality_report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    console.print(
        f"Informe escrito en {config.quality_report_path} "
        f"({len(suspects)} sospechoso(s) de {len(annotated)})."
    )


@app.command()
def inspect(
    source: str = typer.Option(..., "--source"),
    n: int = typer.Option(8, help="Número de imágenes en la lámina"),
) -> None:
    """Genera `grid_<fuente>.png`: por fila, imagen preprocesada, máscara,
    caja elegida dibujada y recorte final (FR-022). La muestra prioriza casos
    sospechosos y de respaldo a Otsu, para que la verificación manual (T031)
    los vea primero."""
    config = CorpusConfig()
    if not config.records_jsonl_path.exists():
        console.print(
            f"[red]No existe {config.records_jsonl_path}. Ejecuta antes "
            "'mammo-corpus build'.[/red]"
        )
        raise typer.Exit(code=1)

    records = [
        record
        for record in _load_latest_records(config)
        if record.source_dataset == source and record.status == "ok"
    ]
    if not records:
        console.print(
            f"[yellow]No hay recortes 'ok' para la fuente '{source}'.[/yellow]"
        )
        raise typer.Exit(code=1)

    def _is_notable(record: RegistroDeRecorte) -> bool:
        suspect, _ = is_suspect(record, params=config.quality)
        return suspect or record.fallback_reason is not None

    records.sort(key=lambda record: record.image_id)
    notable = [record for record in records if _is_notable(record)]
    rest = [record for record in records if record not in notable]
    random.Random(config.seed).shuffle(rest)
    sample = sorted((notable + rest)[:n], key=lambda record: record.image_id)

    catalog_csv = pl.read_csv(config.csv_path, infer_schema_length=0)
    manifest_by_id = {image.image_id: image for image in build_manifest(catalog_csv)}

    config.qc_dir.mkdir(parents=True, exist_ok=True)
    grid = _build_qc_grid(sample, manifest_by_id, config)
    cv2.imwrite(str(config.grid_path(source)), grid)

    console.print(
        f"Lámina escrita en {config.grid_path(source)} ({len(sample)} imagen(es))."
    )


def _resize_square(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _build_qc_grid(
    records: list[RegistroDeRecorte],
    manifest_by_id: dict[str, ImagenMammoBench],
    config: CorpusConfig,
) -> np.ndarray:
    """Una fila por registro: imagen preprocesada, máscara (o negro si falta),
    imagen con la caja elegida dibujada, y el recorte final."""
    size = _GRID_THUMB_PX
    rows = []

    for record in records:
        image_meta = manifest_by_id[record.image_id]
        pixels = read_grayscale(config.mammobench_root / image_meta.preprocessed_path)
        mask_pixels = read_mask(config.mammobench_root / image_meta.mask_path)
        if mask_pixels is None:
            mask_pixels = np.zeros_like(pixels)

        boxed = pixels.copy()
        if record.box is not None:
            box = record.box
            cv2.rectangle(boxed, (box.x0, box.y0), (box.x1, box.y1), 255, 3)

        crop = (
            read_grayscale(Path(record.crop_path))
            if record.crop_path is not None
            else np.zeros_like(pixels)
        )

        tiles = [
            _resize_square(tile, size) for tile in (pixels, mask_pixels, boxed, crop)
        ]
        rows.append(cv2.hconcat(tiles))

    return cv2.vconcat(rows)


if __name__ == "__main__":
    app()
