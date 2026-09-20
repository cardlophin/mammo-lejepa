from __future__ import annotations

import uuid
from pathlib import Path

import polars as pl
import typer
from rich.console import Console

from mammo_lejepa.ssl.baselines import build_conditions, lejepa_encoder
from mammo_lejepa.ssl.config import EvalConfig, TrainConfig
from mammo_lejepa.ssl.diagnostics import embedding_report
from mammo_lejepa.ssl.probe import extract_embeddings, run_probes
from mammo_lejepa.ssl.reporting import (
    ConditionResults,
    load_eval,
    read_history,
    save_eval,
    to_markdown,
)
from mammo_lejepa.ssl.trainer import train

app = typer.Typer(
    help="mammo-ssl: preentrenamiento LeJEPA sobre el corpus mamario.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def pretrain(
    catalog_path: Path = typer.Option(Path("data/corpus/catalog.parquet"), "--catalog"),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        help="Si se omite, crea runs/run_<id> nuevo. Reapuntar al mismo "
        "directorio reanuda desde su último checkpoint válido (FR-016): no "
        "hace falta ningún flag de reanudación aparte.",
    ),
    encoder_name: str = typer.Option("resnet50", "--encoder"),
    seed: int = typer.Option(0),
    device: str | None = typer.Option(
        None, help="cuda/mps/cpu; detectado automáticamente si se omite (D-05)"
    ),
    views: int = typer.Option(4, "--views"),
    lamb: float = typer.Option(0.02),
    num_slices: int = typer.Option(1024, "--num-slices"),
    batch_size: int = typer.Option(256, "--batch-size"),
    epochs: int = typer.Option(100),
    lr: float = typer.Option(2e-3),
    source: list[str] = typer.Option(
        [], "--source", help="Repetible: acota el entrenamiento a estas fuentes"
    ),
    include_suspect: bool = typer.Option(
        True, help="Incluir en el entrenamiento las imágenes marcadas sospechosas"
    ),
) -> None:
    """Preentrena un encoder con el objetivo LeJEPA (SIGReg + invarianza)
    sobre `catalog.parquet` (feature 002). Interrumpible con `Ctrl-C`: guarda
    checkpoint antes de salir y relanzar con el mismo `--run-dir` continúa
    exactamente donde se quedó (FR-015/FR-016)."""
    resolved_run_dir = run_dir or Path("runs") / f"run_{uuid.uuid4().hex[:12]}"

    config = TrainConfig(
        catalog_path=catalog_path,
        run_dir=resolved_run_dir,
        seed=seed,
        device=device,
        encoder_name=encoder_name,
        views=views,
        lamb=lamb,
        num_slices=num_slices,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        sources=tuple(source) if source else None,
        include_suspect=include_suspect,
    )

    console.print(f"run_dir: {resolved_run_dir}")
    console.print(f"Dispositivo: {config.resolved_device()}")

    summary = train(config)

    console.print(
        f"\n[bold]Resumen[/bold] ({summary.exit_reason}): "
        f"{summary.epochs_completed} época(s) completadas. run_id={summary.run_id}"
    )
    if summary.n_excluded_suspect:
        console.print(
            f"Excluidas {summary.n_excluded_suspect} imagen(es) sospechosas "
            "del entrenamiento."
        )
    for metrics in summary.history[-3:]:
        console.print(
            f"  época {metrics.epoch}: loss={metrics.loss_total:.4f} "
            f"(sigreg={metrics.loss_sigreg:.4f}, inv={metrics.loss_invariance:.4f}) "
            f"rango_efectivo={metrics.effective_rank:.2f} "
            f"desviación_isotropía={metrics.isotropy_deviation:.4f}"
        )


@app.command()
def probe(
    run_dir: Path = typer.Option(..., "--run-dir", help="Ejecución con checkpoint"),
    catalog_path: Path = typer.Option(Path("data/corpus/catalog.parquet"), "--catalog"),
    device: str | None = typer.Option(None),
    seed: int = typer.Option(0),
    max_samples: int | None = typer.Option(
        None,
        "--max-samples",
        help="Tope de imágenes por partición (muestreo con semilla)",
    ),
    imagenet: bool = typer.Option(
        True, help="Incluir la línea base preentrenada en ImageNet (descarga pesos)"
    ),
) -> None:
    """Sonda lineal sobre el encoder congelado, bajo el mismo protocolo para
    las tres condiciones —pesos aleatorios, ImageNet y LeJEPA— más la sonda
    de control sobre `source_dataset` (FR-019 a FR-023)."""
    config = EvalConfig(
        catalog_path=catalog_path,
        seed=seed,
        device=device,
        max_samples_per_split=max_samples,
    )
    catalog = pl.read_parquet(catalog_path)
    console.print(f"Dispositivo: {config.resolved_device()}")

    conditions = build_conditions(run_dir, include_imagenet=imagenet, seed=seed)
    checkpoint = lejepa_encoder(run_dir)[2]

    results: ConditionResults = {}
    for name, encoder in conditions.items():
        console.print(f"Sondas: {name}…")
        results[name] = run_probes(encoder, catalog, config=config)

    eval_dir = save_eval(run_dir, results, config, checkpoint=checkpoint)
    console.print(to_markdown(results, control_target=config.control_target))
    console.print(f"Resultados guardados en {eval_dir}")


@app.command()
def diagnose(
    run_dir: Path = typer.Option(..., "--run-dir"),
    catalog_path: Path = typer.Option(Path("data/corpus/catalog.parquet"), "--catalog"),
    device: str | None = typer.Option(None),
    max_samples: int | None = typer.Option(2000, "--max-samples"),
) -> None:
    """Geometría de los embeddings: evolución por época registrada durante el
    entrenamiento y, sobre el checkpoint final, sobre la partición de test
    (FR-025 a FR-027). Avisa si el rango efectivo cae bajo el umbral."""
    config = TrainConfig.from_json((run_dir / "config.json").read_text())
    threshold = config.min_effective_rank

    console.print("[bold]Historial[/bold] (rango efectivo por época)")
    for row in read_history(run_dir):
        flag = (
            " [red]<- bajo el umbral[/red]" if row["effective_rank"] < threshold else ""
        )
        console.print(
            f"  época {row['epoch']}: rango={row['effective_rank']:.2f} "
            f"isotropía={row['isotropy_deviation']:.2f} "
            f"sigreg={row['loss_sigreg']:.3f} inv={row['loss_invariance']:.4f}{flag}"
        )

    eval_config = EvalConfig(
        catalog_path=catalog_path, device=device, max_samples_per_split=max_samples
    )
    encoder, _, _ = lejepa_encoder(run_dir)
    embeddings = extract_embeddings(
        encoder, pl.read_parquet(catalog_path), split="test", config=eval_config
    )
    report = embedding_report(embeddings.features)
    console.print(
        f"\n[bold]Test[/bold] ({embeddings.features.shape[0]} imágenes): "
        f"rango efectivo={report.effective_rank:.2f}, "
        f"entropía espectral={report.spectral_entropy:.3f}, "
        f"desviación de isotropía={report.isotropy_deviation:.2f}, "
        f"norma media={report.mean_norm:.2f}"
    )
    if report.effective_rank < threshold:
        console.print(
            f"[yellow]Aviso:[/yellow] rango efectivo bajo el umbral {threshold} "
            "(colapso incipiente, FR-027)."
        )


@app.command()
def compare(
    run_dir: list[Path] = typer.Option(
        ..., "--run-dir", help="Repetible: ejecuciones con `probe` ya ejecutado"
    ),
) -> None:
    """Tabla comparativa de varias ejecuciones (p. ej. ResNet-50 frente a
    ViT) sobre las mismas condiciones. Las líneas base se toman de la primera
    ejecución; la condición `lejepa` de cada una se etiqueta con su
    directorio."""
    merged: ConditionResults = {}
    for path in run_dir:
        for condition, targets in load_eval(path).items():
            if condition == "lejepa":
                merged[f"lejepa:{path.name}"] = targets
            else:
                merged.setdefault(condition, targets)

    console.print(to_markdown(merged, control_target="source_dataset"))


if __name__ == "__main__":
    app()
