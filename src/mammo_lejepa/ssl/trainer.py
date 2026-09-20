from __future__ import annotations

import contextlib
import json
import random
import signal
import uuid
import warnings
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from mammo_lejepa.ssl.augment import build_view_transform
from mammo_lejepa.ssl.checkpoint import (
    TrainState,
    capture_rng_state,
    latest_valid,
    load,
    restore_rng_state,
    save,
)
from mammo_lejepa.ssl.config import TrainConfig
from mammo_lejepa.ssl.data import CropDataset, collate_views
from mammo_lejepa.ssl.diagnostics import embedding_report
from mammo_lejepa.ssl.encoders import build_encoder
from mammo_lejepa.ssl.objective import lejepa_loss
from mammo_lejepa.ssl.projector import build_projector
from mammo_lejepa.ssl.schedules import warmup_cosine

_DEFAULT_WEIGHT_DECAY: dict[str, float] = {"resnet50": 5e-4, "vit_small": 5e-2}


def _default_weight_decay(encoder_name: str) -> float:
    return _DEFAULT_WEIGHT_DECAY.get(encoder_name, 5e-4)


def _autocast(device: str) -> torch.autocast:
    """Precisión mixta degradable en un único punto (D-05): `bfloat16` en
    CUDA; en cualquier otro dispositivo, un `autocast` desactivado —
    equivalente a `float32` sin cambiar el tipo del cómputo— porque ni CPU ni
    MPS soportan `bfloat16` de la misma forma. Una sola rama, nunca
    `if device == "mps"` repartido por el bucle."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device, enabled=False)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    loss_total: float
    loss_sigreg: float
    loss_invariance: float
    learning_rate: float
    seconds: float
    effective_rank: float
    isotropy_deviation: float


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    epochs_completed: int
    history: tuple[EpochMetrics, ...]
    exit_reason: str
    n_excluded_suspect: int


def train(config: TrainConfig) -> RunSummary:
    """Único punto que conoce a la vez el modelo, los datos, el dispositivo y
    el disco (contracts.md): no contiene lógica del objetivo ni de
    diagnóstico, las llama. Atiende `SIGINT`/`SIGTERM` guardando checkpoint
    antes de salir."""
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    device = config.resolved_device()

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    if config.batch_size < config.min_batch_size:
        warnings.warn(
            f"Lote efectivo ({config.batch_size}) por debajo del mínimo "
            f"recomendado ({config.min_batch_size}) para la estimación de "
            "SIGReg (FR-018): el estadístico puede ser ruidoso.",
            stacklevel=2,
        )

    catalog = pl.read_parquet(config.catalog_path)
    transform = build_view_transform(config.augment)
    dataset = CropDataset(
        catalog,
        split="train",
        views=config.views,
        transform=transform,
        include_suspect=config.include_suspect,
        sources=config.sources,
    )
    if len(dataset) == 0:
        raise ValueError(
            "El conjunto de entrenamiento está vacío tras aplicar los filtros "
            f"(split='train', sources={config.sources})."
        )

    # Sin `generator=` explícito: el muestreo aleatorio del `DataLoader` cae
    # en el generador global de Torch, que es exactamente el que
    # `capture_rng_state`/`restore_rng_state` checkpointean (D-07). Si se le
    # diera un `torch.Generator()` propio, su estado quedaría fuera del
    # checkpoint y la secuencia de lotes no continuaría tras reanudar.
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_views,
    )
    if len(loader) == 0:
        raise ValueError(
            f"El dataset tiene {len(dataset)} imagen(es), menos que "
            f"batch_size={config.batch_size}: ningún lote completo posible."
        )

    encoder, encoder_dim = build_encoder(config.encoder_name)
    projector = build_projector(encoder_dim, out_dim=config.proj_dim)
    encoder.to(device)
    projector.to(device)

    weight_decay = (
        config.weight_decay
        if config.weight_decay is not None
        else _default_weight_decay(config.encoder_name)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "weight_decay": weight_decay},
            {"params": projector.parameters(), "weight_decay": weight_decay},
        ],
        lr=config.lr,
    )

    start_epoch = 0
    step = 0
    checkpoint_dir = config.run_dir / "checkpoints"
    checkpoint_path = latest_valid(checkpoint_dir)
    if checkpoint_path is not None:
        # Siempre a CPU, nunca `map_location=device`: el estado de los
        # generadores aleatorios debe seguir siendo un `ByteTensor` de CPU
        # (`torch.set_rng_state` lo exige), y `map_location` movería TODO el
        # payload al dispositivo, RNG incluido, rompiéndolo. `load_state_dict`
        # ya sabe mover los pesos del modelo/optimizador al dispositivo
        # correcto porque `encoder`/`projector` ya están en él (`.to(device)`
        # más arriba) — verificado con un fallo real en MPS (bug encontrado
        # en T023, ejecución de humo sobre datos reales).
        state = load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(state.model_state)
        projector.load_state_dict(state.projector_state)
        optimizer.load_state_dict(state.optimizer_state)
        restore_rng_state(
            {
                "torch": state.torch_rng_state,
                "torch_cuda": state.torch_cuda_rng_state,
                "numpy": state.numpy_rng_state,
                "python": state.python_rng_state,
            }
        )
        start_epoch = state.epoch + 1
        step = state.step

    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "config.json").write_text(config.to_json())

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    shutdown_requested = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    previous_handlers: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            previous_handlers[sig] = signal.signal(sig, _handle_signal)

    history: list[EpochMetrics] = []
    exit_reason = "completed"
    current_lr = config.lr

    try:
        for epoch in range(start_epoch, config.epochs):
            encoder.train()
            projector.train()
            epoch_start = perf_counter()
            totals = {"total": 0.0, "sigreg": 0.0, "invariance": 0.0}
            n_batches = 0
            last_embeddings: torch.Tensor | None = None

            for views, _labels in loader:
                views = views.to(device)
                n_views, batch_size = views.shape[:2]
                flat_views = views.reshape(n_views * batch_size, *views.shape[2:])

                current_lr = warmup_cosine(
                    step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    base_lr=config.lr,
                    final_lr_ratio=config.final_lr_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                with _autocast(device):
                    embeddings = encoder(flat_views)
                    projections = projector(embeddings)
                    projections = projections.reshape(
                        batch_size, n_views, -1
                    ).transpose(0, 1)
                    loss = lejepa_loss(
                        projections,
                        lamb=config.lamb,
                        num_slices=config.num_slices,
                        num_points=config.num_points,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                optimizer.step()

                step += 1
                n_batches += 1
                totals["total"] += loss.total.item()
                totals["sigreg"] += loss.sigreg_term.item()
                totals["invariance"] += loss.invariance_term.item()
                last_embeddings = embeddings.detach()

                if shutdown_requested:
                    break

            n_batches = max(n_batches, 1)
            with torch.no_grad():
                diagnostics = (
                    embedding_report(last_embeddings)
                    if last_embeddings is not None
                    else embedding_report(torch.zeros(2, encoder_dim, device=device))
                )

            metrics = EpochMetrics(
                epoch=epoch,
                loss_total=totals["total"] / n_batches,
                loss_sigreg=totals["sigreg"] / n_batches,
                loss_invariance=totals["invariance"] / n_batches,
                learning_rate=current_lr,
                seconds=perf_counter() - epoch_start,
                effective_rank=diagnostics.effective_rank,
                isotropy_deviation=diagnostics.isotropy_deviation,
            )
            history.append(metrics)
            _append_history(config.run_dir, metrics)
            if diagnostics.effective_rank < config.min_effective_rank:
                warnings.warn(
                    f"Época {epoch}: rango efectivo de los embeddings "
                    f"{diagnostics.effective_rank:.2f} por debajo del umbral "
                    f"{config.min_effective_rank} — señal de colapso incipiente "
                    "(FR-027).",
                    stacklevel=2,
                )

            rng_state = capture_rng_state()
            checkpoint_state = TrainState(
                epoch=epoch,
                step=step,
                model_state=encoder.state_dict(),
                projector_state=projector.state_dict(),
                optimizer_state=optimizer.state_dict(),
                torch_rng_state=rng_state["torch"],
                torch_cuda_rng_state=rng_state["torch_cuda"],
                numpy_rng_state=rng_state["numpy"],
                python_rng_state=rng_state["python"],
            )
            save(checkpoint_dir / f"epoch_{epoch:04d}.pt", checkpoint_state)

            if shutdown_requested:
                exit_reason = "interrupted"
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]

    return RunSummary(
        run_id=run_id,
        epochs_completed=start_epoch + len(history),
        history=tuple(history),
        exit_reason=exit_reason,
        n_excluded_suspect=dataset.n_excluded_suspect,
    )


def _append_history(run_dir: Any, metrics: EpochMetrics) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "history.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metrics)) + "\n")
