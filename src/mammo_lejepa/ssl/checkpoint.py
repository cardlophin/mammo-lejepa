from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class TrainState:
    """Estado completo de un entrenamiento en un instante dado (D-07): la
    reanudación que no restaura el estado del generador no es reanudación —
    cambia la secuencia de aumentaciones y la curva no continúa."""

    epoch: int
    step: int
    model_state: dict[str, Any]
    projector_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    torch_rng_state: torch.Tensor
    torch_cuda_rng_state: list[torch.Tensor] | None
    numpy_rng_state: dict[str, Any]
    python_rng_state: tuple[Any, ...]


def capture_rng_state() -> dict[str, Any]:
    """Estado de los generadores aleatorios de Python, NumPy y Torch (CPU y
    CUDA si hay), en el formato que espera `restore_rng_state`."""
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save(path: Path, state: TrainState) -> None:
    """Escribe a temporal, `fsync`, renombra: un fichero con nombre
    definitivo es siempre válido (D-07) — nunca puede confundirse con uno
    escrito a medias tras una interrupción."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".part")

    payload = {
        "epoch": state.epoch,
        "step": state.step,
        "model_state": state.model_state,
        "projector_state": state.projector_state,
        "optimizer_state": state.optimizer_state,
        "torch_rng_state": state.torch_rng_state,
        "torch_cuda_rng_state": state.torch_cuda_rng_state,
        "numpy_rng_state": state.numpy_rng_state,
        "python_rng_state": state.python_rng_state,
    }

    with open(temp_path, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())

    temp_path.replace(path)


def load(path: Path, *, map_location: str = "cpu") -> TrainState:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return TrainState(**payload)


def latest_valid(directory: Path) -> Path | None:
    """El checkpoint más reciente por época entre los válidos, o `None` si no
    hay ninguno. Un fichero `epoch_*.pt.part` de una interrupción a medio
    escribir no termina en `.pt`, así que el propio patrón del glob ya lo
    excluye — nunca se confunde con uno completo."""
    if not directory.exists():
        return None

    candidates = list(directory.glob("epoch_*.pt"))
    if not candidates:
        return None

    def epoch_of(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return max(candidates, key=epoch_of)
