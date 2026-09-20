from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from mammo_lejepa.ssl.checkpoint import latest_valid, load
from mammo_lejepa.ssl.config import TrainConfig
from mammo_lejepa.ssl.encoders import build_encoder

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _freeze(encoder: nn.Module) -> nn.Module:
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


class _ImageNetNormalized(nn.Module):
    """Normalización de ImageNet dentro del propio módulo: las tres
    condiciones reciben exactamente los mismos tensores (D-09), y sólo el
    encoder preentrenado en ImageNet, que la espera, la aplica."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder((x - self.mean) / self.std)


def random_encoder(name: str, *, seed: int = 0) -> tuple[nn.Module, int]:
    """Encoder con pesos aleatorios congelados: acota por abajo lo que puede
    valer una sonda lineal sin ningún preentrenamiento (D-09)."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        encoder, out_dim = build_encoder(name)
    return _freeze(encoder), out_dim


def imagenet_encoder(name: str) -> tuple[nn.Module, int]:
    """Encoder preentrenado en ImageNet congelado: dice si preentrenar en el
    dominio aporta algo sobre transferir de imagen natural (D-09). Descarga
    los pesos la primera vez."""
    encoder, out_dim = build_encoder(name, pretrained=True)
    return _freeze(_ImageNetNormalized(encoder)), out_dim


def lejepa_encoder(run_dir: Path) -> tuple[nn.Module, int, Path]:
    """Encoder preentrenado con LeJEPA, cargado del último checkpoint válido
    de `run_dir` con la configuración archivada en `config.json`, congelado."""
    config = TrainConfig.from_json((run_dir / "config.json").read_text())
    checkpoint_path = latest_valid(run_dir / "checkpoints")
    if checkpoint_path is None:
        raise FileNotFoundError(f"{run_dir} no tiene ningún checkpoint válido.")

    state = load(checkpoint_path, map_location="cpu")
    encoder, out_dim = build_encoder(config.encoder_name)
    encoder.load_state_dict(state.model_state)
    return _freeze(encoder), out_dim, checkpoint_path


def build_conditions(
    run_dir: Path, *, include_imagenet: bool = True, seed: int = 0
) -> dict[str, nn.Module]:
    """Las tres condiciones del protocolo (D-09), en el orden en que se
    reportan: aleatorio, ImageNet y LeJEPA."""
    config = TrainConfig.from_json((run_dir / "config.json").read_text())

    conditions: dict[str, nn.Module] = {
        "random": random_encoder(config.encoder_name, seed=seed)[0]
    }
    if include_imagenet:
        conditions["imagenet"] = imagenet_encoder(config.encoder_name)[0]
    conditions["lejepa"] = lejepa_encoder(run_dir)[0]
    return conditions
