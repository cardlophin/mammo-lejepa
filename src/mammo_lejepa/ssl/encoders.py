from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
import torchvision.models as tv_models

_REGISTRY: dict[str, tuple[Callable[..., nn.Module], int]] = {}


def register_encoder(
    name: str, builder: Callable[..., nn.Module], out_dim: int
) -> None:
    """Un encoder es cualquier `nn.Module` cuyo `forward` devuelva `[B, D]`
    con `D == out_dim` (contracts.md). Añadir una arquitectura es registrarla
    aquí; ningún otro módulo cambia (D-04)."""
    _REGISTRY[name] = (builder, out_dim)


def build_encoder(name: str, **kwargs: object) -> tuple[nn.Module, int]:
    if name not in _REGISTRY:
        raise ValueError(
            f"Encoder desconocido: {name!r}. Registrados: {sorted(_REGISTRY)}"
        )
    builder, out_dim = _REGISTRY[name]
    return builder(**kwargs), out_dim


def registered_encoder_names() -> list[str]:
    return sorted(_REGISTRY)


def _build_resnet50(*, pretrained: bool = False) -> nn.Module:
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet50(weights=weights)
    model.fc = nn.Identity()
    return model


register_encoder("resnet50", _build_resnet50, out_dim=2048)
