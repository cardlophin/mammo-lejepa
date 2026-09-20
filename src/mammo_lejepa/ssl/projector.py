from __future__ import annotations

import torch.nn as nn
from torchvision.ops import MLP


def build_projector(
    in_dim: int, *, hidden_dim: int = 2048, out_dim: int = 128
) -> nn.Module:
    """MLP de tres capas con `BatchNorm1d` entre ellas (el mismo diseño que
    `MINIMAL.md` del paquete oficial). La sonda lineal evalúa la salida del
    **encoder**, no la de este proyector: el proyector se descarta tras el
    preentrenamiento, como es estándar en SSL (D-04)."""
    return MLP(in_dim, [hidden_dim, hidden_dim, out_dim], norm_layer=nn.BatchNorm1d)
