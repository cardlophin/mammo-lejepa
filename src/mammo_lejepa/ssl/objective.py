from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LeJEPALoss:
    """Los tres términos por separado (contracts.md): registrarlos aparte es lo
    que permite diagnosticar un entrenamiento que va mal en vez de ver sólo un
    escalar."""

    total: Tensor
    sigreg_term: Tensor
    invariance_term: Tensor


def epps_pulley_quadrature(
    num_points: int = 17, t_max: float = 3.0
) -> tuple[Tensor, Tensor]:
    """Nodos y pesos de la cuadratura trapezoidal sobre `[0, t_max]`, con medio
    peso en los extremos y ya multiplicados por la función característica
    teórica de la normal estándar `φ(t) = exp(-t²/2)` (verificado en T002
    contra `lejepa.univariate.epps_pulley.EppsPulley`: es exactamente su
    `self.weights`). `num_points` debe ser impar — incluye el nodo en `t=0`,
    y la cuadratura explota la simetría par de `φ` integrando sólo `t >= 0`."""
    if num_points % 2 == 0:
        raise ValueError(f"num_points debe ser impar, recibió {num_points}")

    knots = torch.linspace(0.0, t_max, num_points)
    dt = t_max / (num_points - 1)
    base_weights = torch.full((num_points,), 2 * dt)
    base_weights[0] = dt
    base_weights[-1] = dt
    phi = torch.exp(-knots.square() * 0.5)
    weights = base_weights * phi

    return knots, weights


def epps_pulley_statistic(
    projections: Tensor,
    *,
    knots: Tensor,
    weights: Tensor,
) -> Tensor:
    """Estadístico de Epps-Pulley de `projections[..., N]` (N muestras
    unidimensionales en la última dimensión) contra la normal estándar, para
    cada valor de las dimensiones anteriores. **Desviación documentada frente
    al contrato original** (que declaraba `[S, N] -> []` escalar): deja las
    dimensiones anteriores a `N` sin reducir — devuelve `[...]`, no `[]` — para
    que `sigreg` decida cómo promediar entre slices y, si aplica, entre vistas;
    es exactamente cómo lo deja `EppsPulley.forward` del paquete oficial antes
    de que `SlicingUnivariateTest` lo reduzca (verificado en T002)."""
    n = projections.shape[-1]
    phi = torch.exp(-knots.square() * 0.5)

    angle = projections.unsqueeze(-1) * knots  # [..., N, K]
    cos_mean = angle.cos().mean(dim=-2)  # [..., K]
    sin_mean = angle.sin().mean(dim=-2)  # [..., K]
    err = (cos_mean - phi).square() + sin_mean.square()  # [..., K]

    return (err @ weights) * n  # [...]


def sigreg(
    embeddings: Tensor,
    *,
    num_slices: int = 1024,
    num_points: int = 17,
    generator: torch.Generator | None = None,
) -> Tensor:
    """SIGReg (D-01): proyecta `embeddings` (forma `[..., N, D]` — acepta tanto
    `[N, D]` como `[V, N, D]`) sobre `num_slices` direcciones aleatorias
    unitarias, aplica `epps_pulley_statistic` a cada proyección y promedia
    sobre slices y sobre cualquier dimensión anterior (p. ej. `V`). Coste
    lineal en `N` (FR-007): ninguna operación construye una matriz `N x N`.
    Direcciones nuevas en cada llamada — nunca se fijan tras la primera
    (verificado en T002) —, reproducibles si se pasa `generator`."""
    if embeddings.ndim < 2:
        raise ValueError(
            f"sigreg espera al menos [N, D], recibió forma {tuple(embeddings.shape)}"
        )

    knots, weights = epps_pulley_quadrature(num_points)
    knots = knots.to(device=embeddings.device, dtype=embeddings.dtype)
    weights = weights.to(device=embeddings.device, dtype=embeddings.dtype)

    dim = embeddings.shape[-1]
    directions = torch.randn(
        dim,
        num_slices,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    directions = directions / directions.norm(p=2, dim=0, keepdim=True)

    projections = embeddings @ directions  # [..., N, num_slices]
    projections = projections.transpose(-1, -2)  # [..., num_slices, N]

    stats = epps_pulley_statistic(projections, knots=knots, weights=weights)
    return stats.mean()


def invariance(views: Tensor) -> Tensor:
    """Dispersión de las `V` vistas de cada imagen respecto a su media,
    simétrica entre vistas —sin distinguir "globales" de "locales" (D-01/D-02:
    el ejemplo mínimo oficial no lo hace)—. Vale exactamente 0 cuando las `V`
    vistas de cada imagen coinciden. El gradiente fluye por las `V` vistas: no
    hay `detach` en ningún término (FR-005)."""
    if views.ndim != 3:
        raise ValueError(
            f"invariance espera [V, B, D], recibió forma {tuple(views.shape)}"
        )

    mean_view = views.mean(dim=0, keepdim=True)  # [1, B, D]
    return (views - mean_view).square().mean()


def lejepa_loss(
    views: Tensor,
    *,
    lamb: float = 0.02,
    num_slices: int = 1024,
    num_points: int = 17,
    generator: torch.Generator | None = None,
) -> LeJEPALoss:
    """`λ · sigreg(views) + (1 - λ) · invariance(views)` (FR-003), sobre
    `views` de forma `[V, B, D]` — la salida del proyector, no del encoder
    (D-04). `λ=0,02` por defecto, el del ejemplo mínimo oficial (D-03,
    confirmado en T002). No conoce ninguna arquitectura: su única entrada son
    tensores con la forma declarada (FR-004); una forma distinta es
    `ValueError`, no una reinterpretación silenciosa."""
    if views.ndim != 3:
        raise ValueError(
            f"lejepa_loss espera [V, B, D], recibió forma {tuple(views.shape)}"
        )

    sigreg_term = sigreg(
        views, num_slices=num_slices, num_points=num_points, generator=generator
    )
    invariance_term = invariance(views)
    total = lamb * sigreg_term + (1.0 - lamb) * invariance_term

    return LeJEPALoss(
        total=total, sigreg_term=sigreg_term, invariance_term=invariance_term
    )
