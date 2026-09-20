from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class EmbeddingStats:
    """Diagnóstico de la geometría de un lote de embeddings (FR-025):
    registrado periódicamente durante el entrenamiento, no sólo al final
    (FR-026)."""

    effective_rank: float
    spectral_entropy: float
    isotropy_deviation: float
    mean_norm: float


def _singular_values(embeddings: Tensor) -> Tensor:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(centered)


def effective_rank(embeddings: Tensor) -> float:
    """Entropía exponenciada del espectro de valores singulares normalizado
    (Roy & Vetterli, 2007): cercano a `D` para una gaussiana isótropa, cercano
    a 1 cuando los embeddings colapsan en un punto o un subespacio de rango
    bajo — el aviso de colapso incipiente (FR-027) se dispara sobre este
    valor."""
    singular_values = _singular_values(embeddings)
    total = singular_values.sum()
    if total <= 0:
        return 1.0

    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.exp())


def spectral_entropy(embeddings: Tensor) -> float:
    """Entropía normalizada (en `[0, 1]`) del espectro de valores singulares:
    1 para una distribución perfectamente plana (isótropa), cercana a 0
    cuando toda la varianza se concentra en una sola dirección."""
    singular_values = _singular_values(embeddings)
    total = singular_values.sum()
    if total <= 0:
        return 0.0

    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * probabilities.log()).sum()
    max_entropy = torch.log(torch.tensor(float(singular_values.numel())))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def isotropy_deviation(embeddings: Tensor) -> float:
    """Distancia de Frobenius entre la matriz de covarianza normalizada de
    `embeddings` y la identidad, dividida por `sqrt(D)` para que el valor no
    dependa de la dimensión. Cercana a 0 para una gaussiana isótropa."""
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    n = centered.shape[0]
    dim = centered.shape[1]

    covariance = (centered.T @ centered) / max(n - 1, 1)
    variance = covariance.diagonal().mean().clamp_min(1e-12)
    normalized_covariance = covariance / variance

    identity = torch.eye(dim, dtype=embeddings.dtype, device=embeddings.device)
    deviation = torch.linalg.matrix_norm(normalized_covariance - identity, ord="fro")
    return float(deviation / (dim**0.5))


def embedding_report(embeddings: Tensor) -> EmbeddingStats:
    """Las cuatro métricas juntas, para registrar en una sola llamada.
    Ninguna función de este módulo modifica `embeddings` ni requiere
    gradiente."""
    with torch.no_grad():
        return EmbeddingStats(
            effective_rank=effective_rank(embeddings),
            spectral_entropy=spectral_entropy(embeddings),
            isotropy_deviation=isotropy_deviation(embeddings),
            mean_norm=float(embeddings.norm(dim=-1).mean()),
        )
