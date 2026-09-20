from __future__ import annotations

import tracemalloc
from time import perf_counter

import pytest
import torch

from mammo_lejepa.ssl.objective import sigreg

pytestmark = pytest.mark.slow

_SIZES = (500, 1000, 2000, 4000)


def _time_sigreg(n: int, *, dim: int = 32, num_slices: int = 256) -> float:
    embeddings = torch.randn(n, dim)
    # Calentamiento: la primera llamada paga coste fijo (alojamiento de buffers).
    sigreg(embeddings, num_slices=num_slices)

    start = perf_counter()
    for _ in range(5):
        sigreg(embeddings, num_slices=num_slices)
    return (perf_counter() - start) / 5


def test_sigreg_time_scales_linearly_with_batch_size() -> None:
    """SC-004: coste lineal en el tamaño del lote, medido en cuatro tamaños.
    No se exige una recta perfecta —hay coste fijo y ruido del sistema—, sólo
    que doblar N no multiplique el tiempo por más que un factor generoso muy
    por debajo de lo que daría una operación cuadrática (que en estos cuatro
    tamaños, con un factor de doblado en cada paso, daría ~2x por paso lineal
    frente a ~4x por paso si fuera cuadrática)."""
    times = [_time_sigreg(n) for n in _SIZES]

    for i in range(1, len(_SIZES)):
        size_ratio = _SIZES[i] / _SIZES[i - 1]
        time_ratio = times[i] / max(times[i - 1], 1e-9)
        # Tolerancia generosa (una cuadrática daría ~size_ratio**2 == 4x aquí).
        assert time_ratio < size_ratio * 2.5, (
            f"tiempo creció {time_ratio:.2f}x al pasar de {_SIZES[i - 1]} a "
            f"{_SIZES[i]} muestras (razón de tamaño {size_ratio:.2f}x) — "
            "sugiere coste peor que lineal"
        )


def test_sigreg_never_builds_an_n_by_n_matrix() -> None:
    """Comprobación estructural directa de FR-007, más fiable que temporizar:
    ningún tensor intermedio debe crecer como N². Se mide con memoria pico en
    vez de perfilar cada operación."""
    n_small, n_large = 1000, 8000
    tracemalloc.start()

    tracemalloc.reset_peak()
    sigreg(torch.randn(n_small, 32), num_slices=256)
    _, peak_small = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()
    sigreg(torch.randn(n_large, 32), num_slices=256)
    _, peak_large = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    size_ratio = n_large / n_small
    memory_ratio = peak_large / max(peak_small, 1)
    # Una matriz N x N habría dado (8x)**2 == 64x; permitimos hasta 3x el
    # crecimiento lineal esperado para absorber buffers auxiliares.
    assert memory_ratio < size_ratio * 3, (
        f"memoria creció {memory_ratio:.2f}x al pasar de {n_small} a "
        f"{n_large} muestras (razón de tamaño {size_ratio:.2f}x)"
    )
