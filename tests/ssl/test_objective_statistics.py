from __future__ import annotations

import pytest
import torch

from mammo_lejepa.ssl.objective import sigreg


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_sigreg_is_low_for_isotropic_gaussian_samples() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(4000, 32)

    value = sigreg(embeddings, num_slices=512, generator=_generator(1))

    # Sin una referencia absoluta, "bajo" se calibra contra el caso colapsado
    # en los dos tests siguientes (SC-001/SC-002); aquí sólo se acota un techo
    # generoso para detectar una regresión burda.
    assert value.item() < 5.0


def test_sigreg_is_much_higher_for_collapsed_embeddings() -> None:
    torch.manual_seed(0)
    isotropic = torch.randn(4000, 32)
    collapsed = torch.zeros(4000, 32) + 1e-3 * torch.randn(4000, 32)

    iso_value = sigreg(isotropic, num_slices=512, generator=_generator(2))
    collapsed_value = sigreg(collapsed, num_slices=512, generator=_generator(2))

    # SC-002: al menos un orden de magnitud de diferencia.
    assert collapsed_value.item() > 10 * iso_value.item()


def test_sigreg_is_much_higher_for_low_rank_embeddings() -> None:
    torch.manual_seed(0)
    isotropic = torch.randn(4000, 32)

    low_rank_basis = torch.randn(4000, 2)
    projection = torch.randn(2, 32)
    low_rank = low_rank_basis @ projection  # rango <= 2 en 32 dimensiones

    iso_value = sigreg(isotropic, num_slices=512, generator=_generator(3))
    low_rank_value = sigreg(low_rank, num_slices=512, generator=_generator(3))

    assert low_rank_value.item() > 10 * iso_value.item()


def test_sigreg_is_invariant_to_sample_permutation() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(1000, 16)
    permutation = torch.randperm(embeddings.size(0))

    value_original = sigreg(embeddings, num_slices=128, generator=_generator(4))
    value_permuted = sigreg(
        embeddings[permutation], num_slices=128, generator=_generator(4)
    )

    assert value_original.item() == pytest.approx(value_permuted.item(), rel=1e-5)


def test_sigreg_is_invariant_to_dimension_permutation_in_distribution() -> None:
    """Con la MISMA semilla y direcciones aleatorias fijas, permutar las
    dimensiones de entrada SÍ cambia el valor puntual (las direcciones no se
    permutan junto con los datos) — eso no es una fuga de la propiedad, es que
    la invarianza a permutación de dimensiones es una propiedad de la
    distribución sobre direcciones aleatorias isótropas, no del valor exacto
    para una única realización. Se verifica promediando sobre muchas
    realizaciones independientes."""
    torch.manual_seed(0)
    embeddings = torch.randn(1000, 16)
    permutation = torch.randperm(embeddings.size(-1))
    permuted = embeddings[:, permutation]

    seeds = range(30)
    original_values = [
        sigreg(embeddings, num_slices=256, generator=_generator(seed)).item()
        for seed in seeds
    ]
    permuted_values = [
        sigreg(permuted, num_slices=256, generator=_generator(seed)).item()
        for seed in seeds
    ]

    original_mean = sum(original_values) / len(original_values)
    permuted_mean = sum(permuted_values) / len(permuted_values)

    assert original_mean == pytest.approx(permuted_mean, rel=0.15)


def test_sigreg_variance_across_seeds_decreases_with_more_slices() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(2000, 32)

    def spread(num_slices: int) -> float:
        values = [
            sigreg(embeddings, num_slices=num_slices, generator=_generator(seed)).item()
            for seed in range(10)
        ]
        return float(torch.tensor(values).std())

    # Más slices, estimador más estable: el valor no deriva, su varianza baja.
    assert spread(1024) < spread(16)
