from __future__ import annotations

import pytest
import torch

from mammo_lejepa.ssl.diagnostics import (
    effective_rank,
    embedding_report,
    isotropy_deviation,
    spectral_entropy,
)


def test_effective_rank_close_to_dimension_for_isotropic_embeddings() -> None:
    torch.manual_seed(0)
    dim = 32
    embeddings = torch.randn(4000, dim)

    assert effective_rank(embeddings) > 0.9 * dim


def test_effective_rank_close_to_one_for_exact_collapse() -> None:
    embeddings = torch.full((2000, 32), 3.0)

    assert effective_rank(embeddings) == pytest.approx(1.0, abs=1e-3)


def test_effective_rank_close_to_true_rank_for_low_rank_embeddings() -> None:
    torch.manual_seed(0)
    for rank in (1, 2, 5):
        basis = torch.randn(3000, rank)
        projection = torch.randn(rank, 32)
        embeddings = basis @ projection

        assert effective_rank(embeddings) == pytest.approx(rank, rel=0.05, abs=0.1)


def test_isotropy_deviation_near_zero_for_isotropic_embeddings() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(4000, 32)

    assert isotropy_deviation(embeddings) < 0.3


def test_isotropy_deviation_large_for_rank_one_embeddings() -> None:
    torch.manual_seed(0)
    basis = torch.randn(3000, 1)
    projection = torch.randn(1, 32)
    embeddings = basis @ projection

    assert isotropy_deviation(embeddings) > 3.0


def test_spectral_entropy_near_one_for_isotropic_embeddings() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(4000, 32)

    assert spectral_entropy(embeddings) > 0.9


def test_spectral_entropy_near_zero_for_rank_one_embeddings() -> None:
    torch.manual_seed(0)
    basis = torch.randn(3000, 1)
    projection = torch.randn(1, 32)
    embeddings = basis @ projection

    assert spectral_entropy(embeddings) < 0.3


def test_embedding_report_does_not_modify_input_or_require_grad() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(200, 16, requires_grad=True)
    original = embeddings.clone()

    report = embedding_report(embeddings)

    torch.testing.assert_close(embeddings, original)
    assert report.effective_rank > 0
    assert report.mean_norm > 0
