from __future__ import annotations

import pytest
import torch

from mammo_lejepa.ssl.objective import invariance, lejepa_loss, sigreg


def test_lejepa_loss_requires_three_dimensional_views() -> None:
    with pytest.raises(ValueError, match=r"\[V, B, D\]"):
        lejepa_loss(torch.randn(8, 16))  # [B, D], falta V


def test_invariance_requires_three_dimensional_views() -> None:
    with pytest.raises(ValueError, match=r"\[V, B, D\]"):
        invariance(torch.randn(8, 16))


def test_sigreg_requires_at_least_two_dimensions() -> None:
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        sigreg(torch.randn(16))


def test_invariance_zero_for_identical_views() -> None:
    single_view = torch.randn(1, 8, 16)
    views = single_view.expand(4, 8, 16)

    assert invariance(views).item() == pytest.approx(0.0, abs=1e-6)


def test_invariance_grows_with_dispersion() -> None:
    torch.manual_seed(0)
    base = torch.randn(1, 8, 16)
    small_noise = base + 0.01 * torch.randn(4, 8, 16)
    large_noise = base + 1.0 * torch.randn(4, 8, 16)

    assert invariance(small_noise) < invariance(large_noise)


def test_lejepa_loss_returns_three_separate_terms() -> None:
    torch.manual_seed(0)
    views = torch.randn(4, 8, 16)

    loss = lejepa_loss(views, num_slices=64)

    assert loss.total.ndim == 0
    assert loss.sigreg_term.ndim == 0
    assert loss.invariance_term.ndim == 0
    expected_total = 0.02 * loss.sigreg_term + 0.98 * loss.invariance_term
    assert loss.total.item() == pytest.approx(expected_total.item(), rel=1e-5)


def test_lejepa_loss_respects_custom_lambda() -> None:
    torch.manual_seed(0)
    views = torch.randn(4, 8, 16)

    loss = lejepa_loss(views, lamb=0.5, num_slices=64)

    expected_total = 0.5 * loss.sigreg_term + 0.5 * loss.invariance_term
    assert loss.total.item() == pytest.approx(expected_total.item(), rel=1e-5)


def test_no_term_is_detached_from_the_graph() -> None:
    torch.manual_seed(0)
    views = torch.randn(4, 8, 16, requires_grad=True)

    loss = lejepa_loss(views, num_slices=64)

    assert loss.total.requires_grad
    assert loss.sigreg_term.requires_grad
    assert loss.invariance_term.requires_grad


def test_gradients_are_finite_through_a_linear_module() -> None:
    torch.manual_seed(0)
    linear = torch.nn.Linear(8, 16)
    raw_views = torch.randn(4, 8, 8)

    views = linear(raw_views)
    loss = lejepa_loss(views, num_slices=64)
    loss.total.backward()

    for name, param in linear.named_parameters():
        assert param.grad is not None, f"{name} sin gradiente"
        assert torch.isfinite(param.grad).all(), f"{name} con NaN/Inf"


def test_gradients_are_finite_with_small_batch() -> None:
    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 8)
    raw_views = torch.randn(2, 3, 4)  # lote pequeño: B=3

    views = linear(raw_views)
    loss = lejepa_loss(views, num_slices=32)
    loss.total.backward()

    for param in linear.parameters():
        assert torch.isfinite(param.grad).all()


def test_sigreg_is_reproducible_with_the_same_generator() -> None:
    embeddings = torch.randn(200, 16)

    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)

    value_a = sigreg(embeddings, num_slices=64, generator=generator_a)
    value_b = sigreg(embeddings, num_slices=64, generator=generator_b)

    assert value_a.item() == pytest.approx(value_b.item(), rel=1e-9)
