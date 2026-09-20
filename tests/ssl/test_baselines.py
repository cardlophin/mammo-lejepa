from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mammo_lejepa.ssl.baselines import (
    _ImageNetNormalized,
    build_conditions,
    lejepa_encoder,
    random_encoder,
)
from mammo_lejepa.ssl.checkpoint import latest_valid, load
from mammo_lejepa.ssl.config import TrainConfig
from mammo_lejepa.ssl.trainer import train

pytestmark = pytest.mark.slow


@pytest.fixture
def trained_run(tiny_catalog: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    train(
        TrainConfig(
            catalog_path=tiny_catalog,
            run_dir=run_dir,
            device="cpu",
            views=2,
            batch_size=4,
            epochs=1,
            num_slices=16,
            min_batch_size=1,
        )
    )
    return run_dir


def _assert_frozen(encoder: torch.nn.Module) -> None:
    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())


def test_random_encoder_is_frozen_and_seed_deterministic() -> None:
    first, dim = random_encoder("resnet50", seed=3)
    second, _ = random_encoder("resnet50", seed=3)
    other, _ = random_encoder("resnet50", seed=4)

    _assert_frozen(first)
    assert dim == 2048
    a, b, c = (next(e.parameters()) for e in (first, second, other))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_random_encoder_does_not_disturb_global_rng() -> None:
    torch.manual_seed(11)
    expected = torch.rand(3)

    torch.manual_seed(11)
    random_encoder("resnet50", seed=99)

    assert torch.equal(torch.rand(3), expected)


def test_lejepa_encoder_loads_the_latest_checkpoint(trained_run: Path) -> None:
    encoder, _, path = lejepa_encoder(trained_run)

    saved = load(latest_valid(trained_run / "checkpoints"), map_location="cpu")
    assert path == latest_valid(trained_run / "checkpoints")
    _assert_frozen(encoder)
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, saved.model_state[key])


def test_lejepa_encoder_fails_clearly_without_a_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        TrainConfig(catalog_path=tmp_path / "c.parquet", run_dir=tmp_path).to_json()
    )

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        lejepa_encoder(tmp_path)


def test_build_conditions_without_imagenet(trained_run: Path) -> None:
    conditions = build_conditions(trained_run, include_imagenet=False)

    assert list(conditions) == ["random", "lejepa"]
    for encoder in conditions.values():
        _assert_frozen(encoder)


def test_imagenet_wrapper_normalizes_before_the_encoder() -> None:
    class Probe(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    wrapped = _ImageNetNormalized(Probe())
    x = torch.full((1, 3, 2, 2), 0.485)

    out = wrapped(x)

    assert out[0, 0].abs().max() == pytest.approx(0.0, abs=1e-6)
    assert out[0, 1].abs().max() > 0
