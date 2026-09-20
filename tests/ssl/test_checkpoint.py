from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from mammo_lejepa.ssl.checkpoint import (
    TrainState,
    capture_rng_state,
    latest_valid,
    load,
    restore_rng_state,
    save,
)


def _dummy_state(epoch: int, step: int) -> TrainState:
    model = torch.nn.Linear(4, 4)
    return TrainState(
        epoch=epoch,
        step=step,
        model_state=model.state_dict(),
        projector_state={},
        optimizer_state={},
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_state=None,
        numpy_rng_state=np.random.get_state(),
        python_rng_state=random.getstate(),
    )


def test_save_and_load_round_trips_model_state(tmp_path: Path) -> None:
    state = _dummy_state(epoch=3, step=100)
    path = tmp_path / "epoch_0003.pt"

    save(path, state)
    restored = load(path)

    assert restored.epoch == 3
    assert restored.step == 100
    for key, value in state.model_state.items():
        torch.testing.assert_close(restored.model_state[key], value)


def test_save_leaves_no_part_file_after_success(tmp_path: Path) -> None:
    path = tmp_path / "epoch_0001.pt"
    save(path, _dummy_state(1, 10))

    assert path.exists()
    assert not path.with_name(path.name + ".part").exists()


def test_latest_valid_picks_the_highest_epoch(tmp_path: Path) -> None:
    for epoch in (1, 5, 3):
        save(tmp_path / f"epoch_{epoch:04d}.pt", _dummy_state(epoch, epoch * 10))

    result = latest_valid(tmp_path)

    assert result is not None
    assert result.name == "epoch_0005.pt"


def test_latest_valid_ignores_incomplete_part_files(tmp_path: Path) -> None:
    save(tmp_path / "epoch_0001.pt", _dummy_state(1, 10))
    (tmp_path / "epoch_0002.pt.part").write_bytes(b"incomplete")

    result = latest_valid(tmp_path)

    assert result is not None
    assert result.name == "epoch_0001.pt"


def test_latest_valid_returns_none_when_directory_is_empty(tmp_path: Path) -> None:
    assert latest_valid(tmp_path) is None
    assert latest_valid(tmp_path / "does_not_exist") is None


def test_rng_state_round_trip_reproduces_the_same_random_sequence() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    captured = capture_rng_state()
    expected_next_torch = torch.rand(5)
    expected_next_python = [random.random() for _ in range(5)]
    expected_next_numpy = np.random.rand(5)

    restore_rng_state(captured)

    torch.testing.assert_close(torch.rand(5), expected_next_torch)
    assert [random.random() for _ in range(5)] == expected_next_python
    np.testing.assert_allclose(np.random.rand(5), expected_next_numpy)
