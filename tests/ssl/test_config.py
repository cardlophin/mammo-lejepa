from __future__ import annotations

from pathlib import Path

from mammo_lejepa.ssl.config import (
    AugmentConfig,
    EvalConfig,
    TrainConfig,
    resolve_device,
)


def test_resolve_device_returns_explicit_value_unchanged() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"


def test_resolve_device_auto_detects_something_valid() -> None:
    assert resolve_device(None) in {"cuda", "mps", "cpu"}


def test_train_config_round_trips_through_json() -> None:
    config = TrainConfig(
        catalog_path=Path("data/corpus/catalog.parquet"),
        run_dir=Path("runs/run_x"),
        sources=("ddsm", "inbreast"),
        augment=AugmentConfig(resolution=128, crop_scale_min=0.3),
    )

    restored = TrainConfig.from_json(config.to_json())

    assert restored == config
    assert isinstance(restored.catalog_path, Path)
    assert isinstance(restored.sources, tuple)
    assert isinstance(restored.augment, AugmentConfig)


def test_train_config_with_no_sources_round_trips() -> None:
    config = TrainConfig(
        catalog_path=Path("data/corpus/catalog.parquet"), run_dir=Path("runs/run_x")
    )

    restored = TrainConfig.from_json(config.to_json())

    assert restored == config
    assert restored.sources is None


def test_eval_config_round_trips_through_json() -> None:
    config = EvalConfig(catalog_path=Path("data/corpus/catalog.parquet"))

    restored = EvalConfig.from_json(config.to_json())

    assert restored == config
    assert isinstance(restored.catalog_path, Path)
    assert isinstance(restored.targets, tuple)


def test_train_config_json_is_stable_across_runs() -> None:
    config = TrainConfig(
        catalog_path=Path("data/corpus/catalog.parquet"), run_dir=Path("runs/run_x")
    )

    assert config.to_json() == config.to_json()
