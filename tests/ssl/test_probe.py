from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image
from torch import nn

from mammo_lejepa.ssl.config import EvalConfig
from mammo_lejepa.ssl.probe import (
    extract_embeddings,
    linear_probe,
    run_probes,
)


def _tiny_encoder() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
    )


@pytest.fixture
def catalog(tmp_path: Path) -> pl.DataFrame:
    """Clase `bright` = imágenes claras, `dark` = oscuras: separables por un
    encoder trivial. `val` existe para comprobar que la sonda no lo toca."""
    rows = []
    rng = np.random.default_rng(0)
    layout = (
        [("train", "a")] * 24
        + [("train", "b")] * 24
        + [("val", "a")] * 6
        + [("test", "a")] * 12
        + [("test", "b")] * 12
    )
    for i, (split, source) in enumerate(layout):
        bright = i % 2 == 0
        base = 200 if bright else 40
        pixels = np.clip(base + rng.normal(0, 10, (40, 40)), 0, 255).astype(np.uint8)
        path = tmp_path / f"img_{i}.png"
        Image.fromarray(pixels, mode="L").save(path)
        rows.append(
            {
                "image_id": f"img_{i}",
                "patient_key": f"{source}:p{i}",
                "source_dataset": source,
                "split": split,
                "status": "ok",
                "crop_path": str(path),
                "classification": "bright" if bright else "dark",
                "density": "D" if bright else None,
                "birads": "1",
                "suspect": False,
            }
        )
    return pl.DataFrame(rows)


def _config(catalog_path: Path, **overrides: object) -> EvalConfig:
    defaults = dict(
        catalog_path=catalog_path,
        device="cpu",
        resolution=32,
        batch_size=16,
        probe_epochs=200,
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)  # type: ignore[arg-type]


def test_probe_separates_separable_classes(
    catalog: pl.DataFrame, tmp_path: Path
) -> None:
    result = linear_probe(
        _tiny_encoder(), catalog, target="classification", config=_config(tmp_path)
    )

    assert result.accuracy > 0.9
    assert result.balanced_accuracy > 0.9
    assert result.majority_baseline == pytest.approx(0.5)


def test_no_encoder_parameter_receives_gradient(
    catalog: pl.DataFrame, tmp_path: Path
) -> None:
    encoder = _tiny_encoder()

    linear_probe(encoder, catalog, target="classification", config=_config(tmp_path))

    assert all(p.grad is None for p in encoder.parameters())
    assert not encoder.training


def test_partitions_come_from_the_catalog_and_val_is_untouched(
    catalog: pl.DataFrame, tmp_path: Path
) -> None:
    result = linear_probe(
        _tiny_encoder(), catalog, target="classification", config=_config(tmp_path)
    )

    assert result.n_train == catalog.filter(pl.col("split") == "train").height
    assert result.n_test == catalog.filter(pl.col("split") == "test").height
    assert sum(result.n_test_by_source.values()) == result.n_test


def test_only_rows_with_the_label_are_used(
    catalog: pl.DataFrame, tmp_path: Path
) -> None:
    config = _config(tmp_path)

    result = linear_probe(_tiny_encoder(), catalog, target="density", config=config)

    labelled_train = catalog.filter(
        (pl.col("split") == "train") & pl.col("density").is_not_null()
    ).height
    assert result.n_train == labelled_train
    assert result.n_classes == 1


def test_run_probes_always_includes_the_source_control(
    catalog: pl.DataFrame, tmp_path: Path
) -> None:
    results = run_probes(_tiny_encoder(), catalog, config=_config(tmp_path))

    assert set(results) == {"classification", "density", "BIRADS", "source_dataset"}
    assert results["BIRADS"].n_classes == 1


def test_accuracy_is_reported_per_source(catalog: pl.DataFrame, tmp_path: Path) -> None:
    result = linear_probe(
        _tiny_encoder(), catalog, target="classification", config=_config(tmp_path)
    )

    assert set(result.accuracy_by_source) == {"a", "b"}
    assert result.n_test_by_source == {"a": 12, "b": 12}


def test_max_samples_caps_each_split(catalog: pl.DataFrame, tmp_path: Path) -> None:
    config = _config(tmp_path, max_samples_per_split=10)

    embeddings = extract_embeddings(
        _tiny_encoder(), catalog, split="train", config=config
    )

    assert embeddings.features.shape[0] == 10


def test_same_seed_gives_the_same_result(catalog: pl.DataFrame, tmp_path: Path) -> None:
    encoder = _tiny_encoder()
    config = _config(tmp_path)

    first = linear_probe(encoder, catalog, target="classification", config=config)
    second = linear_probe(encoder, catalog, target="classification", config=config)

    assert first == second
    torch.testing.assert_close(
        torch.tensor(first.accuracy), torch.tensor(second.accuracy)
    )


def test_empty_test_partition_fails_with_a_clear_error(
    catalog: pl.DataFrame, tmp_path: Path
) -> None:
    train_only = catalog.filter(pl.col("split") == "train")

    with pytest.raises(ValueError):
        linear_probe(
            _tiny_encoder(),
            train_only,
            target="classification",
            config=_config(tmp_path),
        )
