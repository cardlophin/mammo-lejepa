from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from mammo_lejepa.ssl.augment import build_view_transform
from mammo_lejepa.ssl.config import AugmentConfig
from mammo_lejepa.ssl.data import CropDataset, collate_views


def _write_png(path: Path, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pixels = (rng.random((200, 150)) * 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


@pytest.fixture
def sample_catalog(tmp_path: Path) -> pl.DataFrame:
    rows = []
    for i, (split, source, suspect) in enumerate(
        [
            ("train", "inbreast", False),
            ("train", "inbreast", True),
            ("train", "ddsm", False),
            ("val", "inbreast", False),
            ("test", "ddsm", False),
        ]
    ):
        png_path = tmp_path / f"img_{i}.png"
        _write_png(png_path, seed=i)
        rows.append(
            {
                "image_id": f"img_{i}",
                "patient_key": f"{source}:p{i}",
                "source_dataset": source,
                "split": split,
                "status": "ok",
                "crop_path": str(png_path),
                "classification": "Normal",
                "density": "DENSITY B",
                "birads": "2",
                "suspect": suspect,
            }
        )
    return pl.DataFrame(rows)


def _dataset(catalog: pl.DataFrame, **kwargs) -> CropDataset:
    transform = build_view_transform(AugmentConfig(resolution=32))
    return CropDataset(catalog, transform=transform, **kwargs)


def test_only_rows_from_requested_split_appear(sample_catalog: pl.DataFrame) -> None:
    dataset = _dataset(sample_catalog, split="train", views=2)

    assert len(dataset) == 3
    for row in dataset._rows:
        assert row["split"] == "train"


def test_each_item_has_v_views(sample_catalog: pl.DataFrame) -> None:
    dataset = _dataset(sample_catalog, split="train", views=4)

    views, labels = dataset[0]

    assert views.shape[0] == 4
    assert views.shape[1:] == (3, 32, 32)
    assert labels["image_id"] == "img_0"


def test_suspect_filtering_reports_the_count(sample_catalog: pl.DataFrame) -> None:
    included = _dataset(sample_catalog, split="train", views=1, include_suspect=True)
    excluded = _dataset(sample_catalog, split="train", views=1, include_suspect=False)

    assert len(included) == 3
    assert included.n_excluded_suspect == 0
    assert len(excluded) == 2
    assert excluded.n_excluded_suspect == 1


def test_source_filter_restricts_to_requested_sources(
    sample_catalog: pl.DataFrame,
) -> None:
    dataset = _dataset(sample_catalog, split="train", views=1, sources=["ddsm"])

    assert len(dataset) == 1
    assert dataset._rows[0]["source_dataset"] == "ddsm"


def test_val_and_test_splits_are_isolated(sample_catalog: pl.DataFrame) -> None:
    val_dataset = _dataset(sample_catalog, split="val", views=1)
    test_dataset = _dataset(sample_catalog, split="test", views=1)

    assert len(val_dataset) == 1
    assert len(test_dataset) == 1
    assert val_dataset._rows[0]["image_id"] != test_dataset._rows[0]["image_id"]


def test_collate_views_produces_v_first_shape(sample_catalog: pl.DataFrame) -> None:
    dataset = _dataset(sample_catalog, split="train", views=3)
    batch = [dataset[i] for i in range(len(dataset))]

    views, labels = collate_views(batch)

    assert views.shape == (3, 3, 3, 32, 32)  # [V, B, C, H, W]
    assert len(labels["image_id"]) == 3
