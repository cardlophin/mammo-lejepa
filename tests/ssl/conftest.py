from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image


@pytest.fixture
def tiny_catalog(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)

    rows = []
    for i in range(20):
        png_path = images_dir / f"img_{i}.png"
        pixels = (rng.random((200, 150)) * 255).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(png_path)
        rows.append(
            {
                "image_id": f"img_{i}",
                "patient_key": f"src:p{i}",
                "source_dataset": "src",
                "split": "train",
                "status": "ok",
                "crop_path": str(png_path),
                "classification": "Normal",
                "density": "DENSITY B",
                "birads": "2",
                "suspect": False,
            }
        )

    catalog_path = tmp_path / "catalog.parquet"
    pl.DataFrame(rows).write_parquet(catalog_path)
    return catalog_path
