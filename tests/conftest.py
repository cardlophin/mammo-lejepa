from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mammo_lejepa.config import PipelineConfig


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """`PipelineConfig` de prueba: rutas bajo un directorio temporal, sin
    tocar `data/` real."""
    return PipelineConfig(data_dir=tmp_path / "vindr-mammo")


@pytest.fixture
def sample_annotations() -> pl.DataFrame:
    """Muestra reducida de `breast-level_annotations.csv` en memoria, con dos
    estudios de dos imágenes cada uno, uno por split."""
    return pl.DataFrame(
        {
            "study_id": [
                "study_a",
                "study_a",
                "study_b",
                "study_b",
            ],
            "series_id": [
                "series_a1",
                "series_a1",
                "series_b1",
                "series_b1",
            ],
            "image_id": [
                "image_a_l",
                "image_a_r",
                "image_b_l",
                "image_b_r",
            ],
            "laterality": ["L", "R", "L", "R"],
            "view_position": ["CC", "CC", "MLO", "MLO"],
            "height": [512, 512, 384, 384],
            "width": [384, 384, 512, 512],
            "breast_birads": ["BI-RADS 1", "BI-RADS 1", "BI-RADS 2", "BI-RADS 2"],
            "breast_density": [
                "DENSITY B",
                "DENSITY B",
                "DENSITY C",
                "DENSITY C",
            ],
            "split": ["training", "training", "test", "test"],
        }
    )


@pytest.fixture
def sample_metadata() -> pl.DataFrame:
    """Muestra reducida de `metadata.csv` en memoria, ya desduplicada la
    columna `SOP Instance UID` repetida del CSV real."""
    return pl.DataFrame(
        {
            "SOP Instance UID": [
                "image_a_l",
                "image_a_r",
                "image_b_l",
                "image_b_r",
            ],
        }
    )


@pytest.fixture
def sample_findings() -> pl.DataFrame:
    """Muestra reducida de `finding_annotations.csv`: una fila con caja y una
    sin coordenadas (caso mayoritario del CSV real)."""
    return pl.DataFrame(
        {
            "image_id": ["image_a_l", "image_a_r"],
            "study_id": ["study_a", "study_a"],
            "finding_categories": ["['Mass']", None],
            "finding_birads": ["BI-RADS 4", None],
            "xmin": [50.0, None],
            "ymin": [60.0, None],
            "xmax": [150.0, None],
            "ymax": [160.0, None],
        }
    )
