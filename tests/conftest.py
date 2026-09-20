from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mammo_lejepa.config import CorpusConfig

_CSV_COLUMNS = [
    "source_dataset",
    "preprocessed_image_path",
    "classification",
    "density",
    "BIRADS",
    "mask_path",
    "raw_image_path",
    "laterality",
    "view",
    "source_subjectID",
    "original_source_path",
    "subject_age",
    "abnormality",
    "molecular_subtype",
    "ROI_path",
    "x",
    "y",
    "radius",
    "abnormality id",
    "calc type",
    "calc distribution",
    "subtlety",
    "mass shape",
    "mass margins",
    "cropped_image_file_new",
]


@pytest.fixture
def corpus_config(tmp_path: Path) -> CorpusConfig:
    """`CorpusConfig` de prueba: rutas bajo un directorio temporal, sin tocar
    `data/` real."""
    return CorpusConfig(
        mammobench_root=tmp_path / "Mammo_Bench_v2",
        output_dir=tmp_path / "corpus",
    )


@pytest.fixture
def sample_mammo_bench_rows() -> list[dict[str, str]]:
    """Muestra reducida de `mammo-bench.csv` en memoria: tres fuentes, un
    paciente con clasificaciones mixtas, una fila DMID con anotación de
    lesión (`ROI_path`/`x`/`y`/`radius`) y campos ausentes como cadena vacía
    (igual que el CSV real, antes de normalizar a `None`)."""

    def row(**overrides: str) -> dict[str, str]:
        base = dict.fromkeys(_CSV_COLUMNS, "")
        base.update(overrides)
        return base

    return [
        row(
            source_dataset="inbreast",
            preprocessed_image_path="Preprocessed_Dataset/inbreast/inbreast_0.jpg",
            mask_path="Masks/inbreast/inbreast_0.jpg",
            raw_image_path="Original_Dataset/inbreast/inbreast_0.jpg",
            classification="Normal",
            density="DENSITY B",
            laterality="L",
            view="CC",
            source_subjectID="20586908",
        ),
        row(
            source_dataset="inbreast",
            preprocessed_image_path="Preprocessed_Dataset/inbreast/inbreast_1.jpg",
            mask_path="Masks/inbreast/inbreast_1.jpg",
            raw_image_path="Original_Dataset/inbreast/inbreast_1.jpg",
            classification="Malignant",
            density="DENSITY B",
            laterality="R",
            view="MLO",
            source_subjectID="20586908",
        ),
        row(
            source_dataset="ddsm",
            preprocessed_image_path="Preprocessed_Dataset/ddsm/ddsm_0.jpg",
            mask_path="Masks/ddsm/ddsm_0.jpg",
            raw_image_path="Original_Dataset/ddsm/ddsm_0.jpg",
            classification="Benign",
            laterality="L",
            view="CC",
            source_subjectID="473.0",
        ),
        row(
            source_dataset="ddsm",
            preprocessed_image_path="Preprocessed_Dataset/ddsm/ddsm_1.jpg",
            mask_path="Masks/ddsm/ddsm_1.jpg",
            raw_image_path="Original_Dataset/ddsm/ddsm_1.jpg",
            classification="Benign",
            laterality="R",
            view="CC",
            source_subjectID="510.0",
        ),
        row(
            source_dataset="dmid",
            preprocessed_image_path="Preprocessed_Dataset/dmid/dmid_0.jpg",
            mask_path="Masks/dmid/dmid_0.jpg",
            raw_image_path="Original_Dataset/dmid/dmid_0.jpg",
            classification="Suspicious Malignant",
            BIRADS="4",
            abnormality="Mass",
            laterality="L",
            view="MLO",
            source_subjectID="P_0001",
            ROI_path="Original_Dataset/dmid/dmid_0_ROI.jpg",
            x="120.5",
            y="80.0",
            radius="15.0",
        ),
    ]


@pytest.fixture
def sample_mammo_bench_csv(
    sample_mammo_bench_rows: list[dict[str, str]],
) -> pl.DataFrame:
    return pl.DataFrame(
        sample_mammo_bench_rows, schema={c: pl.Utf8 for c in _CSV_COLUMNS}
    )
