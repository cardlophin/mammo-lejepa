from __future__ import annotations

from pathlib import Path

import cv2
import polars as pl
import pytest

from mammo_lejepa.catalog import consolidate
from mammo_lejepa.config import CorpusConfig
from mammo_lejepa.manifest import build_manifest
from mammo_lejepa.resume import latest_records_by_image, parse_jsonl_records
from mammo_lejepa.runner import run_build
from mammo_lejepa.splits import assign_splits, verify_no_patient_leakage
from tests.fixtures.synthetic_mammogram import make_synthetic_mammogram

pytestmark = pytest.mark.slow

_SOURCES = ["inbreast", "ddsm"]
_PATIENTS_PER_SOURCE = 15
_IMAGES_PER_PATIENT = 2
_CLASSIFICATIONS = ["Normal", "Benign", "Malignant"]


def _write_corpus(root: Path) -> None:
    rows = []
    for source in _SOURCES:
        (root / "Preprocessed_Dataset" / source).mkdir(parents=True, exist_ok=True)
        (root / "Masks" / source).mkdir(parents=True, exist_ok=True)

        for patient_i in range(_PATIENTS_PER_SOURCE):
            for image_i in range(_IMAGES_PER_PATIENT):
                image_id = f"{source}_p{patient_i}_{image_i}"
                sm = make_synthetic_mammogram()
                cv2.imwrite(
                    str(root / "Preprocessed_Dataset" / source / f"{image_id}.jpg"),
                    sm.image,
                )
                assert sm.mask is not None
                cv2.imwrite(str(root / "Masks" / source / f"{image_id}.jpg"), sm.mask)
                rows.append(
                    {
                        "source_dataset": source,
                        "preprocessed_image_path": (
                            f"Preprocessed_Dataset/{source}/{image_id}.jpg"
                        ),
                        "mask_path": f"Masks/{source}/{image_id}.jpg",
                        "raw_image_path": f"Original_Dataset/{source}/{image_id}.jpg",
                        "laterality": "L" if image_i == 0 else "R",
                        "view": "CC",
                        "source_subjectID": f"patient_{patient_i}",
                        "classification": _CLASSIFICATIONS[
                            (patient_i + image_i) % len(_CLASSIFICATIONS)
                        ],
                        "density": "",
                        "BIRADS": "",
                        "abnormality": "",
                        "molecular_subtype": "",
                        "subject_age": "",
                    }
                )

    csv_dir = root / "CSV_Files"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(csv_dir / "mammo-bench.csv")


def test_no_patient_appears_in_more_than_one_split_on_real_catalog(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Mammo_Bench_v2"
    _write_corpus(root)

    config = CorpusConfig(
        mammobench_root=root, output_dir=tmp_path / "corpus", workers=2
    )
    catalog_csv = pl.read_csv(config.csv_path, infer_schema_length=0)
    manifest = build_manifest(catalog_csv)

    summary = run_build(manifest, config=config, run_id="run_integrity")
    assert summary.n_failed == 0

    lines = config.records_jsonl_path.read_text().splitlines()
    records = parse_jsonl_records(lines)
    catalog = consolidate(records)

    latest_records = list(latest_records_by_image(records).values())
    assignment = assign_splits(latest_records, seed=0)
    split_column = catalog["patient_key"].replace_strict(assignment, default=None)
    catalog = catalog.with_columns(split_column.alias("split"))

    assert catalog.filter(pl.col("split").is_null()).height == 0

    pairs = list(
        zip(catalog["patient_key"].to_list(), catalog["split"].to_list(), strict=True)
    )
    verify_no_patient_leakage(pairs)  # no debe lanzar

    # Comprobación directa e independiente (T036): intersección de
    # patient_key entre particiones exactamente vacía.
    by_split = {
        split_name: set(
            catalog.filter(pl.col("split") == split_name)["patient_key"].to_list()
        )
        for split_name in ("train", "val", "test")
    }
    assert by_split["train"] & by_split["val"] == set()
    assert by_split["train"] & by_split["test"] == set()
    assert by_split["val"] & by_split["test"] == set()

    total_patients = len(_SOURCES) * _PATIENTS_PER_SOURCE
    assert sum(len(keys) for keys in by_split.values()) == total_patients
