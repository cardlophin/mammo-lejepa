from __future__ import annotations

import json
from pathlib import Path

import cv2
import polars as pl
import pytest

from mammo_lejepa.config import CorpusConfig
from mammo_lejepa.manifest import build_manifest
from mammo_lejepa.runner import run_build
from tests.fixtures.synthetic_mammogram import make_synthetic_mammogram

pytestmark = pytest.mark.slow

_SOURCES = ["inbreast", "ddsm", "dmid"]
_IMAGES_PER_SOURCE = 4


def _write_corpus(root: Path) -> pl.DataFrame:
    rows: list[dict[str, str]] = []

    for source in _SOURCES:
        (root / "Preprocessed_Dataset" / source).mkdir(parents=True, exist_ok=True)
        (root / "Masks" / source).mkdir(parents=True, exist_ok=True)
        (root / "Original_Dataset" / source).mkdir(parents=True, exist_ok=True)

        for i in range(_IMAGES_PER_SOURCE):
            image_id = f"{source}_{i}"
            sm = make_synthetic_mammogram(include_pectoral=(i == 0))

            preprocessed_rel = f"Preprocessed_Dataset/{source}/{image_id}.jpg"
            mask_rel = f"Masks/{source}/{image_id}.jpg"
            raw_rel = f"Original_Dataset/{source}/{image_id}.jpg"

            cv2.imwrite(str(root / preprocessed_rel), sm.image)
            assert sm.mask is not None
            cv2.imwrite(str(root / mask_rel), sm.mask)
            cv2.imwrite(str(root / raw_rel), sm.image)

            rows.append(
                {
                    "source_dataset": source,
                    "preprocessed_image_path": preprocessed_rel,
                    "mask_path": mask_rel,
                    "raw_image_path": raw_rel,
                    "laterality": "L" if i % 2 == 0 else "R",
                    "view": "CC",
                    "source_subjectID": f"patient_{i}",
                    "classification": "Normal",
                    "density": "",
                    "BIRADS": "",
                    "abnormality": "",
                    "molecular_subtype": "",
                    "subject_age": "",
                }
            )

    csv_dir = root / "CSV_Files"
    csv_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(csv_dir / "mammo-bench.csv")
    return df


def test_build_end_to_end_produces_crops_and_catalog(tmp_path: Path) -> None:
    root = tmp_path / "Mammo_Bench_v2"
    _write_corpus(root)

    config = CorpusConfig(
        mammobench_root=root, output_dir=tmp_path / "corpus", workers=2
    )
    catalog_csv = pl.read_csv(config.csv_path, infer_schema_length=0)
    manifest = build_manifest(catalog_csv)

    assert len(manifest) == len(_SOURCES) * _IMAGES_PER_SOURCE

    summary = run_build(manifest, config=config, run_id="run_test_001")

    expected_total = len(_SOURCES) * _IMAGES_PER_SOURCE
    assert summary.n_total == expected_total
    assert summary.n_ok == expected_total
    assert summary.n_failed == 0
    assert summary.exit_reason == "completed"

    for source in _SOURCES:
        for i in range(_IMAGES_PER_SOURCE):
            crop_path = config.crop_path(source, f"{source}_{i}")
            assert crop_path.exists()
            assert crop_path.stat().st_size > 0

    lines = config.records_jsonl_path.read_text().splitlines()
    assert len(lines) == expected_total

    records = [json.loads(line) for line in lines]
    required_fields = {
        "image_id",
        "source_dataset",
        "source_subject_id",
        "patient_key",
        "laterality",
        "view",
        "status",
        "process_seconds",
        "run_id",
        "code_version",
        "processed_at",
        "box",
        "fallback_reason",
        "mask_area_ratio",
        "mask_otsu_iou",
        "suspect",
        "suspect_reason",
        "crop_path",
        "crop_bytes",
        "classification",
    }
    counts_by_source: dict[str, int] = {}
    for record in records:
        assert required_fields <= record.keys()
        assert record["status"] == "ok"
        assert record["run_id"] == "run_test_001"
        counts_by_source[record["source_dataset"]] = (
            counts_by_source.get(record["source_dataset"], 0) + 1
        )

    assert counts_by_source == {source: _IMAGES_PER_SOURCE for source in _SOURCES}
