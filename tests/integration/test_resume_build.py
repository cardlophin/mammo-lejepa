from __future__ import annotations

from pathlib import Path

import cv2
import polars as pl
import pytest

from mammo_lejepa.config import CorpusConfig
from mammo_lejepa.manifest import build_manifest
from mammo_lejepa.resume import latest_records_by_image, parse_jsonl_records
from mammo_lejepa.runner import run_build
from tests.fixtures.synthetic_mammogram import make_synthetic_mammogram

pytestmark = pytest.mark.slow

_IMAGES = ["inbreast_0", "inbreast_1", "inbreast_2", "inbreast_3"]


def _write_corpus(root: Path) -> None:
    source_dir = root / "Preprocessed_Dataset" / "inbreast"
    mask_dir = root / "Masks" / "inbreast"
    source_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id in _IMAGES:
        sm = make_synthetic_mammogram()
        cv2.imwrite(str(source_dir / f"{image_id}.jpg"), sm.image)
        assert sm.mask is not None
        cv2.imwrite(str(mask_dir / f"{image_id}.jpg"), sm.mask)
        preprocessed_rel = f"Preprocessed_Dataset/inbreast/{image_id}.jpg"
        rows.append(
            {
                "source_dataset": "inbreast",
                "preprocessed_image_path": preprocessed_rel,
                "mask_path": f"Masks/inbreast/{image_id}.jpg",
                "raw_image_path": f"Original_Dataset/inbreast/{image_id}.jpg",
                "laterality": "L",
                "view": "CC",
                "source_subjectID": image_id,
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
    pl.DataFrame(rows).write_csv(csv_dir / "mammo-bench.csv")


def test_interrupted_build_resumes_without_reprocessing(tmp_path: Path) -> None:
    root = tmp_path / "Mammo_Bench_v2"
    _write_corpus(root)

    config = CorpusConfig(
        mammobench_root=root, output_dir=tmp_path / "corpus", workers=2
    )
    catalog_csv = pl.read_csv(config.csv_path, infer_schema_length=0)
    all_images = build_manifest(catalog_csv)

    # "Interrupción a mitad": sólo se lanzan las 2 primeras imágenes.
    first_half = all_images[:2]
    run_build(first_half, config=config, run_id="run_before_interrupt")

    lines_after_first = config.records_jsonl_path.read_text().splitlines()
    assert len(lines_after_first) == 2

    # Al relanzar con el manifiesto COMPLETO, la reanudación integrada en
    # runner.py (T023) descuenta lo ya resuelto automáticamente.
    summary = run_build(all_images, config=config, run_id="run_after_resume")

    assert summary.n_total == 2  # sólo las 2 restantes, no las 4
    assert summary.n_ok == 2

    lines = config.records_jsonl_path.read_text().splitlines()
    assert len(lines) == 4  # 2 del primer run + 2 del segundo, sin duplicar trabajo

    records = parse_jsonl_records(lines)
    latest = latest_records_by_image(records)
    assert len(latest) == 4
    assert all(record.status == "ok" for record in latest.values())

    for image_id in _IMAGES:
        crop_path = config.crop_path("inbreast", image_id)
        assert crop_path.exists()


def test_fully_completed_build_processes_nothing_on_rerun(tmp_path: Path) -> None:
    root = tmp_path / "Mammo_Bench_v2"
    _write_corpus(root)

    config = CorpusConfig(
        mammobench_root=root, output_dir=tmp_path / "corpus", workers=2
    )
    catalog_csv = pl.read_csv(config.csv_path, infer_schema_length=0)
    all_images = build_manifest(catalog_csv)

    first_summary = run_build(all_images, config=config, run_id="run_full")
    assert first_summary.n_ok == len(_IMAGES)

    second_summary = run_build(all_images, config=config, run_id="run_rerun")

    assert second_summary.n_total == 0
    assert second_summary.n_ok == 0

    lines = config.records_jsonl_path.read_text().splitlines()
    assert len(lines) == len(_IMAGES)  # nada se repitió
