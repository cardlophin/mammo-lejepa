from __future__ import annotations

import polars as pl
import pytest

from mammo_lejepa.manifest import build_manifest, missing_files


def test_build_manifest_produces_one_entry_per_row(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_mammo_bench_csv)

    assert len(manifest) == sample_mammo_bench_csv.height


def test_build_manifest_normalizes_empty_strings_to_none(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_mammo_bench_csv)

    # La primera fila de la fixture no trae density/BIRADS/abnormality/etc.
    normal_row = next(img for img in manifest if img.image_id == "inbreast_0")
    assert normal_row.birads is None
    assert normal_row.abnormality is None
    assert normal_row.molecular_subtype is None
    assert normal_row.subject_age is None
    assert normal_row.density == "DENSITY B"  # ésta sí viene rellena


def test_build_manifest_derives_patient_key(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_mammo_bench_csv)

    image = next(img for img in manifest if img.image_id == "inbreast_0")
    assert image.patient_key == "inbreast:20586908"


def test_build_manifest_derives_image_id_from_filename(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_mammo_bench_csv)

    image_ids = {img.image_id for img in manifest}
    assert "inbreast_0" in image_ids
    assert "ddsm_0" in image_ids
    assert "dmid_0" in image_ids


def test_build_manifest_missing_column_names_it(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    incomplete = sample_mammo_bench_csv.drop("mask_path")

    with pytest.raises(ValueError, match="mask_path"):
        build_manifest(incomplete)


def test_build_manifest_filters_by_source_deterministically(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_mammo_bench_csv, sources=["ddsm"])

    assert {img.source_dataset for img in manifest} == {"ddsm"}
    assert [img.image_id for img in manifest] == sorted(
        img.image_id for img in manifest
    )


def test_build_manifest_unknown_source_raises(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="nonexistent"):
        build_manifest(sample_mammo_bench_csv, sources=["nonexistent"])


def test_build_manifest_limit_is_deterministic(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    first = build_manifest(sample_mammo_bench_csv, limit=2)
    second = build_manifest(sample_mammo_bench_csv, limit=2)

    assert [img.image_id for img in first] == [img.image_id for img in second]
    assert len(first) == 2


def test_build_manifest_explicit_image_ids(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_mammo_bench_csv, image_ids=["ddsm_0"])

    assert [img.image_id for img in manifest] == ["ddsm_0"]


def test_build_manifest_unknown_image_id_raises(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="nope"):
        build_manifest(sample_mammo_bench_csv, image_ids=["nope"])


def test_missing_files_reports_absent_preprocessed_images(
    sample_mammo_bench_csv: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_mammo_bench_csv)
    existing = {img.preprocessed_path for img in manifest[1:]}  # falta la primera

    problems = missing_files(manifest, existing)

    assert len(problems) == 1
    assert problems[0][0].image_id == manifest[0].image_id
