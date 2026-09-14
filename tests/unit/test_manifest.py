from __future__ import annotations

import polars as pl
import pytest

from mammo_lejepa.manifest import (
    batch_by_study,
    build_manifest,
    count_metadata_discrepancies,
)


def test_build_manifest_returns_unique_pairs_in_deterministic_order(
    sample_annotations: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_annotations)

    assert manifest.select("study_id", "image_id").is_duplicated().sum() == 0
    assert manifest.get_column("study_id").to_list() == [
        "study_a",
        "study_a",
        "study_b",
        "study_b",
    ]


def test_build_manifest_n_studies_selects_first_in_order(
    sample_annotations: pl.DataFrame,
) -> None:
    manifest = build_manifest(sample_annotations, n_studies=1)

    assert manifest.get_column("study_id").unique().to_list() == ["study_a"]


def test_build_manifest_missing_split_raises(sample_annotations: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="nonexistent"):
        build_manifest(sample_annotations, split="nonexistent")


def test_build_manifest_missing_columns_names_them(
    sample_annotations: pl.DataFrame,
) -> None:
    incomplete = sample_annotations.drop("breast_birads")

    with pytest.raises(ValueError, match="breast_birads"):
        build_manifest(incomplete)


def test_batch_by_study_does_not_assume_four_images(
    sample_annotations: pl.DataFrame,
) -> None:
    three_images = sample_annotations.head(3)  # study_a completo + 1 de study_b
    manifest = build_manifest(three_images)

    batches = batch_by_study(manifest)

    assert sorted(len(b) for b in batches) == [1, 2]
    assert all(
        task.study_id == batch[0].study_id for batch in batches for task in batch
    )


def test_count_metadata_discrepancies_matching_ids_are_zero(
    sample_annotations: pl.DataFrame, sample_metadata: pl.DataFrame
) -> None:
    only_in_metadata, only_in_annotations = count_metadata_discrepancies(
        sample_metadata, sample_annotations
    )

    assert (only_in_metadata, only_in_annotations) == (0, 0)


def test_count_metadata_discrepancies_counts_each_direction(
    sample_annotations: pl.DataFrame, sample_metadata: pl.DataFrame
) -> None:
    metadata_with_extra = pl.concat(
        [sample_metadata, pl.DataFrame({"SOP Instance UID": ["orphan_in_metadata"]})]
    )
    annotations_missing_one = sample_annotations.filter(
        pl.col("image_id") != "image_b_r"
    )

    only_in_metadata, only_in_annotations = count_metadata_discrepancies(
        metadata_with_extra, annotations_missing_one
    )

    assert only_in_metadata == 2  # orphan_in_metadata + image_b_r
    assert only_in_annotations == 0


def test_count_metadata_discrepancies_does_not_alter_manifest(
    sample_annotations: pl.DataFrame, sample_metadata: pl.DataFrame
) -> None:
    before = build_manifest(sample_annotations)
    count_metadata_discrepancies(sample_metadata, sample_annotations)
    after = build_manifest(sample_annotations)

    assert before.equals(after)
