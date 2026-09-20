from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from mammo_lejepa.vindr.models import ImageTask

REQUIRED_ANNOTATION_COLUMNS: frozenset[str] = frozenset(
    {
        "study_id",
        "series_id",
        "image_id",
        "laterality",
        "view_position",
        "height",
        "width",
        "breast_birads",
        "breast_density",
        "split",
    }
)


def build_manifest(
    annotations: pl.DataFrame,
    *,
    split: str | None = None,
    study_ids: Sequence[str] | None = None,
    n_studies: int | None = None,
) -> pl.DataFrame:
    """Construye el manifiesto de trabajo a partir de `breast-level_annotations`
    (FR-003), acotado por split, lista de estudios y número de estudios
    (FR-004). Devuelve pares `(study_id, image_id)` únicos y ordenados de forma
    determinista; `n_studies` selecciona siempre los primeros del orden, nunca
    una muestra aleatoria sin semilla (FR-005)."""
    missing_columns = REQUIRED_ANNOTATION_COLUMNS - set(annotations.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas en breast-level_annotations.csv: "
            f"{sorted(missing_columns)}"
        )

    filtered = annotations

    if split is not None:
        available_splits = filtered.get_column("split").unique().to_list()
        if split not in available_splits:
            raise ValueError(
                f"El split {split!r} no existe. Splits disponibles: "
                f"{sorted(available_splits)}"
            )
        filtered = filtered.filter(pl.col("split") == split)

    if study_ids is not None:
        requested = set(study_ids)
        available = set(filtered.get_column("study_id").to_list())
        missing_studies = requested - available
        if missing_studies:
            raise ValueError(
                f"study_id solicitados no encontrados: {sorted(missing_studies)}"
            )
        filtered = filtered.filter(pl.col("study_id").is_in(list(requested)))

    study_order = filtered.select("study_id").unique().sort("study_id")

    if study_order.is_empty():
        raise ValueError("No hay ningún estudio que cumpla los filtros solicitados.")

    if n_studies is not None:
        study_order = study_order.head(n_studies)

    manifest = filtered.join(study_order, on="study_id", how="inner").sort(
        ["study_id", "image_id"]
    )

    return manifest


def batch_by_study(manifest: pl.DataFrame) -> list[list[ImageTask]]:
    """Agrupa el manifiesto en lotes por estudio, sin asumir un número fijo de
    imágenes por estudio (FR-006). Respeta el orden determinista de
    `build_manifest`."""
    batches: list[list[ImageTask]] = []

    for _, group in manifest.group_by("study_id", maintain_order=True):
        tasks = [
            ImageTask(
                study_id=row["study_id"],
                series_id=row["series_id"],
                image_id=row["image_id"],
                split=row["split"],
                laterality=row["laterality"],
                view_position=row["view_position"],
                breast_birads=row["breast_birads"],
                breast_density=row["breast_density"],
                expected_height=int(row["height"]),
                expected_width=int(row["width"]),
            )
            for row in group.sort("image_id").iter_rows(named=True)
        ]
        batches.append(tasks)

    return batches


def count_metadata_discrepancies(
    metadata: pl.DataFrame,
    annotations: pl.DataFrame,
) -> tuple[int, int]:
    """Compara el `image_id` de `annotations` (`breast-level_annotations.csv`)
    contra el `SOP Instance UID` de `metadata` (`metadata.csv`) —ambos
    identifican la misma imagen bajo un nombre de columna distinto—, y
    devuelve el recuento de discrepancias en cada sentido
    `(only_in_metadata, only_in_annotations)`. No lanza excepción y no
    modifica el manifiesto."""
    metadata_ids = set(metadata.get_column("SOP Instance UID").to_list())
    annotation_ids = set(annotations.get_column("image_id").to_list())

    only_in_metadata = len(metadata_ids - annotation_ids)
    only_in_annotations = len(annotation_ids - metadata_ids)

    return only_in_metadata, only_in_annotations
