from __future__ import annotations

from collections.abc import Container, Sequence

import polars as pl

from mammo_lejepa.models import ImagenMammoBench

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "source_dataset",
        "preprocessed_image_path",
        "mask_path",
        "raw_image_path",
        "laterality",
        "view",
        "source_subjectID",
        "classification",
        "density",
        "BIRADS",
        "abnormality",
        "molecular_subtype",
        "subject_age",
    }
)


def build_manifest(
    catalog_csv: pl.DataFrame,
    *,
    sources: Sequence[str] | None = None,
    limit: int | None = None,
    image_ids: Sequence[str] | None = None,
) -> list[ImagenMammoBench]:
    """Construye el manifiesto de trabajo desde `mammo-bench.csv` (FR-004).
    Normaliza cadenas vacías a `None`, deriva `image_id` (del nombre de
    fichero de `preprocessed_image_path`, sin extensión) y `patient_key`
    (`f"{source_dataset}:{source_subject_id}"`), y produce un orden
    determinista (por `source_dataset` y `preprocessed_image_path`). Acota
    por fuente, por lista explícita de `image_id` y por número de imágenes
    (FR-005). Lanza `ValueError` nombrando la columna, fuente o `image_id`
    ausente."""
    missing_columns = REQUIRED_COLUMNS - set(catalog_csv.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas en mammo-bench.csv: {sorted(missing_columns)}"
        )

    filtered = catalog_csv

    if sources is not None:
        requested_sources = set(sources)
        available_sources = set(filtered.get_column("source_dataset").unique())
        missing_sources = requested_sources - available_sources
        if missing_sources:
            raise ValueError(
                f"Fuentes solicitadas no encontradas: {sorted(missing_sources)}"
            )
        filtered = filtered.filter(
            pl.col("source_dataset").is_in(list(requested_sources))
        )

    ordered = filtered.sort(["source_dataset", "preprocessed_image_path"])
    manifest = [_row_to_image(row) for row in ordered.iter_rows(named=True)]

    if image_ids is not None:
        requested_ids = set(image_ids)
        available_ids = {image.image_id for image in manifest}
        missing_ids = requested_ids - available_ids
        if missing_ids:
            raise ValueError(
                f"image_id solicitados no encontrados: {sorted(missing_ids)}"
            )
        manifest = [image for image in manifest if image.image_id in requested_ids]

    if limit is not None:
        manifest = manifest[:limit]

    return manifest


def missing_files(
    manifest: Sequence[ImagenMammoBench],
    existing: Container[str],
) -> list[tuple[ImagenMammoBench, str]]:
    """Verifica, sin tocar el disco, qué imágenes del manifiesto no tienen su
    fichero preprocesado en `existing` (el conjunto de rutas existentes,
    calculado por el llamador). Devuelve pares `(imagen, qué falta)`
    (FR-006)."""
    problems: list[tuple[ImagenMammoBench, str]] = []

    for image in manifest:
        if image.preprocessed_path not in existing:
            problems.append((image, f"falta preprocessed: {image.preprocessed_path}"))

    return problems


def _row_to_image(row: dict[str, str | None]) -> ImagenMammoBench:
    source_dataset = row["source_dataset"] or ""
    source_subject_id = row["source_subjectID"] or ""
    preprocessed_path = row["preprocessed_image_path"] or ""
    image_id = _stem(preprocessed_path)

    return ImagenMammoBench(
        image_id=image_id,
        source_dataset=source_dataset,
        source_subject_id=source_subject_id,
        patient_key=f"{source_dataset}:{source_subject_id}",
        preprocessed_path=preprocessed_path,
        mask_path=row["mask_path"] or "",
        raw_path=row["raw_image_path"] or "",
        laterality=row["laterality"] or "",
        view=row["view"] or "",
        classification=row["classification"] or "",
        density=_normalize_empty(row["density"]),
        birads=_normalize_empty(row["BIRADS"]),
        abnormality=_normalize_empty(row["abnormality"]),
        molecular_subtype=_normalize_empty(row["molecular_subtype"]),
        subject_age=_normalize_empty(row["subject_age"]),
    )


def _stem(path: str) -> str:
    """Nombre de fichero sin directorio ni extensión, con operaciones de
    cadena puras (sin `pathlib`, prohibido en un módulo puro)."""
    filename = path.rsplit("/", 1)[-1]
    return filename.rsplit(".", 1)[0] if "." in filename else filename


def _normalize_empty(value: str | None) -> str | None:
    """Las etiquetas ausentes vienen como cadena vacía en el CSV; `None`
    significa explícitamente "no anotado", nunca se confunde con un valor
    (data-model.md, notas sobre la fuente)."""
    if value is None or value == "":
        return None
    return value
