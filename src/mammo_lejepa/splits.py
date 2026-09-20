from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from mammo_lejepa.models import RegistroDeRecorte

_CLASSIFICATION_SEVERITY: tuple[str, ...] = (
    "Malignant",
    "Suspicious Malignant",
    "Benign",
    "Normal",
)

_SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def patient_classification(classifications: Sequence[str]) -> str:
    """Reduce las clasificaciones por imagen de un paciente a una sola,
    quedándose con la más severa presente según `_CLASSIFICATION_SEVERITY`
    (spec.md FR-018, Clarifications 2026-09-15). Lanza `ValueError` ante una
    clasificación fuera de ese conjunto cerrado, en vez de ignorarla en
    silencio, o ante una secuencia vacía."""
    if not classifications:
        raise ValueError("classifications no puede estar vacío")

    unknown = sorted(set(classifications) - set(_CLASSIFICATION_SEVERITY))
    if unknown:
        raise ValueError(f"Clasificación fuera del conjunto cerrado: {unknown}")

    for severity in _CLASSIFICATION_SEVERITY:
        if severity in classifications:
            return severity

    raise AssertionError("unreachable: unknown ya descartó valores fuera del conjunto")


def _split_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Reparte `n` unidades en tres tamaños enteros proporcionales a
    `ratios`, sumando exactamente `n` (método del resto mayor: el remanente
    de redondear hacia abajo va a las particiones con mayor parte
    fraccionaria, empezando por `train`)."""
    raw = [ratio * n for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = n - sum(counts)

    order = sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1

    return counts[0], counts[1], counts[2]


def assign_splits(
    records: Sequence[RegistroDeRecorte],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    stratify_by: Sequence[str] = ("source_dataset", "classification"),
) -> dict[str, str]:
    """Asigna cada `patient_key` (nunca la imagen, FR-017) a `train`/`val`/
    `test`. Sólo considera registros `ok` con `classification` presente.
    Misma semilla y mismos datos dan la misma asignación con independencia
    del orden de `records`: cada estrato usa su propio generador aleatorio,
    sembrado con `(seed, estrato)`, sobre la lista de `patient_key` ordenada
    alfabéticamente."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios debe sumar 1.0: {ratios}")

    patient_sources: dict[str, str] = {}
    patient_classifications: dict[str, list[str]] = {}
    for record in records:
        if record.status != "ok" or record.classification is None:
            continue
        patient_sources[record.patient_key] = record.source_dataset
        patient_classifications.setdefault(record.patient_key, []).append(
            record.classification
        )

    strata: dict[tuple[str, ...], list[str]] = {}
    for patient_key, classifications in patient_classifications.items():
        key_parts = []
        for field_name in stratify_by:
            if field_name == "classification":
                key_parts.append(patient_classification(classifications))
            elif field_name == "source_dataset":
                key_parts.append(patient_sources[patient_key])
            else:
                raise ValueError(f"Campo de estratificación desconocido: {field_name}")
        strata.setdefault(tuple(key_parts), []).append(patient_key)

    assignment: dict[str, str] = {}
    for stratum_key in sorted(strata):
        patient_keys = sorted(strata[stratum_key])
        rng = random.Random(f"{seed}:{stratum_key}")
        rng.shuffle(patient_keys)

        sizes = _split_sizes(len(patient_keys), ratios)
        start = 0
        for split_name, size in zip(_SPLIT_NAMES, sizes, strict=True):
            for patient_key in patient_keys[start : start + size]:
                assignment[patient_key] = split_name
            start += size

    return assignment


def verify_no_patient_leakage(pairs: Iterable[tuple[str, str]]) -> None:
    """Lanza `ValueError` si el mismo `patient_key` aparece asociado a más de
    una partición entre `pairs` (FR-019). A diferencia de validar
    `assign_splits` directamente —un `dict` no puede tener, por construcción,
    dos valores para la misma clave—, `pairs` admite repeticiones (una por
    imagen del catálogo, por ejemplo), que es donde una fuga real podría
    colarse tras fusionar asignaciones o al reanudar `split`."""
    seen: dict[str, str] = {}
    for patient_key, split_name in pairs:
        previous = seen.get(patient_key)
        if previous is not None and previous != split_name:
            raise ValueError(
                f"Fuga de paciente: {patient_key} aparece en '{previous}' y "
                f"'{split_name}'"
            )
        seen[patient_key] = split_name
