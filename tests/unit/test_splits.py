from __future__ import annotations

from collections import Counter

import pytest

from mammo_lejepa.models import RegistroDeRecorte
from mammo_lejepa.splits import (
    assign_splits,
    patient_classification,
    verify_no_patient_leakage,
)


def _record(
    *,
    image_id: str,
    source_dataset: str,
    subject_id: str,
    classification: str = "Normal",
) -> RegistroDeRecorte:
    return RegistroDeRecorte(
        image_id=image_id,
        source_dataset=source_dataset,
        source_subject_id=subject_id,
        patient_key=f"{source_dataset}:{subject_id}",
        laterality="L",
        view="CC",
        status="ok",
        process_seconds=1.0,
        run_id="run_x",
        code_version="0.1.0",
        processed_at="2026-01-01T00:00:00+00:00",
        classification=classification,
    )


def _many_patients(n_patients: int, n_sources: int = 2) -> list[RegistroDeRecorte]:
    records = []
    classifications = ["Normal", "Benign", "Malignant"]
    for i in range(n_patients):
        source = f"source_{i % n_sources}"
        records.append(
            _record(
                image_id=f"img_{i}",
                source_dataset=source,
                subject_id=f"patient_{i}",
                classification=classifications[i % len(classifications)],
            )
        )
    return records


# --- patient_classification -------------------------------------------------


def test_patient_classification_keeps_most_severe() -> None:
    assert patient_classification(["Normal", "Benign"]) == "Benign"
    assert patient_classification(["Benign", "Malignant", "Normal"]) == "Malignant"
    assert (
        patient_classification(["Suspicious Malignant", "Benign"])
        == "Suspicious Malignant"
    )


def test_patient_classification_single_value() -> None:
    assert patient_classification(["Normal"]) == "Normal"


def test_patient_classification_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Not-A-Real-Classification"):
        patient_classification(["Normal", "Not-A-Real-Classification"])


def test_patient_classification_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError):
        patient_classification([])


# --- assign_splits ------------------------------------------------------


def test_same_seed_gives_same_assignment_regardless_of_input_order() -> None:
    records = _many_patients(60)
    shuffled = list(reversed(records))

    assignment_a = assign_splits(records, seed=0)
    assignment_b = assign_splits(shuffled, seed=0)

    assert assignment_a == assignment_b


def test_different_seed_can_change_assignment() -> None:
    records = _many_patients(60)

    assignment_a = assign_splits(records, seed=0)
    assignment_b = assign_splits(records, seed=1)

    assert assignment_a != assignment_b


def test_no_patient_appears_in_more_than_one_split() -> None:
    records = _many_patients(60)

    assignment = assign_splits(records, seed=0)

    # Por construcción `assignment` es patient_key -> split (un único valor),
    # así que la comprobación real ocurre en `verify_no_patient_leakage`
    # sobre los pares aplanados (una imagen puede repetir su patient_key).
    pairs = [(record.patient_key, assignment[record.patient_key]) for record in records]
    verify_no_patient_leakage(pairs)


def test_split_proportions_within_tolerance() -> None:
    records = _many_patients(300, n_sources=3)

    assignment = assign_splits(records, seed=0, ratios=(0.8, 0.1, 0.1))

    counts = Counter(assignment.values())
    total = sum(counts.values())

    assert abs(counts["train"] / total - 0.8) < 0.05
    assert abs(counts["val"] / total - 0.1) < 0.05
    assert abs(counts["test"] / total - 0.1) < 0.05


def test_split_assignment_is_by_patient_not_by_image_count() -> None:
    """Un paciente con 14 imágenes cuenta como UN paciente al calcular el
    tamaño de cada partición — nunca como 14 —, así que las proporciones por
    paciente no se rompen aunque el reparto por imagen sí quede desigual."""
    records = [
        _record(image_id=f"heavy_{i}", source_dataset="ddsm", subject_id="heavy")
        for i in range(14)
    ]
    records += [
        _record(image_id=f"light_{i}", source_dataset="ddsm", subject_id=f"light_{i}")
        for i in range(20)
    ]

    assignment = assign_splits(records, seed=0, ratios=(0.8, 0.1, 0.1))

    # 21 pacientes en total (1 "heavy" + 20 "light"): el tamaño de cada
    # partición debe reflejar ese recuento de pacientes, no 34 imágenes.
    counts = Counter(assignment.values())
    assert sum(counts.values()) == 21
    assert 16 <= counts["train"] <= 18
    assert 1 <= counts["val"] <= 3
    assert 1 <= counts["test"] <= 3


def test_assign_splits_rejects_ratios_not_summing_to_one() -> None:
    records = _many_patients(10)

    with pytest.raises(ValueError):
        assign_splits(records, ratios=(0.5, 0.5, 0.5))


# --- verify_no_patient_leakage -------------------------------------------


def test_verify_no_patient_leakage_passes_consistent_pairs() -> None:
    pairs = [("inbreast:p1", "train"), ("inbreast:p1", "train"), ("ddsm:p2", "val")]

    verify_no_patient_leakage(pairs)  # no debe lanzar


def test_verify_no_patient_leakage_raises_on_conflicting_split() -> None:
    pairs = [("inbreast:p1", "train"), ("inbreast:p1", "test")]

    with pytest.raises(ValueError, match="inbreast:p1"):
        verify_no_patient_leakage(pairs)
