from __future__ import annotations

import shutil
from pathlib import Path


def quarantine(dicom_path: Path, quarantine_dir: Path) -> Path:
    """Mueve un DICOM cuyo procesado falló a `quarantine_dir/<study_id>/`,
    preservando el nombre de su directorio de estudio. Único añadido de esta
    feature sobre `storage.py` compartido, que se quedó sin esta función al
    reescribirse para Mammo-Bench por ser específica de DICOM (research.md,
    D-03)."""
    study_dir = quarantine_dir / dicom_path.parent.name
    study_dir.mkdir(parents=True, exist_ok=True)
    destination = study_dir / dicom_path.name
    shutil.move(str(dicom_path), str(destination))
    return destination
