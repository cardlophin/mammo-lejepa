from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np

from mammo_lejepa.models import CajaMamaria


class FailureCategory(StrEnum):
    """Propio de `vindr/`: incluye `NETWORK`/`AUTH`, que el `FailureCategory` de
    Mammo-Bench no tiene porque ese corpus no descarga nada (research.md, D-02)."""

    NETWORK = "NETWORK"
    AUTH = "AUTH"
    DECODE = "DECODE"
    SEGMENTATION = "SEGMENTATION"
    WRITE = "WRITE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ImageTask:
    """La unidad de trabajo. Se deriva del manifiesto y no cambia durante la
    ejecución."""

    study_id: str
    series_id: str
    image_id: str
    split: str
    laterality: str
    view_position: str
    breast_birads: str
    breast_density: str
    expected_height: int
    expected_width: int


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Lo que un worker devuelve al proceso principal tras procesar un DICOM. La
    caja mamaria es `CajaMamaria` (`mammo_lejepa.models`), no un tipo propio de
    VinDr: es lo que permite reutilizar `geometry.py`/`segmentation.py` de
    Mammo-Bench sin modificarlos (research.md, D-01)."""

    status: Literal["ok", "failed"]
    process_seconds: float
    breast_crop: CajaMamaria | None = None
    photometric_interpretation: str | None = None
    transfer_syntax_uid: str | None = None
    window_center: float | None = None
    window_width: float | None = None
    pixel_spacing: tuple[float, float] | None = None
    manufacturer: str | None = None
    model_name: str | None = None
    normalize_low: float | None = None
    normalize_high: float | None = None
    inverted_monochrome1: bool = False
    png_path: str | None = None
    png_bytes: int | None = None
    png_sha256: str | None = None
    failure_category: FailureCategory | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """La fila del catálogo (una línea de `images.jsonl`): une `ImageTask`,
    `ProcessOutcome`, procedencia técnica y resultado de la ejecución."""

    study_id: str
    series_id: str
    image_id: str
    status: Literal["ok", "failed"]
    split: str
    laterality: str
    view_position: str
    breast_birads: str
    breast_density: str
    run_id: str
    pipeline_version: str
    processed_at: str
    download_seconds: float
    process_seconds: float
    dicom_bytes: int
    breast_crop: CajaMamaria | None = None
    photometric_interpretation: str | None = None
    transfer_syntax_uid: str | None = None
    window_center: float | None = None
    window_width: float | None = None
    pixel_spacing: tuple[float, float] | None = None
    manufacturer: str | None = None
    model_name: str | None = None
    normalize_low: float | None = None
    normalize_high: float | None = None
    inverted_monochrome1: bool = False
    png_path: str | None = None
    png_bytes: int | None = None
    png_sha256: str | None = None
    failure_category: FailureCategory | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class DicomPayload:
    """Píxeles en su tipo nativo y cabeceras de procedencia leídas de un DICOM.
    `eq=False` porque comparar arrays de numpy con `==` no devuelve un booleano
    simple."""

    pixels: np.ndarray
    rows: int
    columns: int
    photometric_interpretation: str
    transfer_syntax_uid: str
    window_center: float | None
    window_width: float | None
    pixel_spacing: tuple[float, float] | None
    manufacturer: str | None
    model_name: str | None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Resultado de una descarga individual de DICOM."""

    status: Literal["downloaded", "already_exists"]
    dicom_path: Path
    bytes_downloaded: int
    download_seconds: float


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Resumen de una ejecución completa (una línea de `runs.jsonl`)."""

    run_id: str
    started_at: str
    finished_at: str
    command: str
    pipeline_version: str
    split: str | None
    n_studies: int | None
    downloads: int
    workers: int
    queue_size: int
    margin_px: int
    images_total: int
    images_ok: int
    images_failed: int
    failures_by_category: dict[FailureCategory, int]
    bytes_downloaded: int
    bytes_written: int
    exit_reason: str
