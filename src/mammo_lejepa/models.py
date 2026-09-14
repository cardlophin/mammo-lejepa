from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np


class FailureCategory(StrEnum):
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
class BreastCrop:
    """Resultado geométrico de la detección de la caja mamaria. Con ella y el PNG
    se puede reconstruir cualquier coordenada del original sin volver a descargar
    el DICOM."""

    x0_orig: int
    y0_orig: int
    x1_orig: int
    y1_orig: int
    margin_px: int
    otsu_threshold: float
    source_height: int
    source_width: int
    crop_height: int
    crop_width: int
    area_ratio: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not (0 <= self.x0_orig < self.x1_orig <= self.source_width):
            raise ValueError(
                "BreastCrop inválido: "
                f"0 <= x0_orig({self.x0_orig}) < x1_orig({self.x1_orig}) "
                f"<= source_width({self.source_width}) no se cumple"
            )
        if not (0 <= self.y0_orig < self.y1_orig <= self.source_height):
            raise ValueError(
                "BreastCrop inválido: "
                f"0 <= y0_orig({self.y0_orig}) < y1_orig({self.y1_orig}) "
                f"<= source_height({self.source_height}) no se cumple"
            )
        if self.crop_width != self.x1_orig - self.x0_orig:
            raise ValueError(
                f"crop_width({self.crop_width}) != "
                f"x1_orig - x0_orig({self.x1_orig - self.x0_orig})"
            )
        if self.crop_height != self.y1_orig - self.y0_orig:
            raise ValueError(
                f"crop_height({self.crop_height}) != "
                f"y1_orig - y0_orig({self.y1_orig - self.y0_orig})"
            )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Caja genérica con el espacio de referencia explícito. Se usa tanto para la
    caja mamaria como para los hallazgos."""

    x0: float
    y0: float
    x1: float
    y1: float
    space: Literal["orig", "crop"]


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Lo que un worker devuelve al proceso principal tras procesar un DICOM."""

    status: Literal["ok", "failed"]
    process_seconds: float
    breast_crop: BreastCrop | None = None
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
    """La fila del catálogo (una línea de `images.jsonl`): une ImageTask,
    ProcessOutcome, procedencia técnica y resultado de la ejecución."""

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
    breast_crop: BreastCrop | None = None
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
