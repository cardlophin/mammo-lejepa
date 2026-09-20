from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np


class FailureCategory(StrEnum):
    DECODE = "DECODE"
    SEGMENTATION = "SEGMENTATION"
    WRITE = "WRITE"
    UNKNOWN = "UNKNOWN"


class BoxSource(StrEnum):
    MASK = "MASK"
    OTSU = "OTSU"


class FallbackReason(StrEnum):
    MISSING_MASK = "MISSING_MASK"
    EMPTY_MASK = "EMPTY_MASK"
    SATURATED_MASK = "SATURATED_MASK"
    DEGENERATE_BOX = "DEGENERATE_BOX"


@dataclass(frozen=True, slots=True)
class ImagenMammoBench:
    """Una fila de `mammo-bench.csv`. Inmutable, es la unidad de trabajo."""

    image_id: str
    source_dataset: str
    source_subject_id: str
    patient_key: str
    preprocessed_path: str
    mask_path: str
    raw_path: str
    laterality: str
    view: str
    classification: str
    density: str | None = None
    birads: str | None = None
    abnormality: str | None = None
    molecular_subtype: str | None = None
    subject_age: str | None = None


@dataclass(frozen=True, slots=True)
class CajaMamaria:
    """La caja en coordenadas de la imagen de `Preprocessed_Dataset`, con el
    margen ya aplicado y recortada a los límites de la imagen."""

    x0: int
    y0: int
    x1: int
    y1: int
    margin_px: int
    source: BoxSource
    threshold: float
    image_height: int
    image_width: int
    area_ratio: float

    def __post_init__(self) -> None:
        if not (0 <= self.x0 < self.x1 <= self.image_width):
            raise ValueError(
                "CajaMamaria inválida: "
                f"0 <= x0({self.x0}) < x1({self.x1}) <= image_width"
                f"({self.image_width}) no se cumple"
            )
        if not (0 <= self.y0 < self.y1 <= self.image_height):
            raise ValueError(
                "CajaMamaria inválida: "
                f"0 <= y0({self.y0}) < y1({self.y1}) <= image_height"
                f"({self.image_height}) no se cumple"
            )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Caja genérica con el espacio de referencia explícito (`geometry.py`,
    conservado de 001 para trasladar coordenadas cuando haga falta)."""

    x0: float
    y0: float
    x1: float
    y1: float
    space: Literal["orig", "crop"]


@dataclass(frozen=True, slots=True)
class RegistroDeRecorte:
    """Lo que un trabajador devuelve; acaba siendo una fila de `catalog.parquet`
    (sin `split`, que se incorpora al consolidar)."""

    image_id: str
    source_dataset: str
    source_subject_id: str
    patient_key: str
    laterality: str
    view: str
    status: Literal["ok", "failed"]
    process_seconds: float
    run_id: str
    code_version: str
    processed_at: str
    box: CajaMamaria | None = None
    fallback_reason: FallbackReason | None = None
    mask_area_ratio: float | None = None
    mask_otsu_iou: float | None = None
    suspect: bool = False
    suspect_reason: str | None = None
    crop_path: str | None = None
    crop_bytes: int | None = None
    classification: str | None = None
    density: str | None = None
    birads: str | None = None
    abnormality: str | None = None
    molecular_subtype: str | None = None
    subject_age: str | None = None
    failure_category: FailureCategory | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Resumen de una ejecución completa (una línea de `runs.jsonl`)."""

    run_id: str
    started_at: str
    finished_at: str
    command: str
    code_version: str
    seed: int
    n_total: int
    n_ok: int
    n_failed: int
    failures_by_category: dict[FailureCategory, int]
    n_fallback_otsu: int
    exit_reason: str


@dataclass(frozen=True, slots=True)
class BoxingParams:
    """Parámetros de `boxing.py` (D-01): umbral de la máscara, margen y la
    política de respaldo a Otsu."""

    mask_threshold: int = 128
    margin_px: int = 25
    blur_kernel: int = 5
    close_kernel_ratio: float = 0.006
    mask_area_ratio_min: float = 0.02
    mask_area_ratio_max: float = 0.99


@dataclass(frozen=True, slots=True)
class QualityParams:
    """Parámetros de `quality.py`: rangos que marcan un recorte como
    sospechoso."""

    suspect_area_ratio_min: float = 0.10
    suspect_area_ratio_max: float = 0.98
    suspect_aspect_ratio_min: float = 0.2
    suspect_aspect_ratio_max: float = 5.0
    suspect_iou_threshold: float = 0.3


@dataclass(frozen=True, slots=True, eq=False)
class ImageMaskPayload:
    """Píxeles de imagen y máscara (o `None` si la máscara no existe) en
    escala de grises, `uint8`. `eq=False` porque comparar arrays de numpy con
    `==` no devuelve un booleano simple."""

    image: np.ndarray
    mask: np.ndarray | None
