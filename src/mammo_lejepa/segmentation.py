from __future__ import annotations

import cv2
import numpy as np

from mammo_lejepa.errors import SegmentationError
from mammo_lejepa.models import BoxSource, CajaMamaria


def detect_breast_box(
    image_uint8: np.ndarray,
    *,
    margin_px: int = 25,
    blur_kernel: int = 5,
    close_kernel_ratio: float = 0.006,
) -> CajaMamaria:
    """Detecta el campo mamario mediante umbralización de Otsu, cierre
    morfológico y selección del mayor componente conexo, y deriva de él una
    bounding box rectangular con un margen configurable (FR-008, camino de
    respaldo de `boxing.py`). Lanza `SegmentationError` cuando no hay ningún
    componente conexo válido, la máscara es degenerada, el componente ocupa
    la imagen entera (sin mama detectable) o la caja resultante tiene área
    nula."""
    height, width = image_uint8.shape

    blurred = cv2.GaussianBlur(image_uint8, ksize=(blur_kernel, blur_kernel), sigmaX=0)

    otsu_threshold, binary_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cleaned_mask = _morphological_close(binary_mask, close_kernel_ratio)
    x0, y0, x1, y1 = box_from_binary_mask(cleaned_mask, margin_px=margin_px)

    area_ratio = ((x1 - x0) * (y1 - y0)) / (width * height)

    return CajaMamaria(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        margin_px=margin_px,
        source=BoxSource.OTSU,
        threshold=float(otsu_threshold),
        image_height=height,
        image_width=width,
        area_ratio=area_ratio,
    )


def _morphological_close(
    binary_mask: np.ndarray, close_kernel_ratio: float
) -> np.ndarray:
    height, width = binary_mask.shape
    kernel_size = max(9, int(min(height, width) * close_kernel_ratio))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def box_from_binary_mask(
    binary_mask: np.ndarray, *, margin_px: int
) -> tuple[int, int, int, int]:
    """A partir de una máscara binaria (0/255), se queda con el mayor
    componente conexo y devuelve su bounding box `(x0, y0, x1, y1)` con el
    margen aplicado y recortada a los límites de la máscara. Compartido entre
    `detect_breast_box` (Otsu) y `boxing.box_from_mask` para no duplicar la
    política de selección de componente y margen. Lanza `SegmentationError`
    si no hay componente, si ocupa la máscara entera (sin mama detectable) o
    si el área resultante es nula."""
    height, width = binary_mask.shape

    component = largest_connected_component(binary_mask)
    points = cv2.findNonZero(component)

    if points is None:
        raise SegmentationError(
            "No se encontraron píxeles de mama tras el cierre morfológico: "
            "máscara degenerada."
        )

    x, y, box_width, box_height = cv2.boundingRect(points)

    if x == 0 and y == 0 and x + box_width == width and y + box_height == height:
        raise SegmentationError(
            "El componente conexo mayor ocupa la imagen entera: no hay mama "
            "detectable (máscara degenerada o saturada)."
        )

    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(width, x + box_width + margin_px)
    y1 = min(height, y + box_height + margin_px)

    if x1 <= x0 or y1 <= y0:
        raise SegmentationError(
            f"La caja mamaria resultante tiene área nula: ({x0},{y0})-({x1},{y1})."
        )

    return x0, y0, x1, y1


def largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """Conserva el componente conectado de mayor área; normalmente corresponde
    a la mama y descarta de forma natural texto, letras de orientación y
    marcadores pequeños."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels <= 1:
        raise SegmentationError("No se detectó ningún componente conexo de mama.")

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    largest_mask = np.zeros_like(binary_mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask
