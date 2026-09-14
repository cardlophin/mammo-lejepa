from __future__ import annotations

import cv2
import numpy as np

from mammo_lejepa.errors import SegmentationError
from mammo_lejepa.models import BreastCrop


def detect_breast_box(
    image_uint8: np.ndarray,
    *,
    margin_px: int = 25,
    blur_kernel: int = 5,
    close_kernel_ratio: float = 0.006,
) -> BreastCrop:
    """Detecta el campo mamario mediante umbralización de Otsu, cierre
    morfológico y selección del mayor componente conexo, y deriva de él una
    bounding box rectangular con un margen configurable (FR-012). Lanza
    `SegmentationError` cuando no hay ningún componente conexo válido, la
    máscara es degenerada, el componente ocupa la imagen entera (sin mama
    detectable) o la caja resultante tiene área nula (FR-015)."""
    height, width = image_uint8.shape

    blurred = cv2.GaussianBlur(image_uint8, ksize=(blur_kernel, blur_kernel), sigmaX=0)

    otsu_threshold, binary_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel_size = max(9, int(min(height, width) * close_kernel_ratio))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    breast_component = _largest_connected_component(cleaned_mask)
    breast_points = cv2.findNonZero(breast_component)

    if breast_points is None:
        raise SegmentationError(
            "No se encontraron píxeles de mama tras el cierre morfológico: "
            "máscara degenerada."
        )

    x, y, box_width, box_height = cv2.boundingRect(breast_points)

    if x == 0 and y == 0 and x + box_width == width and y + box_height == height:
        raise SegmentationError(
            "El componente conexo mayor ocupa la imagen entera: no hay mama "
            "detectable (máscara de Otsu degenerada o imagen casi uniforme)."
        )

    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(width, x + box_width + margin_px)
    y1 = min(height, y + box_height + margin_px)

    if x1 <= x0 or y1 <= y0:
        raise SegmentationError(
            f"La caja mamaria resultante tiene área nula: ({x0},{y0})-({x1},{y1})."
        )

    crop_width = x1 - x0
    crop_height = y1 - y0
    area_ratio = (crop_width * crop_height) / (width * height)

    return BreastCrop(
        x0_orig=x0,
        y0_orig=y0,
        x1_orig=x1,
        y1_orig=y1,
        margin_px=margin_px,
        otsu_threshold=float(otsu_threshold),
        source_height=height,
        source_width=width,
        crop_height=crop_height,
        crop_width=crop_width,
        area_ratio=area_ratio,
    )


def _largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """Conserva el componente conectado de mayor área; normalmente corresponde
    a la mama y descarta de forma natural texto DICOM, letras de orientación y
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
