from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mammo_lejepa.errors import SegmentationError
from mammo_lejepa.segmentation import detect_breast_box

TOLERANCE_PX = 2


@dataclass(frozen=True, slots=True)
class _EllipseGeometry:
    center_x: int
    center_y: int
    radius_x: int
    radius_y: int

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        return (
            self.center_x - self.radius_x,
            self.center_y - self.radius_y,
            self.center_x + self.radius_x,
            self.center_y + self.radius_y,
        )


def _synthetic_uint8_image(
    *,
    height: int = 512,
    width: int = 384,
    uniform: bool = False,
    include_orientation_label: bool = False,
) -> tuple[np.ndarray, _EllipseGeometry]:
    """Elipse clara sobre fondo oscuro con geometría conocida, directamente en
    `uint8`: `detect_breast_box` no necesita DICOM ni ventana, sólo una imagen
    ya normalizada (segmentation.py no cambia entre 001 y 002)."""
    background_level = 20
    tissue_level = 220

    center_x, center_y = width // 2, int(height * 0.55)
    radius_x, radius_y = int(width * 0.32), int(height * 0.38)
    geometry = _EllipseGeometry(
        center_x=center_x, center_y=center_y, radius_x=radius_x, radius_y=radius_y
    )

    if uniform:
        image = np.full((height, width), background_level, dtype=np.uint8)
        return image, geometry

    yy, xx = np.mgrid[0:height, 0:width]
    ellipse_mask = (
        ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
    ) <= 1.0

    image = np.full((height, width), background_level, dtype=np.uint8)
    image[ellipse_mask] = tissue_level

    if include_orientation_label:
        label_h = max(6, height // 40)
        label_w = max(6, width // 40)
        image[4 : 4 + label_h, 4 : 4 + label_w] = tissue_level

    return image, geometry


def test_detected_box_contains_the_ellipse_within_tolerance() -> None:
    image, geometry = _synthetic_uint8_image()
    x0_expected, y0_expected, x1_expected, y1_expected = geometry.bounding_box
    margin_px = 25

    crop = detect_breast_box(image, margin_px=margin_px)

    assert crop.x0 <= x0_expected - margin_px + TOLERANCE_PX
    assert crop.y0 <= y0_expected - margin_px + TOLERANCE_PX
    assert crop.x1 >= x1_expected + margin_px - TOLERANCE_PX
    assert crop.y1 >= y1_expected + margin_px - TOLERANCE_PX
    # Y no se pasa de largo más allá de la tolerancia.
    assert crop.x0 >= x0_expected - margin_px - TOLERANCE_PX
    assert crop.y0 >= y0_expected - margin_px - TOLERANCE_PX


def test_isolated_orientation_label_is_excluded_from_the_box() -> None:
    image, geometry = _synthetic_uint8_image(include_orientation_label=True)
    x0_expected, y0_expected, _, _ = geometry.bounding_box

    crop = detect_breast_box(image, margin_px=25)

    # La etiqueta está en la esquina (filas/columnas 4..~15); la caja de la
    # mama no debería empezar ahí salvo que el margen ya lo alcance.
    assert crop.x0 >= min(4, x0_expected - 25)
    assert crop.y0 >= min(4, y0_expected - 25)


def test_uniform_image_raises_segmentation_error() -> None:
    image, _ = _synthetic_uint8_image(uniform=True)

    with pytest.raises(SegmentationError):
        detect_breast_box(image)


def test_margin_is_clipped_to_image_bounds_when_breast_touches_edge() -> None:
    image, _ = _synthetic_uint8_image(height=200, width=150)

    crop = detect_breast_box(image, margin_px=10_000)

    assert crop.x0 == 0
    assert crop.y0 == 0
    assert crop.x1 == image.shape[1]
    assert crop.y1 == image.shape[0]


def test_area_ratio_is_always_populated() -> None:
    image, _ = _synthetic_uint8_image()

    crop = detect_breast_box(image)

    assert 0.0 < crop.area_ratio <= 1.0
