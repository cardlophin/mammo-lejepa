from __future__ import annotations

import numpy as np
import pytest

from mammo_lejepa.errors import SegmentationError
from mammo_lejepa.segmentation import detect_breast_box
from mammo_lejepa.windowing import apply_display_window, normalize_to_uint8
from tests.fixtures.synthetic_dicom import make_synthetic_dicom

TOLERANCE_PX = 2


def _synthetic_uint8_image(**kwargs: object) -> tuple[np.ndarray, object]:
    dataset, geometry = make_synthetic_dicom(**kwargs)
    windowed = apply_display_window(
        dataset.pixel_array,
        window_center=dataset.get("WindowCenter"),
        window_width=dataset.get("WindowWidth"),
        photometric_interpretation=dataset.PhotometricInterpretation,
    )
    normalized, _, _ = normalize_to_uint8(windowed)
    return normalized, geometry


def test_detected_box_contains_the_ellipse_within_tolerance() -> None:
    image, geometry = _synthetic_uint8_image()
    x0_expected, y0_expected, x1_expected, y1_expected = geometry.bounding_box
    margin_px = 25

    crop = detect_breast_box(image, margin_px=margin_px)

    assert crop.x0_orig <= x0_expected - margin_px + TOLERANCE_PX
    assert crop.y0_orig <= y0_expected - margin_px + TOLERANCE_PX
    assert crop.x1_orig >= x1_expected + margin_px - TOLERANCE_PX
    assert crop.y1_orig >= y1_expected + margin_px - TOLERANCE_PX
    # Y no se pasa de largo más allá de la tolerancia.
    assert crop.x0_orig >= x0_expected - margin_px - TOLERANCE_PX
    assert crop.y0_orig >= y0_expected - margin_px - TOLERANCE_PX


def test_isolated_orientation_label_is_excluded_from_the_box() -> None:
    image, geometry = _synthetic_uint8_image(include_orientation_label=True)
    x0_expected, y0_expected, _, _ = geometry.bounding_box

    crop = detect_breast_box(image, margin_px=25)

    # La etiqueta está en la esquina (filas/columnas 4..~15); la caja de la
    # mama no debería empezar ahí salvo que el margen ya lo alcance.
    assert crop.x0_orig >= min(4, x0_expected - 25)
    assert crop.y0_orig >= min(4, y0_expected - 25)


def test_uniform_image_raises_segmentation_error() -> None:
    image, _ = _synthetic_uint8_image(uniform=True)

    with pytest.raises(SegmentationError):
        detect_breast_box(image)


def test_margin_is_clipped_to_image_bounds_when_breast_touches_edge() -> None:
    image, _ = _synthetic_uint8_image(height=200, width=150)

    crop = detect_breast_box(image, margin_px=10_000)

    assert crop.x0_orig == 0
    assert crop.y0_orig == 0
    assert crop.x1_orig == image.shape[1]
    assert crop.y1_orig == image.shape[0]


def test_area_ratio_is_always_populated() -> None:
    image, _ = _synthetic_uint8_image()

    crop = detect_breast_box(image)

    assert 0.0 < crop.area_ratio <= 1.0
