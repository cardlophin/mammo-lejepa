from __future__ import annotations

import numpy as np
import pytest

from mammo_lejepa.geometry import (
    clip_to_crop,
    crop_image,
    to_crop_space,
    to_original_space,
)
from mammo_lejepa.models import BoundingBox, BreastCrop

CROP = BreastCrop(
    x0_orig=50,
    y0_orig=60,
    x1_orig=250,
    y1_orig=260,
    margin_px=25,
    otsu_threshold=100.0,
    source_height=400,
    source_width=400,
    crop_height=200,
    crop_width=200,
    area_ratio=0.25,
)


def test_round_trip_is_identity_for_points_inside_the_crop() -> None:
    original = BoundingBox(x0=80.0, y0=90.0, x1=150.0, y1=160.0, space="orig")

    round_tripped = to_original_space(to_crop_space(original, CROP), CROP)

    assert round_tripped.x0 == pytest.approx(original.x0)
    assert round_tripped.y0 == pytest.approx(original.y0)
    assert round_tripped.x1 == pytest.approx(original.x1)
    assert round_tripped.y1 == pytest.approx(original.y1)


def test_to_crop_space_rejects_a_box_already_in_crop_space() -> None:
    box = BoundingBox(x0=10.0, y0=10.0, x1=20.0, y1=20.0, space="crop")

    with pytest.raises(ValueError, match="orig"):
        to_crop_space(box, CROP)


def test_to_original_space_rejects_a_box_already_in_orig_space() -> None:
    box = BoundingBox(x0=10.0, y0=10.0, x1=20.0, y1=20.0, space="orig")

    with pytest.raises(ValueError, match="crop"):
        to_original_space(box, CROP)


def test_clip_to_crop_marks_partial_boxes_as_clipped() -> None:
    # Caja en espacio 'crop' que sobresale por la derecha (crop_width=200).
    box = BoundingBox(x0=150.0, y0=50.0, x1=250.0, y1=100.0, space="crop")

    clipped_box, was_clipped = clip_to_crop(box, CROP)

    assert was_clipped is True
    assert clipped_box.x1 == 200.0
    assert clipped_box.x0 == 150.0


def test_clip_to_crop_does_not_mark_fully_contained_boxes() -> None:
    box = BoundingBox(x0=10.0, y0=10.0, x1=50.0, y1=50.0, space="crop")

    clipped_box, was_clipped = clip_to_crop(box, CROP)

    assert was_clipped is False
    assert (clipped_box.x0, clipped_box.y0, clipped_box.x1, clipped_box.y1) == (
        10.0,
        10.0,
        50.0,
        50.0,
    )


def test_clip_to_crop_empty_intersection_returns_explicit_degenerate_box() -> None:
    box = BoundingBox(x0=-100.0, y0=-100.0, x1=-50.0, y1=-50.0, space="crop")

    clipped_box, was_clipped = clip_to_crop(box, CROP)

    assert was_clipped is True
    assert (clipped_box.x0, clipped_box.y0, clipped_box.x1, clipped_box.y1) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_crop_image_returns_exactly_crop_height_and_width() -> None:
    image = np.zeros((400, 400), dtype=np.uint8)

    cropped = crop_image(image, CROP)

    assert cropped.shape == (CROP.crop_height, CROP.crop_width)
