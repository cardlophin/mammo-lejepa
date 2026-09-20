from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from mammo_lejepa.ssl.augment import (
    build_eval_transform,
    build_view_transform,
    describe,
)
from mammo_lejepa.ssl.config import AugmentConfig


def _synthetic_image(height: int = 300, width: int = 200) -> Image.Image:
    rng = np.random.default_rng(0)
    pixels = (rng.random((height, width)) * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def test_view_transform_produces_expected_shape_and_range() -> None:
    config = AugmentConfig(resolution=96)
    transform = build_view_transform(config)

    output = transform(_synthetic_image())

    assert output.shape == (3, 96, 96)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_multiple_views_of_the_same_image_differ() -> None:
    torch.manual_seed(0)
    config = AugmentConfig()
    transform = build_view_transform(config)
    image = _synthetic_image()

    views = [transform(image) for _ in range(4)]

    assert not all(torch.equal(views[0], view) for view in views[1:])


def test_eval_transform_is_deterministic() -> None:
    config = AugmentConfig(resolution=64)
    transform = build_eval_transform(config)
    image = _synthetic_image()

    first = transform(image)
    second = transform(image)

    torch.testing.assert_close(first, second)
    assert first.shape == (3, 64, 64)


def test_describe_reflects_the_exact_configuration() -> None:
    config = AugmentConfig(
        resolution=112,
        crop_scale_min=0.3,
        crop_scale_max=0.9,
        horizontal_flip_prob=0.7,
        brightness_jitter=0.15,
        contrast_jitter=0.25,
        gaussian_blur_prob=0.4,
        gaussian_blur_sigma_min=0.2,
        gaussian_blur_sigma_max=0.8,
    )

    spec = describe(config)

    assert spec["resolution"] == 112
    assert spec["random_resized_crop"]["scale"] == [0.3, 0.9]
    assert spec["horizontal_flip"]["probability"] == 0.7
    assert spec["color_jitter"]["brightness"] == 0.15
    assert spec["color_jitter"]["contrast"] == 0.25
    assert spec["gaussian_blur"]["probability"] == 0.4
    assert spec["gaussian_blur"]["sigma_range"] == [0.2, 0.8]


def test_describe_names_every_excluded_transform_with_a_reason() -> None:
    spec = describe(AugmentConfig())

    excluded = spec["excluded"]
    assert "solarize" in excluded
    assert "grayscale_conversion" in excluded
    assert "hue_saturation_jitter" in excluded
    assert "large_rotations" in excluded
    assert all(isinstance(reason, str) and reason for reason in excluded.values())


def test_describe_is_json_serializable() -> None:
    import json

    json.dumps(describe(AugmentConfig()))
