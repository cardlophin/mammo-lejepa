from __future__ import annotations

import numpy as np
import pytest

from mammo_lejepa.vindr.windowing import apply_display_window, normalize_to_uint8


def test_monochrome1_is_inverted_relative_to_monochrome2() -> None:
    # min=0 para que la inversión (basada en el máximo local de la imagen)
    # sea exactamente autoinversa y el contenido resulte comparable.
    raw = np.array([[0.0, 100.0], [200.0, 300.0]])

    mono2 = apply_display_window(
        raw,
        window_center=None,
        window_width=None,
        photometric_interpretation="MONOCHROME2",
    )
    mono1_raw = raw.max() - raw  # mismo contenido visual, codificado como MONOCHROME1
    mono1 = apply_display_window(
        mono1_raw,
        window_center=None,
        window_width=None,
        photometric_interpretation="MONOCHROME1",
    )

    np.testing.assert_allclose(mono1, mono2)


def test_absent_window_is_identity_for_monochrome2() -> None:
    raw = np.array([[10.0, 4000.0], [65000.0, 500.0]])

    result = apply_display_window(
        raw,
        window_center=None,
        window_width=None,
        photometric_interpretation="MONOCHROME2",
    )

    np.testing.assert_allclose(result, raw)


def test_window_clips_to_center_and_width() -> None:
    raw = np.array([[0.0, 50.0, 100.0, 150.0, 200.0]])

    result = apply_display_window(
        raw,
        window_center=100.0,
        window_width=100.0,
        photometric_interpretation="MONOCHROME2",
    )

    np.testing.assert_allclose(result, [[50.0, 50.0, 100.0, 150.0, 150.0]])


def test_normalize_to_uint8_returns_full_range_and_cutoffs() -> None:
    rng = np.random.default_rng(seed=0)
    image = rng.uniform(low=100.0, high=5000.0, size=(64, 64))

    normalized, low, high = normalize_to_uint8(image)

    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255
    assert low < high


def test_normalize_to_uint8_degenerate_percentiles_never_divides_by_zero() -> None:
    image = np.full((16, 16), 42.0)

    normalized, low, high = normalize_to_uint8(image)

    assert normalized.dtype == np.uint8
    assert np.all(normalized == 0)
    assert low == high == 42.0


@pytest.mark.parametrize("high_percentile", [0.5, 0.0])
def test_normalize_to_uint8_degenerate_custom_percentiles(
    high_percentile: float,
) -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])

    normalized, low, high = normalize_to_uint8(
        image, low_percentile=0.5, high_percentile=high_percentile
    )

    assert normalized.dtype == np.uint8
    assert low == 1.0
    assert high == 4.0
