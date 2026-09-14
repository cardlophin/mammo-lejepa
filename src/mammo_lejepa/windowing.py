from __future__ import annotations

import numpy as np

MONOCHROME1 = "MONOCHROME1"


def apply_display_window(
    pixels: np.ndarray,
    *,
    window_center: float | None,
    window_width: float | None,
    photometric_interpretation: str,
) -> np.ndarray:
    """Aplica la ventana de visualización DICOM (lineal, centrada en
    `window_center` con anchura `window_width`) cuando ambos parámetros están
    presentes; es la identidad en caso contrario. Invierte siempre MONOCHROME1
    de modo que el tejido quede claro sobre fondo oscuro, igual que MONOCHROME2
    (FR-010)."""
    image = pixels.astype(np.float32)

    if window_center is not None and window_width is not None and window_width > 0:
        low = window_center - window_width / 2
        high = window_center + window_width / 2
        image = np.clip(image, low, high)

    if photometric_interpretation == MONOCHROME1:
        image = image.max() - image

    return image


def normalize_to_uint8(
    image: np.ndarray,
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> tuple[np.ndarray, float, float]:
    """Normaliza `image` a `uint8` mediante percentiles robustos, para reducir
    el efecto de valores extremos. Devuelve la imagen normalizada junto con los
    dos valores de corte empleados (FR-011), que se registran en el catálogo.
    Si los percentiles degeneran (`high <= low`) recurre al mínimo y al máximo;
    si incluso así la imagen es constante, devuelve ceros sin dividir nunca por
    cero."""
    image = image.astype(np.float32)
    low, high = (
        float(v) for v in np.percentile(image, [low_percentile, high_percentile])
    )

    if high <= low:
        low, high = float(image.min()), float(image.max())

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8), low, high

    normalized = (image - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255).astype(np.uint8), low, high
