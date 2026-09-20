from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mammo_lejepa.errors import DecodeError


def read_grayscale(path: Path) -> np.ndarray:
    """Lee `path` como imagen en escala de grises, `uint8` (FR-006, FR-010).
    Lanza `DecodeError` si el fichero no existe o no puede decodificarse."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise DecodeError(f"No se pudo leer la imagen: {path}")
    return image


def read_mask(path: Path) -> np.ndarray | None:
    """Lee la máscara en `path` como `uint8` en escala de grises. Devuelve
    `None` si el fichero no existe: la ausencia de máscara es un caso
    previsto que dispara el respaldo a Otsu (FR-008), no un error."""
    if not path.exists():
        return None

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise DecodeError(f"No se pudo leer la máscara: {path}")
    return mask
