from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BACKGROUND_LEVEL = 20
TISSUE_LEVEL = 220
# Intensidad del "pectoral" deliberadamente parecida al tejido: es lo que hace
# que Otsu no pueda distinguirlo (Edge Case de spec.md / D-01).
PECTORAL_LEVEL = 210

MaskVariant = Literal["valid", "missing", "empty", "saturated"]


@dataclass(frozen=True, slots=True)
class EllipseGeometry:
    """Geometría conocida de la elipse (mama), sin el triángulo pectoral."""

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


@dataclass(frozen=True, slots=True, eq=False)
class SyntheticMammogram:
    """Par (imagen, máscara) de geometría conocida. `mask` es `None` en la
    variante `missing`. `geometry` describe siempre la elipse (la mama), sin
    el pectoral ni la etiqueta de orientación."""

    image: np.ndarray
    mask: np.ndarray | None
    geometry: EllipseGeometry


def make_synthetic_mammogram(
    *,
    height: int = 512,
    width: int = 384,
    include_orientation_label: bool = False,
    include_pectoral: bool = False,
    mask_variant: MaskVariant = "valid",
) -> SyntheticMammogram:
    """Elipse clara sobre fondo oscuro con geometría conocida. Con
    `include_pectoral=True`, añade un triángulo en la esquina superior
    izquierda con intensidad parecida al tejido, tocando la elipse (mismo
    componente conexo bajo Otsu). La máscara de verdad-terreno (`mask_variant
    ='valid'`) marca sólo la elipse como mama y excluye el triángulo — es la
    prueba directa de D-01: la máscara excluye el pectoral, Otsu no."""
    center_x = int(width * 0.55)
    center_y = int(height * 0.55)
    radius_x = int(width * 0.42)
    radius_y = int(height * 0.4)
    geometry = EllipseGeometry(
        center_x=center_x, center_y=center_y, radius_x=radius_x, radius_y=radius_y
    )

    yy, xx = np.mgrid[0:height, 0:width]
    ellipse_mask = (
        ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
    ) <= 1.0

    image = np.full((height, width), BACKGROUND_LEVEL, dtype=np.uint8)
    image[ellipse_mask] = TISSUE_LEVEL

    if include_pectoral:
        # Triángulo rectángulo en la esquina superior izquierda cuya
        # hipotenusa (x + y = umbral) cruza el interior de la elipse: se
        # solapa de verdad con la mama, en vez de quedar como un componente
        # separado que `largest_connected_component` descartaría (como la
        # etiqueta de orientación). El umbral 0.45*(cx+cy) se verificó
        # empíricamente: separa "toca la elipse" (caja de Otsu sin cambios)
        # de "la caja de Otsu salta a (0,0)".
        pectoral_threshold = int((center_x + center_y) * 0.45)
        pectoral_mask = (xx.astype(np.int64) + yy.astype(np.int64)) < pectoral_threshold
        image[pectoral_mask] = PECTORAL_LEVEL

    if include_orientation_label:
        label_h = max(6, height // 40)
        label_w = max(6, width // 40)
        corner = (
            slice(height - label_h, height),
            slice(width - label_w, width),
        )
        image[corner] = TISSUE_LEVEL

    mask = _build_mask(ellipse_mask, height=height, width=width, variant=mask_variant)

    return SyntheticMammogram(image=image, mask=mask, geometry=geometry)


def _build_mask(
    ellipse_mask: np.ndarray, *, height: int, width: int, variant: MaskVariant
) -> np.ndarray | None:
    if variant == "missing":
        return None

    if variant == "empty":
        return np.zeros((height, width), dtype=np.uint8)

    if variant == "saturated":
        # >99% de área cubierta: dispara el respaldo por FR-008 (D-01: fuera
        # de [0.02, 0.99]).
        return np.full((height, width), 255, dtype=np.uint8)

    # "valid": la máscara de verdad-terreno es sólo la elipse — excluye
    # explícitamente el pectoral, a diferencia de lo que haría Otsu.
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[ellipse_mask] = 255
    return mask
