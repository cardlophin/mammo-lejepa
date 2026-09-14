from __future__ import annotations

import numpy as np

from mammo_lejepa.models import BoundingBox, BreastCrop


def crop_image(image: np.ndarray, crop: BreastCrop) -> np.ndarray:
    """Recorta `image` a la caja de `crop`, sin redimensionar (FR-013). Devuelve
    exactamente `(crop_height, crop_width)`."""
    return image[crop.y0_orig : crop.y1_orig, crop.x0_orig : crop.x1_orig]


def to_crop_space(box: BoundingBox, crop: BreastCrop) -> BoundingBox:
    """Traslada `box` del espacio del DICOM original al espacio del recorte."""
    if box.space != "orig":
        raise ValueError(
            f"to_crop_space espera una caja en espacio 'orig', recibió {box.space!r}."
        )
    return BoundingBox(
        x0=box.x0 - crop.x0_orig,
        y0=box.y0 - crop.y0_orig,
        x1=box.x1 - crop.x0_orig,
        y1=box.y1 - crop.y0_orig,
        space="crop",
    )


def to_original_space(box: BoundingBox, crop: BreastCrop) -> BoundingBox:
    """Traslada `box` del espacio del recorte al espacio del DICOM original."""
    if box.space != "crop":
        raise ValueError(
            f"to_original_space espera una caja en espacio 'crop', recibió "
            f"{box.space!r}."
        )
    return BoundingBox(
        x0=box.x0 + crop.x0_orig,
        y0=box.y0 + crop.y0_orig,
        x1=box.x1 + crop.x0_orig,
        y1=box.y1 + crop.y0_orig,
        space="orig",
    )


def clip_to_crop(box: BoundingBox, crop: BreastCrop) -> tuple[BoundingBox, bool]:
    """Intersecta `box` (en espacio `crop`) con los límites del recorte
    `[0, crop_width) x [0, crop_height)`. Devuelve la caja intersectada y un
    indicador de si hubo recorte; una intersección vacía devuelve una caja
    degenerada explícita `(0,0,0,0)`, nunca `None`."""
    if box.space != "crop":
        raise ValueError(
            f"clip_to_crop espera una caja en espacio 'crop', recibió {box.space!r}."
        )

    x0 = max(box.x0, 0.0)
    y0 = max(box.y0, 0.0)
    x1 = min(box.x1, float(crop.crop_width))
    y1 = min(box.y1, float(crop.crop_height))

    if x1 <= x0 or y1 <= y0:
        return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0, space="crop"), True

    clipped = (x0, y0, x1, y1) != (box.x0, box.y0, box.x1, box.y1)
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, space="crop"), clipped
