from __future__ import annotations

import cv2
import numpy as np

from mammo_lejepa.errors import SegmentationError
from mammo_lejepa.models import BoxingParams, BoxSource, CajaMamaria, FallbackReason
from mammo_lejepa.segmentation import box_from_binary_mask, detect_breast_box


def box_from_mask(
    mask_uint8: np.ndarray,
    *,
    threshold: int = 128,
    margin_px: int = 25,
) -> CajaMamaria:
    """Umbraliza `mask_uint8` a `threshold` (las máscaras de Mammo-Bench son
    JPEG con pérdida, no binarias puras), se queda con el mayor componente
    conexo y aplica el margen (FR-007, D-01). Lanza `SegmentationError` si la
    máscara queda vacía tras umbralizar, si el componente ocupa la imagen
    entera o si la caja resultante tiene área nula."""
    height, width = mask_uint8.shape
    _, binary_mask = cv2.threshold(mask_uint8, threshold, 255, cv2.THRESH_BINARY)

    x0, y0, x1, y1 = box_from_binary_mask(binary_mask, margin_px=margin_px)
    area_ratio = ((x1 - x0) * (y1 - y0)) / (width * height)

    return CajaMamaria(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        margin_px=margin_px,
        source=BoxSource.MASK,
        threshold=float(threshold),
        image_height=height,
        image_width=width,
        area_ratio=area_ratio,
    )


def mask_area_ratio(mask_uint8: np.ndarray, *, threshold: int) -> float:
    """Fracción de píxeles de `mask_uint8` por encima de `threshold`, sin
    seleccionar componente conexo: es la señal cruda que decide si la
    máscara está vacía o saturada (D-01), independiente de si luego produce
    una caja válida."""
    _, binary_mask = cv2.threshold(mask_uint8, threshold, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(binary_mask)) / binary_mask.size


def align_mask_to_image(
    mask_uint8: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    """Realinea `mask_uint8` a la rejilla de píxeles de `image_shape` con el
    vecino más próximo. Mammo-Bench guarda máscara e imagen a resoluciones
    distintas de forma sistemática para varias fuentes — no es un fichero
    corrupto puntual: en una muestra real de `ddsm`, 99 de 100 pares
    imagen/máscara difieren de tamaño (p. ej. `ddsm_0`: imagen 301x109,
    máscara 363x146). Sin este realineado, la caja calculada sobre la
    máscara queda en el espacio de píxeles equivocado y `crop_image`
    produce un recorte desplazado o, en el peor caso, vacío (visto en
    datos reales: `ddsm_10046`, imagen 354x84 frente a máscara 360x240).
    El vecino más próximo evita introducir grises intermedios en una
    máscara casi binaria."""
    if mask_uint8.shape == image_shape:
        return mask_uint8
    height, width = image_shape
    return cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)


def resolve_box(
    image_uint8: np.ndarray,
    mask_uint8: np.ndarray | None,
    *,
    config: BoxingParams,
) -> tuple[CajaMamaria, CajaMamaria | None, FallbackReason | None]:
    """Decide la caja mamaria: la máscara manda, Otsu respalda (D-01).
    Devuelve `(caja_elegida, caja_otsu_o_None, motivo_de_respaldo_o_None)`.
    La caja de Otsu se calcula siempre que exista una máscara válida sobre la
    que umbralizar, para el IoU (FR-009), aunque no sea la elegida. Si la
    máscara no coincide en resolución con la imagen, se realinea primero
    (`align_mask_to_image`) — no se trata como ausente."""
    aligned_mask = (
        align_mask_to_image(mask_uint8, image_uint8.shape)
        if mask_uint8 is not None
        else None
    )
    fallback_reason = _fallback_reason(aligned_mask, config)

    if fallback_reason is None:
        assert aligned_mask is not None
        try:
            mask_box = box_from_mask(
                aligned_mask,
                threshold=config.mask_threshold,
                margin_px=config.margin_px,
            )
        except SegmentationError:
            fallback_reason = FallbackReason.DEGENERATE_BOX
        else:
            otsu_box = detect_breast_box(
                image_uint8,
                margin_px=config.margin_px,
                blur_kernel=config.blur_kernel,
                close_kernel_ratio=config.close_kernel_ratio,
            )
            return mask_box, otsu_box, None

    otsu_box = detect_breast_box(
        image_uint8,
        margin_px=config.margin_px,
        blur_kernel=config.blur_kernel,
        close_kernel_ratio=config.close_kernel_ratio,
    )
    return otsu_box, None, fallback_reason


def _fallback_reason(
    mask_uint8: np.ndarray | None, config: BoxingParams
) -> FallbackReason | None:
    if mask_uint8 is None:
        return FallbackReason.MISSING_MASK

    ratio = mask_area_ratio(mask_uint8, threshold=config.mask_threshold)
    if ratio < config.mask_area_ratio_min:
        return FallbackReason.EMPTY_MASK
    if ratio > config.mask_area_ratio_max:
        return FallbackReason.SATURATED_MASK
    return None


def box_iou(a: CajaMamaria, b: CajaMamaria) -> float:
    """Intersection-over-union entre dos cajas en el mismo espacio de
    coordenadas. Nunca `None` ni `NaN`: `0.0` para cajas disjuntas."""
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
