from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torchvision.transforms import v2

from mammo_lejepa.ssl.config import AugmentConfig

_COLOR_JITTER_PROBABILITY = 0.8
_GAUSSIAN_BLUR_KERNEL_SIZE = 7


def build_view_transform(
    config: AugmentConfig,
) -> Callable[[Image.Image], torch.Tensor]:
    """Canalización de vistas para mamografía (D-06, T012) — decidida y
    justificada aquí, no heredada sin revisión de la receta de imagen
    natural:

    - `RandomResizedCrop` con escala revisada a `[crop_scale_min,
      crop_scale_max]` (por defecto `[0.4, 1.0]`, frente al `[0.08, 1.0]` de
      ImageNet): el corpus ya viene recortado a la mama (feature 002), así
      que una vista del 8% del área puede no contener tejido en absoluto.
    - Volteo horizontal: admisible porque el corpus contiene mamas izquierdas
      y derechas, pero destruye la lateralidad como señal — declarado, no
      ignorado.
    - Jitter suave de brillo/contraste únicamente (sin tono ni saturación:
      la imagen es de un solo canal, agrupado a RGB sólo para compatibilidad
      de forma con encoders preentrenados en ImageNet — D-09).
    - Desenfoque gaussiano suave.
    - **Excluido deliberadamente**: `solarize` (invierte intensidades altas,
      y la intensidad correlaciona con densidad del tejido, una de las
      etiquetas a predecir — invertirla la destruye); conversión a escala de
      grises (la imagen ya lo es; convertir "de nuevo" no aumenta nada);
      rotaciones grandes (la orientación en mamografía es canónica, no
      arbitraria como en fotografía natural)."""
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                config.resolution,
                scale=(config.crop_scale_min, config.crop_scale_max),
            ),
            v2.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=config.brightness_jitter,
                        contrast=config.contrast_jitter,
                    )
                ],
                p=_COLOR_JITTER_PROBABILITY,
            ),
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        kernel_size=_GAUSSIAN_BLUR_KERNEL_SIZE,
                        sigma=(
                            config.gaussian_blur_sigma_min,
                            config.gaussian_blur_sigma_max,
                        ),
                    )
                ],
                p=config.gaussian_blur_prob,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def build_eval_transform(
    config: AugmentConfig,
) -> Callable[[Image.Image], torch.Tensor]:
    """Transformación determinista (sin aleatoriedad) para evaluación: cambio
    de tamaño y recorte central, sin ninguna de las aumentaciones de
    entrenamiento. La misma para el encoder preentrenado y para las líneas
    base (D-09): si difiriera, la comparación no valdría."""
    return v2.Compose(
        [
            v2.Resize(config.resolution),
            v2.CenterCrop(config.resolution),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def describe(config: AugmentConfig) -> dict[str, object]:
    """La configuración exacta y serializable de la canalización de
    entrenamiento (contracts.md): una aumentación que no aparece aquí no
    puede estar en la canalización."""
    return {
        "resolution": config.resolution,
        "random_resized_crop": {
            "scale": [config.crop_scale_min, config.crop_scale_max],
        },
        "horizontal_flip": {
            "probability": config.horizontal_flip_prob,
            "note": "destruye la lateralidad como señal",
        },
        "color_jitter": {
            "probability": _COLOR_JITTER_PROBABILITY,
            "brightness": config.brightness_jitter,
            "contrast": config.contrast_jitter,
        },
        "gaussian_blur": {
            "probability": config.gaussian_blur_prob,
            "kernel_size": _GAUSSIAN_BLUR_KERNEL_SIZE,
            "sigma_range": [
                config.gaussian_blur_sigma_min,
                config.gaussian_blur_sigma_max,
            ],
        },
        "excluded": {
            "solarize": "invertiría intensidades; la intensidad correlaciona "
            "con densidad del tejido, una etiqueta a predecir",
            "grayscale_conversion": "la imagen ya es de un solo canal",
            "hue_saturation_jitter": "no aplica a una imagen sin color",
            "large_rotations": "la orientación en mamografía es canónica",
        },
    }
