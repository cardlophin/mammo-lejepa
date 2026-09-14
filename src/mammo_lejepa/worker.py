from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from mammo_lejepa.config import PipelineConfig
from mammo_lejepa.dicom_io import read_dicom
from mammo_lejepa.errors import DecodeError, classify_exception
from mammo_lejepa.geometry import crop_image
from mammo_lejepa.models import ImageTask, ProcessOutcome
from mammo_lejepa.segmentation import detect_breast_box
from mammo_lejepa.storage import write_png_atomic
from mammo_lejepa.windowing import apply_display_window, normalize_to_uint8


def process_dicom(
    task: ImageTask, dicom_path: Path, config: PipelineConfig
) -> ProcessOutcome:
    """Encadena lectura, ventana, normalización, detección, recorte y
    escritura del PNG. Nunca lanza excepciones al proceso padre: cualquier
    fallo se captura y se devuelve como un `ProcessOutcome` fallido con su
    categoría y su traza (contrato de `worker.py`)."""
    start = perf_counter()

    try:
        payload = read_dicom(dicom_path)

        if (
            payload.rows != task.expected_height
            or payload.columns != task.expected_width
        ):
            # Discrepancia CSV vs. DICOM real: se registra como fallo, nunca
            # se corrige en silencio (data-model.md, notas sobre las fuentes).
            raise DecodeError(
                f"Dimensiones declaradas ({task.expected_height}x"
                f"{task.expected_width}) no coinciden con las reales del "
                f"DICOM ({payload.rows}x{payload.columns})."
            )

        windowed = apply_display_window(
            payload.pixels,
            window_center=payload.window_center,
            window_width=payload.window_width,
            photometric_interpretation=payload.photometric_interpretation,
        )
        normalized, normalize_low, normalize_high = normalize_to_uint8(
            windowed,
            low_percentile=config.low_percentile,
            high_percentile=config.high_percentile,
        )

        crop = detect_breast_box(
            normalized,
            margin_px=config.margin_px,
            blur_kernel=config.blur_kernel,
            close_kernel_ratio=config.close_kernel_ratio,
        )
        cropped_image = crop_image(normalized, crop)

        destination = config.processed_dir / task.study_id / f"{task.image_id}.png"
        png_bytes = write_png_atomic(cropped_image, destination)
        png_sha256 = _sha256_of_array(cropped_image)

        return ProcessOutcome(
            status="ok",
            process_seconds=perf_counter() - start,
            breast_crop=crop,
            photometric_interpretation=payload.photometric_interpretation,
            transfer_syntax_uid=payload.transfer_syntax_uid,
            window_center=payload.window_center,
            window_width=payload.window_width,
            pixel_spacing=payload.pixel_spacing,
            manufacturer=payload.manufacturer,
            model_name=payload.model_name,
            normalize_low=normalize_low,
            normalize_high=normalize_high,
            inverted_monochrome1=payload.photometric_interpretation == "MONOCHROME1",
            png_path=str(destination),
            png_bytes=png_bytes,
            png_sha256=png_sha256,
        )
    except Exception as error:  # contrato: nunca propaga al proceso padre
        return ProcessOutcome(
            status="failed",
            process_seconds=perf_counter() - start,
            failure_category=classify_exception(error),
            error_message=str(error)[:2000],
        )


def _sha256_of_array(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()
