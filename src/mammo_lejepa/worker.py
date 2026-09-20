from __future__ import annotations

from time import perf_counter

from mammo_lejepa.boxing import box_iou, mask_area_ratio, resolve_box
from mammo_lejepa.config import CorpusConfig
from mammo_lejepa.errors import classify_exception
from mammo_lejepa.geometry import crop_image
from mammo_lejepa.image_io import read_grayscale, read_mask
from mammo_lejepa.models import ImagenMammoBench, RegistroDeRecorte
from mammo_lejepa.storage import write_png_atomic


def process_image(image: ImagenMammoBench, config: CorpusConfig) -> RegistroDeRecorte:
    """Lee imagen y máscara, resuelve la caja mamaria (máscara u Otsu de
    respaldo, D-01), recorta y escribe el PNG de forma atómica. Nunca lanza
    excepciones al proceso padre: cualquier fallo se captura y se devuelve
    como un `RegistroDeRecorte` fallido con su categoría. `run_id`,
    `code_version` y `processed_at` quedan vacíos: el escritor único
    (`runner.py`) los añade antes de persistir, igual que en 001."""
    start = perf_counter()

    try:
        pixels = read_grayscale(config.mammobench_root / image.preprocessed_path)
        mask_pixels = read_mask(config.mammobench_root / image.mask_path)

        chosen_box, otsu_box, fallback_reason = resolve_box(
            pixels, mask_pixels, config=config.boxing
        )

        mask_ratio = (
            mask_area_ratio(mask_pixels, threshold=config.boxing.mask_threshold)
            if mask_pixels is not None
            else None
        )
        mask_otsu_iou = box_iou(chosen_box, otsu_box) if otsu_box is not None else None

        cropped = crop_image(pixels, chosen_box)
        destination = config.crop_path(image.source_dataset, image.image_id)
        crop_bytes = write_png_atomic(cropped, destination)

        return RegistroDeRecorte(
            image_id=image.image_id,
            source_dataset=image.source_dataset,
            source_subject_id=image.source_subject_id,
            patient_key=image.patient_key,
            laterality=image.laterality,
            view=image.view,
            status="ok",
            process_seconds=perf_counter() - start,
            run_id="",
            code_version="",
            processed_at="",
            box=chosen_box,
            fallback_reason=fallback_reason,
            mask_area_ratio=mask_ratio,
            mask_otsu_iou=mask_otsu_iou,
            crop_path=str(destination),
            crop_bytes=crop_bytes,
            classification=image.classification,
            density=image.density,
            birads=image.birads,
            abnormality=image.abnormality,
            molecular_subtype=image.molecular_subtype,
            subject_age=image.subject_age,
        )
    except Exception as error:  # contrato: nunca propaga al proceso padre
        return RegistroDeRecorte(
            image_id=image.image_id,
            source_dataset=image.source_dataset,
            source_subject_id=image.source_subject_id,
            patient_key=image.patient_key,
            laterality=image.laterality,
            view=image.view,
            status="failed",
            process_seconds=perf_counter() - start,
            run_id="",
            code_version="",
            processed_at="",
            failure_category=classify_exception(error),
            error_message=str(error)[:2000],
        )
