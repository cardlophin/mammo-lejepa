from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)


@dataclass(frozen=True, slots=True)
class EllipseGeometry:
    """Geometría conocida de la elipse sintética, en coordenadas de imagen."""

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


def _make_pixel_array(
    height: int,
    width: int,
    *,
    bits_allocated: int,
    photometric_interpretation: str,
    include_orientation_label: bool,
    uniform: bool,
) -> tuple[np.ndarray, EllipseGeometry]:
    max_value = (1 << bits_allocated) - 1
    background_level = int(max_value * 0.05)
    tissue_level = int(max_value * 0.8)

    center_x, center_y = width // 2, int(height * 0.55)
    radius_x, radius_y = int(width * 0.32), int(height * 0.38)
    geometry = EllipseGeometry(
        center_x=center_x, center_y=center_y, radius_x=radius_x, radius_y=radius_y
    )

    if uniform:
        array = np.full((height, width), background_level, dtype=np.float64)
    else:
        yy, xx = np.mgrid[0:height, 0:width]
        ellipse_mask = (
            ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
        ) <= 1.0

        array = np.full((height, width), background_level, dtype=np.float64)
        array[ellipse_mask] = tissue_level

        if include_orientation_label:
            label_h = max(6, height // 40)
            label_w = max(6, width // 40)
            array[4 : 4 + label_h, 4 : 4 + label_w] = tissue_level

    # MONOCHROME1: el valor mínimo se muestra como blanco, así que el tejido
    # (que debe verse claro tras la ventana) se codifica con valores RAW bajos.
    if photometric_interpretation == "MONOCHROME1":
        array = max_value - array

    pixels = np.clip(array, 0, max_value).astype(np.uint16)
    return pixels, geometry


def make_synthetic_dicom(
    *,
    height: int = 512,
    width: int = 384,
    photometric_interpretation: str = "MONOCHROME2",
    bits_allocated: int = 16,
    include_window: bool = True,
    include_orientation_label: bool = False,
    uniform: bool = False,
    laterality: str = "L",
    view_position: str = "CC",
    manufacturer: str = "SYNTH",
) -> tuple[FileDataset, EllipseGeometry]:
    """Fábrica de DICOM sintéticos: elipse clara sobre fondo oscuro con
    geometría conocida. Soporta MONOCHROME1/MONOCHROME2, 12/16 bits, con y sin
    `WindowCenter`/`WindowWidth`, una variante con una etiqueta de orientación
    simulada en una esquina (componente conexo pequeño y separado) y una
    variante `uniform` (sin mama detectable, para los tests de fallo de
    segmentación)."""
    pixels, geometry = _make_pixel_array(
        height,
        width,
        bits_allocated=bits_allocated,
        photometric_interpretation=photometric_interpretation,
        include_orientation_label=include_orientation_label,
        uniform=uniform,
    )

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()

    dataset.Modality = "MG"
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.SamplesPerPixel = 1
    dataset.Rows = height
    dataset.Columns = width
    dataset.BitsAllocated = 16
    dataset.BitsStored = bits_allocated
    dataset.HighBit = bits_allocated - 1
    dataset.PixelRepresentation = 0
    dataset.ImageLaterality = laterality
    dataset.ViewPosition = view_position
    dataset.Manufacturer = manufacturer
    dataset.ManufacturerModelName = "SyntheticGenerator"
    dataset.ImagerPixelSpacing = [0.1, 0.1]

    if include_window:
        max_value = (1 << bits_allocated) - 1
        dataset.WindowCenter = max_value / 2
        dataset.WindowWidth = max_value

    dataset.PixelData = pixels.tobytes()

    return dataset, geometry


def save_dicom(dataset: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(path, enforce_file_format=True)
    return path


def make_corrupt_dicom_bytes() -> bytes:
    """Bytes que parecen un DICOM (preámbulo + `DICM`) pero cuya cabecera no
    puede decodificarse: para los tests de aislamiento de fallos."""
    return b"\x00" * 128 + b"DICM" + b"\xff" * 64
