from __future__ import annotations

from pathlib import Path
from typing import Any

import pydicom
from pydicom.multival import MultiValue

from mammo_lejepa.errors import DecodeError
from mammo_lejepa.models import DicomPayload

_DECODER_HINTS: dict[str, str] = {
    "1.2.840.10008.1.2.4.90": "pylibjpeg-openjpeg (JPEG 2000, Lossless)",
    "1.2.840.10008.1.2.4.91": "pylibjpeg-openjpeg (JPEG 2000)",
    "1.2.840.10008.1.2.4.70": "pylibjpeg-libjpeg (JPEG Lossless, Process 14, SV1)",
    "1.2.840.10008.1.2.4.57": "pylibjpeg-libjpeg (JPEG Lossless, Process 14)",
    "1.2.840.10008.1.2.4.80": "pylibjpeg-libjpeg (JPEG-LS Lossless)",
    "1.2.840.10008.1.2.4.81": "pylibjpeg-libjpeg (JPEG-LS Near-Lossless)",
}


def read_dicom(path: Path) -> DicomPayload:
    """Lee un DICOM y devuelve sus píxeles en tipo nativo junto con las
    cabeceras de procedencia. Traduce cualquier fallo de decodificación a
    `DecodeError` nombrando la sintaxis de transferencia y el paquete ausente,
    en lugar de dejar pasar un `AttributeError` opaco (FR-010, edge case de
    sintaxis sin decodificador)."""
    dataset = pydicom.dcmread(path)

    try:
        pixels = dataset.pixel_array
    except Exception as error:
        raise DecodeError(_decode_failure_message(dataset, path, error)) from error

    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax_uid = (
        str(file_meta.TransferSyntaxUID) if file_meta is not None else ""
    )

    pixel_spacing_raw = dataset.get("PixelSpacing") or dataset.get("ImagerPixelSpacing")
    pixel_spacing = (
        (float(pixel_spacing_raw[0]), float(pixel_spacing_raw[1]))
        if pixel_spacing_raw is not None
        else None
    )

    return DicomPayload(
        pixels=pixels,
        rows=int(dataset.Rows),
        columns=int(dataset.Columns),
        photometric_interpretation=str(dataset.get("PhotometricInterpretation", "")),
        transfer_syntax_uid=transfer_syntax_uid,
        window_center=_first_float(dataset.get("WindowCenter")),
        window_width=_first_float(dataset.get("WindowWidth")),
        pixel_spacing=pixel_spacing,
        manufacturer=_optional_str(dataset.get("Manufacturer")),
        model_name=_optional_str(dataset.get("ManufacturerModelName")),
    )


def _decode_failure_message(
    dataset: pydicom.Dataset, path: Path, error: Exception
) -> str:
    transfer_syntax = getattr(
        getattr(dataset, "file_meta", None), "TransferSyntaxUID", None
    )
    if transfer_syntax is not None:
        ts_repr = f"{transfer_syntax} ({transfer_syntax.name})"
        hint = _DECODER_HINTS.get(str(transfer_syntax), "pylibjpeg o python-gdcm")
    else:
        ts_repr = "desconocida"
        hint = "pylibjpeg o python-gdcm"

    return (
        f"No se pudo decodificar {path.name}: sintaxis de transferencia "
        f"{ts_repr}. Falta el decodificador ({hint}). Causa original: {error}"
    )


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, MultiValue | list):
        return float(value[0]) if len(value) else None
    return float(value)


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
