from __future__ import annotations

from mammo_lejepa.models import FailureCategory


class MammoCorpusError(Exception):
    """Base de todas las excepciones propias del preprocesado."""


class DecodeError(MammoCorpusError):
    """La imagen o la máscara no pudo decodificarse (fichero corrupto o
    formato no soportado)."""


class SegmentationError(MammoCorpusError):
    """No se pudo detectar un campo mamario válido, ni desde la máscara ni
    desde Otsu."""


class WriteError(MammoCorpusError):
    """Fallo al escribir el PNG o el registro en disco."""


_EXCEPTION_TO_CATEGORY: dict[type[Exception], FailureCategory] = {
    DecodeError: FailureCategory.DECODE,
    SegmentationError: FailureCategory.SEGMENTATION,
    WriteError: FailureCategory.WRITE,
}


def classify_exception(error: Exception) -> FailureCategory:
    """Mapea una excepción a su `FailureCategory`. Cualquier excepción no
    reconocida se clasifica como `UNKNOWN` en lugar de propagarse."""
    for exc_type, category in _EXCEPTION_TO_CATEGORY.items():
        if isinstance(error, exc_type):
            return category
    return FailureCategory.UNKNOWN
