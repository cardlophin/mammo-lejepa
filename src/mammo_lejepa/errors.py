from __future__ import annotations

from mammo_lejepa.models import FailureCategory


class MammoETLError(Exception):
    """Base de todas las excepciones propias del pipeline."""


class AuthError(MammoETLError):
    """Sesión de PhysioNet inválida o caducada. Motivo de aborto global."""


class NetworkError(MammoETLError):
    """Fallo de red transitorio o permanente durante la descarga."""


class DecodeError(MammoETLError):
    """El DICOM no pudo decodificarse (sintaxis de transferencia, decodificador
    ausente, cabecera corrupta)."""


class SegmentationError(MammoETLError):
    """No se pudo detectar un campo mamario válido en la imagen."""


class WriteError(MammoETLError):
    """Fallo al escribir el PNG o el registro en disco."""


_EXCEPTION_TO_CATEGORY: dict[type[Exception], FailureCategory] = {
    AuthError: FailureCategory.AUTH,
    NetworkError: FailureCategory.NETWORK,
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


_REDACTED_PLACEHOLDER = "***REDACTED***"


def redact_secrets(text: str, *secrets: str | None) -> str:
    """Sustituye cualquier aparición literal de un secreto (cookie sessionid,
    usuario, contraseña) por un marcador. Se aplica a mensajes de error antes
    de registrarlos, para que nunca aparezcan credenciales en la salida, en
    los mensajes de error ni en los artefactos generados (FR-031). Los
    mensajes ya se construyen sin interpolar credenciales; esto es una
    segunda capa de defensa ante un error inesperado de una librería
    externa."""
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _REDACTED_PLACEHOLDER)
    return redacted
