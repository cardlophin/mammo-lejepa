from __future__ import annotations

from mammo_lejepa.errors import DecodeError, SegmentationError, WriteError
from mammo_lejepa.vindr.models import FailureCategory

__all__ = [
    "AuthError",
    "DecodeError",
    "NetworkError",
    "SegmentationError",
    "VindrError",
    "WriteError",
    "classify_exception",
    "redact_secrets",
]


class VindrError(Exception):
    """Base de las excepciones propias del ETL de VinDr-Mammo."""


class AuthError(VindrError):
    """Sesión de PhysioNet inválida o caducada. Motivo de aborto global."""


class NetworkError(VindrError):
    """Fallo de red transitorio o permanente durante la descarga."""


_EXCEPTION_TO_CATEGORY: dict[type[Exception], FailureCategory] = {
    AuthError: FailureCategory.AUTH,
    NetworkError: FailureCategory.NETWORK,
    DecodeError: FailureCategory.DECODE,
    SegmentationError: FailureCategory.SEGMENTATION,
    WriteError: FailureCategory.WRITE,
}


def classify_exception(error: Exception) -> FailureCategory:
    """Mapea una excepción a su `FailureCategory` propio de VinDr (con
    `AUTH`/`NETWORK`, que el de Mammo-Bench no tiene — research.md D-02).
    Cualquier excepción no reconocida se clasifica como `UNKNOWN` en lugar de
    propagarse."""
    for exc_type, category in _EXCEPTION_TO_CATEGORY.items():
        if isinstance(error, exc_type):
            return category
    return FailureCategory.UNKNOWN


_REDACTED_PLACEHOLDER = "***REDACTED***"


def redact_secrets(text: str, *secrets: str | None) -> str:
    """Sustituye cualquier aparición literal de un secreto (cookie sessionid,
    usuario, contraseña) por un marcador. Se aplica a mensajes de error antes
    de registrarlos, para que nunca aparezcan credenciales en la salida, en
    los mensajes de error ni en los artefactos generados (FR-026). Los
    mensajes ya se construyen sin interpolar credenciales; esto es una
    segunda capa de defensa ante un error inesperado de una librería
    externa."""
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _REDACTED_PLACEHOLDER)
    return redacted
