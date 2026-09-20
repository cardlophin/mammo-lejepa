from __future__ import annotations

from mammo_lejepa.errors import DecodeError, SegmentationError, WriteError
from mammo_lejepa.vindr.errors import (
    AuthError,
    NetworkError,
    classify_exception,
    redact_secrets,
)
from mammo_lejepa.vindr.models import FailureCategory


def test_classify_exception_maps_auth_and_network() -> None:
    assert classify_exception(AuthError("x")) is FailureCategory.AUTH
    assert classify_exception(NetworkError("x")) is FailureCategory.NETWORK


def test_classify_exception_delegates_shared_categories() -> None:
    assert classify_exception(DecodeError("x")) is FailureCategory.DECODE
    assert classify_exception(SegmentationError("x")) is FailureCategory.SEGMENTATION
    assert classify_exception(WriteError("x")) is FailureCategory.WRITE


def test_classify_exception_unknown_for_unrecognized() -> None:
    assert classify_exception(ValueError("x")) is FailureCategory.UNKNOWN


def test_redact_secrets_replaces_each_non_empty_secret() -> None:
    text = "fallo con usuario=alice contraseña=hunter2 cookie=abc123"

    redacted = redact_secrets(text, "alice", "hunter2", "abc123")

    assert "alice" not in redacted
    assert "hunter2" not in redacted
    assert "abc123" not in redacted
    assert redacted.count("***REDACTED***") == 3


def test_redact_secrets_ignores_empty_and_none_secrets() -> None:
    text = "sin secretos que ocultar"

    redacted = redact_secrets(text, "", None)

    assert redacted == text
