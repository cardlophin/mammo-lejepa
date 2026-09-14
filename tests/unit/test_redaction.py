from __future__ import annotations

from mammo_lejepa.config import PhysioNetCredentials
from mammo_lejepa.errors import redact_secrets
from mammo_lejepa.models import ProcessOutcome
from mammo_lejepa.pipeline import _redact_outcome


def test_redact_secrets_replaces_each_occurrence() -> None:
    text = "fallo con cookie=abc123 y usuario=cardlophin en la traza"

    redacted = redact_secrets(text, "cardlophin", "abc123")

    assert "abc123" not in redacted
    assert "cardlophin" not in redacted
    assert "***REDACTED***" in redacted


def test_redact_secrets_ignores_falsy_secrets() -> None:
    text = "mensaje sin secretos"

    redacted = redact_secrets(text, None, "")

    assert redacted == text


def test_redact_secrets_leaves_unrelated_text_untouched() -> None:
    text = "HTTP 503 al descargar https://physionet.org/files/x.dicom"

    redacted = redact_secrets(text, "sekrit-cookie-value")

    assert redacted == text


def test_pipeline_redact_outcome_scrubs_error_message() -> None:
    credentials = PhysioNetCredentials(
        username="cardlophin", password="hunter2", session_id="abc123sessioncookie"
    )
    outcome = ProcessOutcome(
        status="failed",
        process_seconds=0.0,
        error_message="fallo inesperado: cookie abc123sessioncookie caducada",
    )

    redacted = _redact_outcome(outcome, credentials)

    assert redacted.error_message is not None
    assert "abc123sessioncookie" not in redacted.error_message
    assert "***REDACTED***" in redacted.error_message


def test_pipeline_redact_outcome_is_noop_without_error_message() -> None:
    credentials = PhysioNetCredentials(username="u", password="p", session_id="s")
    outcome = ProcessOutcome(status="ok", process_seconds=1.0)

    redacted = _redact_outcome(outcome, credentials)

    assert redacted is outcome
