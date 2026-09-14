from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from mammo_lejepa.models import FailureCategory, ImageRecord, ImageTask
from mammo_lejepa.resume import (
    latest_records_by_image,
    parse_jsonl_records,
    pending_tasks,
    summarize_failures,
)


def _task(image_id: str, study_id: str = "study_a") -> ImageTask:
    return ImageTask(
        study_id=study_id,
        series_id=f"{study_id}-series",
        image_id=image_id,
        split="training",
        laterality="L",
        view_position="CC",
        breast_birads="BI-RADS 1",
        breast_density="DENSITY B",
        expected_height=100,
        expected_width=100,
    )


def _record(
    image_id: str,
    *,
    status: str = "ok",
    processed_at: str = "2026-01-01T00:00:00+00:00",
    failure_category: FailureCategory | None = None,
) -> ImageRecord:
    return ImageRecord(
        study_id="study_a",
        series_id="study_a-series",
        image_id=image_id,
        status=status,
        split="training",
        laterality="L",
        view_position="CC",
        breast_birads="BI-RADS 1",
        breast_density="DENSITY B",
        run_id="run_x",
        pipeline_version="0.1.0",
        processed_at=processed_at,
        download_seconds=1.0,
        process_seconds=1.0,
        dicom_bytes=1000,
        failure_category=failure_category,
    )


def test_ok_record_with_png_is_not_pending() -> None:
    manifest = [_task("img_1")]
    completed = {"img_1": _record("img_1", status="ok")}

    result = pending_tasks(manifest, completed, existing_pngs={"img_1"})

    assert result == []


def test_ok_record_without_png_returns_to_pending() -> None:
    manifest = [_task("img_1")]
    completed = {"img_1": _record("img_1", status="ok")}

    result = pending_tasks(manifest, completed, existing_pngs=set())

    assert result == [manifest[0]]


def test_failed_record_only_returns_in_retry_mode() -> None:
    manifest = [_task("img_1")]
    completed = {"img_1": _record("img_1", status="failed")}

    # Modo normal: no se reintenta automáticamente lo fallido.
    result = pending_tasks(manifest, completed, existing_pngs=set())
    assert result == [manifest[0]]

    # En modo reintento explícito, el llamador pasa `completed={}` para que
    # la imagen vuelva a la lista de pendientes (comportamiento de `retry`).
    result_retry = pending_tasks(manifest, completed={}, existing_pngs=set())
    assert result_retry == [manifest[0]]


def test_unresolved_task_is_pending() -> None:
    manifest = [_task("img_1"), _task("img_2")]
    completed = {"img_1": _record("img_1", status="ok")}

    result = pending_tasks(manifest, completed, existing_pngs={"img_1"})

    assert [task.image_id for task in result] == ["img_2"]


def test_parse_jsonl_discards_truncated_last_line() -> None:
    complete_json = json.dumps(asdict(_record("img_1")), default=str)
    truncated = '{"study_id": "study_a", "image_id": "img_2", "stat'

    records = parse_jsonl_records([complete_json, truncated])

    assert len(records) == 1
    assert records[0].image_id == "img_1"


def test_parse_jsonl_raises_on_malformed_middle_line() -> None:
    complete_json = json.dumps(asdict(_record("img_1")), default=str)
    malformed = "not-json-at-all"

    with pytest.raises(json.JSONDecodeError):
        parse_jsonl_records([malformed, complete_json])


def test_latest_records_by_image_keeps_most_recent() -> None:
    older = _record("img_1", processed_at="2026-01-01T00:00:00+00:00")
    newer = _record("img_1", processed_at="2026-01-02T00:00:00+00:00")

    latest = latest_records_by_image([older, newer])

    assert latest["img_1"] is newer


def test_summarize_failures_counts_by_category() -> None:
    records = [
        _record("img_1", status="failed", failure_category=FailureCategory.DECODE),
        _record("img_2", status="failed", failure_category=FailureCategory.DECODE),
        _record("img_3", status="failed", failure_category=FailureCategory.NETWORK),
        _record("img_4", status="ok"),
    ]

    summary = summarize_failures(records)

    assert summary == {
        FailureCategory.DECODE: 2,
        FailureCategory.NETWORK: 1,
    }
