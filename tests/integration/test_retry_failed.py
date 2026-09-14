from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydicom.filebase import DicomBytesIO
from pydicom.filewriter import dcmwrite

from mammo_lejepa.config import PhysioNetCredentials, PipelineConfig
from mammo_lejepa.models import ImageTask
from mammo_lejepa.pipeline import run_pipeline
from mammo_lejepa.resume import latest_records_by_image, parse_jsonl_records
from tests.fixtures.synthetic_dicom import make_synthetic_dicom

pytestmark = pytest.mark.slow

_CREDENTIALS = PhysioNetCredentials(username="u", password="p", session_id="s")
_IMAGES = [("study_1", "image_ok"), ("study_1", "image_flaky")]


def _tasks() -> list[ImageTask]:
    return [
        ImageTask(
            study_id=study_id,
            series_id=f"{study_id}-series",
            image_id=image_id,
            split="training",
            laterality="L",
            view_position="CC",
            breast_birads="BI-RADS 1",
            breast_density="DENSITY B",
            expected_height=256,
            expected_width=192,
        )
        for study_id, image_id in _IMAGES
    ]


async def _validate_access_handler(request: web.Request) -> web.Response:
    return web.Response(text="study_id,image_id\n", content_type="text/csv")


def _dicom_bytes() -> bytes:
    dataset, _ = make_synthetic_dicom(height=256, width=192)
    buffer = DicomBytesIO()
    dcmwrite(buffer, dataset, enforce_file_format=True)
    return buffer.getvalue()


async def test_retry_only_processes_failed_and_replaces_its_record(
    tmp_path: Path,
) -> None:
    flaky_should_fail = {"value": True}

    async def image_handler(request: web.Request) -> web.Response:
        image_id = request.match_info["image_id"]
        if image_id == "image_flaky" and flaky_should_fail["value"]:
            return web.Response(status=404, text="not found (permanent for retry test)")
        return web.Response(body=_dicom_bytes(), content_type="application/dicom")

    app = web.Application()
    app.router.add_get("/files/ds/v1/images/{study_id}/{image_id}.dicom", image_handler)
    app.router.add_get(
        "/files/ds/v1/breast-level_annotations.csv", _validate_access_handler
    )

    async with TestServer(app) as server:
        config = PipelineConfig(
            data_dir=tmp_path / "vindr-mammo",
            base_url=str(server.make_url("/")).rstrip("/"),
            dataset_name="ds",
            dataset_version="v1",
            downloads=2,
            workers=2,
            queue_size=2,
            max_retries=0,  # que falle rápido, sin reintento de red interno
        )

        first_summary = await run_pipeline(
            _tasks(),
            config=config,
            credentials=_CREDENTIALS,
            run_id="run_first",
            command="run",
        )
        assert first_summary.images_ok == 1
        assert first_summary.images_failed == 1

        # El fallo "se resuelve": la red ya sirve el DICOM normalmente.
        flaky_should_fail["value"] = False

        retry_tasks = [task for task in _tasks() if task.image_id == "image_flaky"]
        retry_summary = await run_pipeline(
            retry_tasks,
            config=config,
            credentials=_CREDENTIALS,
            run_id="run_retry",
            command="retry",
        )

    assert retry_summary.images_total == 1
    assert retry_summary.images_ok == 1

    lines = config.images_jsonl_path.read_text().splitlines()
    all_records = parse_jsonl_records(lines)
    flaky_records = [r for r in all_records if r.image_id == "image_flaky"]

    # Dos líneas en el JSONL append-only (fallo original + reintento), pero
    # la resolución "más reciente gana" sólo ve la exitosa.
    assert len(flaky_records) == 2
    assert flaky_records[0].status == "failed"
    assert flaky_records[1].status == "ok"

    latest = latest_records_by_image(all_records)
    assert latest["image_flaky"].status == "ok"
    assert latest["image_ok"].status == "ok"

    raw_lines = json.loads(lines[-1])
    assert raw_lines["run_id"] == "run_retry"
