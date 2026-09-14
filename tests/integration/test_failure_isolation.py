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
from tests.fixtures.synthetic_dicom import (
    make_corrupt_dicom_bytes,
    make_synthetic_dicom,
)

pytestmark = pytest.mark.slow

_CREDENTIALS = PhysioNetCredentials(username="u", password="p", session_id="s")
_GOOD_IMAGES = [("study_1", "image_ok_a"), ("study_1", "image_ok_b")]
_CORRUPT_IMAGE = ("study_1", "image_corrupt")


def _tasks() -> list[ImageTask]:
    ids = [*_GOOD_IMAGES[:1], _CORRUPT_IMAGE, *_GOOD_IMAGES[1:]]
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
        for study_id, image_id in ids
    ]


async def _validate_access_handler(request: web.Request) -> web.Response:
    return web.Response(text="study_id,image_id\n", content_type="text/csv")


async def _image_handler(request: web.Request) -> web.Response:
    image_id = request.match_info["image_id"]
    if image_id == _CORRUPT_IMAGE[1]:
        return web.Response(
            body=make_corrupt_dicom_bytes(), content_type="application/dicom"
        )

    dataset, _ = make_synthetic_dicom(height=256, width=192)
    buffer = DicomBytesIO()
    dcmwrite(buffer, dataset, enforce_file_format=True)
    return web.Response(body=buffer.getvalue(), content_type="application/dicom")


async def test_corrupt_dicom_is_quarantined_and_batch_continues(
    tmp_path: Path,
) -> None:
    app = web.Application()
    app.router.add_get(
        "/files/ds/v1/images/{study_id}/{image_id}.dicom", _image_handler
    )
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
        )

        summary = await run_pipeline(
            _tasks(),
            config=config,
            credentials=_CREDENTIALS,
            run_id="run_isolation",
            command="run",
        )

    # El lote continúa: las buenas se resuelven pese al fallo de la corrupta.
    assert summary.images_total == 3
    assert summary.images_ok == 2
    assert summary.images_failed == 1

    quarantined = list(config.quarantine_dir.rglob("*.dicom"))
    assert len(quarantined) == 1
    assert quarantined[0].name == f"{_CORRUPT_IMAGE[1]}.dicom"

    assert list(config.dicom_dir.rglob("*.dicom")) == []
    assert list(config.dicom_dir.rglob("*.part")) == []

    lines = config.images_jsonl_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    corrupt_record = next(r for r in records if r["image_id"] == _CORRUPT_IMAGE[1])

    assert corrupt_record["status"] == "failed"
    assert corrupt_record["failure_category"] == "DECODE"
    assert corrupt_record["error_message"]
