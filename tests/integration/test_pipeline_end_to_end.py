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
from tests.fixtures.synthetic_dicom import make_synthetic_dicom

pytestmark = pytest.mark.slow

_CREDENTIALS = PhysioNetCredentials(username="u", password="p", session_id="s")
_IMAGES = [
    ("study_1", "image_1a"),
    ("study_1", "image_1b"),
    ("study_2", "image_2a"),
]


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


async def _handler(request: web.Request) -> web.Response:
    dataset, _ = make_synthetic_dicom(height=256, width=192)

    buffer = DicomBytesIO()
    dcmwrite(buffer, dataset, enforce_file_format=True)
    return web.Response(body=buffer.getvalue(), content_type="application/dicom")


async def _validate_access_handler(request: web.Request) -> web.Response:
    return web.Response(text="study_id,image_id\n", content_type="text/csv")


async def test_pipeline_end_to_end_produces_png_and_catalog(tmp_path: Path) -> None:
    app = web.Application()
    app.router.add_get("/files/ds/v1/images/{study_id}/{image_id}.dicom", _handler)
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
            run_id="run_test_001",
            command="run",
        )

    assert summary.images_total == 3
    assert summary.images_ok == 3
    assert summary.images_failed == 0

    for study_id, image_id in _IMAGES:
        png_path = config.processed_dir / study_id / f"{image_id}.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0

    assert list(config.dicom_dir.rglob("*.dicom")) == []
    assert list(config.dicom_dir.rglob("*.part")) == []

    lines = config.images_jsonl_path.read_text().splitlines()
    assert len(lines) == 3

    records = [json.loads(line) for line in lines]
    required_fields = {
        "study_id",
        "series_id",
        "image_id",
        "status",
        "png_path",
        "png_bytes",
        "png_sha256",
        "breast_crop",
        "split",
        "laterality",
        "view_position",
        "breast_birads",
        "breast_density",
        "photometric_interpretation",
        "transfer_syntax_uid",
        "run_id",
        "pipeline_version",
        "processed_at",
    }
    for record in records:
        assert required_fields <= record.keys()
        assert record["status"] == "ok"
        assert record["breast_crop"]["otsu_threshold"] > 0
