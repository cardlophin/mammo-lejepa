from __future__ import annotations

import asyncio
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
_N_IMAGES = 12
# Imagen grande a propósito: Otsu + cierre morfológico tardan más que una
# descarga localhost casi instantánea, para que el procesado sea el cuello
# de botella y se manifieste la contrapresión de la cola acotada.
_HEIGHT, _WIDTH = 2200, 1700


def _tasks() -> list[ImageTask]:
    return [
        ImageTask(
            study_id=f"study_{i}",
            series_id=f"study_{i}-series",
            image_id=f"image_{i}",
            split="training",
            laterality="L",
            view_position="CC",
            breast_birads="BI-RADS 1",
            breast_density="DENSITY B",
            expected_height=_HEIGHT,
            expected_width=_WIDTH,
        )
        for i in range(_N_IMAGES)
    ]


async def _validate_access_handler(request: web.Request) -> web.Response:
    return web.Response(text="study_id,image_id\n", content_type="text/csv")


async def _image_handler(request: web.Request) -> web.Response:
    dataset, _ = make_synthetic_dicom(height=_HEIGHT, width=_WIDTH)
    buffer = DicomBytesIO()
    dcmwrite(buffer, dataset, enforce_file_format=True)
    return web.Response(body=buffer.getvalue(), content_type="application/dicom")


async def test_dicom_file_count_never_exceeds_the_calculated_ceiling(
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
            downloads=4,
            workers=1,
            queue_size=2,
        )

        max_observed = 0
        stop = asyncio.Event()

        async def sample_dicom_dir() -> None:
            nonlocal max_observed
            while not stop.is_set():
                count = sum(1 for _ in config.dicom_dir.rglob("*.dicom"))
                max_observed = max(max_observed, count)
                await asyncio.sleep(0.002)

        sampler = asyncio.create_task(sample_dicom_dir())
        try:
            summary = await run_pipeline(
                _tasks(),
                config=config,
                credentials=_CREDENTIALS,
                run_id="run_backpressure",
                command="run",
            )
        finally:
            stop.set()
            await sampler

    assert summary.images_ok == _N_IMAGES

    # D-03: techo = descargas simultáneas + tamaño de cola + workers, en
    # ficheros DICOM en vuelo (no depende de _N_IMAGES).
    ceiling_files = config.downloads + config.queue_size + config.workers
    assert max_observed <= ceiling_files
