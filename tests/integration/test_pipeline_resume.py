from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydicom.filebase import DicomBytesIO
from pydicom.filewriter import dcmwrite

from mammo_lejepa.catalog import consolidate
from mammo_lejepa.config import PhysioNetCredentials, PipelineConfig
from mammo_lejepa.models import ImageTask
from mammo_lejepa.pipeline import run_pipeline
from mammo_lejepa.resume import (
    latest_records_by_image,
    parse_jsonl_records,
    pending_tasks,
)
from tests.fixtures.synthetic_dicom import make_synthetic_dicom

pytestmark = pytest.mark.slow

_CREDENTIALS = PhysioNetCredentials(username="u", password="p", session_id="s")
_ALL_IMAGES = [
    ("study_1", "image_a"),
    ("study_1", "image_b"),
    ("study_2", "image_c"),
    ("study_2", "image_d"),
]


def _tasks(pairs: list[tuple[str, str]]) -> list[ImageTask]:
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
        for study_id, image_id in pairs
    ]


async def _validate_access_handler(request: web.Request) -> web.Response:
    return web.Response(text="study_id,image_id\n", content_type="text/csv")


_DOWNLOAD_COUNTS: dict[str, int] = {}


async def _image_handler(request: web.Request) -> web.Response:
    image_id = request.match_info["image_id"]
    _DOWNLOAD_COUNTS[image_id] = _DOWNLOAD_COUNTS.get(image_id, 0) + 1
    dataset, _ = make_synthetic_dicom(height=256, width=192)
    buffer = DicomBytesIO()
    dcmwrite(buffer, dataset, enforce_file_format=True)
    return web.Response(body=buffer.getvalue(), content_type="application/dicom")


async def test_interrupted_run_resumes_without_redownloading(tmp_path: Path) -> None:
    _DOWNLOAD_COUNTS.clear()

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

        # "Interrupción a mitad": sólo se lanzan las 2 primeras imágenes.
        first_half = _tasks(_ALL_IMAGES[:2])
        await run_pipeline(
            first_half,
            config=config,
            credentials=_CREDENTIALS,
            run_id="run_before_interrupt",
            command="run",
        )

        # Al relanzar, resume.pending_tasks calcula lo que falta sobre el
        # manifiesto COMPLETO (las 4 imágenes), no sólo la primera mitad.
        lines = config.images_jsonl_path.read_text().splitlines()
        completed = latest_records_by_image(parse_jsonl_records(lines))
        existing_pngs = {p.stem for p in config.processed_dir.rglob("*.png")}

        all_tasks = _tasks(_ALL_IMAGES)
        remaining = pending_tasks(all_tasks, completed, existing_pngs)

        assert {t.image_id for t in remaining} == {"image_c", "image_d"}

        await run_pipeline(
            remaining,
            config=config,
            credentials=_CREDENTIALS,
            run_id="run_after_resume",
            command="resume",
        )

    # Cero descargas repetidas: cada imagen se pidió a la red exactamente una vez.
    assert _DOWNLOAD_COUNTS == {"image_a": 1, "image_b": 1, "image_c": 1, "image_d": 1}

    # El catálogo consolidado no tiene duplicados: una fila por imagen procesada.
    lines = config.images_jsonl_path.read_text().splitlines()
    all_records = parse_jsonl_records(lines)
    images_df, _ = consolidate(all_records, _empty_findings())

    assert images_df.height == 4
    assert images_df.get_column("image_id").n_unique() == 4
    assert set(images_df.get_column("status").to_list()) == {"ok"}


def _empty_findings():
    import polars as pl

    schema = {
        "image_id": pl.Utf8,
        "study_id": pl.Utf8,
        "finding_categories": pl.Utf8,
        "finding_birads": pl.Utf8,
        "xmin": pl.Float64,
        "ymin": pl.Float64,
        "xmax": pl.Float64,
        "ymax": pl.Float64,
    }
    return pl.DataFrame(schema=schema)
