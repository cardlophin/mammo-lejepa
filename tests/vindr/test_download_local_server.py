from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from mammo_lejepa.vindr.config import PhysioNetCredentials, PipelineConfig
from mammo_lejepa.vindr.download import build_client_session, download_dicom
from mammo_lejepa.vindr.errors import AuthError, NetworkError
from mammo_lejepa.vindr.models import ImageTask

pytestmark = pytest.mark.slow

_ROUTE = "/files/ds/v1/images/{study_id}/{image_id}.dicom"
_CREDENTIALS = PhysioNetCredentials(username="u", password="p", session_id="s")


def _task() -> ImageTask:
    return ImageTask(
        study_id="study_x",
        series_id="series_x",
        image_id="image_x",
        split="training",
        laterality="L",
        view_position="CC",
        breast_birads="BI-RADS 1",
        breast_density="DENSITY B",
        expected_height=100,
        expected_width=100,
    )


def _config(tmp_path: Path, server: TestServer, **overrides: object) -> PipelineConfig:
    base_url = str(server.make_url("/")).rstrip("/")
    defaults = dict(
        data_dir=tmp_path / "vindr-mammo",
        base_url=base_url,
        dataset_name="ds",
        dataset_version="v1",
        max_retries=2,
        backoff_factor=0.01,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


async def test_download_is_atomic_no_part_file_after_success(tmp_path: Path) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"FAKE-DICOM-BYTES", content_type="application/dicom")

    app = web.Application()
    app.router.add_get(_ROUTE, handler)

    async with TestServer(app) as server:
        config = _config(tmp_path, server)
        async with build_client_session(_CREDENTIALS, config) as session:
            destination = config.dicom_dir / "study_x" / "image_x.dicom"
            result = await download_dicom(session, _task(), destination, config)

    assert result.status == "downloaded"
    assert destination.read_bytes() == b"FAKE-DICOM-BYTES"
    assert not destination.with_name(destination.name + ".part").exists()


async def test_transient_503_is_retried_and_eventually_succeeds(
    tmp_path: Path,
) -> None:
    attempts = {"count": 0}

    async def handler(request: web.Request) -> web.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return web.Response(status=503, text="try again")
        return web.Response(body=b"OK-AFTER-RETRY")

    app = web.Application()
    app.router.add_get(_ROUTE, handler)

    async with TestServer(app) as server:
        config = _config(tmp_path, server)
        async with build_client_session(_CREDENTIALS, config) as session:
            destination = config.dicom_dir / "study_x" / "image_x.dicom"
            result = await download_dicom(session, _task(), destination, config)

    assert result.status == "downloaded"
    assert attempts["count"] == 2
    assert destination.read_bytes() == b"OK-AFTER-RETRY"


async def test_persistent_failure_leaves_no_part_file(tmp_path: Path) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=500, text="always broken")

    app = web.Application()
    app.router.add_get(_ROUTE, handler)

    async with TestServer(app) as server:
        config = _config(tmp_path, server)
        destination = config.dicom_dir / "study_x" / "image_x.dicom"
        async with build_client_session(_CREDENTIALS, config) as session:
            with pytest.raises(NetworkError):
                await download_dicom(session, _task(), destination, config)

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()


async def test_403_raises_auth_error_without_retrying(tmp_path: Path) -> None:
    attempts = {"count": 0}

    async def handler(request: web.Request) -> web.Response:
        attempts["count"] += 1
        return web.Response(status=403, text="forbidden")

    app = web.Application()
    app.router.add_get(_ROUTE, handler)

    async with TestServer(app) as server:
        config = _config(tmp_path, server)
        destination = config.dicom_dir / "study_x" / "image_x.dicom"
        async with build_client_session(_CREDENTIALS, config) as session:
            with pytest.raises(AuthError):
                await download_dicom(session, _task(), destination, config)

    assert attempts["count"] == 1


async def test_html_response_raises_auth_error(tmp_path: Path) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            text="<html>session expired</html>", content_type="text/html"
        )

    app = web.Application()
    app.router.add_get(_ROUTE, handler)

    async with TestServer(app) as server:
        config = _config(tmp_path, server)
        destination = config.dicom_dir / "study_x" / "image_x.dicom"
        async with build_client_session(_CREDENTIALS, config) as session:
            with pytest.raises(AuthError):
                await download_dicom(session, _task(), destination, config)


async def test_already_downloaded_file_is_skipped(tmp_path: Path) -> None:
    async def handler(request: web.Request) -> web.Response:
        raise AssertionError("no debería llamarse: el fichero ya existe")

    app = web.Application()
    app.router.add_get(_ROUTE, handler)

    async with TestServer(app) as server:
        config = _config(tmp_path, server)
        destination = config.dicom_dir / "study_x" / "image_x.dicom"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"already here")

        async with build_client_session(_CREDENTIALS, config) as session:
            result = await download_dicom(session, _task(), destination, config)

    assert result.status == "already_exists"
