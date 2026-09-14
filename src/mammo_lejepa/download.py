from __future__ import annotations

import asyncio
import random
from pathlib import Path
from time import perf_counter

import aiofiles
import aiohttp
from yarl import URL

from mammo_lejepa.config import PhysioNetCredentials, PipelineConfig
from mammo_lejepa.errors import AuthError, NetworkError
from mammo_lejepa.models import DownloadResult, ImageTask

_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
_CHUNK_SIZE = 1024 * 1024


def build_cookie_jar(session_id: str, base_url: str) -> aiohttp.CookieJar:
    """Restringe la cookie `sessionid` al dominio de PhysioNet."""
    cookie_jar = aiohttp.CookieJar()
    cookie_jar.update_cookies({"sessionid": session_id}, response_url=URL(base_url))
    return cookie_jar


def build_client_session(
    credentials: PhysioNetCredentials, config: PipelineConfig
) -> aiohttp.ClientSession:
    """Construye la sesión `aiohttp` autenticada, con `TCPConnector` acotado a
    `config.downloads` (FR-005, FR-006)."""
    timeout = aiohttp.ClientTimeout(
        total=config.total_timeout_seconds,
        connect=config.connect_timeout_seconds,
        sock_read=config.read_timeout_seconds,
    )
    headers = {
        "User-Agent": "mammo-etl/0.1.0 (authorized VinDr-Mammo download)",
        "Accept": "*/*",
        "Referer": config.content_url,
    }
    connector = aiohttp.TCPConnector(
        limit=config.downloads,
        limit_per_host=config.downloads,
        ttl_dns_cache=300,
    )
    cookie_jar = build_cookie_jar(credentials.session_id, config.base_url)

    return aiohttp.ClientSession(
        headers=headers,
        cookie_jar=cookie_jar,
        timeout=timeout,
        connector=connector,
    )


async def validate_access(
    session: aiohttp.ClientSession, config: PipelineConfig
) -> None:
    """Confirma que la cookie tiene acceso efectivo al dataset con una
    petición ligera antes de iniciar el lote (FR-006)."""
    test_url = f"{config.files_base_url}/breast-level_annotations.csv"

    async with session.get(test_url, allow_redirects=True) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        _raise_for_auth_failure(response.status, content_type)

        if response.status != 200:
            raise NetworkError(
                f"HTTP {response.status} al validar el acceso a {test_url}."
            )


async def download_dicom(
    session: aiohttp.ClientSession,
    task: ImageTask,
    destination: Path,
    config: PipelineConfig,
) -> DownloadResult:
    """Descarga el DICOM de `task` de forma atómica (fichero temporal +
    renombrado): un fichero con nombre definitivo es siempre un fichero
    completo (FR-007). Reintenta los fallos transitorios con retroceso
    exponencial y jitter, respetando `Retry-After` (FR-008), y distingue los
    fallos permanentes (401, 403, 404, HTML) elevándolos como `AuthError` o
    `NetworkError` sin reintentar."""
    if destination.exists() and destination.stat().st_size > 0:
        return DownloadResult(
            status="already_exists",
            dicom_path=destination,
            bytes_downloaded=destination.stat().st_size,
            download_seconds=0.0,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".part")
    url = config.dicom_url(task.study_id, task.image_id)
    start = perf_counter()
    max_attempts = config.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        try:
            downloaded_bytes = await _attempt_download(
                session, url, temp_path, attempt, max_attempts, config
            )
        except _RetryableFailure as retry_signal:
            await asyncio.sleep(retry_signal.delay_seconds)
            continue
        except AuthError:
            temp_path.unlink(missing_ok=True)
            raise
        except NetworkError:
            temp_path.unlink(missing_ok=True)
            raise

        temp_path.replace(destination)
        return DownloadResult(
            status="downloaded",
            dicom_path=destination,
            bytes_downloaded=downloaded_bytes,
            download_seconds=perf_counter() - start,
        )

    # Inalcanzable: el último intento siempre retorna o lanza (nunca reintenta).
    raise NetworkError(f"Fallo desconocido al descargar {url}.")


class _RetryableFailure(Exception):
    """Señal interna: reintentar tras `delay_seconds`. Nunca cruza la frontera
    pública de `download_dicom`."""

    def __init__(self, delay_seconds: float) -> None:
        super().__init__(f"reintentar en {delay_seconds:.1f}s")
        self.delay_seconds = delay_seconds


async def _attempt_download(
    session: aiohttp.ClientSession,
    url: str,
    temp_path: Path,
    attempt: int,
    max_attempts: int,
    config: PipelineConfig,
) -> int:
    is_last_attempt = attempt == max_attempts

    try:
        async with session.get(url, allow_redirects=True) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            _raise_for_auth_failure(response.status, content_type)

            if response.status in _TRANSIENT_STATUS:
                if is_last_attempt:
                    raise NetworkError(
                        f"HTTP {response.status} persistente tras {attempt} "
                        f"intentos: {url}"
                    )
                raise _RetryableFailure(_retry_delay(response, attempt, config))

            if response.status != 200:
                body_preview = (await response.text(errors="replace"))[:300]
                raise NetworkError(
                    f"HTTP {response.status} al descargar {url}: {body_preview}"
                )

            downloaded_bytes = 0
            async with aiofiles.open(temp_path, mode="wb") as handle:
                async for chunk in response.content.iter_chunked(_CHUNK_SIZE):
                    await handle.write(chunk)
                    downloaded_bytes += len(chunk)

    except (aiohttp.ClientError, TimeoutError) as error:
        if is_last_attempt:
            raise NetworkError(
                f"Fallo de red persistente tras {attempt} intentos en {url}: {error}"
            ) from error
        raise _RetryableFailure(_backoff_seconds(attempt, config)) from error

    if downloaded_bytes == 0:
        raise NetworkError(f"Se descargó un fichero vacío de {url}.")

    return downloaded_bytes


def _raise_for_auth_failure(status: int, content_type: str) -> None:
    if status == 401:
        raise AuthError(
            "401 Unauthorized: la cookie sessionid de PhysioNet no es válida "
            "o ha caducado. Renueva la cookie en tu navegador y actualiza .env."
        )
    if status == 403:
        raise AuthError(
            "403 Forbidden: la cookie sessionid no tiene acceso a VinDr-Mammo "
            "o ha caducado. Renueva la cookie en tu navegador y actualiza .env."
        )
    if "text/html" in content_type:
        raise AuthError(
            "PhysioNet devolvió HTML en lugar del recurso esperado: "
            "probablemente la sesión ha caducado. Renueva la cookie sessionid."
        )


def _retry_delay(
    response: aiohttp.ClientResponse, attempt: int, config: PipelineConfig
) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return _backoff_seconds(attempt, config)


def _backoff_seconds(attempt: int, config: PipelineConfig) -> float:
    base = config.backoff_factor**attempt
    jitter = random.uniform(0, base * 0.25)
    return base + jitter
