from __future__ import annotations

import asyncio
import os
from pathlib import Path
from time import perf_counter

import aiofiles
import aiohttp
from dotenv import load_dotenv
from physionet.api.utils import get_credentials_from_env
from yarl import URL

load_dotenv()


DICOM_URL = (
    "https://physionet.org/files/vindr-mammo/1.0.0/images/"
    "0028fb2c7f0b3a5cb9a80cb0e1cdbb91/"
    "7fc1f1bb8bb1a7efaf7104e49c4d8b86.dicom"
)

OUTPUT_PATH = Path(
    "data/vindr-mammo/aiohttp_benchmark/7fc1f1bb8bb1a7efaf7104e49c4d8b86.dicom"
)

CHUNK_SIZE = 1024 * 1024

CONNECT_TIMEOUT_SECONDS = 20
READ_TIMEOUT_SECONDS = 300
TOTAL_TIMEOUT_SECONDS = 600


def load_credentials() -> tuple[str, str, str]:
    """
    Carga username/password usando el helper de PhysioNet y carga
    PHYSIONET_SESSIONID desde .env.

    Para los ficheros restringidos de VinDr-Mammo, la cookie sessionid
    es la credencial que reproduce la sesión autorizada del navegador.
    """
    username, password = get_credentials_from_env()
    session_id = os.getenv("PHYSIONET_SESSIONID")

    if not username:
        raise RuntimeError(
            "No se encontró PHYSIONET_USERNAME. Comprueba el archivo .env."
        )

    if not password:
        raise RuntimeError(
            "No se encontró PHYSIONET_PASSWORD. Comprueba el archivo .env."
        )

    if not session_id:
        raise RuntimeError(
            "No se encontró PHYSIONET_SESSIONID. "
            "Copia la cookie sessionid desde el navegador."
        )

    return username, password, session_id


def create_cookie_jar(session_id: str) -> aiohttp.CookieJar:
    """
    Registra sessionid sólo para el dominio de PhysioNet.
    """
    cookie_jar = aiohttp.CookieJar()

    cookie_jar.update_cookies(
        {
            "sessionid": session_id,
        },
        response_url=URL("https://physionet.org/"),
    )

    return cookie_jar


async def benchmark_download() -> None:
    username, _, session_id = load_credentials()

    print(f"Usuario cargado con PhysioNet loader: {username!r}")
    print("Downloader: Python asyncio + aiohttp")
    print(f"URL: {DICOM_URL}")
    print(f"Destino: {OUTPUT_PATH}")
    print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = OUTPUT_PATH.with_suffix(".dicom.part")

    if OUTPUT_PATH.exists():
        print("Eliminando archivo previo para un benchmark limpio...")
        OUTPUT_PATH.unlink()

    if temporary_path.exists():
        temporary_path.unlink()

    timeout = aiohttp.ClientTimeout(
        total=TOTAL_TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
        sock_read=READ_TIMEOUT_SECONDS,
    )

    headers = {
        "User-Agent": (
            "aiohttp-vindr-mammo-benchmark/1.0 (authorized dataset download)"
        ),
        "Accept": "*/*",
        "Referer": ("https://physionet.org/content/vindr-mammo/1.0.0/"),
    }

    cookie_jar = create_cookie_jar(session_id)

    total_start = perf_counter()

    connector = aiohttp.TCPConnector(
        limit=1,
        limit_per_host=1,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(
        headers=headers,
        cookie_jar=cookie_jar,
        timeout=timeout,
        connector=connector,
    ) as session:
        response_start = perf_counter()

        async with session.get(
            DICOM_URL,
            allow_redirects=True,
        ) as response:
            response_elapsed = perf_counter() - response_start

            content_type = response.headers.get(
                "Content-Type",
                "unknown",
            )

            content_length = response.headers.get("Content-Length")

            print(f"HTTP status: {response.status}")
            print(f"Final URL: {response.url}")
            print(f"Content-Type: {content_type}")
            print(f"Content-Length: {content_length or 'no disponible'}")

            if response.status == 401:
                raise PermissionError(
                    "401 Unauthorized: tu cookie sessionid no es válida o ha caducado."
                )

            if response.status == 403:
                raise PermissionError(
                    "403 Forbidden: tu cookie sessionid no tiene acceso "
                    "a VinDr-Mammo o ha caducado."
                )

            if response.status != 200:
                body_preview = await response.text(errors="replace")

                raise RuntimeError(
                    f"HTTP {response.status} al descargar DICOM.\n"
                    f"Respuesta: {body_preview[:500]}"
                )

            if "text/html" in content_type.lower():
                body_preview = await response.text(errors="replace")

                raise RuntimeError(
                    "PhysioNet devolvió HTML en lugar de un DICOM.\n"
                    f"Respuesta: {body_preview[:500]}"
                )

            download_start = perf_counter()

            downloaded_bytes = 0
            chunks = 0

            async with aiofiles.open(
                temporary_path,
                mode="wb",
            ) as output_file:
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    await output_file.write(chunk)

                    downloaded_bytes += len(chunk)
                    chunks += 1

            download_elapsed = perf_counter() - download_start

    if downloaded_bytes == 0:
        temporary_path.unlink(missing_ok=True)

        raise RuntimeError("Se descargó un archivo vacío.")

    temporary_path.replace(OUTPUT_PATH)

    total_elapsed = perf_counter() - total_start

    mebibytes = downloaded_bytes / (1024 * 1024)

    throughput_mib_per_second = (
        mebibytes / download_elapsed if download_elapsed > 0 else 0.0
    )

    print()
    print("Descarga aiohttp terminada correctamente")
    print(f"Archivo: {OUTPUT_PATH}")
    print(f"Bytes escritos: {downloaded_bytes:,}")
    print(f"Chunks recibidos: {chunks:,}")
    print(f"Tiempo hasta respuesta HTTP: {response_elapsed:.3f} s")
    print(f"Tiempo de streaming: {download_elapsed:.3f} s")
    print(f"Tiempo total: {total_elapsed:.3f} s")
    print(f"Throughput efectivo: {throughput_mib_per_second:.2f} MiB/s")


def main() -> None:
    asyncio.run(benchmark_download())


if __name__ == "__main__":
    main()
