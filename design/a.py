from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final

import polars as pl
import requests
from dotenv import load_dotenv
from physionet.api.utils import get_credentials_from_env
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

BASE_URL: Final = "https://physionet.org"
DATASET_NAME: Final = "vindr-mammo"
DATASET_VERSION: Final = "1.0.0"

FILES_BASE_URL: Final = f"{BASE_URL}/files/{DATASET_NAME}/{DATASET_VERSION}"

DATA_DIR: Final = Path("data/vindr-mammo")

ANNOTATIONS_PATH: Final = DATA_DIR / "csv" / "breast-level_annotations.csv"

# Empieza bajo para validar que todo funciona.
SPLIT: Final = "training"
N_STUDIES: Final = 5
MAX_WORKERS: Final = 2

CHUNK_SIZE: Final = 1024 * 1024
CONNECT_TIMEOUT: Final = 20
READ_TIMEOUT: Final = 300

thread_local = threading.local()


# ---------------------------------------------------------------------------
# Credenciales y sesión autenticada
# ---------------------------------------------------------------------------


def load_physionet_credentials() -> tuple[str, str, str]:
    """
    Carga username/password con el loader oficial del paquete physionet
    y la cookie de sesión desde el archivo .env.
    """
    username, password = get_credentials_from_env()
    session_id = os.getenv("PHYSIONET_SESSIONID")

    if not username:
        raise RuntimeError(
            "PHYSIONET_USERNAME no está disponible. Comprueba tu archivo .env."
        )

    if not password:
        raise RuntimeError(
            "PHYSIONET_PASSWORD no está disponible. Comprueba tu archivo .env."
        )

    if not session_id:
        raise RuntimeError(
            "PHYSIONET_SESSIONID no está disponible. "
            "Copia la cookie sessionid de PhysioNet desde tu navegador "
            "y guárdala en .env."
        )

    return username, password, session_id


def create_physionet_session() -> requests.Session:
    """
    Crea una sesión requests autenticada mediante la cookie sessionid
    del navegador. Username/password se cargan y validan usando
    get_credentials_from_env(), pero la descarga protegida se autoriza
    mediante la cookie de sesión.
    """
    username, _, session_id = load_physionet_credentials()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
    )

    session = requests.Session()

    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Referer": (f"{BASE_URL}/content/{DATASET_NAME}/{DATASET_VERSION}/"),
        }
    )

    session.cookies.set(
        "sessionid",
        session_id,
        domain="physionet.org",
        path="/",
    )

    return session


def get_worker_session() -> requests.Session:
    """
    requests.Session no debe compartirse entre threads.

    Cada worker obtiene una sesión independiente, pero todas comparten
    la misma PHYSIONET_SESSIONID cargada desde el entorno.
    """
    if not hasattr(thread_local, "session"):
        thread_local.session = create_physionet_session()

    return thread_local.session


# ---------------------------------------------------------------------------
# Validación previa
# ---------------------------------------------------------------------------


def assert_vindr_access() -> None:
    """
    Comprueba que la cookie actual tiene acceso efectivo al dataset
    antes de lanzar muchas solicitudes DICOM.
    """
    test_url = f"{FILES_BASE_URL}/breast-level_annotations.csv"

    session = create_physionet_session()

    try:
        with session.get(
            test_url,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
        ) as response:
            print(f"Validando acceso con URL: {test_url}")
            print(f"Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type')}")

            if response.status_code == 401:
                raise PermissionError(
                    "401 Unauthorized. La cookie no es válida o ha expirado."
                )

            if response.status_code == 403:
                raise PermissionError(
                    "403 Forbidden. La cookie sessionid no tiene acceso "
                    "efectivo a VinDr-Mammo o ha expirado. "
                    "Abre PhysioNet en el navegador, verifica que puedes "
                    "descargar un archivo y copia una sessionid nueva."
                )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            if "text/html" in content_type:
                raise RuntimeError(
                    "PhysioNet devolvió HTML en vez del CSV durante "
                    "la validación. Probablemente la sesión no es válida."
                )

        print("Acceso a VinDr-Mammo validado correctamente.\n")

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Manifiesto de imágenes
# ---------------------------------------------------------------------------


def build_manifest() -> pl.DataFrame:
    """
    Selecciona los primeros N_STUDIES estudios del split indicado
    y devuelve las combinaciones únicas (study_id, image_id).
    """
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"No existe el CSV de anotaciones:\n{ANNOTATIONS_PATH}\n\n"
            "Descarga primero breast-level_annotations.csv."
        )

    annotations = pl.read_csv(ANNOTATIONS_PATH)

    required_columns = {"study_id", "image_id", "split"}
    missing_columns = required_columns - set(annotations.columns)

    if missing_columns:
        raise ValueError(
            "Faltan columnas necesarias en breast-level_annotations.csv: "
            f"{sorted(missing_columns)}"
        )

    selected_studies = (
        annotations.filter(pl.col("split") == SPLIT)
        .select("study_id")
        .unique()
        .sort("study_id")
        .head(N_STUDIES)
    )

    if selected_studies.is_empty():
        available_splits = annotations.get_column("split").unique().to_list()

        raise ValueError(
            f"No hay estudios con split={SPLIT!r}. "
            f"Splits disponibles: {available_splits}"
        )

    manifest = (
        annotations.join(selected_studies, on="study_id", how="inner")
        .select("study_id", "image_id")
        .unique()
        .sort(["study_id", "image_id"])
    )

    return manifest


def save_manifest(manifest: pl.DataFrame) -> Path:
    manifest_path = DATA_DIR / (f"manifest_{SPLIT}_{N_STUDIES}_studies.csv")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(manifest_path)

    return manifest_path


# ---------------------------------------------------------------------------
# Descarga de DICOM
# ---------------------------------------------------------------------------


def download_dicom(
    study_id: str,
    image_id: str,
) -> tuple[str, str, str, int]:
    """
    Descarga un DICOM y lo escribe de forma atómica:

    1. Descarga a un archivo .part.
    2. Sólo cuando termina, mueve .part al .dicom final.
    3. Si el .dicom final ya existe y no está vacío, lo omite.
    """
    destination = DATA_DIR / "images" / study_id / f"{image_id}.dicom"

    temporary_path = destination.with_suffix(".dicom.part")

    if destination.exists() and destination.stat().st_size > 0:
        return study_id, image_id, "already_exists", destination.stat().st_size

    dicom_url = f"{FILES_BASE_URL}/images/{study_id}/{image_id}.dicom"

    destination.parent.mkdir(parents=True, exist_ok=True)

    session = get_worker_session()

    try:
        with session.get(
            dicom_url,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
        ) as response:
            if response.status_code == 401:
                raise PermissionError(
                    "401 Unauthorized. La cookie PhysioNet ha expirado."
                )

            if response.status_code == 403:
                raise PermissionError(
                    "403 Forbidden. La cookie PhysioNet no permite "
                    "descargar este DICOM o ha expirado."
                )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            if "text/html" in content_type:
                preview = response.text[:300]

                raise RuntimeError(
                    f"Se recibió HTML en lugar de un DICOM. Respuesta: {preview}"
                )

            with temporary_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        output_file.write(chunk)

        downloaded_size = temporary_path.stat().st_size

        if downloaded_size == 0:
            raise RuntimeError("El archivo DICOM descargado está vacío.")

        temporary_path.replace(destination)

        return study_id, image_id, "downloaded", downloaded_size

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Ejecución del batch
# ---------------------------------------------------------------------------


def main() -> None:
    username, _, _ = load_physionet_credentials()

    print(f"Usuario cargado con PhysioNet loader: {username!r}")
    print(f"Dataset: {DATASET_NAME} v{DATASET_VERSION}")
    print(f"Split: {SPLIT}")
    print(f"Estudios solicitados: {N_STUDIES}")
    print(f"Workers: {MAX_WORKERS}")
    print()

    assert_vindr_access()

    manifest = build_manifest()
    manifest_path = save_manifest(manifest)

    rows = list(manifest.iter_rows(named=True))

    print(f"Manifiesto guardado en: {manifest_path}")
    print(f"Imágenes únicas a descargar: {len(rows)}")
    print()

    downloaded = 0
    already_exists = 0
    failed_downloads: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {
            executor.submit(
                download_dicom,
                row["study_id"],
                row["image_id"],
            ): row
            for row in rows
        }

        for index, future in enumerate(
            as_completed(future_to_row),
            start=1,
        ):
            row = future_to_row[future]

            try:
                study_id, image_id, status, size = future.result()

                if status == "downloaded":
                    downloaded += 1

                    print(
                        f"[{index}/{len(rows)}] DESCARGADA "
                        f"{study_id}/{image_id}.dicom "
                        f"({size:,} bytes)"
                    )

                elif status == "already_exists":
                    already_exists += 1

                    print(
                        f"[{index}/{len(rows)}] YA EXISTÍA "
                        f"{study_id}/{image_id}.dicom "
                        f"({size:,} bytes)"
                    )

            except Exception as error:
                failed_downloads.append(
                    {
                        "study_id": row["study_id"],
                        "image_id": row["image_id"],
                        "error": str(error),
                    }
                )

                print(
                    f"[{index}/{len(rows)}] ERROR "
                    f"{row['study_id']}/{row['image_id']}: {error}"
                )

    print("\nResumen de descarga")
    print(f"Descargadas ahora: {downloaded}")
    print(f"Ya existentes: {already_exists}")
    print(f"Fallidas: {len(failed_downloads)}")

    if failed_downloads:
        failed_path = DATA_DIR / "failed_downloads.csv"

        pl.DataFrame(failed_downloads).write_csv(failed_path)

        print(f"Fallos guardados en: {failed_path}")

        raise RuntimeError(
            f"El batch terminó con {len(failed_downloads)} descargas fallidas."
        )


if __name__ == "__main__":
    main()
