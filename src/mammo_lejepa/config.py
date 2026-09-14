from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PhysioNetCredentials:
    username: str
    password: str
    session_id: str


def load_credentials() -> PhysioNetCredentials:
    """Carga las credenciales de PhysioNet desde `.env` y el entorno. Los
    mensajes de error nombran la variable ausente pero nunca revelan valores
    (FR-031)."""
    from dotenv import load_dotenv
    from physionet.api.utils import get_credentials_from_env

    load_dotenv()

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
            "PHYSIONET_SESSIONID no está disponible. Copia la cookie sessionid "
            "de PhysioNet desde tu navegador y guárdala en .env."
        )

    return PhysioNetCredentials(
        username=username, password=password, session_id=session_id
    )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuración efectiva de una ejecución. Se construye a partir de la CLI
    y del entorno; ninguna constante de comportamiento se edita en el código
    fuente (constitución, principio II)."""

    data_dir: Path = field(default_factory=lambda: Path("data/vindr-mammo"))

    # Concurrencia (D-03)
    downloads: int = 6
    workers: int = 4
    queue_size: int = 12

    # Segmentación y normalización
    margin_px: int = 25
    low_percentile: float = 0.5
    high_percentile: float = 99.5
    blur_kernel: int = 5
    close_kernel_ratio: float = 0.006

    # PhysioNet
    base_url: str = "https://physionet.org"
    dataset_name: str = "vindr-mammo"
    dataset_version: str = "1.0.0"
    connect_timeout_seconds: float = 20.0
    read_timeout_seconds: float = 300.0
    total_timeout_seconds: float = 600.0
    max_retries: int = 5
    backoff_factor: float = 1.5

    # D-03: máximo DICOM observado en el dataset (~35 MB)
    max_dicom_bytes_estimate: int = 35 * 1024 * 1024

    @property
    def csv_dir(self) -> Path:
        return self.data_dir / "csv"

    @property
    def breast_level_annotations_path(self) -> Path:
        return self.csv_dir / "breast-level_annotations.csv"

    @property
    def finding_annotations_path(self) -> Path:
        return self.csv_dir / "finding_annotations.csv"

    @property
    def metadata_path(self) -> Path:
        return self.csv_dir / "metadata.csv"

    @property
    def dicom_dir(self) -> Path:
        return self.data_dir / "images" / "dicom"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "images" / "quarantine"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "images" / "processed"

    @property
    def catalog_dir(self) -> Path:
        return self.data_dir / "catalog"

    @property
    def images_jsonl_path(self) -> Path:
        return self.catalog_dir / "images.jsonl"

    @property
    def images_parquet_path(self) -> Path:
        return self.catalog_dir / "images.parquet"

    @property
    def findings_parquet_path(self) -> Path:
        return self.catalog_dir / "findings.parquet"

    @property
    def runs_jsonl_path(self) -> Path:
        return self.catalog_dir / "runs.jsonl"

    @property
    def files_base_url(self) -> str:
        return f"{self.base_url}/files/{self.dataset_name}/{self.dataset_version}"

    @property
    def content_url(self) -> str:
        return f"{self.base_url}/content/{self.dataset_name}/{self.dataset_version}/"

    def dicom_url(self, study_id: str, image_id: str) -> str:
        return f"{self.files_base_url}/images/{study_id}/{image_id}.dicom"

    @property
    def disk_ceiling_bytes(self) -> int:
        """D-03: el disco intermedio máximo es una función conocida de los
        parámetros de concurrencia, no del tamaño del dataset (constitución,
        principio IV)."""
        in_flight = self.downloads + self.queue_size + self.workers
        return in_flight * self.max_dicom_bytes_estimate
