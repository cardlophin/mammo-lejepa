from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from mammo_lejepa.errors import WriteError
from mammo_lejepa.models import ImageRecord, RunSummary

_LOCK_FILE_NAME = ".run.lock"


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Objeto no serializable en JSONL: {type(value)!r}")


def write_png_atomic(image: np.ndarray, destination: Path) -> int:
    """Codifica `image` como PNG y lo escribe en un fichero temporal,
    sincroniza y renombra al nombre definitivo. Un fichero con nombre
    definitivo es siempre un fichero completo (FR-007 aplicado al PNG). Se
    codifica en memoria con `cv2.imencode` (en vez de `cv2.imwrite` sobre el
    `.part`) porque OpenCV elige el códec por la extensión del fichero, y
    `<nombre>.png.part` no es una extensión reconocida."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".part")

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise WriteError(f"cv2.imencode no pudo codificar la imagen para {destination}")

    try:
        with open(temp_path, "wb") as handle:
            handle.write(encoded.tobytes())
            handle.flush()
            os.fsync(handle.fileno())

        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return destination.stat().st_size


def _append_json_line(payload: object, jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(payload), ensure_ascii=False, default=_json_default)  # pyright: ignore[reportArgumentType]

    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_record(record: ImageRecord, jsonl_path: Path) -> None:
    """Escribe una única línea JSONL con `record` y vacía el búfer del sistema
    operativo antes de retornar: cuando la llamada retorna, el registro es
    duradero (FR-020)."""
    _append_json_line(record, jsonl_path)


def append_run_summary(summary: RunSummary, jsonl_path: Path) -> None:
    """Escribe una línea de `runs.jsonl` (D-07). Se llama dos veces por
    ejecución: al inicio (con los recuentos aún a cero) y al final, de modo
    que una línea de inicio sin su línea de fin delata una ejecución que no
    terminó de forma ordenada."""
    _append_json_line(summary, jsonl_path)


@contextmanager
def acquire_output_lock(directory: Path) -> Iterator[None]:
    """Impide dos ejecuciones simultáneas sobre `directory` (FR-028). Falla de
    inmediato si ya hay otra ejecución activa; el bloqueo se libera
    automáticamente si el proceso termina o se interrumpe."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / _LOCK_FILE_NAME
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(
            f"Ya hay otra ejecución activa sobre {directory}. Si estás seguro de "
            f"que no es así, borra {lock_path} manualmente."
        ) from None

    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())

    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def quarantine(dicom_path: Path, quarantine_dir: Path) -> Path:
    """Mueve un DICOM cuyo procesado falló a `quarantine_dir/<study_id>/`,
    preservando el nombre de su directorio de estudio (FR-018)."""
    study_dir = quarantine_dir / dicom_path.parent.name
    study_dir.mkdir(parents=True, exist_ok=True)
    destination = study_dir / dicom_path.name
    shutil.move(str(dicom_path), str(destination))
    return destination


def clean_orphan_part_files(directory: Path) -> int:
    """Borra ficheros `.part` huérfanos de una interrupción anterior. Se ejecuta
    al arrancar, antes de aceptar el bloqueo de salida."""
    if not directory.exists():
        return 0

    removed = 0
    for part_file in directory.rglob("*.part"):
        part_file.unlink(missing_ok=True)
        removed += 1

    return removed
