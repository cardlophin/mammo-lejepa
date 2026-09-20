# Migración: retirada de `001-vindr-streaming-etl`

**Feature**: 002-mammobench-corpus | **Fecha**: 2026-09-15

La feature 001 quedó implementada por completo (58 de 59 tareas). El origen de datos
cambia, así que su código de red y de DICOM deja de tener función. El principio de
gobernanza de la constitución obliga a eliminarlo en la misma entrega que introduce
su sustituto: el historial de git conserva lo borrado, el árbol de trabajo no.

## Qué se elimina

| Ruta | Motivo |
|---|---|
| `src/mammo_lejepa/download.py` | Descarga autenticada de PhysioNet. El corpus es local. |
| `src/mammo_lejepa/dicom_io.py` | No hay DICOM: Mammo-Bench se distribuye en JPEG. |
| `src/mammo_lejepa/windowing.py` | Ventana DICOM y MONOCHROME1 sin equivalente en JPEG de 8 bits. |
| `src/mammo_lejepa/findings.py` | Remapeo de `finding_annotations.csv`, específico de VinDr. |
| `src/mammo_lejepa/pipeline.py` | Orquestación productor/consumidor para red; se sustituye por un pool de procesos sobre ficheros locales. |
| `tests/fixtures/synthetic_dicom.py` | Fábrica de DICOM sintéticos. |
| `tests/unit/test_windowing.py`, `tests/unit/test_findings.py`, `tests/unit/test_redaction.py` | Prueban módulos retirados; ya no hay credenciales que sanear. |
| `tests/integration/test_download_local_server.py`, `test_pipeline_end_to_end.py`, `test_pipeline_resume.py`, `test_backpressure.py`, `test_failure_isolation.py`, `test_retry_failed.py` | Prueban la descarga y la contrapresión de red. |
| `design/a.py`, `design/b.py`, `design/c.py` | Prototipos de descarga y de Otsu; su contenido vive ya en `segmentation.py`. |
| `data/vindr-mammo/`, `data/mammobench.zip` | Datos del origen anterior y el comprimido ya extraído (8,3 GB recuperables). |
| `.env` con credenciales de PhysioNet | Sin uso. |

Dependencias a retirar de `pyproject.toml`: `aiohttp`, `aiofiles`, `physionet`,
`pydicom`, `pylibjpeg`, `pylibjpeg-openjpeg`, `requests`, `beautifulsoup4`, `dotenv`
y `pytest-asyncio`.

## Qué se conserva íntegro

| Ruta | Función en la feature 002 |
|---|---|
| `src/mammo_lejepa/segmentation.py` | Detección por Otsu, ahora como camino de respaldo y como referencia para medir la discrepancia con la máscara. |
| `src/mammo_lejepa/geometry.py` | Recorte y traslación de coordenadas; el contrato no cambia. |
| `src/mammo_lejepa/storage.py` | Escritura atómica de PNG, JSONL con vaciado, bloqueo del directorio de salida. |
| `src/mammo_lejepa/resume.py` | Lógica pura de trabajo pendiente y resumen de fallos. |
| `tests/unit/test_geometry.py`, `test_segmentation.py`, `test_resume.py` | Siguen siendo válidos tal cual. |
| `tests/unit/test_purity.py` | Se actualiza la lista de módulos puros; el test en sí se mantiene. |

## Qué se adapta

| Ruta | Cambio |
|---|---|
| `src/mammo_lejepa/models.py` | Se retiran los campos propios de DICOM (`photometric_interpretation`, `transfer_syntax_uid`, ventana, `normalize_*`) y se añaden `source_dataset`, `box_source`, `mask_otsu_iou`, `split`. |
| `src/mammo_lejepa/config.py` | Desaparecen credenciales, URLs y parámetros de descarga; entran rutas de Mammo-Bench, umbral de máscara, margen, procesos y semilla. |
| `src/mammo_lejepa/manifest.py` | Se reescribe sobre `mammo-bench.csv` en lugar de `breast-level_annotations.csv`. |
| `src/mammo_lejepa/catalog.py` | Mismo patrón de consolidación, esquema nuevo y sin catálogo de hallazgos. |
| `src/mammo_lejepa/worker.py` | Entrada JPEG en vez de DICOM; máscara primero, Otsu de respaldo. |
| `src/mammo_lejepa/cli.py` | Subcomandos nuevos; desaparecen `doctor`, `retry` y la validación de cookie. |
| `README.md` | Reescritura completa: el proyecto ya no trata sobre PhysioNet. |
| `.gitignore` | Añadir `runs/`, `checkpoints/`, `*.ckpt`. |

## Qué se archiva

`specs/001-vindr-streaming-etl/` **no se borra**. Queda como registro de la decisión
y de su alcance, con la cabecera `Status: Superseded by 002-mammobench-corpus`. Es
documentación histórica, no código muerto.
