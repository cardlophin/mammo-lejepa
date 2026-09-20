# Implementation Plan: VinDr-Mammo como conjunto de evaluación downstream

**Branch**: `004-vindr-downstream` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-vindr-downstream/spec.md`

## Summary

Reconstruir el ETL de VinDr-Mammo —descarga desde PhysioNet, recorte de la región
mamaria y catalogación con hallazgos remapeados— como un subpaquete aislado
(`src/mammo_lejepa/vindr/`), a partir del código de la extinta feature 001 que sigue
íntegro en el historial de git (commit `0de1617`), adaptado al modelo de datos
actual del repositorio sin tocarlo. El resultado deja de ser el corpus de
preentrenamiento (rol que hoy tiene Mammo-Bench, feature 002) y pasa a ser el único
conjunto del proyecto con cajas de hallazgo, para una evaluación de localización que
se especificará aparte cuando la feature 003 produzca un checkpoint.

## Technical Context

**Language/Version**: Python 3.12, `uv`

**Primary Dependencies**: `aiohttp`+`aiofiles` (descarga concurrente), `pydicom` +
`pylibjpeg`/`pylibjpeg-openjpeg` (decodificación DICOM, incluida JPEG 2000
Lossless), `python-dotenv` (carga de `.env`), `physionet` (validación de
credenciales), `polars` (manifiesto y catálogo), `opencv-python` (recorte,
segmentación), `typer`+`rich` (CLI). Desarrollo: `pytest`+`pytest-asyncio`.

**Storage**: `data/vindr-mammo/` — DICOM efímero (se borra tras procesarse con
éxito), PNG recortados a resolución nativa, catálogo en Parquet/JSONL. Fuera del
control de versiones.

**Testing**: `pytest`, con `pytest-asyncio` para el productor/consumidor de
descarga; servidor HTTP local sintético para las pruebas de descarga (sin red real
ni credenciales en la suite); DICOM sintéticos de geometría conocida para el resto.

**Target Platform**: macOS/Linux, un único proceso de usuario (sin GPU, sin MPS/CUDA:
esta feature es E/S y CPU, no entrenamiento).

**Project Type**: biblioteca con CLI (subpaquete adicional dentro del paquete
`mammo_lejepa` existente).

**Performance Goals**: sin objetivo de throughput propio más allá de no
desperdiciar el techo de conexiones/procesos configurado; el cuello de botella real
es la red de PhysioNet, no el procesado local.

**Constraints**: nunca más de `downloads + queue_size + workers` DICOM en disco a
la vez (techo de disco intermedio calculado, no proporcional al tamaño del
dataset); nunca revelar credenciales en salida, registro o artefacto generado;
recorte siempre a resolución nativa, nunca reescalado al construir el corpus
(spec.md, Clarifications 2026-09-16; constitución, "Restricciones del dominio":
"el redimensionado... ocurre en tiempo de entrenamiento, nunca escrito sobre el
corpus").

**Scale/Scope**: 20.000 imágenes, 5.000 estudios (4 imágenes/estudio), 2.254
hallazgos con caja, ~300 GB de origen en PhysioNet, split oficial 16.000/4.000
imágenes (training/test).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principio | Cumplimiento |
|---|---|
| I. Motor puro / E-S | La geometría (`geometry.py`), la segmentación (`segmentation.py`) y la construcción del manifiesto (`vindr/manifest.py`) son puras y se reutilizan o se adaptan sin tocar su naturaleza pura. La red (`vindr/download.py`), el disco (`vindr/dicom_io.py`, `storage.py`), y la CLI (`vindr/cli.py`) son los únicos módulos de frontera. |
| II. Reproducibilidad | Un único objeto de configuración (`vindr/config.py:PipelineConfig`) con valores por defecto declarados; `runs.jsonl` registra configuración efectiva, versión y semilla implícita (la selección determinista de estudios) de cada ejecución. |
| III. Tests | Manifiesto, remapeado de coordenadas y ventana/normalización se prueban sin red ni GPU, con DICOM sintéticos de geometría conocida; la descarga se prueba contra un servidor HTTP local, nunca contra PhysioNet real. |
| IV. Cómputo acotado y reanudable | Techo de disco intermedio calculado a partir de la concurrencia configurada, no del tamaño del dataset (heredado de la feature 001, D-03 de su plan); reanudación verificada por test tras interrupción. |
| V. Trazabilidad | Toda coordenada de hallazgo se guarda en ambos sistemas de referencia (original y recorte) con la transformación que las vincula; el recorte guarda su margen, umbral y dimensiones nativas. |
| VI. Evaluación honesta | Partición reutilizada del split oficial (FR-020), verificada a nivel de `study_id` nunca de imagen (FR-021); heterogeneidad de resolución nativa entre fabricantes declarada como riesgo para la evaluación futura (spec.md, Edge Cases). |

**Resultado**: sin violaciones. La única decisión que se aparta de lo obvio —no
reescalar el recorte al construir el corpus, aunque el encoder que se evaluará
entrene a 128×128— está mandatada por la propia constitución (Restricciones del
dominio, fidelidad radiológica) y documentada en el research.md de esta feature.

## Project Structure

### Documentation (this feature)

```text
specs/004-vindr-downstream/
├── plan.md              # This file
├── research.md          # Phase 0 output: D-01 a D-06
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── module-contracts.md
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
src/mammo_lejepa/
├── __init__.py           # CORPUS_VERSION (002) + PIPELINE_VERSION (esta feature)
├── errors.py             # sin cambios: DecodeError/SegmentationError/WriteError/classify_exception de Mammo-Bench
├── geometry.py            # sin cambios: crop_image/to_crop_space/to_original_space/clip_to_crop sobre CajaMamaria
├── segmentation.py        # sin cambios: detect_breast_box (Otsu) ya devuelve CajaMamaria
├── storage.py              # sin cambios: write_png_atomic/append_record/acquire_output_lock/clean_orphan_part_files
├── resume.py               # sin cambios: pending_tasks/latest_records_by_image/summarize_failures (duck-typing sobre .image_id/.status/.processed_at)
│                           # (parse_jsonl_records de este módulo NO se reutiliza: reconstruye
│                           #  RegistroDeRecorte hardcodeado, no es genérico — research.md D-08)
├── models.py, config.py, manifest.py, catalog.py, worker.py, cli.py, quality.py,
│   boxing.py, splits.py, image_io.py, runner.py    # Mammo-Bench (002), sin tocar
│
└── vindr/                 # esta feature — mismo patrón que el subpaquete ssl/ (003)
    ├── __init__.py
    ├── config.py           # PhysioNetCredentials, load_credentials, PipelineConfig
    ├── models.py           # FailureCategory (propio, con AUTH/NETWORK), ImageTask, ProcessOutcome,
    │                       # ImageRecord, DicomPayload, DownloadResult, RunSummary — usa CajaMamaria
    │                       # y BoundingBox del models.py compartido para la caja y los hallazgos
    ├── errors.py           # AuthError, NetworkError, redact_secrets, classify_exception propio;
    │                       # reexporta DecodeError/SegmentationError/WriteError del errors.py compartido
    ├── storage.py          # quarantine() — lo único que storage.py compartido no tiene
    ├── windowing.py        # PURO: apply_display_window, normalize_to_uint8 (sin dependencias del paquete)
    ├── dicom_io.py         # read_dicom (usa DecodeError compartido)
    ├── manifest.py         # PURO: build_manifest, batch_by_study, count_metadata_discrepancies
    ├── download.py         # build_client_session, validate_access, download_dicom
    ├── worker.py           # process_dicom (usa geometry/segmentation compartidos)
    ├── pipeline.py         # run_pipeline: orquestación asyncio + ProcessPoolExecutor
    ├── findings.py         # PURO: parse_finding_categories, remap_findings (usa geometry compartido)
    ├── catalog.py          # consolidate, validate_catalog_coherence
    └── cli.py              # mammo-etl: doctor, run, resume, retry, consolidate, inspect

tests/vindr/                # mismo patrón que tests/ssl/ (003): un directorio plano, sin unit/integration
├── conftest.py
├── fixtures/
│   └── synthetic_dicom.py
├── test_windowing.py
├── test_manifest.py
├── test_findings.py
├── test_dicom_io.py
├── test_errors.py
├── test_download_local_server.py    # servidor HTTP local, marcado @pytest.mark.slow
├── test_pipeline_end_to_end.py      # marcado @pytest.mark.slow
├── test_pipeline_resume.py          # marcado @pytest.mark.slow
└── test_catalog.py
```

**Structure Decision**: subpaquete `vindr/` dentro del paquete existente, con su
propio `models.py`/`config.py`/`errors.py` (no se pueden compartir literalmente con
Mammo-Bench: sus tipos son distintos por naturaleza, no por descuido — VinDr habla
de descargas HTTP con reintento y autenticación, Mammo-Bench no tiene red en
absoluto). Se reutilizan sin adaptar los módulos genuinamente genéricos
(`geometry.py`, `segmentation.py`, `storage.py`, `resume.py`, `errors.py` en su
parte no ligada a red/autenticación) porque VinDr adopta `CajaMamaria` como su
propio tipo de caja mamaria en vez de reinventar el `BreastCrop` de la extinta
feature 001 — ver D-01 en research.md. `tests/vindr/` es un directorio plano (no
`tests/unit/vindr/` + `tests/integration/vindr/`), igual que `tests/ssl/`
planeado para la feature 003: es la convención ya establecida para un subpaquete
aislado en este repositorio.

## Complexity Tracking

Sin violaciones de la constitución. No procede.
