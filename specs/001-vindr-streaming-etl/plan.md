# Implementation Plan: ETL en streaming de VinDr-Mammo a recortes mamarios catalogados

**Branch**: `001-vindr-streaming-etl` | **Date**: 2026-09-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-vindr-streaming-etl/spec.md`

## Summary

Convertir los ~300 GB de DICOM de VinDr-Mammo en un corpus local de recortes mamarios
en PNG sin pérdida, catalogados y con las anotaciones de hallazgos trasladadas al
sistema de coordenadas del recorte, sin materializar nunca más de unos cientos de
megabytes de DICOM simultáneamente.

El enfoque es un pipeline productor/consumidor: un bucle `asyncio` descarga en
paralelo desde PhysioNet y alimenta una cola acotada; un `ProcessPoolExecutor`
consume de ella, decodifica el DICOM, detecta la caja del campo mamario con Otsu y
mayor componente conexo, recorta a resolución nativa y escribe el PNG; el proceso
principal escribe el registro JSONL y borra el DICOM. La contrapresión de la cola es
lo que impone el techo de disco; el registro incremental es lo que permite reanudar.

El repositorio ya contiene los tres prototipos que validan las piezas por separado:
`design/a.py` (descarga autenticada y manifiesto), `design/b.py` (descarga asíncrona
con `aiohttp`/`aiofiles`) y `design/c.py` (detección de la caja mamaria). Este plan
los industrializa: los convierte en módulos con lógica pura separada de la E/S,
añade estado, reanudación, catálogo y CLI, y los somete a tests.

## Technical Context

**Language/Version**: Python 3.12 (`.python-version`), gestión con `uv`

**Primary Dependencies**: `aiohttp` + `aiofiles` (descarga), `pydicom` +
`pylibjpeg`/`pylibjpeg-openjpeg` (decodificación; ver D-01), `opencv-python`
(Otsu, morfología, componentes conexos, escritura PNG), `numpy`, `polars` (CSV,
Parquet, consolidación), `python-dotenv`, `typer` o `argparse` (CLI), `rich`
(progreso), `pytest` (tests)

**Storage**: sistema de ficheros local. Parquet para manifiestos y catálogos
consolidados, JSONL *append-only* para el estado incremental, PNG para los derivados

**Testing**: `pytest` con DICOM sintéticos y servidor `aiohttp` local; sin red ni
credenciales

**Target Platform**: macOS y Linux, ejecución local en un portátil

**Project Type**: aplicación de línea de comandos con núcleo de biblioteca

**Performance Goals**: ≥ 80 % del ancho de banda del benchmark de `design/b.py`
sostenido durante la ejecución completa; procesado de imagen que no se convierta en
el cuello de botella con 4 workers

**Constraints**: pico de disco intermedio < 1 GB con los parámetros por defecto;
memoria residente acotada (nunca más de `workers + cola` imágenes descomprimidas en
memoria); cortesía con PhysioNet (concurrencia ≤ 8, retroceso exponencial)

**Scale/Scope**: 20.000 imágenes, 5.000 estudios, ~300 GB de origen, ~20-40 GB de
corpus de salida estimado

## Constitution Check

*GATE: debe pasarse antes de la fase 0 y volver a comprobarse tras la fase 1.*

| Principio | Cumplimiento en este plan |
|---|---|
| I. Motor puro separado de la E/S | `segmentation.py`, `geometry.py`, `windowing.py`, `manifest.py` y `resume.py` son funciones puras sobre `np.ndarray` y dataclasses. La red vive en `download.py`, el disco en `storage.py`, la orquestación en `pipeline.py`. Ningún módulo puro importa `aiohttp`, `pathlib` ni `os`. |
| II. CLI y reproducibilidad | Subcomandos `run`, `resume`, `retry`, `consolidate`, `inspect`, `doctor`. Toda la configuración en una dataclass `PipelineConfig` construida a partir de CLI + entorno; ninguna constante de comportamiento se edita en el código. Cada ejecución escribe su `run_id`, sus parámetros efectivos y su versión en `runs.jsonl`. |
| III. Tests con pytest | Unitarios exhaustivos en geometría, segmentación, manifiesto y reanudación; integración con servidor local y DICOM sintéticos. Ningún test toca PhysioNet. |
| IV. Streaming acotado | Cola `asyncio.Queue(maxsize=…)` con contrapresión real, borrado inmediato tras confirmación, techo de disco calculado y mostrado al arrancar (D-03). |
| V. Trazabilidad y reversibilidad | `BreastCrop` completo en cada fila del catálogo, coordenadas con sufijo de espacio explícito, funciones de traslación directa e inversa con test de ida y vuelta, `pipeline_version` en cada registro. |

**Resultado**: sin violaciones. La sección *Complexity Tracking* queda vacía.

## Project Structure

### Documentation (this feature)

```text
specs/001-vindr-streaming-etl/
├── spec.md
├── plan.md              # este fichero
├── research.md          # decisiones D-01…D-10
├── data-model.md        # entidades y esquemas de los artefactos
├── quickstart.md        # puesta en marcha y verificación visual
├── contracts/
│   └── module-contracts.md   # firmas y contratos de los módulos puros
└── tasks.md
```

### Source Code (repository root)

```text
src/mammo_lejepa/
├── __init__.py
├── config.py            # PipelineConfig, carga de entorno, rutas, validación
├── models.py            # ImageTask, BreastCrop, BoundingBox, ProcessOutcome, FailureCategory
├── manifest.py          # PURO: construcción y filtrado del manifiesto desde los CSV
├── windowing.py         # PURO: ventana DICOM, MONOCHROME1, normalización a uint8
├── segmentation.py      # PURO: Otsu, morfología, mayor componente conexo, caja mamaria
├── geometry.py          # PURO: traslación de coordenadas orig↔crop, recorte, intersección
├── findings.py          # PURO: parseo de finding_annotations y remapeo de cajas
├── resume.py            # PURO: trabajo pendiente a partir de manifiesto + registros
├── dicom_io.py          # E/S: lectura pydicom, extracción de cabeceras de procedencia
├── download.py          # E/S: sesión aiohttp, cookie, descarga atómica, reintentos
├── storage.py           # E/S: escritura atómica de PNG, JSONL, Parquet, borrado, bloqueo
├── worker.py            # función de proceso: bytes DICOM → PNG en disco + BreastCrop
├── pipeline.py          # orquestación productor/consumidor, contrapresión, resumen
├── catalog.py           # consolidación JSONL → Parquet, unión con hallazgos
└── cli.py               # subcomandos y salida por consola

tests/
├── conftest.py              # fábricas de DICOM sintéticos y directorios temporales
├── unit/
│   ├── test_windowing.py
│   ├── test_segmentation.py
│   ├── test_geometry.py
│   ├── test_findings.py
│   ├── test_manifest.py
│   └── test_resume.py
├── integration/
│   ├── test_download_local_server.py
│   ├── test_pipeline_end_to_end.py
│   └── test_catalog_consolidation.py
└── fixtures/
    └── synthetic_dicom.py

design/                  # prototipos originales, se conservan como referencia
```

**Structure Decision**: proyecto único con paquete instalable bajo `src/`, coherente
con `pyproject.toml` y con `uv`. La frontera entre módulos puros y módulos de E/S es
visible en la propia lista de ficheros —los seis primeros no importan nada de red ni
de disco—, lo que convierte el principio I en algo comprobable con una regla de
linting y no sólo en una intención. `design/` se mantiene intacto: es el registro de
lo que ya se validó experimentalmente.

## Flujo de ejecución

1. **Arranque**: `cli.run` construye `PipelineConfig`, valida credenciales, adquiere
   el bloqueo del directorio de salida, genera `run_id` y muestra el techo de disco
   calculado.
2. **Validación de acceso**: una petición al CSV de anotaciones confirma que la
   cookie es válida. Un fallo aquí termina la ejecución con código distinto de cero.
3. **Manifiesto**: `manifest.build` lee los CSV con Polars, aplica los filtros de
   alcance y produce la lista de `ImageTask` agrupada en lotes por estudio. Se
   persiste en `catalog/manifest_<run_id>.parquet`.
4. **Reanudación**: `resume.pending_tasks` descuenta las imágenes ya resueltas con
   éxito cuyo PNG existe físicamente.
5. **Productores**: N corrutinas toman tareas de una cola de entrada, descargan el
   DICOM de forma atómica y publican la ruta en una `asyncio.Queue` acotada. La cola
   llena bloquea a los productores: ahí está la contrapresión.
6. **Consumidores**: el bucle principal toma de la cola y despacha al
   `ProcessPoolExecutor` mediante `run_in_executor`. El worker decodifica, detecta la
   caja, recorta, escribe el PNG de forma atómica y devuelve un `ProcessOutcome`.
7. **Confirmación**: el proceso principal escribe la línea JSONL, la vacía al disco y
   borra el DICOM; si el resultado es un fallo, mueve el DICOM a cuarentena.
8. **Cierre**: se drena la cola, se cierra el pool y la sesión, se escribe la línea de
   `runs.jsonl` y se imprime el resumen por categoría de fallo. `SIGINT` provoca un
   cierre ordenado por esta misma vía.
9. **Consolidación**: `cli.consolidate` compacta el JSONL a Parquet y genera el
   catálogo de hallazgos remapeados.

## Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| La cookie de PhysioNet caduca a mitad de una ejecución de horas | Alto | Detección temprana por tipo de contenido y estado HTTP, aborto ordenado, reanudación sin pérdida (US2) |
| Falta el decodificador JPEG 2000 | Alto: 0 % de éxito | Dependencia declarada y subcomando `doctor` que verifica el decodificador antes de descargar nada (D-01) |
| El remapeo de coordenadas tiene un desplazamiento sistemático | Alto y silencioso | Función pura, tests de ida y vuelta, verificación visual obligatoria en el criterio SC-006 |
| Cajas mamarias infladas por marcadores o artefactos | Medio | `crop_area_ratio` registrado en cada fila para auditar la distribución tras la ejecución (D-06) |
| El servidor limita la tasa de peticiones | Medio | Concurrencia conservadora, retroceso exponencial con jitter, respeto de `Retry-After` |
| Deriva entre las dimensiones del CSV y las reales del DICOM | Medio | Contraste obligatorio; la discrepancia se registra como fallo, no se corrige en silencio |

## Complexity Tracking

Sin violaciones de la constitución. No procede.
