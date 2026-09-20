---
description: "Tareas de implementación — ETL en streaming de VinDr-Mammo"
---

# Tasks: ETL en streaming de VinDr-Mammo a recortes mamarios catalogados

**Input**: documentos de diseño en `/specs/001-vindr-streaming-etl/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`,
`contracts/module-contracts.md`

**Tests**: incluidos y obligatorios en la lógica crítica, según el principio III de la
constitución.

## Formato: `[ID] [P?] [Story] Descripción`

- **[P]**: puede ejecutarse en paralelo (ficheros distintos, sin dependencias)
- **[Story]**: historia de usuario a la que pertenece
- Las rutas de fichero son exactas y relativas a la raíz del repositorio

---

## Phase 1: Setup (infraestructura compartida)

**Propósito**: dejar el esqueleto del paquete y las herramientas listas.

- [X] T001 Crear el árbol `src/mammo_lejepa/` con `__init__.py` que expone
  `PIPELINE_VERSION = "0.1.0"`, y el árbol `tests/unit`, `tests/integration`,
  `tests/fixtures` con sus `__init__.py`.
- [X] T002 Declarar el paquete en `pyproject.toml`: `[project.scripts] mammo-etl =
  "mammo_lejepa.cli:app"`, `[tool.hatch.build]` o equivalente apuntando a `src/`, y
  `requires-python = ">=3.12"`.
- [X] T003 [P] Añadir dependencias con `uv add`: `pylibjpeg`, `pylibjpeg-openjpeg`,
  `rich`, `typer`; y de desarrollo: `pytest`, `pytest-asyncio`, `ruff`.
- [X] T004 [P] Configurar `ruff` en `pyproject.toml`: longitud de línea 88, reglas
  `E,F,I,UP,B,SIM,RUF`, y `from __future__ import annotations` obligatorio.
- [X] T005 [P] Configurar `pytest` en `pyproject.toml`: `testpaths = ["tests"]`,
  `asyncio_mode = "auto"`, marcador `slow` para los tests de integración.

---

## Phase 2: Foundational (prerrequisitos bloqueantes)

**Propósito**: entidades, configuración y primitivas de E/S que toda historia
necesita.

**⚠️ CRÍTICO**: ninguna historia de usuario puede empezar hasta terminar esta fase.

- [X] T006 Definir en `src/mammo_lejepa/models.py` las dataclasses inmutables
  `ImageTask`, `BreastCrop`, `BoundingBox`, `ImageRecord`, `ProcessOutcome`,
  `DownloadResult`, `RunSummary` y la enumeración `FailureCategory`
  (`NETWORK`, `AUTH`, `DECODE`, `SEGMENTATION`, `WRITE`, `UNKNOWN`), según
  `data-model.md`. `BreastCrop` valida sus invariantes en `__post_init__`.
- [X] T007 [P] Definir en `src/mammo_lejepa/errors.py` la jerarquía de excepciones:
  `MammoETLError` y sus derivadas `AuthError`, `NetworkError`, `DecodeError`,
  `SegmentationError`, `WriteError`, más la función que mapea excepción →
  `FailureCategory`.
- [X] T008 Implementar `src/mammo_lejepa/config.py`: `PipelineConfig` con rutas
  derivadas de la raíz del proyecto, parámetros de concurrencia
  (`downloads=6`, `workers=4`, `queue_size=12`), `margin_px=25`, percentiles de
  normalización, URLs base de PhysioNet, y carga de credenciales desde el entorno
  con mensajes de error que no revelan sus valores. Incluye el cálculo del techo de
  disco de D-03 como propiedad.
- [X] T009 [P] Implementar en `src/mammo_lejepa/storage.py` la escritura atómica:
  `write_png_atomic`, `append_record` con vaciado de búfer, `acquire_output_lock`
  basado en un fichero de bloqueo con el PID, `quarantine` y la limpieza de ficheros
  `.part` huérfanos al arrancar.
- [X] T010 [P] Implementar `src/mammo_lejepa/dicom_io.py`: `read_dicom` que devuelve
  píxeles y cabeceras de procedencia, y que traduce el fallo por decodificador
  ausente a un `DecodeError` que nombra la sintaxis de transferencia y el paquete que
  falta.
- [X] T011 [P] Crear `tests/fixtures/synthetic_dicom.py`: fábrica de DICOM sintéticos
  con elipse clara sobre fondo oscuro, geometría conocida, variantes
  MONOCHROME1/MONOCHROME2, 12 y 16 bits, con y sin `WindowCenter`, y una variante con
  una etiqueta de orientación simulada en una esquina (componente conexo pequeño y
  separado).
- [X] T012 Escribir `tests/conftest.py` con los *fixtures* de directorio temporal de
  trabajo, `PipelineConfig` de prueba y los CSV de anotaciones reducidos en memoria.
- [X] T013 Implementar el subcomando `doctor` en `src/mammo_lejepa/cli.py`:
  comprueba CSV, decodificador JPEG 2000 (decodificando un DICOM sintético
  comprimido), permisos de escritura, espacio libre frente al techo de disco y
  validez de la cookie. **Es la primera cosa ejecutable del proyecto** y bloquea el
  riesgo D-01.
- [X] T014 Registrar el esqueleto de la CLI en `src/mammo_lejepa/cli.py` con los
  subcomandos `run`, `resume`, `retry`, `consolidate`, `inspect`, `doctor` y sus
  parámetros, todos delegando aún en funciones no implementadas.
- [X] T015 Añadir a `.gitignore` el directorio `data/` completo (constitución,
  "Restricciones del dominio": ni las imágenes ni los CSV de anotaciones se
  redistribuyen — no sólo `images/` y `catalog/`) y `*.part`.

**Checkpoint**: `uv run mammo-etl doctor` responde y diagnostica el entorno real.

---

## Phase 3: User Story 1 — Recortes mamarios verificables de una muestra (P1) 🎯 MVP

**Goal**: descargar N estudios, detectar la caja mamaria, recortar, escribir el PNG y
registrar la fila del catálogo.

**Independent Test**: `mammo-etl run --n-studies 5 --split training` produce los PNG y
las líneas del JSONL, y `images/dicom/` queda vacío.

### Tests de la historia 1 (escribir primero, deben fallar)

- [X] T016 [P] [US1] `tests/unit/test_windowing.py`: MONOCHROME1 queda invertido
  respecto a MONOCHROME2 para el mismo contenido; la ventana ausente es la identidad;
  `normalize_to_uint8` devuelve rango `[0, 255]` y los cortes empleados; percentiles
  degenerados no dividen por cero.
- [X] T017 [P] [US1] `tests/unit/test_segmentation.py`: sobre la elipse sintética de
  geometría conocida, la caja detectada contiene la elipse y no supera sus límites más
  el margen (tolerancia de ±2 px); la etiqueta de orientación aislada queda fuera de
  la caja; una imagen uniforme lanza `SegmentationError`; el margen se recorta a los
  límites de la imagen cuando la mama toca el borde.
- [X] T018 [P] [US1] `tests/unit/test_geometry.py`: `to_crop_space` ∘
  `to_original_space` es la identidad para puntos contenidos; una caja en el espacio
  equivocado lanza `ValueError`; `clip_to_crop` marca correctamente las cajas
  parciales y devuelve caja degenerada explícita para la intersección vacía;
  `crop_image` devuelve exactamente `(crop_height, crop_width)`.
- [X] T019 [P] [US1] `tests/unit/test_manifest.py`: pares únicos y orden
  determinista; `n_studies` selecciona los primeros del orden; split inexistente y
  columnas ausentes lanzan `ValueError` con el nombre de la columna; los lotes por
  estudio no asumen cuatro imágenes; `image_id` de `breast-level_annotations.csv`
  presentes/ausentes frente al `SOP Instance UID` (desduplicado) de `metadata.csv`
  se cuentan como discrepancias sin alterar el manifiesto ni lanzar excepción
  (FR-032).
- [X] T020 [P] [US1] `tests/integration/test_download_local_server.py`: contra un
  servidor `aiohttp` local, la descarga es atómica (no queda `.part` tras un fallo
  inyectado), un 503 se reintenta y acaba en éxito, un 403 eleva `AuthError` y una
  respuesta `text/html` también.

### Implementación de la historia 1

- [X] T021 [P] [US1] Implementar `src/mammo_lejepa/windowing.py` según el contrato,
  portando y limpiando `normalize_uint8` y `load_dicom_image` de `design/c.py`.
- [X] T022 [P] [US1] Implementar `src/mammo_lejepa/segmentation.py`: `detect_breast_box`
  portando `get_breast_bounding_box` y `largest_connected_component` de `design/c.py`,
  devolviendo un `BreastCrop` completo con `otsu_threshold` y `area_ratio`.
- [X] T023 [P] [US1] Implementar `src/mammo_lejepa/geometry.py` con las cuatro
  funciones del contrato.
- [X] T024 [P] [US1] Implementar `src/mammo_lejepa/manifest.py`: `build_manifest` y
  `batch_by_study` con Polars, portando la lógica de `design/a.py` y añadiendo la
  unión con las etiquetas clínicas (`laterality`, `view_position`, `breast_birads`,
  `breast_density`) para construir los `ImageTask`. Añadir
  `count_metadata_discrepancies`: compara el `image_id` de
  `breast-level_annotations.csv` contra el `SOP Instance UID` (columna repetida en
  la cabecera de `metadata.csv`; leer una sola vez, desduplicada) y devuelve el
  recuento de discrepancias en cada sentido sin tocar el manifiesto (FR-032);
  `cli.py` lo registra como aviso al arrancar `run`/`resume` (T028).
- [X] T025 [US1] Implementar `src/mammo_lejepa/download.py`: sesión `aiohttp` con
  `CookieJar` restringido al dominio, cabeceras y `TCPConnector` acotado portando
  `design/b.py`; `validate_access`; `download_dicom` atómico con reintentos,
  retroceso exponencial con jitter y respeto de `Retry-After`; clasificación de
  errores en `AuthError` / `NetworkError`.
- [X] T026 [US1] Implementar `src/mammo_lejepa/worker.py`: `process_dicom` encadena
  `read_dicom` → ventana → normalización → `detect_breast_box` → `crop_image` →
  `write_png_atomic`, contrasta las dimensiones reales con las declaradas en el CSV y
  devuelve siempre un `ProcessOutcome`, sin propagar excepciones.
- [X] T027 [US1] Implementar `src/mammo_lejepa/pipeline.py` en su versión mínima:
  productores `asyncio`, `asyncio.Queue` acotada, `ProcessPoolExecutor`,
  y el orden estricto PNG → línea JSONL → borrado del DICOM (D-08).
- [X] T028 [US1] Conectar el subcomando `run` en `src/mammo_lejepa/cli.py`: construir
  la configuración, validar el acceso, generar `run_id`, imprimir el techo de disco,
  imprimir el aviso de `count_metadata_discrepancies` (FR-032) y lanzar el pipeline.
- [X] T029 [US1] Añadir el progreso en consola con `rich`: imágenes resueltas, tasa
  de MiB/s, fallos acumulados y estimación de tiempo restante.
- [X] T030 [US1] `tests/integration/test_pipeline_end_to_end.py`: contra el servidor
  local que sirve DICOM sintéticos, una ejecución completa produce los PNG
  esperados, una línea JSONL por imagen con todos los campos de `data-model.md`, y
  cero ficheros en el directorio `dicom/`.

**Checkpoint**: MVP funcional. Existe corpus verificable de una muestra real.

---

## Phase 4: User Story 2 — Dataset completo, reanudable y con techo de disco (P2)

**Goal**: que la ejecución sobre 5.000 estudios sea interrumpible, reanudable y con
consumo de disco acotado.

**Independent Test**: interrumpir a mitad, relanzar, comprobar cero descargas
repetidas y ningún duplicado en el catálogo.

### Tests de la historia 2

- [X] T031 [P] [US2] `tests/unit/test_resume.py`: una imagen con registro `ok` pero
  sin PNG vuelve a la lista de pendientes; con registro y PNG no vuelve; una imagen
  fallida sólo vuelve en modo reintento; un JSONL con una última línea truncada se
  lee sin error y descartando esa línea.
- [X] T032 [P] [US2] `tests/integration/test_pipeline_resume.py`: cancelar el pipeline
  a mitad y relanzarlo produce un catálogo sin duplicados y no vuelve a descargar lo
  ya resuelto.
- [X] T033 [P] [US2] `tests/integration/test_backpressure.py`: con `queue_size`
  pequeño y un procesado artificialmente lento, el número de ficheros simultáneos en
  `dicom/` nunca supera el techo calculado.

### Implementación de la historia 2

- [X] T034 [P] [US2] Implementar `src/mammo_lejepa/resume.py`: `pending_tasks`,
  lectura tolerante del JSONL (línea truncada descartada) y `summarize_failures`.
- [X] T035 [US2] Integrar la reanudación en `pipeline.py` y añadir el subcomando
  `resume` en `cli.py`, que reutiliza el manifiesto de la última ejecución.
- [X] T036 [US2] Añadir el bloqueo del directorio de salida en el arranque del `run` y
  del `resume`, con mensaje claro cuando ya hay otra ejecución activa.
- [X] T037 [US2] Implementar el cierre ordenado ante `SIGINT`/`SIGTERM`: dejar de
  encolar, drenar lo que está en vuelo, escribir sus registros, cerrar el pool y la
  sesión, y registrar `exit_reason` en `runs.jsonl`.
- [X] T038 [US2] Implementar el aborto global ante `AuthError`: cancelar los
  productores, drenar la cola y terminar con un mensaje que explique cómo renovar la
  cookie y que el trabajo hecho está a salvo.
- [X] T039 [US2] Escribir `runs.jsonl` al inicio y al final de cada ejecución con los
  parámetros efectivos y los recuentos, según `data-model.md`.
- [X] T040 [US2] Persistir el manifiesto efectivo en
  `data/vindr-mammo/catalog/manifest_<run_id>.parquet`.

**Checkpoint**: el pipeline es apto para una ejecución de horas sobre los 300 GB.

---

## Phase 5: User Story 3 — Catálogo consolidado listo para entrenamiento (P3)

**Goal**: Parquet de imágenes y Parquet de hallazgos con coordenadas remapeadas.

**Independent Test**: dibujar las cajas del Parquet sobre el PNG recortado y ver que
encajan sobre la lesión.

### Tests de la historia 3

- [X] T041 [P] [US3] `tests/unit/test_findings.py`: `parse_finding_categories`
  interpreta `"['Mass']"` y `"['Mass', 'Suspicious Calcification']"` sin `eval`; las
  filas sin coordenadas se descartan y se contabilizan, no producen filas nulas; el
  remapeo desplaza correctamente las cajas; una caja parcialmente fuera queda marcada
  como `clipped`; una caja totalmente fuera se conserva con `fully_contained=false`.
- [X] T042 [P] [US3] `tests/integration/test_catalog_consolidation.py`: la
  consolidación es idempotente, resuelve duplicados quedándose con el
  `processed_at` más reciente, y las imágenes sin hallazgos no aparecen en
  `findings.parquet`.

### Implementación de la historia 3

- [X] T043 [P] [US3] Implementar `src/mammo_lejepa/findings.py` según el contrato,
  con lectura de `finding_annotations.csv` mediante Polars.
- [X] T044 [US3] Implementar `src/mammo_lejepa/catalog.py`: consolidación de
  `images.jsonl` a `images.parquet` con desduplicación, y generación de
  `findings.parquet` uniendo hallazgos con los `BreastCrop` del catálogo.
- [X] T045 [US3] Conectar el subcomando `consolidate` en `cli.py`, con un resumen de
  filas escritas y de discrepancias encontradas.
- [X] T046 [US3] Implementar el subcomando `inspect`: lámina de control de calidad con
  original, caja detectada, recorte y cajas de hallazgo remapeadas, usando
  `matplotlib`. Requiere conservar temporalmente los DICOM de las imágenes
  inspeccionadas o trabajar sobre las de cuarentena.
- [X] T047 [US3] Añadir una validación de coherencia del catálogo: toda fila `ok`
  tiene su PNG en disco con el tamaño registrado, y toda caja de hallazgo remapeada
  cae dentro de las dimensiones del recorte.

**Checkpoint**: el corpus es directamente consumible por un `Dataset` de PyTorch.

---

## Phase 6: User Story 4 — Cuarentena, diagnóstico y reintento (P4)

**Goal**: que los fallos no bloqueen, queden diagnosticables y se puedan reintentar
sin volver a recorrer el dataset.

**Independent Test**: inyectar un DICOM corrupto, ver que el pipeline continúa y que
`retry --failed` lo vuelve a intentar.

### Tests de la historia 4

- [X] T048 [P] [US4] `tests/integration/test_failure_isolation.py`: un DICOM corrupto
  en mitad de un lote no detiene la ejecución, queda en `quarantine/` y su registro
  tiene `failure_category = DECODE` con la traza.
- [X] T049 [P] [US4] `tests/integration/test_retry_failed.py`: el reintento procesa
  sólo las imágenes fallidas y sustituye sus registros previos.

### Implementación de la historia 4

- [X] T050 [US4] Integrar la cuarentena en el pipeline: ante un `ProcessOutcome`
  fallido, mover el DICOM a `data/vindr-mammo/images/quarantine/<study_id>/` en lugar
  de borrarlo.
- [X] T051 [US4] Implementar el subcomando `retry --failed [--category ...]`, que
  reconstruye la lista de pendientes a partir de los registros fallidos.
- [X] T052 [US4] Implementar el resumen final por categoría de error, con la ruta del
  registro y el recuento de DICOM en cuarentena.
- [X] T053 [US4] Garantizar que ningún mensaje de error, traza o registro contiene la
  cookie ni las credenciales: función de saneado aplicada antes de escribir, con su
  test en `tests/unit/test_redaction.py`.

**Checkpoint**: el pipeline es operable en producción sin supervisión continua.

---

## Phase 7: Cierre y calidad

- [X] T054 [P] Añadir un test que verifique el principio I: los módulos puros
  (`manifest`, `windowing`, `segmentation`, `geometry`, `findings`, `resume`) no
  importan `aiohttp`, `requests`, `pydicom`, `os` ni `pathlib`, comprobado por
  análisis del AST en `tests/unit/test_purity.py`.
- [X] T055 [P] Redactar `README.md` con el propósito del corpus, el procedimiento de
  la cookie de PhysioNet, los comandos y la advertencia de no redistribución.
- [X] T056 [P] Documentar en el README el esquema de `images.parquet` y
  `findings.parquet` para quien escriba el `Dataset`.
- [X] T057 Ejecutar `ruff format` y `ruff check --fix` sobre todo el árbol y dejar la
  suite en verde.
- [ ] T058 Ejecución de validación real sobre 50 estudios: medir el pico de disco, la
  tasa de fallos y el rendimiento frente a `design/b.py`, y contrastar con SC-002,
  SC-005 y SC-008. **Parcialmente cubierto**: el usuario ejecutó
  `mammo-etl run --n-studies 5 --split training` dos veces de forma independiente
  (20 imágenes reales cada vez, 0 fallos, 394.9 MB descargados / 22.8 MB escritos ⇒
  5.8 % — cumple SC-003). Sigue pendiente la escala de 50 estudios: el ancho de
  banda de esta sesión (~50–150 KB/s) lo hace inviable en el tiempo disponible.
  Pendiente de ejecución manual por el usuario con
  `mammo-etl run --n-studies 50 --split training`.
- [X] T059 Registrar en `research.md` la `TransferSyntaxUID` observada (cierre del
  punto *a verificar* de D-01) y revisar la distribución de `crop_area_ratio` en
  busca de cajas infladas (D-06). Hecho sobre la muestra real de 4 imágenes
  disponible en esta sesión; repetir con la muestra de T058 cuando se ejecute.

## Correcciones post-implementación

- [X] T060 **FR-033** (reportado por el usuario tras usar la herramienta):
  `run --n-studies N` invocado varias veces sin `--study-id` explícito
  reseleccionaba siempre los mismos primeros `N` estudios en orden alfabético
  —confirmado revisando `runs.jsonl` y los `manifest_<run_id>.parquet`
  persistidos: tres ejecuciones de `--n-studies 5` produjeron el mismo
  manifiesto de 5 estudios—, en vez de avanzar sobre estudios nuevos. Añadida
  `exclude_completed_studies` (pura) en `manifest.py`, con test en
  `tests/unit/test_manifest.py`, y conectada en `cli.py run`: sin
  `--study-id`, excluye del CSV de anotaciones los estudios cuyas imágenes
  están todas ya resueltas con éxito antes de aplicar `--n-studies`/split.
  Verificado además contra el dataset y catálogo reales de esta sesión: tras
  excluir los 5 estudios ya completados, `--n-studies 5` selecciona 5 estudios
  nuevos sin solape.

---

## Dependencias

- **Fase 1 → Fase 2 → Fases 3-6 → Fase 7**: las fases 1 y 2 bloquean todo lo demás.
- **US1 (Fase 3)** no depende de ninguna otra historia: es el MVP.
- **US2 (Fase 4)** depende de US1: reanuda lo que US1 produce.
- **US3 (Fase 5)** depende de US1 (necesita `BreastCrop` en el catálogo); es
  independiente de US2 y puede desarrollarse en paralelo a ella.
- **US4 (Fase 6)** depende de US1; independiente de US2 y US3.
- Dentro de cada historia, los tests marcados [P] se escriben antes que la
  implementación correspondiente y deben fallar primero.

## Estrategia de ejecución sugerida

1. Fases 1 y 2 completas, con `doctor` funcionando: cierra el riesgo del decodificador
   antes de escribir una línea de pipeline.
2. Fase 3 completa y **verificación visual sobre 5 estudios reales**: no escalar
   hasta que la caja mamaria sea correcta a ojo.
3. Fase 4, y sólo entonces lanzar la ejecución larga.
4. Fases 5 y 6 en paralelo mientras la ejecución larga avanza.
5. Fase 7 con los datos reales ya en disco.

---

## Trazabilidad requisito → tarea

| FR | Tareas | FR | Tareas |
|---|---|---|---|
| FR-001 | T019, T024 | FR-017 | T027, T030 |
| FR-002 | T024, T028 | FR-018 | T048, T050 |
| FR-003 | T024 | FR-019 | T008, T033 |
| FR-004 | T040 | FR-020 | T009, T027 |
| FR-005 | T025, T027 | FR-021 | T006, T026, T030 |
| FR-006 | T008, T013, T025 | FR-022 | T024 |
| FR-007 | T020, T025 | FR-023 | T041, T043 |
| FR-008 | T020, T025 | FR-024 | T023, T041, T043 |
| FR-009 | T038 | FR-025 | T044, T045 |
| FR-010 | T016, T021 | FR-026 | T042, T044 |
| FR-011 | T016, T021 | FR-027 | T031, T034 |
| FR-012 | T017, T022 | FR-028 | T009, T036 |
| FR-013 | T018, T023 | FR-029 | T029, T052 |
| FR-014 | T009, T026 | FR-030 | T049, T051 |
| FR-015 | T017, T022, T059 | FR-031 | T008, T053 |
| FR-016 | T027, T033 | | |
| FR-032 | T019, T024, T028 | FR-033 | T060 |

## Trazabilidad criterio de éxito → verificación

| SC | Cómo se verifica |
|---|---|
| SC-001 | T046 (`inspect`) sobre la muestra de 5 estudios |
| SC-002 | T033 en test, T058 en ejecución real |
| SC-003 | T058, comparando bytes descargados y bytes escritos de `runs.jsonl` |
| SC-004 | T032 |
| SC-005 | T058, recuento de cuarentena sobre 50 estudios |
| SC-006 | T046, muestreo de al menos 30 hallazgos |
| SC-007 | `uv run pytest tests/unit` en T057 |
| SC-008 | T058, contra el benchmark de `design/b.py` |
