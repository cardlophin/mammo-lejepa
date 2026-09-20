---

description: "Task list for 004-vindr-downstream"
---

# Tasks: VinDr-Mammo como conjunto de evaluación downstream

**Input**: Design documents from `/specs/004-vindr-downstream/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/module-contracts.md, quickstart.md

**Tests**: incluidos — constitución, principio III, exige tests sobre la lógica
crítica (manifiesto, coordenadas) y principio I exige que sea posible sin red ni
GPU.

**Organización**: por historia de usuario (spec.md). El código nuevo vive en
`src/mammo_lejepa/vindr/`; los tests en `tests/vindr/` (directorio plano, igual que
`tests/ssl/` planeado para la feature 003).

## Phase 1: Setup

- [X] T001 Recuperar del commit `0de1617` (sólo como referencia de partida, no para
  restaurar en su ubicación original) el contenido de `download.py`, `dicom_io.py`,
  `windowing.py`, `findings.py`, `pipeline.py`, `worker.py`, `manifest.py`,
  `catalog.py`, `cli.py` y sus tests, con `git show 0de1617:<ruta>`.
- [X] T002 Restaurar en `pyproject.toml` las dependencias retiradas (`aiohttp`,
  `aiofiles`, `physionet`, `pydicom`, `pylibjpeg`, `pylibjpeg-openjpeg`,
  `python-dotenv`) y `pytest-asyncio` con `asyncio_mode = "auto"` en
  `[tool.pytest.ini_options]`; añadir `mammo-etl = "mammo_lejepa.vindr.cli:app"` en
  `[project.scripts]` **junto a** `mammo-corpus`, sin sustituirlo. No restaurar
  `beautifulsoup4` (research.md, D-07: sin uso real ni en la propia feature 001).
- [X] T003 `uv sync` y comprobar que `mammo-corpus --help` (feature 002) sigue
  funcionando sin cambios tras añadir las dependencias de esta feature.

**Checkpoint**: dependencias instaladas, ningún código nuevo todavía.

---

## Phase 2: Foundational (bloqueante para todas las historias)

**Propósito**: los tipos y la configuración propios de `vindr/` de los que depende
todo lo demás.

- [X] T004 [P] Crear `src/mammo_lejepa/vindr/__init__.py` (paquete vacío).
- [X] T005 [P] Añadir `PIPELINE_VERSION = "0.1.0"` a `src/mammo_lejepa/__init__.py`
  (junto a `CORPUS_VERSION`, sin tocarla).
- [X] T006 [P] Crear `src/mammo_lejepa/vindr/models.py`: `FailureCategory` propio
  (`NETWORK`, `AUTH`, `DECODE`, `SEGMENTATION`, `WRITE`, `UNKNOWN`), `ImageTask`,
  `ProcessOutcome`, `ImageRecord`, `DicomPayload`, `DownloadResult`, `RunSummary` —
  adaptados del `models.py` de `0de1617` (T001) usando `mammo_lejepa.models.
  CajaMamaria` y `mammo_lejepa.models.BoundingBox` en vez de `BreastCrop` propio
  (research.md D-01).
- [X] T007 [P] Crear `src/mammo_lejepa/vindr/errors.py`: `VindrError`, `AuthError`,
  `NetworkError`, `redact_secrets`, `classify_exception` propio — reexportando
  `DecodeError`/`SegmentationError`/`WriteError` de `mammo_lejepa.errors` en vez de
  duplicarlos (research.md D-02).
- [X] T008 [P] Crear `src/mammo_lejepa/vindr/storage.py`: sólo `quarantine()`
  (research.md D-03).
- [X] T009 [P] Crear `src/mammo_lejepa/vindr/config.py`: `PhysioNetCredentials`,
  `load_credentials()`, `PipelineConfig` (data-model.md) — adaptado del `config.py`
  de `0de1617`.
- [X] T010 [P] `tests/vindr/fixtures/synthetic_dicom.py`: adaptar sin cambios de
  fondo desde `0de1617` (fixture autocontenida, sólo numpy).
- [X] T011 [P] `tests/vindr/conftest.py`: fixture `pipeline_config` (tmp_path) y
  fixture de credenciales de prueba, análogas a `tests/conftest.py` de la feature
  002.
- [X] T012 [P] `tests/vindr/test_errors.py`: `classify_exception` mapea
  `AuthError`→`AUTH`, `NetworkError`→`NETWORK`, y delega correctamente en
  `DecodeError`/`SegmentationError`/`WriteError` compartidos; `redact_secrets`
  sustituye cada secreto no vacío y dejar los vacíos/`None` intactos.

**Checkpoint**: `import mammo_lejepa.vindr.config` y `.models` y `.errors`
funcionan; ningún módulo de Mammo-Bench se tocó.

---

## Phase 3: User Story 1 - Reconstruir el corpus de recortes de VinDr (Priority: P1) 🎯 MVP

**Goal**: `mammo-etl run` descarga, decodifica, recorta y cataloga
incrementalmente un subconjunto de VinDr-Mammo, reanudable e interrumpible.

**Independent Test**: con credenciales vigentes, `mammo-etl run --split training
--n-studies 5` produce un PNG por imagen procesada con éxito y un resumen sin
ambigüedad; relanzarlo no reprocesa nada.

### Tests para User Story 1

- [X] T013 [P] [US1] `tests/vindr/test_windowing.py`: adaptar desde `0de1617`
  (import `mammo_lejepa.vindr.windowing`) — MONOCHROME1 invertido, ventana ausente
  es identidad, recorte a `[low, high]`, normalización nunca divide por cero.
- [X] T014 [P] [US1] `tests/vindr/test_dicom_io.py`: `read_dicom` sobre un DICOM
  sintético (`make_synthetic_dicom`) devuelve un `DicomPayload` correcto; una
  sintaxis de transferencia sin decodificador lanza `DecodeError` nombrando la
  sintaxis y el paquete ausente.
- [X] T015 [P] [US1] `tests/vindr/test_manifest.py`: adaptar desde `0de1617` —
  columna requerida ausente lanza `ValueError` nombrándola; split inexistente y
  `study_id` no encontrados lanzan `ValueError`; `n_studies` selecciona siempre los
  primeros de un orden determinista; `batch_by_study` no asume 4 imágenes/estudio;
  `count_metadata_discrepancies` cuenta en ambos sentidos sin lanzar.
- [X] T016 [US1] `tests/vindr/test_download_local_server.py` (marcar
  `@pytest.mark.slow`): adaptar desde `0de1617` contra un servidor HTTP local —
  descarga atómica (`.part` + renombrado), reintento con retroceso en 429/503,
  `AuthError` sin reintentar en 401/403 y ante HTML, `download_dicom` no repite una
  descarga ya completa.
- [X] T017 [US1] `tests/vindr/test_pipeline_end_to_end.py` (marcar
  `@pytest.mark.slow`): adaptar desde `0de1617` — sobre DICOM sintéticos servidos
  localmente, `run_pipeline` produce un PNG y un registro `ok` por imagen, respeta
  `config.queue_size`/`config.workers`, y termina con `exit_reason="completed"`.
- [X] T018 [US1] `tests/vindr/test_pipeline_resume.py` (marcar `@pytest.mark.slow`):
  adaptar desde `0de1617` — interrumpir a mitad y relanzar con el manifiesto
  completo no reprocesa lo ya resuelto con éxito (usa `resume.pending_tasks`
  compartido).

### Implementación de User Story 1

- [X] T019 [P] [US1] Crear `src/mammo_lejepa/vindr/windowing.py`: PURO, sin
  cambios de fondo desde `0de1617`.
- [X] T020 [P] [US1] Crear `src/mammo_lejepa/vindr/dicom_io.py`: `read_dicom`,
  usando `DecodeError` de `mammo_lejepa.errors` (compartido) y `DicomPayload` de
  `vindr.models`.
- [X] T021 [US1] Crear `src/mammo_lejepa/vindr/manifest.py`: `build_manifest`,
  `batch_by_study`, `count_metadata_discrepancies` — adaptado desde `0de1617`
  (depende de T006).
- [X] T022 [US1] Crear `src/mammo_lejepa/vindr/download.py`: `build_cookie_jar`,
  `build_client_session`, `validate_access`, `download_dicom` — adaptado desde
  `0de1617` importando `vindr.config`/`vindr.errors`/`vindr.models` (depende de
  T006, T007, T009).
- [X] T023 [US1] Crear `src/mammo_lejepa/vindr/worker.py`: `process_dicom` —
  adaptado desde el `worker.py` de `0de1617`, usando `detect_breast_box`
  (`segmentation.py` compartido) y `crop_image` (`geometry.py` compartido) sin
  modificarlos; `write_png_atomic` de `storage.py` compartido (depende de T006,
  T019, T020).
- [X] T024 [US1] Crear `src/mammo_lejepa/vindr/pipeline.py`: `run_pipeline` —
  adaptado desde `0de1617`, usando `append_record` de `storage.py` compartido y
  `quarantine` de `vindr/storage.py` (T008); `redact_secrets` de `vindr/errors.py`
  antes de escribir cualquier `error_message` (depende de T006-T009, T022, T023).
- [X] T025 [US1] Crear `src/mammo_lejepa/vindr/cli.py` con la app `mammo-etl` y los
  subcomandos `doctor`, `run`, `resume`, `retry` — adaptados desde `0de1617`;
  `doctor` distingue explícitamente CSV ausentes de cookie caducada/credenciales
  ausentes (FR-022), cada comprobación en su propia línea de salida.
- [ ] T026 [US1] Verificación manual con credenciales reales (no automatizable sin
  ellas): `uv run mammo-etl doctor` diagnostica correctamente CSV ausentes; una vez
  presentes, `uv run mammo-etl run --split training --n-studies 3` produce PNG y
  catálogo incremental.

**Checkpoint**: `mammo-etl run`/`resume`/`retry`/`doctor` funcionales de forma
independiente del resto de historias.

---

## Phase 4: User Story 2 - Cajas de hallazgo en el espacio del recorte (Priority: P2)

**Goal**: `mammo-etl consolidate` produce `images.parquet` y `findings.parquet`
con las cajas de hallazgo remapeadas al espacio del recorte, y `mammo-etl inspect`
permite verificarlo visualmente.

**Independent Test**: sobre registros de imágenes ya procesadas (fixture, sin
depender de un `run` real), `remap_findings` traslada correctamente coordenadas
conocidas y marca los hallazgos clippeados; `consolidate` produce ambos Parquet
coherentes.

### Tests para User Story 2

- [X] T027 [P] [US2] `tests/vindr/test_findings.py`: adaptar desde `0de1617`
  (import `mammo_lejepa.vindr.findings`, tipo `CajaMamaria` en vez de
  `BreastCrop`) — `parse_finding_categories` interpreta el literal sin `eval`;
  `remap_findings` traslada coordenadas conocidas al espacio del recorte, marca
  `clipped`/`fully_contained` correctamente, descarta filas sin coordenadas y
  filas sin caja conocida.
- [X] T028 [P] [US2] `tests/vindr/test_catalog.py`: `consolidate` deduplica por
  `processed_at`, aplana la `CajaMamaria` a columnas `crop_*` (`crop.threshold`,
  no `crop.otsu_threshold`; `crop.image_height`/`image_width`, no
  `source_height`/`source_width` — research.md D-01); `validate_catalog_coherence`
  detecta un PNG ausente, un tamaño distinto del registrado, y una caja de
  hallazgo remapeada fuera de las dimensiones del recorte.

### Implementación de User Story 2

- [X] T029 [US2] Crear `src/mammo_lejepa/vindr/findings.py`: PURO,
  `parse_finding_categories`, `remap_findings` — adaptado desde `0de1617` usando
  `CajaMamaria`/`BoundingBox` de `mammo_lejepa.models` y `to_crop_space`/
  `clip_to_crop` de `geometry.py` compartido sin modificarlos (depende de T006).
- [X] T030 [US2] Crear `src/mammo_lejepa/vindr/catalog.py`: `consolidate`,
  `validate_catalog_coherence` — adaptado desde `0de1617` (depende de T006, T029,
  `resume.py` compartido).
- [X] T031 [US2] Añadir a `src/mammo_lejepa/vindr/cli.py` el subcomando
  `consolidate` (FR-018/FR-019) y el subcomando `inspect` (FR-028/SC-004):
  descarga temporalmente el DICOM de una muestra ya catalogada (se borró tras
  procesarse), dibuja la caja mamaria y los hallazgos remapeados, y borra el DICOM
  de nuevo al terminar — adaptado desde `0de1617` (depende de T030).
- [ ] T032 [US2] Verificación manual con credenciales reales: sobre el corpus de
  la US1, `uv run mammo-etl consolidate` seguido de `uv run mammo-etl inspect
  --n 20` — inspeccionar visualmente que los hallazgos quedan dibujados en la
  posición correcta sobre el recorte (SC-004).

**Checkpoint**: catálogo de imágenes y de hallazgos coherente y verificable
visualmente, independiente de la US3.

---

## Phase 5: User Story 3 - Particiones por estudio verificadas (Priority: P3)

**Goal**: la columna `split` del catálogo es la oficial, y su ausencia de fuga
entre `training`/`test` a nivel de `study_id` se verifica automáticamente, no sólo
se documenta.

**Independent Test**: sobre un `images.parquet` fixture con una fuga deliberada
(una imagen de un estudio con `split` distinto al resto de imágenes del mismo
estudio), la verificación falla nombrando el `study_id`; sobre uno sin fuga, pasa.

### Tests para User Story 3

- [X] T033 [P] [US3] `tests/vindr/test_catalog.py` (añadir casos):
  `verify_no_study_leakage` pasa sobre un catálogo donde cada estudio tiene un
  único `split`; lanza `ValueError` nombrando el `study_id` cuando dos imágenes
  del mismo estudio declaran particiones distintas.

### Implementación de User Story 3

- [X] T034 [US3] Añadir `verify_no_study_leakage(images_df: pl.DataFrame) -> None`
  a `src/mammo_lejepa/vindr/catalog.py` (FR-021): agrupa por `study_id`, comprueba
  que `split` tiene un único valor por grupo, lanza `ValueError` nombrando el
  `study_id` en conflicto en caso contrario.
- [X] T035 [US3] Llamar a `verify_no_study_leakage` desde el subcomando
  `consolidate` de `src/mammo_lejepa/vindr/cli.py`, tras escribir
  `images.parquet`, informando el resultado en consola (FR-021, SC-003).

**Checkpoint**: las tres historias completas y verificables de forma
independiente entre sí.

---

## Phase 6: Polish

- [X] T036 [P] `ruff format` y `ruff check --fix` sobre `src/mammo_lejepa/vindr/`
  y `tests/vindr/`; comprobar que no afecta a ningún fichero de la feature 002.
- [X] T037 `uv run pytest tests/vindr -q -m "not slow"` en verde, sin red ni
  credenciales (constitución, principio III); `uv run pytest` (todo el repo) sigue
  en verde para la feature 002.
- [X] T038 `uv run pytest tests/vindr -q` (incluye los `@pytest.mark.slow` contra
  el servidor HTTP local) en verde.
- [ ] T039 Ejecución real acotada con credenciales vigentes (bloqueada mientras no
  se disponga de una cookie de sesión renovada): `mammo-etl run --split training
  --n-studies 20`, `consolidate`, verificación de partición e `inspect`,
  siguiendo `quickstart.md` paso a paso.

**Checkpoint**: `catalog/images.parquet` y `catalog/findings.parquet` listos para
que una feature de evaluación futura los consuma.

---

## Dependencies & Execution Order

- **Setup (Fase 1)**: sin dependencias.
- **Foundational (Fase 2)**: depende de la Fase 1. Bloquea las tres historias.
- **US1 (Fase 3)**: depende de la Fase 2. Sin dependencia de US2/US3.
- **US2 (Fase 4)**: depende de la Fase 2 y de tener registros de imagen que
  consolidar (en la práctica, de la US1 ejecutada al menos una vez; sus tests
  usan fixtures, no un `run` real, así que es implementable en paralelo con US1).
- **US3 (Fase 5)**: depende de US2 (opera sobre `images.parquet`, que produce
  `consolidate`).
- **Polish (Fase 6)**: depende de las tres historias.

### Oportunidades de paralelismo

- T004-T012 (Fase 2) son en su mayoría `[P]`: distintos ficheros, sin
  dependencias entre sí salvo T012 (test) sobre T007.
- T013-T015 (tests puros de US1) son `[P]` entre sí.
- T019-T020 (US1) son `[P]` entre sí; T021-T024 tienen dependencias secuenciales
  ya anotadas.
- T027-T028 (tests de US2) son `[P]` entre sí.

## Implementation Strategy

**MVP**: Fases 1-3 (Setup, Foundational, US1) — `mammo-etl run`/`resume`/`retry`/
`doctor` funcionando de forma reanudable, sin hallazgos ni verificación de
partición todavía.

**Incremental**: US2 añade el catálogo de hallazgos remapeados (lo que distingue a
VinDr de Mammo-Bench); US3 añade la puerta de calidad de partición. Cada historia
es un incremento demostrable sin romper la anterior.
