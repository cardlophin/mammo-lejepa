---
description: "Tareas de implementación — corpus de recortes desde Mammo-Bench"
---

# Tasks: Corpus de recortes mamarios desde Mammo-Bench

**Input**: documentos de diseño en `/specs/002-mammobench-corpus/`

**Prerequisites**: `spec.md`, `plan.md`, `migration.md`, `data-model.md`,
`contracts/module-contracts.md`

## Formato: `[ID] [P?] [Story] Descripción`

- **[P]**: puede ejecutarse en paralelo (ficheros distintos, sin dependencias)
- Rutas exactas y relativas a la raíz del repositorio

---

## Phase 1: Retirada del origen anterior

**Propósito**: dejar el árbol limpio antes de construir encima. Hacerlo primero evita
que el código nuevo herede por inercia abstracciones de la descarga en red.

- [X] T001 Crear la rama `002-mammobench-corpus` y confirmar que el árbol está limpio
  (hay cambios sin confirmar en `README.md`, `specs/001/`, `cli.py`, `manifest.py` y
  `tests/unit/test_manifest.py`).
- [X] T002 Eliminar los módulos retirados: `src/mammo_lejepa/download.py`,
  `dicom_io.py`, `windowing.py`, `findings.py`, `pipeline.py`.
- [X] T003 [P] Eliminar los tests de los módulos retirados:
  `tests/fixtures/synthetic_dicom.py`, `tests/unit/test_windowing.py`,
  `test_findings.py`, `test_redaction.py`, y los seis tests de integración de
  descarga y contrapresión listados en `migration.md`.
- [X] T004 [P] Eliminar `design/a.py`, `design/b.py`, `design/c.py`.
- [X] T005 [P] Retirar de `pyproject.toml` las dependencias sin uso: `aiohttp`,
  `aiofiles`, `physionet`, `pydicom`, `pylibjpeg`, `pylibjpeg-openjpeg`, `requests`,
  `beautifulsoup4`, `dotenv`, `pytest-asyncio`; y la opción `asyncio_mode` de pytest.
  Ejecutar `uv sync` y comprobar que la resolución no arrastra nada más.
- [X] T006 [P] Marcar `specs/001-vindr-streaming-etl/spec.md` con
  `Status: Superseded by 002-mammobench-corpus` y una línea explicando el motivo.
- [X] T007 Liberar disco: eliminar `data/vindr-mammo/` y `data/mammobench.zip`
  (8,3 GB ya extraídos), y vaciar `.env` de las credenciales de PhysioNet.
- [X] T008 Comprobar que `uv run pytest` sigue en verde con lo que queda
  (`test_geometry`, `test_segmentation`, `test_resume`, `test_purity`); corregir
  únicamente las importaciones rotas por las eliminaciones, sin añadir
  funcionalidad.

**Checkpoint**: árbol sin código de red ni de DICOM, suite reducida en verde.

---

## Phase 2: Foundational

- [X] T009 Reescribir `src/mammo_lejepa/config.py`: `CorpusConfig` con raíz de
  Mammo-Bench, directorio de salida `data/corpus/`, `mask_threshold=128`,
  `margin_px=25`, `workers`, `seed`, `ratios` de partición y parámetros de sospecha.
  Fuera credenciales, URLs y todo lo relativo a la descarga.
- [X] T010 Adaptar `src/mammo_lejepa/models.py` según `data-model.md`: retirar los
  campos DICOM, añadir `ImagenMammoBench`, `CajaMamaria`, `BoxSource`,
  `FallbackReason`, `RegistroDeRecorte`, `patient_key`. Invariantes en
  `__post_init__`.
- [X] T011 [P] Crear `tests/fixtures/synthetic_mammogram.py`: fábrica de pares
  (imagen, máscara) de geometría conocida — elipse clara sobre fondo oscuro, variante
  con etiqueta de orientación en una esquina, variante con una banda triangular en la
  esquina superior simulando músculo pectoral que **la máscara excluye y Otsu no**,
  variante de máscara vacía, saturada y ausente.
- [X] T012 Reescribir `tests/conftest.py` sobre las nuevas fixtures y un CSV de
  Mammo-Bench reducido en memoria.
- [X] T013 [P] Implementar `src/mammo_lejepa/image_io.py` según contrato.
- [X] T014 Actualizar `tests/unit/test_purity.py` con la lista nueva de módulos puros
  (`manifest`, `boxing`, `segmentation`, `geometry`, `splits`, `quality`, `resume`).
- [X] T015 Actualizar `.gitignore`: añadir `runs/`, `checkpoints/`, `*.ckpt`.

**Checkpoint**: entidades y fixtures listas; ninguna historia empezada.

---

## Phase 3: User Story 1 — Corpus de recortes (P1) 🎯 MVP

### Tests primero

- [X] T016 [P] [US1] `tests/unit/test_manifest.py` (reescritura): 19.731 filas del CSV
  de muestra producen 19.731 entradas; cadenas vacías normalizadas a `None`;
  `patient_key` compuesto; `image_id` derivado del nombre de fichero; columna ausente
  lanza `ValueError` nombrándola; los filtros por fuente y por límite son
  deterministas.
- [X] T017 [P] [US1] `tests/unit/test_boxing.py`: la caja de la máscara sintética
  coincide con la geometría conocida ±2 px; **la caja de la máscara excluye la banda
  pectoral y la de Otsu no** —es el test que justifica toda la decisión D-01—;
  máscara vacía, saturada y ausente disparan el respaldo con el `FallbackReason`
  correcto; `box_iou` da 1.0 para cajas idénticas, 0.0 para disjuntas y el valor
  esperado para un solape conocido.
- [X] T018 [P] [US1] `tests/integration/test_build_end_to_end.py`: sobre un corpus
  sintético de 12 imágenes en 3 fuentes, el build produce 12 PNG, 12 líneas de JSONL
  con todos los campos de `data-model.md`, y un resumen por fuente.

### Implementación

- [X] T019 [P] [US1] Reescribir `src/mammo_lejepa/manifest.py` sobre
  `mammo-bench.csv` con Polars, según contrato.
- [X] T020 [P] [US1] Implementar `src/mammo_lejepa/boxing.py`: `box_from_mask`,
  `resolve_box` con la política de respaldo, `box_iou`. Reutiliza `segmentation.py`
  sin modificarlo.
- [X] T021 [US1] Reescribir `src/mammo_lejepa/worker.py`: lectura de imagen y
  máscara, `resolve_box`, `crop_image`, PNG atómico, registro con métricas de
  calidad. Nunca propaga excepciones.
- [X] T022 [US1] Implementar `src/mammo_lejepa/runner.py`: `ProcessPoolExecutor`,
  escritor único de JSONL con vaciado, progreso `rich`, cierre ordenado ante
  `SIGINT`.
- [X] T023 [US1] Integrar la reanudación existente (`resume.py`) en `runner.py`:
  descuenta las imágenes con registro `ok` y PNG presente.
- [X] T024 [US1] Reescribir `src/mammo_lejepa/cli.py` con el subcomando `build` y
  retirar `doctor`, `run`, `resume`, `retry` y todo rastro de validación de cookie.
- [X] T025 [US1] Escribir `runs.jsonl` al inicio y al final, con la configuración
  efectiva y la semilla.
- [X] T026 [US1] `tests/integration/test_resume_build.py`: interrumpir y relanzar no
  reprocesa ni duplica.

**Checkpoint**: `uv run mammo-corpus build` produce el corpus completo.

---

## Phase 4: User Story 2 — Calidad del recorte (P2)

- [X] T027 [P] [US2] `tests/unit/test_quality.py`: `is_suspect` marca área fuera de
  rango, aspecto imposible, IoU bajo y caja que toca los cuatro bordes, devolviendo
  el motivo; `summarize_by_source` calcula los percentiles esperados sobre registros
  sintéticos.
- [X] T028 [P] [US2] Implementar `src/mammo_lejepa/quality.py` según contrato.
- [X] T029 [US2] Subcomando `qc`: genera `quality_report.md` y
  `quality_by_source.parquet` con percentiles de `crop_area_ratio` y `mask_otsu_iou`,
  recuento de respaldos y lista de sospechosos por ruta.
- [X] T030 [US2] Subcomando `inspect`: lámina `grid_<fuente>.png` con muestra
  estratificada — imagen, máscara, caja dibujada y recorte.
- [X] T031 [US2] **Verificación manual obligatoria**: inspeccionar las seis láminas y
  responder por escrito en `qc/findings.md` a dos preguntas — ¿las máscaras excluyen
  realmente el pectoral en cada fuente?, ¿qué pasa con las máscaras de `cdd-cesm`
  que cubren el 80 % de la imagen? No se continúa a la fase 5 sin esa respuesta.

**Checkpoint**: se sabe con datos cuánto se puede fiar el corpus, por fuente.

---

## Phase 5: User Story 3 — Catálogo y particiones (P3)

- [X] T032 [P] [US3] `tests/unit/test_splits.py`: misma semilla da la misma
  asignación con independencia del orden de entrada; ningún paciente en dos
  particiones; proporciones dentro de tolerancia; un paciente con 14 imágenes no
  rompe las proporciones por imagen más allá de lo tolerado.
- [X] T033 [P] [US3] Implementar `src/mammo_lejepa/splits.py` con `assign_splits` y
  `verify_no_patient_leakage`.
- [X] T034 [US3] Adaptar `src/mammo_lejepa/catalog.py`: consolidación idempotente del
  JSONL a `catalog.parquet` con desduplicación por `processed_at`, nulos explícitos y
  exclusión de las 7 columnas vacías del CSV.
- [X] T035 [US3] Subcomandos `consolidate` y `split`, incorporando la columna `split`
  al catálogo y escribiendo `splits.parquet`.
- [X] T036 [US3] `tests/integration/test_splits_integrity.py`: sobre el catálogo real
  ya construido, intersección de `patient_key` entre particiones exactamente vacía.
- [X] T037 [US3] Validación de coherencia del catálogo: toda fila `ok` tiene su PNG
  con el tamaño registrado y dimensiones que coinciden con la caja.

**Checkpoint**: `catalog.parquet` listo para el `Dataset` de la feature 003.

---

## Phase 6: Cierre

- [X] T038 [P] Reescribir `README.md`: Mammo-Bench, las seis fuentes y sus licencias,
  la advertencia de no redistribución, los comandos y el esquema del catálogo.
- [X] T039 [P] Renombrar el script de la CLI en `pyproject.toml` de `mammo-etl` a
  `mammo-corpus`, que es lo que hace ahora.
- [X] T040 `ruff format` y `ruff check --fix` sobre todo el árbol; suite en verde.
- [X] T041 Ejecución real completa sobre las 19.731 imágenes: medir tiempo, tasa de
  respaldo a Otsu y número de sospechosos, y contrastar con SC-002, SC-003 y SC-006.
- [X] T042 Registrar en `plan.md` los valores observados de IoU por fuente, cerrando
  la verificación de D-01 y D-02.

---

## Dependencias

- Fase 1 → Fase 2 → Fase 3 → (Fases 4 y 5 en paralelo) → Fase 6.
- US1 es el MVP y no depende de nada más.
- US2 y US3 dependen ambas de US1 y son independientes entre sí.
- **T031 es una puerta de calidad humana**: bloquea la fase 5 aunque el código esté
  listo, porque construir particiones sobre un corpus mal recortado es trabajo tirado.

## Trazabilidad requisito → tarea

| FR | Tareas | FR | Tareas |
|---|---|---|---|
| FR-001 | T002, T003, T004 | FR-014 | T010, T021 |
| FR-002 | T005 | FR-015 | T019, T034 |
| FR-003 | T008, T020 | FR-016 | T034, T035 |
| FR-004 | T019, T016 | FR-017 | T033, T032 |
| FR-005 | T019, T024 | FR-018 | T033, T032 |
| FR-006 | T019, T029 | FR-019 | T033, T036 |
| FR-007 | T020, T017 | FR-020 | T009, T035 |
| FR-008 | T020, T017 | FR-021 | T028, T029 |
| FR-009 | T020, T017 | FR-022 | T030, T031 |
| FR-010 | T021 | FR-023 | T028, T034 |
| FR-011 | T021 | FR-024 | T023, T026 |
| FR-012 | T021 | FR-025 | T022 |
| FR-013 | T022 | FR-026 | T022, T029 |
