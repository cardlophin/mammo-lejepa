---
description: "Tareas de implementación — preentrenamiento LeJEPA"
---

# Tasks: Preentrenamiento LeJEPA sobre el corpus mamario

**Input**: documentos de diseño en `/specs/003-lejepa-pretraining/`

**Prerequisites**: feature 002 terminada (`catalog.parquet` con particiones)

---

## Phase 1: Setup

- [X] T001 Añadir dependencias con `uv add`: `torch`, `torchvision`, `timm`; y de
  desarrollo `lejepa` (sólo para la validación cruzada). Crear `src/mammo_lejepa/ssl/`
  y `tests/ssl/`. **Nota**: `lejepa` no está publicado en PyPI (`pip install lejepa`
  del propio README da 404 real, verificado) — se añadió como dependencia de git
  (`git+https://github.com/rbalestr-lab/lejepa`, ver `[tool.uv.sources]` en
  `pyproject.toml`), y necesitó `scipy` como dependencia de desarrollo adicional no
  declarada por el propio paquete (import roto sin ella).
- [X] T002 **Leer el código del paquete `lejepa` instalado** y anotar en `plan.md`,
  cerrando D-01: nodos y pesos de la cuadratura, normalización de las proyecciones,
  si se estandarizan los embeddings, si las direcciones se remuestrean cada paso, y
  la forma exacta del término de invarianza del ejemplo mínimo. Sin esto, la
  implementación propia sería una conjetura. Cerrado leyendo `lejepa/univariate/
  epps_pulley.py`, `lejepa/multivariate/slicing.py` y `MINIMAL.md` del repositorio;
  confirma `λ=0,02`/`V=4` ya escritos en D-02/D-03, descarta una lectura previa
  (vía HTML de arXiv) que sugería `λ=0,05` y un esquema multi-crop — ver D-01.
- [X] T003 [P] Implementar `ssl/config.py`: `TrainConfig`, `EvalConfig`,
  `AugmentConfig`, serializables por completo a JSON, con semilla y dispositivo.

---

## Phase 2: User Story 1 — Objetivo propio y verificado (P1) 🎯 MVP

### Tests primero

- [X] T004 [P] [US1] `tests/ssl/test_objective.py`: formas y errores — `[V,B,D]`
  obligatorio, una forma distinta lanza `ValueError`; `invariance` vale 0 con vistas
  idénticas y crece con la dispersión; la pérdida es diferenciable y ningún término
  lleva `detach`.
- [X] T005 [P] [US1] Propiedades estadísticas de SIGReg: bajo con muestras
  `N(0, I)`; al menos un orden de magnitud mayor con embeddings colapsados en un
  punto y con embeddings en un subespacio de rango bajo; invariante a permutación de
  muestras y de dimensiones; estable al variar `num_slices`; reproducible con el
  mismo `generator`.
- [X] T006 [P] [US1] Gradientes: retropropagar sobre un módulo lineal de prueba y
  comprobar que todos los parámetros reciben gradiente finito, sin `NaN` ni `Inf`,
  también con lotes pequeños.
- [X] T007 [P] [US1] `tests/ssl/test_objective_vs_official.py`: la pérdida propia y la
  del paquete oficial coinciden sobre los mismos tensores dentro de la tolerancia
  declarada, con semilla fijada. **La tolerancia se justifica por escrito en el
  propio test.**
- [X] T008 [P] [US1] `tests/ssl/test_scaling.py`: tiempo y memoria del objetivo
  crecen linealmente con `N` medido en al menos cuatro tamaños de lote.

### Implementación

- [X] T009 [US1] Implementar `ssl/objective.py`: `epps_pulley_statistic`, `sigreg`,
  `invariance`, `lejepa_loss`, según contrato y según lo anotado en T002.
- [X] T010 [P] [US1] Implementar `ssl/diagnostics.py` y su test
  `tests/ssl/test_diagnostics.py` sobre casos conocidos (isótropo → rango efectivo
  cercano a `D`; colapsado → cercano a 1).
- [X] T011 [P] [US1] Implementar `ssl/schedules.py` y su test: función pura del paso,
  calentamiento lineal y descenso coseno hasta `base_lr/1000`.

**Checkpoint**: el objetivo está implementado y se sabe que es correcto, sin haber
gastado una sola época de cómputo.

---

## Phase 3: User Story 2 — Entrenamiento reanudable y agnóstico (P2)

- [X] T012 [US2] **Decidir y documentar la canalización de aumentaciones para
  mamografía** (D-06), justificando cada transformación incluida y cada una de las
  descartadas de la receta de imagen natural. Implementar `ssl/augment.py` con
  `describe`.
- [X] T013 [P] [US2] `tests/ssl/test_augment.py`: se producen `V` vistas distintas de
  la misma imagen; la salida tiene la forma y el rango esperados; `describe` refleja
  exactamente la configuración aplicada.
- [X] T014 [P] [US2] Implementar `ssl/encoders.py` con ResNet-50 registrado, y
  `ssl/projector.py` (MLP de tres capas).
- [X] T015 [US2] Implementar `ssl/data.py`: `CropDataset` sobre `catalog.parquet`,
  respeto estricto del `split`, filtro por fuente y por sospechosos, `collate` que
  agrupa `[V, B, ...]`.
- [X] T016 [P] [US2] `tests/ssl/test_data.py`: sólo aparecen filas del split pedido;
  cada elemento trae `V` vistas; el filtrado de sospechosos informa del recuento.
- [X] T017 [US2] Implementar `ssl/checkpoint.py` con guardado atómico y estado
  completo, incluidos los generadores aleatorios.
- [X] T018 [P] [US2] `tests/ssl/test_checkpoint.py`: entrenar 2 pasos, guardar,
  recargar y comprobar que el paso siguiente es idéntico al que se habría dado sin
  interrupción — incluida la secuencia de aumentaciones.
- [X] T019 [US2] Implementar `ssl/trainer.py`: bucle, AdamW, programador, precisión
  mixta degradable en un único punto, registro por época de la pérdida y sus dos
  términos, y del diagnóstico de embeddings.
- [X] T020 [US2] Aviso explícito cuando el lote efectivo esté por debajo del mínimo
  recomendado para la estimación de SIGReg (FR-018).
- [X] T021 [US2] Subcomando `pretrain` en `ssl/cli.py`, con detección automática de
  dispositivo. **Sin flag `--resume` aparte** (desviación documentada del texto
  original de la tarea): reanudar es automático en `train()` — relanzar con el
  mismo `--run-dir` ya continúa desde el último checkpoint válido, mismo patrón
  que `mammo-corpus build`/`mammo-etl run` en las features 002/004. Añadido el
  script `mammo-ssl` en `pyproject.toml`.
- [X] T022 [US2] Comprobación entre dispositivos: las primeras iteraciones en CPU y
  en MPS coinciden dentro de la tolerancia numérica, sin ramas de código distintas.
  `tests/ssl/test_device_agnostic.py`; tolerancia relativa del 15%, calibrada contra
  la desviación observada empíricamente entre backends (hasta ~3% sobre datos
  sintéticos) — no hay reproducibilidad bit a bit entre CPU y MPS, ni falta que hace.
- [X] T023 [US2] Ejecución de humo: 3 épocas sobre 500 imágenes reales del corpus
  (`data/corpus/catalog.parquet`, feature 002) en el portátil (MPS), interrumpir
  (simulado truncando checkpoints/historial, no cambiando `epochs` — cambiar
  `epochs` entre la interrupción y la reanudación cambia el horizonte del
  programador de tasa de aprendizaje y compara dos entrenamientos distintos, no
  una interrupción real) y reanudar. **Bug real encontrado y corregido**:
  `trainer.py` cargaba el checkpoint con `map_location=device`, que movía TODO el
  payload —incluido el estado de los generadores aleatorios, que debe seguir
  siendo un `ByteTensor` de CPU para `torch.set_rng_state`— al dispositivo de
  entrenamiento; en MPS esto lanzaba `TypeError: RNG state must be a
  torch.ByteTensor`. Corregido a `map_location="cpu"` siempre (`load_state_dict`
  ya mueve los pesos al dispositivo correcto porque el modelo ya está en él).
  Regresión cubierta en `test_resume_works_on_a_non_cpu_device`
  (`test_device_agnostic.py`). Tras la corrección: la reanudación real sobre datos
  reales completa la época 2 con `loss_total=0.2624` frente a `0.2635` de la
  ejecución sin interrumpir — próximos, no idénticos, por el mismo motivo que
  T022 (no determinismo de MPS entre llamadas, no un fallo de la reanudación; el
  test en CPU, `test_resume_reproduces_the_exact_continuation`, sí exige y logra
  igualdad exacta).

**Checkpoint**: hay un modelo entrenándose de verdad, y se puede parar y seguir.

---

## Phase 4: User Story 3 — Evaluación con líneas base (P3)

- [X] T024 [P] [US3] Implementar `ssl/probe.py`: sonda lineal sobre encoder
  congelado, con verificación explícita de ausencia de gradientes en el encoder.
- [X] T025 [P] [US3] `tests/ssl/test_probe.py`: ningún parámetro del encoder recibe
  gradiente durante la sonda; las particiones proceden del catálogo y no se
  reparticiona.
- [X] T026 [US3] Implementar `ssl/baselines.py`: encoder aleatorio congelado y
  encoder ImageNet congelado, bajo **el mismo** `EvalConfig`.
- [X] T027 [US3] Sondas de `classification`, `density` y `BIRADS`, con recuento de
  muestras y desagregación por `source_dataset`.
- [X] T028 [US3] Sonda de control sobre `source_dataset` (D-10), reportada siempre
  junto a las demás.
- [X] T029 [US3] Implementar `ssl/reporting.py`: tabla comparativa de las tres
  condiciones por tarea, historial de entrenamiento y registro de la ejecución.
- [X] T030 [US3] Subcomandos `probe`, `diagnose` y `compare`.
- [ ] T031 [US3] **Lectura crítica de los resultados**, por escrito en
  `runs/<run_id>/findings.md`: ¿supera LeJEPA a los pesos aleatorios? ¿supera a
  ImageNet congelado? ¿cuánto acierta la sonda de fuente? ¿se sostiene el rango
  efectivo? Un resultado negativo se registra igual.

---

## Phase 5: User Story 4 — Agnosticismo de arquitectura (P4)

- [ ] T032 [P] [US4] Registrar un ViT pequeño en `ssl/encoders.py` vía `timm`, con su
  decaimiento de peso propio (5e-2 frente a 5e-4 de ResNet).
- [ ] T033 [US4] `tests/ssl/test_encoder_agnostic.py`: el objetivo produce pérdida y
  gradientes con ambos encoders **sin ninguna modificación en `objective.py`**;
  además, todo encoder registrado cumple la firma `[B, D]`.
- [ ] T034 [US4] Entrenar el ViT con idéntica configuración salvo el encoder y
  añadirlo a la tabla comparativa.

---

## Phase 6: Cierre

- [ ] T035 [P] Ablación con y sin DDSM sobre el mismo protocolo, aprovechando el
  filtro por fuente (FR-012).
- [ ] T036 [P] Documentar en el README el flujo completo: corpus → preentrenamiento →
  sonda, con los comandos y la tabla de resultados.
- [ ] T037 `ruff format`, `ruff check --fix`, suite completa en verde.
- [ ] T038 Registrar en `plan.md` las respuestas obtenidas en T002, cerrando D-01 con
  los valores reales en lugar de la referencia bibliográfica.

---

## Dependencias

- Fase 1 → Fase 2 → Fase 3 → Fase 4 → Fase 5 → Fase 6.
- **T002 bloquea T009**: no se implementa el estadístico antes de haber leído la
  referencia.
- US1 es independiente del corpus: se puede completar entera sin que la feature 002
  esté terminada, lo que permite trabajar en paralelo.
- US3 depende de un checkpoint de US2.
- US4 depende de que US1 esté cerrada; su valor es demostrativo.

## Trazabilidad requisito → tarea

| FR | Tareas | FR | Tareas |
|---|---|---|---|
| FR-001 | T002, T009, T005 | FR-015 | T017, T018 |
| FR-002 | T009, T004 | FR-016 | T017, T018, T021 |
| FR-003 | T009 | FR-017 | T019 |
| FR-004 | T009, T033 | FR-018 | T020 |
| FR-005 | T004, T009 | FR-019 | T024, T025 |
| FR-006 | T007 | FR-020 | T027 |
| FR-007 | T008 | FR-021 | T027, T029 |
| FR-008 | T015, T016 | FR-022 | T026 |
| FR-009 | T012, T013 | FR-023 | T028 |
| FR-010 | T012, T029 | FR-024 | T015, T025 |
| FR-011 | T015, T016 | FR-025 | T010 |
| FR-012 | T015, T035 | FR-026 | T019 |
| FR-013 | T019, T022 | FR-027 | T019, T030 |
| FR-014 | T019, T032 | | |
