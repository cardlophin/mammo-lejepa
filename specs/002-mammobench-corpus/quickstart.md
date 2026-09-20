# Quickstart

**Feature**: 002-mammobench-corpus

## 1. Requisitos previos

```bash
uv sync
```

No hace falta ninguna credencial: el corpus es local. Comprueba que
`data/Mammo_Bench_v2/` está completo (19.731 filas en
`CSV_Files/mammo-bench.csv`, `Preprocessed_Dataset/` y `Masks/` presentes).

## 2. Primera ejecución de muestra

```bash
uv run mammo-corpus build --source inbreast --limit 8
```

Resultado esperado: un PNG por imagen en `data/corpus/crops/inbreast/`, una línea por
imagen en `data/corpus/records.jsonl` con `box_source` (`mask` u `otsu`) y
`mask_otsu_iou`, y un resumen en consola sin fallos.

## 3. Verificación visual (obligatoria antes de escalar)

```bash
uv run mammo-corpus inspect --source inbreast --n 8
```

Genera `data/corpus/qc/grid_inbreast.png`: una lámina con la imagen original, la
máscara, la caja dibujada y el recorte. Repite por cada una de las seis fuentes antes
de dar el corpus por bueno — es la comprobación de SC-004 y del riesgo "las máscaras
no eliminan el pectoral" (plan.md).

## 4. Ejecución completa

```bash
uv run mammo-corpus build --workers 8
```

Interrumpible con `Ctrl-C`; relanzar el mismo comando continúa donde se quedó sin
reprocesar lo ya hecho (FR-024). Con 8 procesos, las 19.731 imágenes deben terminar
en menos de 60 minutos (SC-006).

## 5. Control de calidad

```bash
uv run mammo-corpus qc
```

Genera `data/corpus/qc/quality_report.md` y `quality_by_source.parquet`: por fuente,
percentiles de `crop_area_ratio` y de `mask_otsu_iou`, recuento de respaldos a Otsu y
lista de casos sospechosos por ruta (FR-021).

**Puerta manual obligatoria** (tasks.md T031): inspecciona las seis láminas del paso
3 y responde por escrito en `data/corpus/qc/findings.md` si las máscaras excluyen
realmente el pectoral en cada fuente y qué está pasando con las máscaras de
`cdd-cesm` que cubren el 80 % de la imagen. No continúes al paso 6 sin esa respuesta.

## 6. Catálogo y particiones

```bash
uv run mammo-corpus consolidate   # records.jsonl -> catalog.parquet
uv run mammo-corpus split --seed 0 --ratios 0.8 0.1 0.1
```

`split` asigna train/val/test por `patient_key` (`source_dataset` +
`source_subject_id`), estratificando por fuente y por la clasificación más severa de
cada paciente (spec.md, Clarifications 2026-09-15), y escribe `splits.parquet`.
Verifica automáticamente que ningún paciente aparece en más de una partición
(FR-019, SC-005).

## 7. Tests

```bash
uv run pytest                # toda la suite, sin disco real ni GPU
uv run pytest tests/unit -q  # sólo lógica pura, < 30 s (SC-007)
```
