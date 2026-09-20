# Modelo de datos

**Feature**: 002-mammobench-corpus | **Fecha**: 2026-09-15

Coordenadas: origen arriba-izquierda, cajas semiabiertas `[x0, x1) × [y0, y1)` en
píxeles enteros del espacio de `Preprocessed_Dataset`.

## Entidades en memoria

### `ImagenMammoBench`

Una fila de `mammo-bench.csv`. Inmutable, es la unidad de trabajo.

| Campo | Tipo | Nota |
|---|---|---|
| `image_id` | `str` | derivado del nombre de fichero, p. ej. `inbreast_0` |
| `source_dataset` | `str` | `ddsm` \| `cmmd` \| `kau-bcmd` \| `cdd-cesm` \| `dmid` \| `inbreast` |
| `source_subject_id` | `str` | único **sólo dentro de su fuente** |
| `patient_key` | `str` | `f"{source_dataset}:{source_subject_id}"` — la clave real de paciente |
| `preprocessed_path`, `mask_path`, `raw_path` | `Path` | relativas a la raíz de Mammo-Bench |
| `laterality`, `view` | `str` | `L`/`R`, `CC`/`MLO`; `view` está vacío en una fila |
| `classification` | `str` | presente en las 19.731 filas |
| `density`, `birads`, `abnormality`, `molecular_subtype`, `subject_age` | `str \| None` | ausencia explícita, no cadena vacía |

### `CajaMamaria`

| Campo | Tipo | Significado |
|---|---|---|
| `x0`, `y0`, `x1`, `y1` | `int` | caja con margen ya aplicado y recortada a la imagen |
| `margin_px` | `int` | margen configurado |
| `source` | `BoxSource` | `MASK` \| `OTSU` |
| `threshold` | `float` | umbral de la máscara o el que eligió Otsu |
| `image_height`, `image_width` | `int` | dimensiones nativas de la imagen de entrada |
| `area_ratio` | `float` | área de la caja / área de la imagen |

Invariantes: `0 ≤ x0 < x1 ≤ image_width`, `0 ≤ y0 < y1 ≤ image_height`.

### `BoxSource`

Enumeración cerrada `MASK` / `OTSU`. El motivo del respaldo se guarda aparte en
`fallback_reason`: `MISSING_MASK`, `EMPTY_MASK`, `SATURATED_MASK`, `DEGENERATE_BOX`.

Máscara e imagen no siempre comparten resolución (sistemático en `ddsm`: 99 de 100
pares muestreados difieren de tamaño) — no se trata como máscara ausente, se realinea
con el vecino más próximo antes de umbralizar (`boxing.align_mask_to_image`, D-07).

### `RegistroDeRecorte`

Lo que un trabajador devuelve y lo que acaba siendo una fila del catálogo.

## Artefactos en disco

```text
data/
├── Mammo_Bench_v2/                      # entrada, sólo lectura
│   ├── CSV_Files/mammo-bench.csv        # 19.731 filas
│   ├── Preprocessed_Dataset/<fuente>/   # entrada del recorte
│   ├── Masks/<fuente>/                  # fuente primaria de la caja
│   └── Original_Dataset/<fuente>/       # respaldo de auditoría, no se toca
└── corpus/                              # salida de esta feature
    ├── crops/<fuente>/<image_id>.png
    ├── records.jsonl                    # incremental, append-only
    ├── catalog.parquet                  # consolidado, una fila por imagen
    ├── splits.parquet                   # patient_key -> split
    ├── runs.jsonl                       # una línea por ejecución
    └── qc/
        ├── quality_report.md
        ├── quality_by_source.parquet
        └── grid_<fuente>.png
```

### `catalog.parquet` — una fila por imagen

**Identidad**: `image_id`, `source_dataset`, `source_subject_id`, `patient_key`,
`laterality`, `view`.

**Resultado**: `status` (`ok` \| `failed`), `failure_category`, `error_message`,
`crop_path`, `crop_bytes`.

**Geometría**: `image_height`, `image_width`, `crop_x0`, `crop_y0`, `crop_x1`,
`crop_y1`, `crop_height`, `crop_width`, `crop_margin_px`, `crop_area_ratio`.

**Calidad**: `box_source`, `fallback_reason`, `threshold`, `mask_area_ratio`,
`mask_otsu_iou`, `suspect` (bool), `suspect_reason`.

**Etiquetas**: `classification`, `density`, `birads`, `abnormality`,
`molecular_subtype`, `subject_age`.

**Procedencia**: `run_id`, `code_version`, `processed_at` (UTC ISO-8601),
`process_seconds`.

**Partición**: `split` (`train` \| `val` \| `test`), incorporada al consolidar.

Clave de unicidad: `image_id`. Ante duplicados gana el `processed_at` más reciente.

### `splits.parquet`

`patient_key`, `source_dataset`, `split`, `n_images`, `seed`. Es el artefacto
auditable de la partición: permite comprobar la ausencia de fuga sin recorrer el
catálogo entero.

La estratificación por `classification` opera sobre esta clasificación reducida por
paciente, no sobre `catalog.parquet.classification` (que es por imagen): un paciente
con imágenes de más de un `classification` se estratifica por la más severa presente
—`Malignant` > `Suspicious Malignant` > `Benign` > `Normal`— (spec.md, FR-018,
Clarifications 2026-09-15; verificado sobre el CSV real: 535 de 5.860 pacientes,
9,1 %, tienen más de un valor). `splits.parquet` no persiste esta clasificación
reducida como columna aparte: es un valor intermedio de `assign_splits`, no un dato
de catálogo.

### `quality_by_source.parquet`

Una fila por fuente: `n_images`, `n_ok`, `n_failed`, `n_fallback_otsu`, `n_suspect`,
percentiles 5/25/50/75/95 de `crop_area_ratio` y de `mask_otsu_iou`, y mediana de la
resolución nativa y de la del recorte.

### `runs.jsonl`

`run_id`, `started_at`, `finished_at`, `command`, `code_version`, configuración
efectiva completa, `seed`, `n_total`, `n_ok`, `n_failed`, recuentos por categoría,
`exit_reason`.

## Notas sobre la fuente

- El CSV trae 25 columnas, de las que 7 están **completamente vacías** en las 19.731
  filas (`abnormality id`, `calc type`, `calc distribution`, `subtlety`,
  `mass shape`, `mass margins`, `cropped_image_file_new`). No se copian al catálogo.
- `ROI_path`, `x`, `y`, `radius` sólo tienen valor en unas 270 filas de `dmid`. Se
  copian tal cual, **sin remapear** a coordenadas del recorte: queda fuera del
  alcance de esta feature y el catálogo lo declara.
- Los campos ausentes vienen como cadena vacía; el catálogo los convierte a nulo
  explícito para que `null` signifique "no anotado" y no se confunda con un valor.
- `Original_Dataset` tiene 20.000 ficheros frente a las 19.731 filas: los 269
  sobrantes son ficheros `_ROI` de `dmid`. El manifiesto nunca lista directorios.
