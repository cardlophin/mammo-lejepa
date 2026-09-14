# Phase 1 — Modelo de datos

**Feature**: 001-vindr-streaming-etl | **Fecha**: 2026-09-14

Convención de coordenadas para todo el documento: `(x, y)` con origen en la esquina
superior izquierda, `x` creciente hacia la derecha, `y` creciente hacia abajo. Las
cajas son semiabiertas `[x0, x1) × [y0, y1)`, expresadas en píxeles enteros. Todo
campo de coordenadas declara su espacio de referencia en el nombre: sufijo `_orig`
para el espacio del DICOM original y `_crop` para el espacio del recorte.

## Entidades en memoria (dataclasses inmutables, lógica pura)

### `ImageTask`

La unidad de trabajo. Se deriva del manifiesto y no cambia durante la ejecución.

| Campo | Tipo | Origen |
|---|---|---|
| `study_id` | `str` | `breast-level_annotations.csv` |
| `series_id` | `str` | `breast-level_annotations.csv` |
| `image_id` | `str` | `breast-level_annotations.csv` |
| `split` | `str` | `breast-level_annotations.csv` |
| `laterality` | `str` | `L` / `R` |
| `view_position` | `str` | `CC` / `MLO` |
| `breast_birads` | `str` | etiqueta a nivel de mama |
| `breast_density` | `str` | `DENSITY A`…`D` |
| `expected_height` | `int` | filas declaradas en el CSV |
| `expected_width` | `int` | columnas declaradas en el CSV |

### `BreastCrop`

El resultado geométrico de la detección. Es la entidad que hace reversible la
transformación: con ella y el PNG se puede reconstruir cualquier coordenada del
original sin volver a descargar el DICOM.

| Campo | Tipo | Significado |
|---|---|---|
| `x0_orig`, `y0_orig`, `x1_orig`, `y1_orig` | `int` | caja en el espacio del DICOM |
| `margin_px` | `int` | margen añadido a la caja ajustada |
| `otsu_threshold` | `float` | umbral seleccionado por Otsu |
| `source_height`, `source_width` | `int` | dimensiones reales del DICOM |
| `crop_height`, `crop_width` | `int` | dimensiones del recorte |
| `scale` | `float` | 1.0 en esta versión (sin redimensionado) |
| `area_ratio` | `float` | área del recorte / área original |

Invariantes: `0 ≤ x0 < x1 ≤ source_width`, `0 ≤ y0 < y1 ≤ source_height`,
`crop_width == x1_orig - x0_orig`, `crop_height == y1_orig - y0_orig`.

### `BoundingBox`

Caja genérica con el espacio de referencia explícito. Se usa tanto para la caja
mamaria como para los hallazgos.

| Campo | Tipo |
|---|---|
| `x0`, `y0`, `x1`, `y1` | `float` |
| `space` | `"orig"` \| `"crop"` |

### `ProcessOutcome`

Lo que un worker devuelve al proceso principal. Una de dos variantes: éxito con su
`BreastCrop` y las estadísticas de escritura, o fallo con su categoría y su traza.

Categorías de fallo (enumeración cerrada, base del resumen final y del reintento
selectivo): `NETWORK`, `AUTH`, `DECODE`, `SEGMENTATION`, `WRITE`, `UNKNOWN`.

## Artefactos en disco

```text
data/vindr-mammo/
├── csv/                                   # entrada, ya presente
│   ├── breast-level_annotations.csv       # 20.000 filas — inventario y etiquetas
│   ├── finding_annotations.csv            # 20.486 filas; sólo 2.254 con caja
│   └── metadata.csv                       # 20.000 filas — cabeceras DICOM
├── images/
│   ├── dicom/<study_id>/<image_id>.dicom  # efímero: existe segundos
│   ├── quarantine/<study_id>/…            # DICOM que fallaron el procesado
│   └── processed/<study_id>/<image_id>.png
└── catalog/
    ├── manifest_<run_id>.parquet          # inventario inmutable de la ejecución
    ├── images.jsonl                       # incremental, append-only
    ├── images.parquet                     # consolidado
    ├── findings.parquet                   # consolidado
    └── runs.jsonl                         # una línea por ejecución
```

### `images.jsonl` / `images.parquet` — una fila por imagen resuelta

**Identidad**: `image_id`, `study_id`, `series_id`.

**Resultado**: `status` (`ok` \| `failed`), `failure_category`, `error_message`
(truncado y sin credenciales), `png_path` (relativa a la raíz del repositorio),
`png_bytes`, `png_sha256`.

**Etiquetas clínicas**: `split`, `laterality`, `view_position`, `breast_birads`,
`breast_density`.

**Geometría**: en `images.jsonl` viaja como un objeto anidado `breast_crop`
(los campos de `BreastCrop`: `x0_orig`, `y0_orig`, `x1_orig`, `y1_orig`,
`margin_px`, `otsu_threshold`, `source_height`, `source_width`, `crop_height`,
`crop_width`, `area_ratio`, `scale`; `null` si `status="failed"`), que la
consolidación (`catalog.py`) aplana en `images.parquet` con el prefijo
`crop_`: `crop_x0`, `crop_y0`, `crop_x1`, `crop_y1`, `crop_height`,
`crop_width`, `crop_margin_px`, `crop_area_ratio`, `crop_otsu_threshold`,
`crop_scale` (además de `source_height`/`source_width` sin prefijo).

**Procedencia técnica**: `photometric_interpretation`, `transfer_syntax_uid`,
`window_center`, `window_width`, `pixel_spacing`, `manufacturer`, `model_name`,
`normalize_low`, `normalize_high`, `inverted_monochrome1` (bool).

**Ejecución**: `run_id`, `pipeline_version`, `processed_at` (UTC, ISO-8601),
`download_seconds`, `process_seconds`, `dicom_bytes`.

Clave de unicidad tras la consolidación: `image_id`. Ante duplicados gana el
`processed_at` más reciente.

### `findings.parquet` — una fila por hallazgo anotado

Se deriva de `finding_annotations.csv` unido por `image_id` con el catálogo de
imágenes, y sólo contiene hallazgos de imágenes procesadas con éxito. De las 20.486
filas del CSV, únicamente 2.254 traen coordenadas: el resto corresponde a imágenes
sin lesión anotada y no genera fila aquí. Hay 11 categorías de hallazgo distintas.

| Campo | Significado |
|---|---|
| `finding_id` | identificador estable: `image_id` + índice del hallazgo |
| `image_id`, `study_id` | claves de unión |
| `finding_categories` | lista de categorías (el CSV la trae como literal de lista Python: requiere parseo, no `eval`) |
| `finding_birads` | BI-RADS del hallazgo |
| `xmin_orig`, `ymin_orig`, `xmax_orig`, `ymax_orig` | caja original (float en el CSV) |
| `xmin_crop`, `ymin_crop`, `xmax_crop`, `ymax_crop` | caja trasladada al recorte |
| `fully_contained` | la caja original cabía entera en el recorte |
| `clipped` | la caja se intersectó con los límites del recorte |
| `area_orig`, `area_crop_space` | áreas, para detectar recortes patológicos |

Un hallazgo cuya intersección con el recorte sea vacía se conserva con
`fully_contained = false`, `clipped = true` y coordenadas nulas: es una señal de que
la detección de la caja mamaria falló para esa imagen, y perderlo silenciosamente
ocultaría el defecto.

### `runs.jsonl` — una línea por ejecución

`run_id`, `started_at`, `finished_at`, `command`, `pipeline_version`, parámetros
efectivos (split, número de estudios, concurrencia, margen, tamaño de cola),
`images_total`, `images_ok`, `images_failed`, recuentos por categoría de fallo,
`bytes_downloaded`, `bytes_written`, `exit_reason`.

## Transformaciones

**Original → recorte**: `x_crop = x_orig - crop_x0`, `y_crop = y_orig - crop_y0`,
seguido de intersección con `[0, crop_width) × [0, crop_height)`.

**Recorte → original**: `x_orig = x_crop + crop_x0`, `y_orig = y_crop + crop_y0`.

La composición de ambas es la identidad para cualquier punto contenido en el recorte:
esta propiedad es la comprobación de ida y vuelta exigida por los tests.

## Notas sobre las fuentes

- `breast-level_annotations.csv` tiene 20.000 filas de datos y es el inventario
  autoritativo: 5.000 estudios con exactamente cuatro imágenes cada uno en los
  ficheros actuales, pero el código no codifica ese número como invariante.
- `finding_annotations.csv` tiene una fila por imagen, no por lesión: 20.486 filas
  para 20.000 imágenes, de las que 2.254 llevan coordenadas. Las filas sin caja son
  imágenes sin lesión anotada.
- `metadata.csv` repite la columna `SOP Instance UID` dos veces en su cabecera; hay
  que leerla con nombres de columna explícitos o desduplicarlos al cargarla.
- `finding_annotations.csv` trae `finding_categories` como cadena con formato de
  lista de Python (`"['Mass']"`); se parsea con `ast.literal_eval` o con una
  expresión regular, nunca con `eval`.
- Las dimensiones declaradas en los CSV (`height`, `width`) deben contrastarse con
  las reales del DICOM; una discrepancia invalida el remapeo de coordenadas y debe
  registrarse como fallo, no corregirse en silencio.
