# Modelo de datos

**Feature**: 004-vindr-downstream | **Fecha**: 2026-09-16

Coordenadas: origen arriba-izquierda, cajas semiabiertas `[x0, x1) × [y0, y1)`. Toda
caja declara su espacio de referencia (`orig` = DICOM original, `crop` = recorte).

## Entidades en memoria (`vindr/models.py`)

### `ImageTask`

Una imagen del manifiesto. Inmutable, no cambia durante la ejecución.

| Campo | Tipo | Nota |
|---|---|---|
| `study_id`, `series_id`, `image_id` | `str` | identidad; `study_id` es la unidad de partición |
| `split` | `str` | `training` \| `test`, oficial de `breast-level_annotations.csv` |
| `laterality`, `view_position` | `str` | `L`/`R`; `CC`/`MLO` |
| `breast_birads`, `breast_density` | `str` | etiquetas oficiales a nivel de mama |
| `expected_height`, `expected_width` | `int` | declaradas en el CSV; se contrastan contra el DICOM real |

### Caja mamaria — reutiliza `mammo_lejepa.models.CajaMamaria`

Ver `specs/002-mammobench-corpus/data-model.md`. Sin cambios: `x0, y0, x1, y1,
margin_px, source, threshold, image_height, image_width, area_ratio`. `source` es
siempre `OTSU` en esta feature — VinDr no trae máscara de segmentación, a diferencia
de Mammo-Bench (D-01 de research.md).

### `ProcessOutcome`

Lo que un trabajador devuelve tras procesar un DICOM ya descargado.

| Campo | Tipo | Nota |
|---|---|---|
| `status` | `"ok" \| "failed"` | |
| `breast_crop` | `CajaMamaria \| None` | |
| `photometric_interpretation`, `transfer_syntax_uid` | `str \| None` | procedencia técnica del DICOM |
| `window_center`, `window_width` | `float \| None` | ventana DICOM aplicada, si la había |
| `pixel_spacing` | `tuple[float, float] \| None` | mm/píxel, si el DICOM la declara |
| `manufacturer`, `model_name` | `str \| None` | del equipo — relevante para el riesgo de resolución nativa heterogénea (spec.md, Edge Cases) |
| `normalize_low`, `normalize_high` | `float \| None` | percentiles usados al normalizar a 8 bits |
| `inverted_monochrome1` | `bool` | si se invirtió por ser `MONOCHROME1` |
| `png_path`, `png_bytes`, `png_sha256` | `str \| int \| None` | del recorte, si tuvo éxito |
| `failure_category`, `error_message` | — | si falló |

### `ImageRecord`

Una fila de `images.jsonl`/`images.parquet`: une `ImageTask`, `ProcessOutcome`,
procedencia de la ejecución (`run_id`, `pipeline_version`, `processed_at`,
`download_seconds`, `process_seconds`, `dicom_bytes`).

### `DicomPayload`

Píxeles en su tipo nativo y cabeceras leídas de un DICOM (`rows`, `columns`,
`photometric_interpretation`, `transfer_syntax_uid`, `window_center`,
`window_width`, `pixel_spacing`, `manufacturer`, `model_name`).

### `DownloadResult`

Resultado de una descarga individual (`status`: `downloaded` \| `already_exists`,
`dicom_path`, `bytes_downloaded`, `download_seconds`).

### `RunSummary`

Resumen de una ejecución completa (una línea de `runs.jsonl`): `run_id`,
`started_at`, `finished_at`, `command`, `pipeline_version`, `split`, `n_studies`,
configuración de concurrencia efectiva, recuentos de éxito/fallo por categoría,
bytes descargados/escritos, `exit_reason`.

### `FailureCategory` (propio de `vindr/`)

`NETWORK`, `AUTH`, `DECODE`, `SEGMENTATION`, `WRITE`, `UNKNOWN` — distinto del
`FailureCategory` de Mammo-Bench (D-02 de research.md).

## Artefactos en disco

```text
data/vindr-mammo/
├── csv/
│   ├── breast-level_annotations.csv     # 20.000 filas — inventario, etiquetas, split oficial
│   ├── finding_annotations.csv          # 20.486 filas; sólo 2.254 con caja de hallazgo
│   └── metadata.csv                     # 20.000 filas — cabeceras DICOM, para detectar discrepancias
├── images/
│   ├── dicom/<study_id>/<image_id>.dicom       # efímero: existe sólo mientras se procesa
│   ├── quarantine/<study_id>/…                 # DICOM que fallaron el procesado
│   └── processed/<study_id>/<image_id>.png     # recorte, resolución NATIVA (D-04)
└── catalog/
    ├── images.jsonl              # incremental, append-only
    ├── images.parquet            # consolidado, una fila por imagen, incluye `split`
    ├── findings.parquet          # consolidado, una fila por hallazgo con coordenadas
    ├── runs.jsonl                # una línea por ejecución
    └── qc_grid.png               # lámina de verificación visual (FR-028)
```

### `images.parquet` — una fila por imagen

**Identidad**: `study_id`, `series_id`, `image_id`, `split`.

**Resultado**: `status`, `failure_category`, `error_message`, `png_path`,
`png_bytes`, `png_sha256`.

**Etiquetas clínicas**: `laterality`, `view_position`, `breast_birads`,
`breast_density`.

**Geometría**: `source_height`, `source_width`, `crop_x0`, `crop_y0`, `crop_x1`,
`crop_y1`, `crop_height`, `crop_width`, `crop_margin_px`, `crop_area_ratio`,
`crop_otsu_threshold`.

**Procedencia técnica**: `photometric_interpretation`, `transfer_syntax_uid`,
`window_center`, `window_width`, `pixel_spacing`, `manufacturer`, `model_name`,
`normalize_low`, `normalize_high`, `inverted_monochrome1`.

**Ejecución**: `run_id`, `pipeline_version`, `processed_at`, `download_seconds`,
`process_seconds`, `dicom_bytes`.

Clave de unicidad: `image_id`. Ante duplicados gana el `processed_at` más reciente
(idempotente, FR-018).

### `findings.parquet` — una fila por hallazgo con coordenadas

| Campo | Significado |
|---|---|
| `finding_id` | `image_id` + índice del hallazgo dentro de la imagen |
| `image_id`, `study_id` | claves de unión con `images.parquet` |
| `finding_categories` | lista de categorías del hallazgo |
| `finding_birads` | BI-RADS del hallazgo (distinto del BI-RADS a nivel de mama) |
| `xmin_orig`…`ymax_orig` | caja en el espacio del DICOM original |
| `xmin_crop`…`ymax_crop` | caja trasladada al espacio del recorte (resolución nativa, D-04) |
| `fully_contained` | la caja original cabía entera en el recorte |
| `clipped` | la caja se intersectó con los límites del recorte |
| `area_orig`, `area_crop_space` | áreas, para detectar recortes patológicos |

Sólo contiene hallazgos de imágenes procesadas con éxito y con coordenadas
anotadas: de las 20.486 filas de `finding_annotations.csv`, únicamente 2.254 traen
coordenadas (spec.md); el resto son imágenes sin lesión y no generan fila aquí, no
es un error (FR-016). Un hallazgo cuya caja quede fuera del recorte se conserva
marcado, nunca se elimina (FR-017).

### `runs.jsonl`

`run_id`, `started_at`, `finished_at`, `command`, `pipeline_version`,
configuración efectiva completa (`split`, `n_studies`, `downloads`, `workers`,
`queue_size`, `margin_px`), recuentos de éxito/fallo, bytes, `exit_reason`.

## Notas sobre la fuente

- `breast-level_annotations.csv` es la fuente de verdad del inventario y del split
  oficial: 20.000 filas, 20.000 `image_id` únicos, 5.000 estudios, exactamente 4
  imágenes por estudio, 16.000 en `training` y 4.000 en `test` (verificado sobre
  los ficheros originales en `specs/001-vindr-streaming-etl/spec.md`). El código no
  codifica el 4 como invariante.
- Todos los `image_id` de `finding_annotations.csv` existen en
  `breast-level_annotations.csv`.
- `metadata.csv` identifica la imagen por `SOP Instance UID`, no por `image_id`;
  ambos identifican lo mismo bajo un nombre de columna distinto
  (`count_metadata_discrepancies`).
