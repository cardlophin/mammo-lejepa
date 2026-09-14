# mammo-lejepa

Pipeline de extracción, recorte y catalogación de [VinDr-Mammo](https://physionet.org/content/vindr-mammo/1.0.0/)
(PhysioNet v1.0.0) para preentrenamiento auto-supervisado sobre imagen mamaria.

Convierte los ~300 GB de DICOM del dataset en un corpus local de recortes
mamarios en PNG sin pérdida, catalogados junto con las anotaciones clínicas y
las bounding boxes de hallazgos trasladadas al sistema de coordenadas del
recorte — sin materializar nunca más de unos cientos de megabytes de DICOM
simultáneamente. Ver `specs/001-vindr-streaming-etl/` para la especificación,
el plan y las tareas completas.

## ⚠️ Datos clínicos y licencia

VinDr-Mammo se distribuye bajo el acuerdo de uso de datos de PhysioNet. Las
imágenes, los derivados (PNG, Parquet) y los CSV de anotaciones **no se
redistribuyen ni se suben a ningún servicio de terceros**. Todo `data/` está
fuera del control de versiones (`.gitignore`).

## Requisitos previos

```bash
uv sync
```

Necesitas una cuenta de PhysioNet con acceso concedido a VinDr-Mammo. Crea un
`.env` en la raíz del repositorio (nunca versionado):

```
PHYSIONET_USERNAME=tu_usuario
PHYSIONET_PASSWORD=tu_contraseña
PHYSIONET_SESSIONID=cookie_de_sesion
```

La cookie `sessionid` se copia del navegador con una sesión de PhysioNet
activa y acceso concedido a VinDr-Mammo. Caduca: si `mammo-etl doctor` falla
con un error de cookie, renuévala en el navegador y actualiza `.env`.

Los tres CSV de anotaciones (`breast-level_annotations.csv`,
`finding_annotations.csv`, `metadata.csv`) deben estar ya presentes en
`data/vindr-mammo/csv/`.

## Comandos

```bash
uv run mammo-etl doctor                              # comprueba el entorno, sin descargar nada
uv run mammo-etl run --split training --n-studies 5   # procesa una muestra
uv run mammo-etl inspect --n 8                        # lámina de control visual (SC-001, SC-006)
uv run mammo-etl run --split training                 # dataset completo, interrumpible con Ctrl-C
uv run mammo-etl resume                                # reanuda la última ejecución
uv run mammo-etl retry --failed                        # reintenta sólo lo fallido
uv run mammo-etl consolidate                            # JSONL -> images.parquet + findings.parquet
```

Ver `specs/001-vindr-streaming-etl/quickstart.md` para el flujo completo paso
a paso.

## Tests

```bash
uv run pytest                # toda la suite, sin red ni credenciales
uv run pytest tests/unit -q  # sólo lógica pura, < 30 s
```

## Esquema del catálogo consolidado

### `catalog/images.parquet` — una fila por imagen procesada

| Grupo | Campos |
|---|---|
| Identidad | `study_id`, `series_id`, `image_id` |
| Resultado | `status` (`ok`\|`failed`), `png_path`, `png_bytes`, `png_sha256`, `failure_category`, `error_message` |
| Etiquetas clínicas | `split`, `laterality`, `view_position`, `breast_birads`, `breast_density` |
| Geometría | `source_height`, `source_width`, `crop_x0`, `crop_y0`, `crop_x1`, `crop_y1`, `crop_height`, `crop_width`, `crop_margin_px`, `crop_area_ratio`, `crop_otsu_threshold`, `crop_scale` |
| Procedencia técnica | `photometric_interpretation`, `transfer_syntax_uid`, `window_center`, `window_width`, `pixel_spacing`, `manufacturer`, `model_name`, `normalize_low`, `normalize_high`, `inverted_monochrome1` |
| Ejecución | `run_id`, `pipeline_version`, `processed_at`, `download_seconds`, `process_seconds`, `dicom_bytes` |

Fila `ok`: coordenadas de recorte con sufijo de espacio explícito
(`x0_orig`→`crop_x0`, en el sistema de coordenadas del DICOM original, con
`margin_px` ya aplicado). Con `crop_x0/y0/x1/y1` y el PNG puede reconstruirse
cualquier coordenada original sin volver a descargar el DICOM.

### `catalog/findings.parquet` — una fila por hallazgo anotado

| Campo | Significado |
|---|---|
| `finding_id` | `image_id` + índice del hallazgo dentro de la imagen |
| `image_id`, `study_id` | claves de unión con `images.parquet` |
| `finding_categories` | lista de categorías del hallazgo |
| `finding_birads` | BI-RADS del hallazgo |
| `xmin_orig`…`ymax_orig` | caja en el espacio del DICOM original |
| `xmin_crop`…`ymax_crop` | caja trasladada al espacio del recorte |
| `fully_contained` | la caja original cabía entera en el recorte |
| `clipped` | la caja se intersectó con los límites del recorte |
| `area_orig`, `area_crop_space` | áreas, para detectar recortes patológicos |

Sólo contiene hallazgos de imágenes procesadas con éxito y con coordenadas
anotadas (la mayoría de filas de `finding_annotations.csv` no tienen lesión:
no generan fila aquí, no es un error). Un hallazgo cuya caja quede fuera del
recorte se conserva marcado, nunca se elimina.
# mammo-lejepa
