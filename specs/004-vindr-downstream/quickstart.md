# Quickstart

**Feature**: 004-vindr-downstream

## 1. Requisitos previos

```bash
uv sync
```

Necesitas una cuenta de PhysioNet con acceso concedido a VinDr-Mammo. `.env` en la
raíz del repositorio (nunca versionado) debe tener:

```
PHYSIONET_USERNAME=tu_usuario
PHYSIONET_PASSWORD=tu_contraseña
PHYSIONET_SESSIONID=cookie_de_sesion
```

La cookie `sessionid` se copia del navegador con una sesión de PhysioNet activa y
acceso concedido. Caduca de forma independiente del usuario/contraseña.

## 2. Diagnóstico del entorno (sin descargar nada)

```bash
uv run mammo-etl doctor
```

Debe distinguir con claridad dos estados posibles antes de intentar nada más
(FR-022, SC-005):

- **CSV de anotaciones ausentes**: hay que descargarlos de PhysioNet a
  `data/vindr-mammo/csv/` antes de continuar.
- **Cookie de sesión caducada**: usuario y contraseña pueden ser correctos y aun así
  fallar; renueva la cookie desde el navegador y actualiza `.env`.

## 3. Primera ejecución de muestra

```bash
uv run mammo-etl run --split training --n-studies 5
```

Resultado esperado: un PNG por imagen descargada y procesada con éxito en
`data/vindr-mammo/images/processed/<study_id>/`, una línea en
`data/vindr-mammo/catalog/images.jsonl` por imagen, y un resumen en consola sin
ambigüedad sobre qué se procesó y qué falló.

## 4. Ejecución completa

```bash
uv run mammo-etl run
```

Interrumpible con `Ctrl-C`; `uv run mammo-etl resume` continúa donde se quedó sin
volver a descargar ni reprocesar lo ya resuelto con éxito (FR-023).

## 5. Consolidación y coherencia

```bash
uv run mammo-etl consolidate
```

Genera `data/vindr-mammo/catalog/images.parquet` y `findings.parquet` (FR-018),
remapeando las cajas de hallazgo al espacio del recorte (FR-016/FR-017), y valida
la coherencia del catálogo (FR-019): todo `status="ok"` tiene su PNG en disco con
el tamaño registrado, y toda caja de hallazgo remapeada cae dentro de las
dimensiones del recorte.

## 6. Verificación de partición

La partición (`split`) es la oficial de `breast-level_annotations.csv`, incorporada
a `images.parquet` al consolidar. Verificar la ausencia de fuga de estudio
(FR-021, SC-003):

```python
import polars as pl

images = pl.read_parquet("data/vindr-mammo/catalog/images.parquet")
by_split = images.group_by("study_id").agg(pl.col("split").n_unique().alias("n"))
assert by_split.filter(pl.col("n") > 1).height == 0
```

## 7. Verificación visual (obligatoria antes de fiarse del remapeado)

```bash
uv run mammo-etl inspect --n 20 --output data/vindr-mammo/catalog/qc_grid.png
```

Genera una lámina con el DICOM original, la caja detectada y el recorte con los
hallazgos remapeados dibujados encima (FR-028, SC-004). El DICOM ya se borró tras
procesarse con éxito; `inspect` lo vuelve a descargar temporalmente sólo para esta
verificación y lo elimina de nuevo al terminar.

## 8. Tests

```bash
uv run pytest tests/vindr -q          # suite de esta feature
uv run pytest tests/vindr -q -m "not slow"   # sin servidor HTTP local ni pipeline completo
uv run pytest                          # toda la suite del repositorio, ambas features
```

Ninguna prueba de `tests/vindr/` depende de la red real ni de credenciales de
PhysioNet: la descarga se prueba contra un servidor HTTP local sintético.
