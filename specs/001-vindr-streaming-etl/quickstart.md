# Quickstart

**Feature**: 001-vindr-streaming-etl

## 1. Requisitos previos

```bash
uv sync
```

Dependencias que hay que añadir a las ya presentes en `pyproject.toml`:
`pylibjpeg`, `pylibjpeg-openjpeg` (decodificación JPEG 2000), `pytest`, `rich` y la
biblioteca de CLI elegida.

`.env` en la raíz del repositorio, nunca versionado:

```
PHYSIONET_USERNAME=...
PHYSIONET_PASSWORD=...
PHYSIONET_SESSIONID=...
```

La cookie `sessionid` se copia del navegador con una sesión de PhysioNet activa y
acceso concedido a VinDr-Mammo. Caduca: si el `doctor` falla con 403, hay que
renovarla.

## 2. Comprobación del entorno

```bash
uv run mammo-etl doctor
```

Verifica, en este orden y sin descargar ninguna imagen: presencia de los tres CSV,
integridad del manifiesto, disponibilidad del decodificador JPEG 2000, escritura en
los directorios de salida, validez de la cookie y espacio libre frente al techo de
disco calculado.

## 3. Primera ejecución de muestra

```bash
uv run mammo-etl run --split training --n-studies 5
```

Resultado esperado: un PNG por imagen en
`data/vindr-mammo/images/processed/<study_id>/`, una línea por imagen en
`data/vindr-mammo/catalog/images.jsonl`, `data/vindr-mammo/images/dicom/` vacío al
terminar y un resumen en consola sin fallos.

## 4. Verificación visual (obligatoria antes de escalar)

```bash
uv run mammo-etl inspect --n 8 --output data/vindr-mammo/catalog/qc_grid.png
```

Genera una lámina con el DICOM original, la caja detectada y el recorte, más las
cajas de hallazgos remapeadas cuando existan. Es la comprobación de los criterios
SC-001 y SC-006: si las cajas de hallazgo no caen sobre la lesión, el remapeo está
mal y no se debe escalar.

## 5. Ejecución completa

```bash
uv run mammo-etl run --split training --downloads 6 --workers 4 --queue-size 12
```

Interrumpible con `Ctrl-C`. Para continuar:

```bash
uv run mammo-etl resume
```

## 6. Fallos y consolidación

```bash
uv run mammo-etl retry --failed          # reintenta sólo lo fallido
uv run mammo-etl consolidate             # JSONL -> images.parquet + findings.parquet
```

## 7. Tests

```bash
uv run pytest                            # toda la suite, sin red ni credenciales
uv run pytest tests/unit -q              # sólo lógica pura, < 30 s
```
