# Contratos de módulos

**Feature**: 002-mammobench-corpus

Los módulos marcados **PURO** no importan `cv2.imread`/`imwrite`, `pathlib`, `os` ni
nada de red, y no producen efectos observables fuera de su valor de retorno.

## `manifest.py` — PURO

```python
def build_manifest(
    catalog_csv: pl.DataFrame,
    *,
    sources: Sequence[str] | None = None,
    limit: int | None = None,
    image_ids: Sequence[str] | None = None,
) -> list[ImagenMammoBench]: ...


def missing_files(
    manifest: Sequence[ImagenMammoBench],
    existing: Container[str],
) -> list[tuple[ImagenMammoBench, str]]: ...
```

Contrato: `build_manifest` normaliza las cadenas vacías a `None`, deriva `image_id` y
`patient_key`, y produce un orden determinista. Lanza `ValueError` nombrando la
columna ausente. `missing_files` no toca el disco: recibe el conjunto de rutas
existentes y devuelve los pares (imagen, qué falta).

## `boxing.py` — PURO

```python
def align_mask_to_image(
    mask_uint8: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray: ...


def box_from_mask(
    mask_uint8: np.ndarray,
    *,
    threshold: int = 128,
    margin_px: int = 25,
) -> CajaMamaria: ...


def resolve_box(
    image_uint8: np.ndarray,
    mask_uint8: np.ndarray | None,
    *,
    config: BoxingParams,
) -> tuple[CajaMamaria, CajaMamaria | None, FallbackReason | None]: ...


def box_iou(a: CajaMamaria, b: CajaMamaria) -> float: ...
```

Contrato: `align_mask_to_image` reescala la máscara a la resolución de la imagen con
el vecino más próximo cuando difieren (D-07: sistemático en `ddsm`, no un caso raro);
es un no-op si ya coinciden. `resolve_box` la llama siempre antes de umbralizar, así
que la resolución no dispara por sí sola el respaldo a Otsu. `box_from_mask`
umbraliza, se queda con el mayor componente conexo, aplica el margen y recorta a los
límites; lanza `SegmentationError` si la máscara queda vacía tras umbralizar.
`resolve_box` devuelve la caja elegida, la caja de Otsu cuando ha podido calcularse
—siempre que haya máscara válida, para el IoU— y el motivo del respaldo cuando lo
haya. La política de respaldo está aquí y en ningún otro sitio. `box_iou` devuelve
0.0 para cajas disjuntas, nunca `None` ni `NaN`.

## `splits.py` — PURO

```python
_CLASSIFICATION_SEVERITY: tuple[str, ...] = (
    "Malignant",
    "Suspicious Malignant",
    "Benign",
    "Normal",
)


def patient_classification(classifications: Sequence[str]) -> str: ...


def assign_splits(
    records: Sequence[RegistroDeRecorte],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    stratify_by: Sequence[str] = ("source_dataset", "classification"),
) -> dict[str, str]: ...


def verify_no_patient_leakage(pairs: Iterable[tuple[str, str]]) -> None: ...
```

Contrato: la unidad de asignación es `patient_key`, jamás la imagen. Misma semilla,
misma asignación, con independencia del orden en que lleguen los registros.

`classification` es un atributo por imagen, no por paciente (spec.md, FR-018,
Clarifications 2026-09-15): antes de estratificar, `assign_splits` reduce las
clasificaciones de cada paciente a una sola llamando a `patient_classification`, que
devuelve la más severa presente según `_CLASSIFICATION_SEVERITY` (`Malignant` >
`Suspicious Malignant` > `Benign` > `Normal`). Lanza `ValueError` ante una
clasificación fuera de ese conjunto cerrado, en vez de ignorarla en silencio.

**Desviación documentada** (mismo patrón que `runner.run_build`): `verify_no_patient_
leakage` recibe `Iterable[tuple[patient_key, split]]`, no el `Mapping[str, str]` que
devuelve `assign_splits`. Un `dict` no puede tener dos valores para la misma clave, así
que validarlo directamente sería una tautología; `pairs` admite repeticiones —una por
fila del catálogo, por ejemplo— que es donde una fuga real podría colarse (T036: sobre
`catalog.parquet` ya construido, comprueba que ningún `patient_key` tiene más de un
`split` entre sus filas). Lanza `ValueError` nombrando el `patient_key` en conflicto.

## `quality.py` — PURO

```python
def summarize_by_source(records: Sequence[RegistroDeRecorte]) -> pl.DataFrame: ...


def is_suspect(
    record: RegistroDeRecorte, *, params: QualityParams
) -> tuple[bool, str | None]: ...
```

Contrato: `is_suspect` marca área fuera de `[0,10, 0,98]`, relación de aspecto fuera
de `[0,2, 5,0]`, IoU máscara-Otsu por debajo del umbral configurado, o caja que toca
los cuatro bordes. Devuelve el motivo además del indicador. **No filtra nada**:
decidir qué se hace con un caso sospechoso es del entrenamiento.

## `catalog.py` — E/S

```python
def consolidate(records: Iterable[RegistroDeRecorte]) -> pl.DataFrame: ...
def validate_catalog_coherence(catalog: pl.DataFrame) -> list[str]: ...
```

Contrato: `consolidate` deduplica por `image_id` quedándose con el `processed_at` más
reciente (FR-016, idempotente) y aplana `CajaMamaria` a columnas `crop_*`; no incluye
`split` — lo añade el subcomando `split` al escribir `catalog.parquet` (T035).
`validate_catalog_coherence` es E/S porque relee cada PNG `ok` del disco para
comprobar tamaño y dimensiones contra lo registrado (T037); devuelve la lista de
problemas, vacía si el catálogo es coherente.

## `image_io.py` — E/S

```python
def read_grayscale(path: Path) -> np.ndarray: ...
def read_mask(path: Path) -> np.ndarray | None: ...
```

Contrato: devuelven `uint8` en escala de grises. `read_mask` devuelve `None` si el
fichero no existe, en lugar de lanzar: la ausencia de máscara es un caso previsto que
dispara el respaldo, no un error.

## `worker.py` — proceso hijo

```python
def process_image(
    image: ImagenMammoBench, config: CorpusConfig
) -> RegistroDeRecorte: ...
```

Contrato: lee imagen y máscara, resuelve la caja, recorta, escribe el PNG de forma
atómica y devuelve el registro. **Nunca lanza al proceso padre**: todo fallo se
captura y se devuelve con su categoría. No escribe en el JSONL: el escritor es único
y vive en el padre.

## `runner.py` — orquestación

```python
def run_build(
    manifest: Sequence[ImagenMammoBench],
    *,
    config: CorpusConfig,
    run_id: str,
    command: str = "build",
    on_result: Callable[[RegistroDeRecorte], None] | None = None,
) -> RunSummary: ...
```

Contrato: recibe ya construido el manifiesto candidato —leer `mammo-bench.csv` y
aplicar `--source`/`--limit`/`--image-id` es responsabilidad de `cli.py`—, y lo
primero que hace es descontar de él las imágenes ya resueltas con éxito cuyo recorte
existe físicamente (FR-024), reutilizando `resume.pending_tasks` tal cual. Reparte lo
pendiente entre procesos, recoge registros conforme llegan, los escribe en el JSONL
con vaciado y mantiene el progreso (`on_result`, si se da, se invoca tras cada
imagen). Atiende `SIGINT`/`SIGTERM` dejando terminar el trabajo en vuelo, cancelando
el resto y devolviendo `exit_reason="interrupted"`; el JSONL queda siempre
consistente. Sin lógica de imagen: toda decisión sobre la caja está en `boxing.py`.
