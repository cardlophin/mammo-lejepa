# Contratos de módulos

**Feature**: 001-vindr-streaming-etl

Firmas normativas de la frontera entre módulos. La implementación puede cambiar por
dentro; estas firmas y sus contratos no, sin actualizar antes este documento. Los
módulos marcados **PURO** no importan `aiohttp`, `requests`, `os`, `pathlib` ni
`pydicom`, y no producen efectos observables fuera de su valor de retorno.

## `manifest.py` — PURO

```python
def build_manifest(
    annotations: pl.DataFrame,
    *,
    split: str | None = None,
    study_ids: Sequence[str] | None = None,
    n_studies: int | None = None,
) -> pl.DataFrame: ...


def batch_by_study(manifest: pl.DataFrame) -> list[list[ImageTask]]: ...


def count_metadata_discrepancies(
    metadata: pl.DataFrame,
    annotations: pl.DataFrame,
) -> tuple[int, int]: ...  # (only_in_metadata, only_in_annotations)
```

Contrato: `build_manifest` devuelve pares `(study_id, image_id)` únicos y ordenados de
forma determinista. Lanza `ValueError` si faltan columnas requeridas, si el split
solicitado no existe o si algún `study_id` pedido no está presente. `n_studies`
selecciona los primeros estudios del orden determinista, nunca una muestra aleatoria
sin semilla. Los lotes de `batch_by_study` respetan ese mismo orden y no asumen
ningún tamaño fijo.

`count_metadata_discrepancies` (FR-032) compara el conjunto de `image_id` de
`annotations` (`breast-level_annotations.csv`) contra el conjunto de
`SOP Instance UID` de `metadata` (`metadata.csv`; esa columna está repetida en la
cabecera del CSV con el mismo valor — se lee una sola vez, desduplicada) y devuelve
el recuento de discrepancias en cada sentido. No lanza excepción y no influye en el
resultado de `build_manifest`, que sigue usando exclusivamente `annotations` como
fuente del manifiesto (FR-001); es responsabilidad del llamador (`cli.py`, T028)
registrar el resultado como aviso.

## `windowing.py` — PURO

```python
def apply_display_window(
    pixels: np.ndarray,
    *,
    window_center: float | None,
    window_width: float | None,
    photometric_interpretation: str,
) -> np.ndarray: ...


def normalize_to_uint8(
    image: np.ndarray,
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> tuple[np.ndarray, float, float]: ...
```

Contrato: `apply_display_window` invierte MONOCHROME1 de modo que el tejido quede
siempre claro sobre fondo oscuro, y es la identidad cuando faltan los parámetros de
ventana. `normalize_to_uint8` devuelve la imagen en `uint8` junto con los dos valores
de corte empleados, que se registran en el catálogo; si los percentiles degeneran
(`high <= low`) recurre al mínimo y al máximo y jamás divide por cero.

## `segmentation.py` — PURO

```python
def detect_breast_box(
    image_uint8: np.ndarray,
    *,
    margin_px: int = 25,
    blur_kernel: int = 5,
    close_kernel_ratio: float = 0.006,
) -> BreastCrop: ...
```

Contrato: devuelve un `BreastCrop` con la caja ya expandida por el margen y recortada
a los límites de la imagen. Lanza `SegmentationError` cuando no existe ningún
componente conexo, cuando la máscara es degenerada o cuando la caja resultante tiene
área nula. No escribe nada, no dibuja nada y no depende del nombre del fichero. El
`area_ratio` se calcula siempre, también cuando es sospechosamente alto: filtrar es
decisión del consumidor, no de esta función.

## `geometry.py` — PURO

```python
def crop_image(image: np.ndarray, crop: BreastCrop) -> np.ndarray: ...


def to_crop_space(box: BoundingBox, crop: BreastCrop) -> BoundingBox: ...


def to_original_space(box: BoundingBox, crop: BreastCrop) -> BoundingBox: ...


def clip_to_crop(box: BoundingBox, crop: BreastCrop) -> tuple[BoundingBox, bool]: ...
```

Contrato: `to_crop_space` y `to_original_space` son inversas exactas para cualquier
punto contenido en el recorte —comprobación de ida y vuelta obligatoria en los
tests—. `to_crop_space` exige una caja en espacio `orig` y devuelve una en espacio
`crop`; recibir una caja en el espacio equivocado es un `ValueError`, no una
conversión silenciosa. `clip_to_crop` devuelve la caja intersectada y un indicador de
si hubo recorte; una intersección vacía devuelve una caja degenerada explícita, nunca
`None`.

## `findings.py` — PURO

```python
def parse_finding_categories(raw: str) -> list[str]: ...


def remap_findings(
    findings: pl.DataFrame,
    crops: Mapping[str, BreastCrop],
) -> pl.DataFrame: ...
```

Contrato: `parse_finding_categories` interpreta el literal de lista de Python del CSV
sin usar `eval`. `remap_findings` filtra primero a las filas con coordenadas y cuya
imagen tiene un `BreastCrop` conocido —las filas sin caja son imágenes sin lesión
anotada y no generan hallazgo (FR-023)—, y para cada una devuelve una fila con el
esquema de `findings.parquet` de `data-model.md` (`finding_id`, `image_id`,
`study_id`, categorías y BI-RADS del hallazgo, caja en `_orig` y en `_crop`,
`fully_contained`, `clipped`, `area_orig`, `area_crop_space`); los atributos por
imagen (lateralidad, proyección, etiquetas clínicas…) no se repiten aquí, se
recuperan uniendo por `image_id` con `images.parquet`. Un hallazgo cuya caja quede
parcial o totalmente fuera del recorte se conserva marcado, nunca se elimina
(FR-024).

## `resume.py` — PURO

```python
def parse_jsonl_records(lines: Iterable[str]) -> list[ImageRecord]: ...


def latest_records_by_image(
    records: Iterable[ImageRecord],
) -> dict[str, ImageRecord]: ...


def pending_tasks(
    manifest: Sequence[ImageTask],
    completed: Mapping[str, ImageRecord],
    existing_pngs: Container[str],
) -> list[ImageTask]: ...


def summarize_failures(
    records: Iterable[ImageRecord],
) -> dict[FailureCategory, int]: ...
```

Contrato: `parse_jsonl_records` recibe líneas ya leídas de disco por el llamador (no
abre ficheros: sigue siendo puro) y descarta sin excepción una última línea truncada;
cualquier otra línea malformada sí propaga `json.JSONDecodeError`, porque no se
explica por una interrupción normal. `latest_records_by_image` resuelve duplicados
quedándose con el `processed_at` más reciente (compartido por `catalog.py` para
FR-026). `pending_tasks`: una imagen sólo se considera resuelta si su registro más
reciente es `ok` **y** su `image_id` figura en `existing_pngs` (el conjunto de
`image_id` cuyo PNG existe físicamente, verificado por el llamador). Las imágenes
fallidas vuelven a la lista de pendientes salvo en modo reintento explícito. Ninguna
de estas funciones accede al disco.

## `catalog.py` — E/S (consolidación)

```python
def consolidate(
    records: Iterable[ImageRecord],
    finding_rows: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]: ...  # (images_df, findings_df)
```

Contrato: recibe ya parseados los `ImageRecord` de `images.jsonl` (vía
`resume.parse_jsonl_records`, leído por el llamador) y el `finding_annotations.csv`
crudo; no abre ficheros él mismo, aunque no se clasifica como puro porque su
propósito es preparar artefactos para escribir en disco. `images_df` sigue el
esquema plano de `images.parquet` de `data-model.md` (aplana `BreastCrop` con el
prefijo `crop_`), deduplicado por `image_id` con `resume.latest_records_by_image`
(FR-026). `findings_df` es el resultado de `findings.remap_findings` usando los
`BreastCrop` de las imágenes con `status="ok"` como mapa de recorte (FR-023). El
llamador (`cli.py consolidate`) escribe ambos con `pl.DataFrame.write_parquet`.

## `dicom_io.py` — E/S

```python
def read_dicom(path: Path) -> DicomPayload: ...
```

Contrato: devuelve los píxeles en su tipo nativo y las cabeceras de procedencia
(interpretación fotométrica, sintaxis de transferencia, ventana, espaciado de píxel,
fabricante, modelo). Traduce cualquier fallo de decodificación a `DecodeError`
nombrando la sintaxis de transferencia y el decodificador ausente cuando esa sea la
causa.

## `download.py` — E/S

```python
async def validate_access(
    session: aiohttp.ClientSession, config: PipelineConfig
) -> None: ...


async def download_dicom(
    session: aiohttp.ClientSession,
    task: ImageTask,
    destination: Path,
    config: PipelineConfig,
) -> DownloadResult: ...
```

Contrato: escritura atómica vía fichero temporal; un fichero con nombre definitivo es
siempre un fichero completo. Las respuestas 401, 403 y las de tipo `text/html` se
elevan como `AuthError`, que el orquestador trata como motivo de aborto global; los
errores transitorios se reintentan internamente con retroceso exponencial y jitter
antes de propagarse como `NetworkError`. Ningún mensaje de error incluye la cookie ni
las credenciales.

## `storage.py` — E/S

```python
def write_png_atomic(image: np.ndarray, destination: Path) -> int: ...
def append_record(record: ImageRecord, jsonl_path: Path) -> None: ...
def acquire_output_lock(directory: Path) -> AbstractContextManager[None]: ...
def quarantine(dicom_path: Path, quarantine_dir: Path) -> Path: ...
```

Contrato: `write_png_atomic` escribe en temporal, sincroniza y renombra, y devuelve
los bytes escritos. `append_record` escribe una única línea y vacía el búfer antes de
retornar: cuando la llamada retorna, el registro es duradero. `acquire_output_lock`
falla de inmediato si ya hay otra ejecución activa sobre el mismo directorio.

## `worker.py` — proceso hijo

```python
def process_dicom(
    task: ImageTask, dicom_path: Path, config: PipelineConfig
) -> ProcessOutcome: ...
```

Contrato: ejecuta lectura, ventana, normalización, detección, recorte y escritura del
PNG, y devuelve un `ProcessOutcome` serializable. **Nunca lanza excepciones al
proceso padre**: cualquier fallo se captura y se devuelve con su categoría y su
traza. No borra el DICOM: esa decisión pertenece al orquestador, que es quien sabe si
el registro ya está confirmado.

## `pipeline.py` — orquestación

```python
async def run_pipeline(
    tasks: Sequence[ImageTask],
    *,
    config: PipelineConfig,
    credentials: PhysioNetCredentials,
    run_id: str,
    command: str,
    on_result: Callable[[ProcessOutcome], None] | None = None,
    shutdown_event: asyncio.Event | None = None,
    split: str | None = None,
    n_studies: int | None = None,
) -> RunSummary: ...
```

Contrato: es el único punto que conoce a la vez la red, el pool de procesos y el
disco. Recibe ya construidos los `ImageTask` a procesar —construir el manifiesto,
aplicar la reanudación y decidir `run_id`/`command` es responsabilidad del llamador
(`cli.py`)—, de modo que `pipeline.py` no necesita leer CSV ni conocer los filtros de
alcance. Garantiza el orden PNG → registro → borrado (D-08) e impone la contrapresión
de la cola. `on_result`, si se da, se invoca tras cada imagen resuelta para la barra
de progreso (FR-029). `split`/`n_studies` sólo se copian tal cual al `RunSummary`
devuelto (el llamador es quien los conoce; `pipeline.py` no filtra con ellos). Si
`shutdown_event` se marca (p.ej. por un manejador de `SIGINT`/`SIGTERM` en `cli.py`),
los productores dejan de encolar nuevas descargas sin cancelar el trabajo en vuelo:
los consumidores siguen drenando la cola hasta vaciarla y escriben sus registros antes
de terminar con `exit_reason="interrupted"` (T037). No contiene lógica de negocio:
toda decisión sobre la imagen o sobre las coordenadas está
delegada en los módulos puros.
