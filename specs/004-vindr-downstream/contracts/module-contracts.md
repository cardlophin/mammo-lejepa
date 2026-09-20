# Contratos de módulos

**Feature**: 004-vindr-downstream

Los módulos marcados **PURO** no importan red, `pathlib`/`os` para E/S, ni producen
efectos observables fuera de su valor de retorno. Los módulos ya existentes que se
reutilizan sin cambios (`geometry.py`, `segmentation.py`, `storage.py`, `resume.py`,
`errors.py` compartido) están documentados en
`specs/002-mammobench-corpus/contracts/module-contracts.md`; aquí sólo se listan las
diferencias de uso y los módulos nuevos de `vindr/`.

## `vindr/manifest.py` — PURO

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
) -> tuple[int, int]: ...
```

Contrato: `build_manifest` lanza `ValueError` nombrando la columna ausente, el split
inexistente o los `study_id` no encontrados. La selección por `n_studies` toma
siempre los primeros de un orden determinista (`study_id` ordenado), nunca una
muestra aleatoria. `batch_by_study` agrupa sin asumir un número fijo de imágenes por
estudio. `count_metadata_discrepancies` no lanza y no modifica el manifiesto: sólo
cuenta.

## `vindr/windowing.py` — PURO

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

Contrato: `apply_display_window` es la identidad si falta `window_center` o
`window_width`; invierte siempre `MONOCHROME1` para que el tejido quede claro sobre
fondo oscuro, igual que `MONOCHROME2`. `normalize_to_uint8` nunca divide por cero:
si los percentiles degeneran recurre al mínimo/máximo, y si aun así la imagen es
constante devuelve ceros.

## `vindr/dicom_io.py` — E/S

```python
def read_dicom(path: Path) -> DicomPayload: ...
```

Contrato: lanza `DecodeError` (reexportado del `errors.py` compartido) nombrando la
sintaxis de transferencia y el decodificador ausente si `pixel_array` falla, en vez
de dejar pasar la excepción opaca de `pydicom`.

## `vindr/download.py` — E/S

```python
def build_cookie_jar(session_id: str, base_url: str) -> aiohttp.CookieJar: ...
def build_client_session(
    credentials: PhysioNetCredentials, config: PipelineConfig
) -> aiohttp.ClientSession: ...
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

Contrato: `validate_access` y `download_dicom` lanzan `AuthError` ante HTTP
401/403 o una respuesta HTML (sesión caducada) — **sin reintentar**— y `NetworkError`
ante fallos transitorios (429/500/502/503/504, errores de conexión) tras agotar los
reintentos con retroceso exponencial y `Retry-After`. `download_dicom` es atómico:
escribe en un `.part` y renombra al terminar; si el destino ya existe con tamaño no
nulo, no vuelve a descargar (`status="already_exists"`).

## `vindr/worker.py` — proceso hijo

```python
def process_dicom(
    task: ImageTask, dicom_path: Path, config: PipelineConfig
) -> ProcessOutcome: ...
```

Contrato: encadena `read_dicom` → `apply_display_window` → `normalize_to_uint8` →
`detect_breast_box` (`segmentation.py` compartido, D-01) → `crop_image`
(`geometry.py` compartido) → `write_png_atomic` (`storage.py` compartido). Lanza
`DecodeError` explícito si las dimensiones declaradas en `ImageTask` no coinciden
con las reales del DICOM. **Nunca lanza al proceso padre**: todo fallo se captura y
se devuelve como `ProcessOutcome` fallido con su categoría (`vindr.errors.
classify_exception`).

## `vindr/pipeline.py` — orquestación

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

Contrato: productores `asyncio` (descarga, `config.downloads` concurrentes)
alimentan una cola acotada (`config.queue_size`) que consumen procesos de un
`ProcessPoolExecutor` (`config.workers`) — D-06 de research.md. Garantiza el orden
PNG → registro JSONL → borrado del DICOM local (o cuarentena si falló). Si
`shutdown_event` se marca, los productores dejan de encolar nuevo trabajo pero los
consumidores drenan lo que ya está en la cola antes de terminar con
`exit_reason="interrupted"`; el JSONL queda siempre consistente. Si algún productor
encuentra un `AuthError`, se propaga tras drenar — el trabajo ya completado no se
pierde.

## `vindr/findings.py` — PURO

```python
def parse_finding_categories(raw: str | None) -> list[str]: ...


def remap_findings(
    findings: pl.DataFrame,
    crops: Mapping[str, CajaMamaria],
) -> pl.DataFrame: ...
```

Contrato: idéntico en comportamiento al de la extinta feature 001 (FR-016/FR-017);
sólo cambia el tipo de `crops` a `CajaMamaria` (D-01). `remap_findings` usa
`to_crop_space`/`clip_to_crop` del `geometry.py` compartido sin modificarlos. Sólo
las filas con las cuatro coordenadas y cuya imagen tiene una caja conocida
(procesada con éxito) generan fila; el resto se descarta aquí, contabilizado por el
llamador. Un hallazgo clippeado o totalmente fuera del recorte se conserva marcado.

## `vindr/catalog.py` — E/S

```python
def parse_jsonl_records(lines: Iterable[str]) -> list[ImageRecord]: ...


def consolidate(
    records: Iterable[ImageRecord],
    finding_rows: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]: ...


def verify_no_study_leakage(images_df: pl.DataFrame) -> None: ...


def validate_catalog_coherence(
    images_df: pl.DataFrame, findings_df: pl.DataFrame
) -> list[str]: ...
```

**Desviación descubierta durante la implementación** (no prevista en el plan
inicial): `mammo_lejepa.resume.parse_jsonl_records` reconstruye
`RegistroDeRecorte`/`CajaMamaria` de forma hardcodeada en `_record_from_dict` —no
es genérica por duck-typing como el resto de `resume.py`—, así que no sirve para
deserializar `ImageRecord` de VinDr. `vindr/catalog.py` define su propia versión.
`pending_tasks`, `latest_records_by_image` y `summarize_failures` de
`mammo_lejepa.resume` sí son genéricas (sólo acceden a `.image_id`/`.status`/
`.processed_at`/`.failure_category`) y se reutilizan sin cambios ni copia.

Contrato: `consolidate` deduplica por `image_id` quedándose con el `processed_at`
más reciente (FR-018, idempotente) y aplana la `CajaMamaria` de cada imagen exitosa
a columnas `crop_*`. `verify_no_study_leakage` agrupa por `study_id` y lanza
`ValueError` nombrándolo si dos imágenes del mismo estudio declaran particiones
distintas (FR-021). `validate_catalog_coherence` relee cada PNG `ok` del disco para
comprobar tamaño contra lo registrado, y comprueba que toda caja de hallazgo
remapeada cae dentro de las dimensiones del recorte que declara la imagen
(FR-019); devuelve la lista de problemas, vacía si el catálogo es coherente.

## `vindr/errors.py` — E/S (define excepciones, sin tocar disco)

```python
class VindrError(Exception): ...


class AuthError(VindrError): ...


class NetworkError(VindrError): ...


# DecodeError, SegmentationError, WriteError: reexportados de mammo_lejepa.errors


def classify_exception(error: Exception) -> FailureCategory: ...


def redact_secrets(text: str, *secrets: str | None) -> str: ...
```

Contrato: `classify_exception` reconoce `AuthError`→`AUTH`, `NetworkError`→
`NETWORK`, y delega en el mapeo compartido para `DecodeError`/`SegmentationError`/
`WriteError`; cualquier otra excepción se clasifica `UNKNOWN`. `redact_secrets`
sustituye cualquier aparición literal de un secreto por un marcador — segunda capa
de defensa aplicada antes de `append_record`, por si una librería externa incluyera
texto inesperado en una excepción (FR-026).

## `vindr/config.py` — E/S

```python
@dataclass(frozen=True, slots=True)
class PhysioNetCredentials:
    username: str
    password: str
    session_id: str


def load_credentials() -> PhysioNetCredentials: ...


@dataclass(frozen=True, slots=True)
class PipelineConfig: ...  # ver data-model.md
```

Contrato: `load_credentials` lee `.env` y el entorno; lanza `RuntimeError` nombrando
la variable ausente (`PHYSIONET_USERNAME`/`PHYSIONET_PASSWORD`/
`PHYSIONET_SESSIONID`) sin revelar ningún valor cargado (FR-026). La cookie
caducada no se detecta aquí —la ausencia de la variable y la caducidad de su
contenido son fallos distintos, FR-022— sino en `download.validate_access`, que sí
hace una petición real.

## `vindr/cli.py` — E/S (comando `mammo-etl`)

Subcomandos: `doctor` (FR-022), `run` (FR-003 a FR-013), `resume` (FR-023), `retry`
(FR-024), `consolidate` (FR-018/FR-019), `inspect` (FR-028). `doctor` comprueba, en
este orden y sin descargar ninguna imagen: presencia de los tres CSV, decodificador
JPEG 2000 funcional, permisos de escritura, espacio libre frente al techo de disco
calculado, credenciales cargadas, y validez de la cookie con una petición real —
cada comprobación se reporta de forma independiente, así que un CSV ausente y una
cookie caducada nunca se confunden en el mismo mensaje (FR-022, SC-005).
