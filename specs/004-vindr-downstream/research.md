# Phase 0 — Investigación y decisiones técnicas

**Feature**: 004-vindr-downstream

## D-01 — `vindr/` adopta `CajaMamaria`, no reinventa `BreastCrop`

**Decisión**: el subpaquete `vindr/` usa `mammo_lejepa.models.CajaMamaria` (y
`mammo_lejepa.models.BoundingBox` para las cajas de hallazgo) como su propia
representación de la caja mamaria, en vez de restaurar el `BreastCrop` de la
extinta feature 001.

**Justificación**: verificado por diff contra el commit `0de1617`, `geometry.py` y
`segmentation.py` ya no operan sobre `BreastCrop` — se reescribieron para
Mammo-Bench y ahora construyen y consumen `CajaMamaria` directamente
(`detect_breast_box` ya devuelve una `CajaMamaria`; `crop_image`,
`to_crop_space`, `to_original_space` y `clip_to_crop` ya esperan sus campos
`x0/y0/x1/y1/image_height/image_width`, no los de `BreastCrop`
—`x0_orig/y0_orig/source_height/source_width/crop_width/crop_height`—). Restaurar
`BreastCrop` obligaría a: (a) mantener una segunda copia de `geometry.py`/
`segmentation.py` dentro de `vindr/`, violando "sin duplicar", o (b) tocar los
módulos compartidos de Mammo-Bench para que acepten ambos tipos, violando el
aislamiento entre features que motiva tener un subpaquete en primer lugar. Adoptar
`CajaMamaria` tal cual permite llamar a `detect_breast_box`/`crop_image`/
`to_crop_space`/`clip_to_crop` sin modificarlos ni una línea.

**Consecuencia**: `CajaMamaria` no tiene el campo `scale` que sí tenía `BreastCrop`
(por defecto `1.0`, nunca usado por ninguna función de `geometry.py` en ninguna de
las dos versiones — vestigial). Se pierde sin sustituto; no hacía nada.
`vindr/catalog.py` aplana `crop.threshold` (antes `otsu_threshold`) y
`crop.image_height`/`crop.image_width` (antes `source_height`/`source_width`) en
vez de sus nombres originales.

**Alternativas descartadas**: mantener `BreastCrop` con su propio `geometry.py`
duplicado dentro de `vindr/` — descartado por duplicar ~80 líneas de lógica de
coordenadas ya probada, con el riesgo de que diverjan silenciosamente con el
tiempo.

## D-02 — `errors.py` compartido no cambia; lo específico de red vive en `vindr/errors.py`

**Decisión**: `mammo_lejepa/errors.py` (Mammo-Bench) no se modifica. `vindr/errors.py`
define `AuthError`, `NetworkError`, `redact_secrets` y su propio `classify_exception`
—que mapea a un `FailureCategory` propio de `vindr/models.py` con seis valores
(`NETWORK`, `AUTH`, `DECODE`, `SEGMENTATION`, `WRITE`, `UNKNOWN`)—, reexportando
`DecodeError`, `SegmentationError` y `WriteError` desde el `errors.py` compartido en
vez de duplicarlos.

**Justificación**: verificado por diff, el `errors.py` compartido perdió
`AuthError`/`NetworkError`/`redact_secrets` al reescribirse para Mammo-Bench (que no
tiene red), y su `FailureCategory` (en `mammo_lejepa.models`) sólo tiene
`DECODE`/`SEGMENTATION`/`WRITE`/`UNKNOWN` — sin `AUTH`/`NETWORK`. Añadir esos dos
valores al `FailureCategory` compartido filtraría un concepto —fallo de
autenticación/red— que Mammo-Bench, al ser puramente local, no puede producir nunca:
ensancharía la superficie de un tipo compartido por un caso que sólo una de las dos
features puede dar. `DecodeError`/`SegmentationError`/`WriteError` sí significan
exactamente lo mismo en ambos dominios (fallo al decodificar una imagen, al
segmentar la mama, al escribir un derivado), así que se reutilizan literalmente, sin
redefinirlos.

**Alternativas descartadas**: extender el `FailureCategory` compartido con
`AUTH`/`NETWORK` — descartado por lo anterior (ensancha un tipo de Mammo-Bench con
un concepto que nunca puede darse en Mammo-Bench). Duplicar
`DecodeError`/`SegmentationError`/`WriteError` dentro de `vindr/errors.py` —
descartado por "sin duplicar": son el mismo concepto, no un parecido superficial.

## D-03 — `quarantine()` vive en `vindr/storage.py`, no vuelve al `storage.py` compartido

**Decisión**: `mammo_lejepa/storage.py` no se modifica. `vindr/storage.py` es un
módulo nuevo y pequeño con la única función `quarantine(dicom_path, quarantine_dir)`.

**Justificación**: verificado por diff, `quarantine()` se eliminó del `storage.py`
compartido durante la feature 002 por ser código muerto específico de DICOM (mueve
un fichero de entrada que falló a un directorio de cuarentena; Mammo-Bench no
descarga nada, no tiene nada que poner en cuarentena). El resto de `storage.py`
(`write_png_atomic`, `append_record`, `append_run_summary`, `acquire_output_lock`,
`clean_orphan_part_files`) son utilidades de fichero genéricas que no mencionan
ningún tipo de ninguna de las dos features en su lógica (sólo en una anotación de
tipo de `append_record`, que Python no comprueba en tiempo de ejecución) y se
reutilizan sin cambios.

## D-04 — El recorte se guarda a resolución nativa, nunca reescalado

**Decisión**: `vindr/worker.py` guarda el PNG recortado a la resolución nativa del
DICOM de origen (menos el fondo descartado por la detección de la caja mamaria).
Ninguna función de esta feature reescala el recorte a 128×128 ni a ninguna otra
resolución fija.

**Justificación**: spec.md (Clarifications 2026-09-16) ya documenta el motivo desde
la perspectiva de producto — una microcalcificación puede ocupar un puñado de
píxeles o menos a 128×128, destruyendo la señal de localización que es la razón de
ser de este dataset. Desde la constitución, la sección "Restricciones del dominio"
(Fidelidad radiológica) ya lo exige con carácter general: "el recorte es la única
pérdida geométrica admitida... el redimensionado... ocurre en tiempo de
entrenamiento, nunca escrito sobre el corpus". Esta decisión no es una elección
libre de esta feature: es la aplicación de una regla ya vigente en el proyecto.

**Consecuencia técnica**: como el hallazgo se remapea al espacio del recorte en
píxeles absolutos (no en fracción del ancho/alto), reescalar más adelante a
cualquier resolución que pida un protocolo de evaluación es multiplicar las cuatro
coordenadas por `nueva_resolución / crop_width` (y análogo en `y`) — no hace falta
guardar nada adicional para dejarlo trivial.

## D-05 — Partición por `study_id`, split oficial reutilizado y verificado

**Decisión**: la columna `split` del catálogo es la que ya trae
`breast-level_annotations.csv` (`training`/`test`), no una partición calculada por
esta feature. `vindr/manifest.py`/`vindr/catalog.py` no implementan ningún
`assign_splits` propio.

**Justificación**: verificado sobre `specs/001-vindr-streaming-etl/`, el split
oficial ya es determinista y ya está a nivel de estudio (16.000 imágenes/4.000
estudios en `training`, 4.000 imágenes/1.000 estudios en `test`, sin solapamiento
documentado). El release público de VinDr-Mammo no expone un identificador de
paciente distinto del estudio, así que no hay una unidad más fina que
`assign_splits` (de `mammo_lejepa.splits`, feature 002) pudiera usar que no sea ya
la que usa el split oficial. Inventar una partición nueva no añadiría rigor:
duplicaría con peor información la que ya existe. Lo que esta feature sí añade es
la **verificación** —no presente en el 001 original— de que la intersección de
`study_id` entre ambas particiones es vacía, tratándolo como una comprobación
independiente en vez de una propiedad asumida del CSV de origen (constitución,
principio VI).

**Alternativas descartadas**: recalcular la partición con `mammo_lejepa.splits.
assign_splits`, estratificando por hallazgo/BI-RADS — descartado porque
invalidaría la comparabilidad con cualquier resultado publicado sobre el split
oficial de VinDr-Mammo, sin ninguna ganancia de rigor a cambio.

## D-06 — Concurrencia productor/consumidor con `asyncio`, distinta de la de Mammo-Bench

**Decisión**: `vindr/pipeline.py` conserva el modelo de la extinta feature 001:
productores `asyncio` (descarga) alimentando una cola acotada que consumen
procesos de un `ProcessPoolExecutor` (decodificación + recorte + escritura). No se
adopta el `ProcessPoolExecutor` + `as_completed` más simple de `mammo_lejepa.
runner` (feature 002).

**Justificación**: son dos problemas de forma distinta. Mammo-Bench no tiene red:
leer un fichero local y procesarlo es "un problema vergonzosamente paralelo" sin
contrapresión que gestionar, de ahí que su `runner.py` sea un `ProcessPoolExecutor`
liso. VinDr sí tiene red: descargar y procesar avanzan a ritmos distintos y no
acotados de antemano (la red es más lenta e impredecible que el disco local), así
que hace falta desacoplar productor (descarga) de consumidor (proceso) con una cola
de tamaño fijo — es exactamente lo que ya justificaba D-03 del plan.md original de
la feature 001 (techo de disco intermedio en función de la concurrencia, no del
tamaño del dataset). Forzar el patrón de 002 aquí perdería esa contrapresión sin
ninguna ganancia de simplicidad real, porque la complejidad de coordinar red y
proceso no desaparece, sólo se movería a un sitio peor.

## D-07 — Restauración aditiva de dependencias y scripts, nunca sustitutiva

**Decisión**: `pyproject.toml` recupera exactamente las dependencias que la feature
002 retiró (`aiohttp`, `aiofiles`, `physionet`, `pydicom`, `pylibjpeg`,
`pylibjpeg-openjpeg`, `python-dotenv`, `pytest-asyncio` con `asyncio_mode="auto"`),
y añade `mammo-etl` como un **segundo** script de consola junto a `mammo-corpus`,
sin sustituirlo. `beautifulsoup4` —presente en el `pyproject.toml` original de la
feature 001 pero sin ningún uso real en su código, verificado por búsqueda de
`bs4`/`BeautifulSoup` en todo el commit `0de1617`— no se restaura: era peso muerto
incluso en la feature 001, y añadirlo de vuelta no serviría a ningún requisito de
ésta.

**Justificación**: las features 002 y 004 conviven en el mismo repositorio; ninguna
debe poder romper la instalación o la CLI de la otra.

## D-08 — `resume.parse_jsonl_records` no es reutilizable; el resto de `resume.py` sí

**Decisión**: `vindr/catalog.py` define su propio `parse_jsonl_records`/
`_record_from_dict` para `ImageRecord`. `pending_tasks`, `latest_records_by_image` y
`summarize_failures` de `mammo_lejepa.resume` se reutilizan literalmente, sin copia
ni adaptación.

**Justificación**: descubierto durante la implementación (no estaba en el plan
inicial de D-01 a D-07). `mammo_lejepa.resume.parse_jsonl_records` delega en
`_record_from_dict`, que construye `RegistroDeRecorte(**kwargs)` de forma
hardcodeada — no es genérico por duck-typing como el resto del módulo. Las otras
tres funciones sólo leen atributos (`.image_id`, `.status`, `.processed_at`,
`.failure_category`) de los objetos que reciben, sin importar su tipo concreto; se
comprobó que funcionan sin cambios sobre `ImageRecord` de `vindr/models.py`. Este es
el mismo patrón de "reutilizable donde es genérico, no donde no lo es" que ya
motivó D-01/D-02/D-03; se documenta aparte porque no se detectó hasta escribir
`tests/vindr/test_pipeline_resume.py`, no durante la investigación inicial.
