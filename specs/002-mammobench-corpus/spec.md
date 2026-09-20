# Feature Specification: Corpus de recortes mamarios desde Mammo-Bench

**Feature Branch**: `002-mammobench-corpus`

**Created**: 2026-09-15

**Status**: Ready for planning

**Supersedes**: `001-vindr-streaming-etl` (retirada completa; ver [migration.md](./migration.md))

**Input**: Sustituir el origen de datos del proyecto: se abandona la descarga de
VinDr-Mammo desde PhysioNet y se pasa a un corpus local ya descargado, Mammo-Bench
v2, del que hay que producir recortes de la región mamaria listos para
preentrenamiento auto-supervisado con LeJEPA.

## Contexto verificado sobre los datos

Comprobado directamente sobre `data/Mammo_Bench_v2/` el 2026-09-15:

- **19.731 imágenes únicas** de **5.860 pacientes**, seis fuentes: `ddsm` (10.400),
  `cmmd` (5.202), `kau-bcmd` (2.206), `cdd-cesm` (1.003), `dmid` (510), `inbreast`
  (410). VinDr-Mammo **no** está entre ellas.
- Los 59.462 ficheros `.jpg` del árbol son la misma imagen por triplicado:
  `Original_Dataset/` (20.000, incluidos 269 ficheros `_ROI` de DMID),
  `Preprocessed_Dataset/` (19.731) y `Masks/` (19.731).
- **Cada imagen trae su máscara de mama**, generada con OpenBreast e incluyendo
  eliminación del músculo pectoral. Están guardadas en JPEG, con pérdida: no son
  binarias puras y hay que umbralizarlas.
- `Preprocessed_Dataset` **no está recortado** salvo en `ddsm`: para las otras cinco
  fuentes tiene exactamente las mismas dimensiones que `Original_Dataset`.
- **Heterogeneidad extrema de resolución**: lado corto mediano de 165 px en `ddsm`
  (el 53 % del corpus) frente a 4.748 px en `dmid`, 2.816 en `kau-bcmd`, 2.560 en
  `inbreast`, 1.914 en `cmmd` y 1.290 en `cdd-cesm`.
- Etiquetas disponibles: `classification` en las 19.731 filas (Normal 5.243, Benign
  6.069, Malignant 8.184, Suspicious Malignant 235), `density` en 14.441, `BIRADS`
  en 4.097, `abnormality` en 5.712, `molecular_subtype` en 2.956.

## Clarifications

### Session 2026-09-15

- Q: `splits.py` asigna particiones a nivel de paciente y estratifica por
  `(source_dataset, classification)`, pero `classification` es un atributo por
  imagen, no por paciente. Verificado sobre el CSV real: 535 de 5.860 pacientes
  (9,1 %) tienen imágenes con más de un `classification` distinto (p. ej. una
  imagen Benign y otra Malignant del mismo paciente). ¿Qué `classification`
  debe representar a ese paciente a efectos de estratificación? → A: Gana la
  más severa — orden `Malignant` > `Suspicious Malignant` > `Benign` >
  `Normal`; un paciente con alguna imagen maligna se estratifica como
  Malignant.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obtener el corpus de recortes completo (Priority: P1)

Como ingeniero que va a preentrenar un modelo, quiero convertir las 19.731 imágenes
de Mammo-Bench en recortes de la región mamaria guardados en disco junto con un
catálogo, para que el `Dataset` de entrenamiento se limite a leer ficheros sin tomar
ninguna decisión sobre la imagen.

**Why this priority**: es el producto de esta feature. Sin corpus no hay
entrenamiento.

**Independent Test**: ejecutar el preprocesado completo y comprobar que existe un
recorte por cada fila del CSV, que el catálogo tiene 19.731 filas y que una muestra
inspeccionada visualmente contiene la mama sin músculo pectoral ni fondo sobrante.

**Acceptance Scenarios**:

1. **Given** el árbol `data/Mammo_Bench_v2/` completo, **When** se ejecuta el
   preprocesado, **Then** se produce un recorte por imagen del CSV, un catálogo con
   una fila por imagen y un resumen con los recuentos por fuente.
2. **Given** una imagen con máscara válida, **When** se procesa, **Then** la caja de
   recorte procede de la máscara umbralizada y el registro indica `box_source =
   "mask"`.
3. **Given** una imagen cuya máscara está vacía, saturada o es degenerada, **When**
   se procesa, **Then** el sistema recurre a la detección por Otsu, el recorte se
   produce igualmente y el registro indica `box_source = "otsu"`.
4. **Given** una ejecución interrumpida, **When** se relanza, **Then** continúa desde
   el último estado confirmado sin reprocesar lo ya hecho.

---

### User Story 2 - Saber cuánto se puede fiar del recorte (Priority: P2)

Como responsable del corpus, quiero una medida objetiva de la calidad de cada
recorte y de la discrepancia entre la máscara provista y mi propia detección por
Otsu, para poder decidir con datos si una fuente concreta está mal segmentada en vez
de descubrirlo cuando el modelo ya esté entrenado.

**Why this priority**: el corpus alimenta todo lo demás; un recorte sistemáticamente
malo en una fuente contamina el preentrenamiento y es invisible en el agregado.

**Independent Test**: generar el informe de calidad y comprobar que da, por fuente,
la distribución de fracción de área recortada y el IoU entre la caja de la máscara y
la caja de Otsu, con los casos extremos identificados por nombre de fichero.

**Acceptance Scenarios**:

1. **Given** el corpus procesado, **When** se pide el informe de calidad, **Then**
   incluye por fuente la mediana y los percentiles de `crop_area_ratio`, el IoU
   máscara-Otsu y el recuento de recortes que recurrieron al respaldo.
2. **Given** un recorte cuya área es inverosímil —menos del 10 % o más del 98 % de la
   imagen— **When** se genera el informe, **Then** aparece marcado como sospechoso
   con su ruta, sin ser eliminado automáticamente del corpus.
3. **Given** una muestra estratificada por fuente, **When** se pide la lámina de
   control visual, **Then** se obtiene una rejilla con original, máscara, caja y
   recorte para inspección directa.

---

### User Story 3 - Particiones y catálogo listos para entrenar (Priority: P3)

Como ingeniero que va a escribir el `Dataset`, quiero un catálogo Parquet con las
particiones ya asignadas y sin fuga de paciente, para que la evaluación posterior sea
defendible y no dependa de que me acuerde de hacerlo bien.

**Why this priority**: se construye sobre US1 y determina la validez de todas las
métricas de la feature 003.

**Independent Test**: comprobar que la intersección de `source_subjectID` entre
particiones es vacía y que las proporciones de clase se mantienen dentro de una
tolerancia entre particiones.

**Acceptance Scenarios**:

1. **Given** el catálogo procesado, **When** se generan las particiones, **Then**
   ningún `source_subjectID` aparece en más de una partición.
2. **Given** las particiones generadas, **When** se comparan las proporciones de
   `classification` y de `source_dataset` entre ellas, **Then** se mantienen dentro
   de la tolerancia configurada.
3. **Given** la misma semilla, **When** se regeneran las particiones, **Then** son
   idénticas.
4. **Given** el catálogo final, **When** lo lee el `Dataset` de la feature 003,
   **Then** dispone de ruta del recorte, fuente, resolución nativa, etiquetas
   disponibles y partición, sin tener que abrir ningún CSV original.

---

### Edge Cases

- **Máscara en JPEG con artefactos de compresión**: los píxeles no son 0 o 255 sino
  un continuo. Umbralizar por el valor medio del rango sin más produce bordes
  dentados y componentes espurios; hay que quedarse con el mayor componente conexo de
  la máscara umbralizada, igual que se hace con Otsu.
- **Máscara de `cdd-cesm` que cubre el 80 % de la imagen**: observado en el muestreo.
  Puede ser correcto (imágenes ya recortadas en origen) o un fallo de segmentación;
  el informe de calidad debe permitir distinguirlo antes de dar el corpus por bueno.
- **Imágenes de DDSM de 91×227 px**: un recorte sobre ellas deja muy pocos píxeles.
  Hay que registrar la resolución nativa y decidir en el entrenamiento, no descartar
  en silencio.
- **Ficheros `_ROI` de DMID**: 269 ficheros en `Original_Dataset` que no son
  mamografías y no aparecen en el CSV. El manifiesto se construye desde el CSV, nunca
  listando el directorio.
- **Ficheros `.DS_Store`** dentro de `Preprocessed_Dataset`: el listado de directorio
  no es fuente de verdad.
- **Filas del CSV cuyo fichero no existe en disco**: se contabilizan y se informan;
  el corpus declara cuántas filas quedaron sin imagen.
- **Paciente con imágenes en dos fuentes distintas**: `source_subjectID` sólo es
  único dentro de su fuente. La clave de paciente es el par
  `(source_dataset, source_subjectID)`.

## Requirements *(mandatory)*

### Retirada del origen anterior

- **FR-001**: El sistema NO DEBE conservar código, tests, dependencias ni
  configuración destinados a descargar o decodificar datos de PhysioNet. La lista
  exacta de ficheros a eliminar y a conservar está en `migration.md`.
- **FR-002**: Las dependencias que dejan de usarse (`aiohttp`, `aiofiles`,
  `physionet`, `pydicom`, `pylibjpeg`, `pylibjpeg-openjpeg`, `requests`,
  `beautifulsoup4`) DEBEN retirarse de `pyproject.toml`.
- **FR-003**: La lógica de detección por Otsu, la geometría de recorte, la escritura
  atómica, la reanudación y el modelo de trazabilidad DEBEN conservarse y
  reutilizarse.

### Manifiesto y selección

- **FR-004**: El manifiesto DEBE construirse a partir de
  `data/Mammo_Bench_v2/CSV_Files/mammo-bench.csv`, nunca listando directorios.
- **FR-005**: El sistema DEBE permitir acotar el trabajo por fuente, por número de
  imágenes y por lista explícita de identificadores, y por defecto DEBE abarcar las
  19.731 filas.
- **FR-006**: El sistema DEBE verificar la existencia en disco de la imagen y de la
  máscara de cada fila, e informar de las ausencias sin abortar.

### Extracción de la región mamaria

- **FR-007**: La caja de recorte DEBE derivarse por defecto de la máscara provista:
  umbralizada, quedándose con el mayor componente conexo y con un margen
  configurable.
- **FR-008**: El sistema DEBE recurrir a la detección por Otsu cuando la máscara
  falte, esté vacía, esté saturada o produzca una caja degenerada, y DEBE registrar
  en cada fila cuál de los dos caminos se usó.
- **FR-009**: El sistema DEBE calcular, para toda imagen con máscara válida, el IoU
  entre la caja derivada de la máscara y la caja que habría dado Otsu, y almacenarlo
  como medida de discrepancia, con independencia de cuál se haya usado.
- **FR-010**: El recorte DEBE aplicarse sobre la imagen de `Preprocessed_Dataset` y
  guardarse a resolución nativa, sin redimensionar.
- **FR-011**: El sistema NO DEBE aplicar la máscara como multiplicación sobre los
  píxeles: el derivado es un recorte rectangular, no una segmentación con fondo
  anulado.
- **FR-012**: El sistema DEBE guardar el recorte como PNG de 8 bits sin pérdida en
  `data/corpus/crops/<source_dataset>/<image_id>.png`.

### Catálogo y trazabilidad

- **FR-013**: El sistema DEBE escribir un registro incremental JSONL, una línea por
  imagen resuelta, inmediatamente después de resolverla.
- **FR-014**: Cada registro DEBE incluir: `image_id`, `source_dataset`,
  `source_subjectID`, `laterality`, `view`, estado, ruta y tamaño del recorte,
  dimensiones nativas y del recorte, la caja con su margen, `box_source`,
  `crop_area_ratio`, `mask_otsu_iou`, la versión del código y la marca temporal.
- **FR-015**: Cada registro DEBE incluir las etiquetas disponibles —`classification`,
  `density`, `BIRADS`, `abnormality`, `molecular_subtype`, `subject_age`— con
  distinción explícita entre ausente y presente.
- **FR-016**: El sistema DEBE ofrecer una consolidación idempotente del JSONL a
  `data/corpus/catalog.parquet`.

### Particiones

- **FR-017**: El sistema DEBE generar particiones train/val/test cuya unidad sea el
  paciente, identificado por el par `(source_dataset, source_subject_id)`.
- **FR-018**: Las particiones DEBEN ser deterministas dada una semilla, y estar
  estratificadas por `source_dataset` y por `classification`. Como
  `classification` es un atributo por imagen, el sistema DEBE derivar de él un
  único valor por paciente antes de estratificar, quedándose con el más severo
  presente entre sus imágenes según el orden `Malignant` > `Suspicious
  Malignant` > `Benign` > `Normal`.
- **FR-019**: El sistema DEBE verificar y dejar constancia de que la intersección de
  pacientes entre particiones es vacía.
- **FR-020**: Las proporciones por defecto DEBEN ser configurables, con 80/10/10 como
  valor inicial.

### Control de calidad

- **FR-021**: El sistema DEBE producir un informe de calidad con, por fuente, la
  distribución de `crop_area_ratio`, la de `mask_otsu_iou`, el recuento de respaldos
  a Otsu y la lista de casos sospechosos.
- **FR-022**: El sistema DEBE producir una lámina visual de control con muestra
  estratificada por fuente: imagen, máscara, caja y recorte.
- **FR-023**: Los casos sospechosos NO DEBEN eliminarse automáticamente del corpus:
  se marcan en el catálogo mediante un indicador para que el entrenamiento pueda
  filtrarlos por configuración.

### Operación

- **FR-024**: El sistema DEBE ser reanudable: al arrancar descuenta las imágenes ya
  resueltas cuyo recorte existe físicamente.
- **FR-025**: El sistema DEBE paralelizar el procesado en varios procesos, con el
  número de trabajadores configurable.
- **FR-026**: El sistema DEBE emitir progreso legible y un resumen final por fuente y
  por categoría de error.

### Key Entities

- **ImagenMammoBench**: una fila del CSV; su fuente, paciente, lateralidad, vista,
  rutas de imagen original, preprocesada y máscara, y sus etiquetas.
- **CajaMamaria**: la caja en coordenadas de la imagen, su margen, su origen
  (`mask` u `otsu`), el umbral empleado y las dimensiones resultantes.
- **RegistroDeRecorte**: la fila del catálogo que une imagen, caja, métricas de
  calidad, etiquetas y procedencia.
- **Partición**: la asignación train/val/test a nivel de paciente, con su semilla y
  sus proporciones.

## Success Criteria *(mandatory)*

- **SC-001**: El corpus contiene un recorte por cada una de las 19.731 filas del CSV,
  o bien el catálogo explica cada ausencia.
- **SC-002**: Menos del 2 % de las imágenes recurre al respaldo por Otsu.
- **SC-003**: La mediana del IoU entre la caja de la máscara y la de Otsu supera 0,80
  en todas las fuentes salvo aquellas donde el informe documente y explique la
  discrepancia.
- **SC-004**: En una muestra estratificada de 60 recortes inspeccionados visualmente,
  el 100 % contiene la mama completa y ninguno conserva una banda de músculo pectoral
  comparable en área al tejido.
- **SC-005**: La intersección de pacientes entre particiones es exactamente vacía, y
  se verifica automáticamente.
- **SC-006**: El preprocesado completo de las 19.731 imágenes termina en menos de 60
  minutos con 8 procesos en el portátil de trabajo.
- **SC-007**: La lógica pura se ejecuta bajo pytest, sin disco ni GPU, en menos de 30
  segundos.
- **SC-008**: Tras una interrupción y un relanzamiento, ninguna imagen se procesa dos
  veces y el catálogo no tiene filas duplicadas.

## Assumptions

- `data/Mammo_Bench_v2/` está completo en local y no hay que descargar nada.
- `Preprocessed_Dataset` es la entrada de referencia para el recorte; `Original_Dataset`
  se conserva sólo como respaldo de auditoría.
- Las máscaras representan tejido mamario con músculo pectoral ya eliminado, según la
  documentación de Mammo-Bench; la feature verifica esa afirmación por muestreo antes
  de darla por buena (T‑QC).
- Las anotaciones de lesión (`x`, `y`, `radius`, `ROI_path`), presentes en apenas 270
  filas, quedan fuera del alcance de esta feature: se copian al catálogo tal cual, sin
  remapear a coordenadas del recorte.
- El redimensionado a la resolución de entrada del modelo y las aumentaciones son
  responsabilidad de la feature 003, no de este corpus.
- DDSM se incluye pese a su baja resolución; la decisión de entrenar a 128×128 la
  hace viable y el catálogo conserva la resolución nativa para poder ablacionar.
