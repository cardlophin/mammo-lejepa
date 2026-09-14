# Feature Specification: ETL en streaming de VinDr-Mammo a recortes mamarios catalogados

**Feature Branch**: `001-vindr-streaming-etl`

**Created**: 2026-09-14

**Status**: Ready for planning

**Input**: Extraer las 20.000 imágenes de VinDr-Mammo (PhysioNet v1.0.0, ~300 GB de
DICOM) sin materializar el dataset completo en disco: descargar de forma asíncrona,
detectar el campo mamario, recortarlo, guardarlo como PNG de 8 bits a resolución
nativa y eliminar el DICOM de origen, conservando un catálogo con las anotaciones
clínicas y las bounding boxes de hallazgos expresadas en el sistema de coordenadas
del recorte.

## Clarifications

### Session 2026-09-14

- Q: The edge case about metadata.csv vs. breast-level_annotations.csv discrepancies says the manifest must "declare an explicit join policy and count discrepancies," but FR-001 builds the manifest solely from breast-level_annotations.csv and metadata.csv is never joined anywhere in the plan or tasks. What should happen with metadata.csv? → A: Add warn-only validation — at manifest-build time, compare image_id sets between metadata.csv and breast-level_annotations.csv, log/count any discrepancies, and continue without aborting the run.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obtener recortes mamarios verificables de una muestra (Priority: P1)

Como ingeniero que prepara el corpus de preentrenamiento, quiero ejecutar el
pipeline sobre un puñado de estudios y obtener, en cuestión de minutos, los PNG
recortados junto con su registro de metadatos, para confirmar con mis propios ojos
que la detección del campo mamario es correcta antes de comprometer horas de
descarga.

**Why this priority**: es el núcleo irreductible del sistema —descarga, decodifica,
recorta, persiste y registra—. Sin esta pieza no existe producto; con ella sola ya
se obtiene valor, porque un corpus reducido permite validar la calidad del recorte y
prototipar el `Dataset` de entrenamiento.

**Independent Test**: ejecutar `mammo-etl run --n-studies 5 --split training` con
credenciales válidas y comprobar que `data/vindr-mammo/images/processed/` contiene un
PNG por imagen del manifiesto, que cada PNG muestra la mama completa sin bandas
negras laterales, y que `data/vindr-mammo/catalog/images.jsonl` tiene una línea por
imagen con su geometría de recorte.

**Acceptance Scenarios**:

1. **Given** credenciales válidas de PhysioNet en `.env` y los CSV de anotaciones
   presentes, **When** se ejecuta el pipeline sobre 5 estudios, **Then** se generan
   los PNG recortados de las imágenes de esos estudios, se registra una línea por
   imagen en el catálogo incremental y no queda ningún `.dicom` en disco.
2. **Given** una imagen con interpretación fotométrica MONOCHROME1, **When** se
   procesa, **Then** el PNG resultante muestra el tejido mamario claro sobre fondo
   oscuro, igual que las imágenes MONOCHROME2.
3. **Given** un DICOM cuyo recorte produce una imagen de dimensiones menores que la
   original, **When** se inspecciona su registro, **Then** contiene la caja de
   recorte `(x0, y0, x1, y1)`, el margen aplicado, el umbral de Otsu empleado y las
   dimensiones original y de salida.
4. **Given** la cookie de sesión de PhysioNet caducada, **When** se lanza la
   ejecución, **Then** el pipeline aborta antes de intentar la primera descarga con
   un mensaje que indica cómo renovar la cookie, y no deja artefactos a medio
   escribir.

---

### User Story 2 - Procesar el dataset completo sin agotar el disco y poder reanudar (Priority: P2)

Como ingeniero con un portátil de capacidad limitada, quiero lanzar el procesado de
los 5.000 estudios sabiendo que el disco nunca superará un techo conocido y que, si
el proceso se interrumpe —corte de red, cierre del portátil, cookie caducada—, al
relanzarlo continuará exactamente donde se quedó sin volver a descargar lo ya hecho.

**Why this priority**: convierte un prototipo en una herramienta utilizable sobre los
300 GB reales. Depende de US1 pero es independiente en su verificación: se comprueba
con el comportamiento del sistema, no con la calidad de la imagen.

**Independent Test**: lanzar el pipeline sobre 50 estudios, interrumpirlo con SIGINT
a mitad, medir el disco ocupado por `images/dicom/` durante la ejecución, relanzarlo
y comprobar que sólo se descargan las imágenes que faltaban y que el catálogo final
no tiene duplicados.

**Acceptance Scenarios**:

1. **Given** una ejecución en curso sobre el dataset completo, **When** se observa
   `images/dicom/` en cualquier instante, **Then** el número de DICOM presentes no
   supera el techo derivado de los parámetros de concurrencia, con independencia de
   cuántas imágenes se lleven procesadas.
2. **Given** una ejecución interrumpida con 1.200 imágenes completadas, **When** se
   relanza con los mismos parámetros, **Then** el pipeline informa de las 1.200 ya
   presentes, no vuelve a descargarlas y procesa únicamente las restantes.
3. **Given** una imagen cuyo PNG ya existe pero cuyo registro no se llegó a escribir,
   **When** se relanza el pipeline, **Then** esa imagen se vuelve a procesar y el
   catálogo queda consistente, sin líneas huérfanas ni duplicadas.
4. **Given** una interrupción durante la escritura de un PNG, **When** se relanza,
   **Then** no queda ningún fichero parcial que pueda confundirse con un derivado
   válido.

---

### User Story 3 - Disponer de un catálogo consolidado listo para el entrenamiento (Priority: P3)

Como ingeniero que va a escribir el `Dataset` de PyTorch, quiero un único fichero
Parquet con una fila por imagen procesada —ruta del PNG, dimensiones, etiquetas
clínicas y geometría del recorte— y otro con una fila por hallazgo anotado con sus
coordenadas ya trasladadas al espacio del recorte, para no tener que recalcular nada
en tiempo de entrenamiento ni volver a tocar los DICOM.

**Why this priority**: es lo que hace que el corpus sea utilizable por el modelo. Se
puede construir y verificar sobre la salida de US1 sin necesidad de haber procesado
el dataset completo.

**Independent Test**: tras procesar una muestra que incluya al menos un estudio con
hallazgos anotados, ejecutar `mammo-etl consolidate` y verificar, dibujando las cajas
del Parquet sobre el PNG recortado, que encajan sobre la lesión.

**Acceptance Scenarios**:

1. **Given** un hallazgo anotado en `finding_annotations.csv` con coordenadas en
   píxeles del DICOM original, **When** se consolida el catálogo, **Then** existe una
   fila con las coordenadas originales y las coordenadas en el espacio del recorte,
   y ambas identifican la misma región anatómica.
2. **Given** una imagen sin hallazgos —`finding_annotations.csv` contiene una fila
   por imagen, y sólo 2.254 de sus 20.486 filas traen coordenadas—, **When** se
   consolida, **Then** aparece en el catálogo de imágenes y no genera ninguna fila en
   el catálogo de hallazgos.
3. **Given** un catálogo consolidado, **When** se comprueban sus filas, **Then** cada
   una incluye `split`, `laterality`, `view_position`, `breast_birads` y
   `breast_density` procedentes de las anotaciones oficiales, y la ruta del PNG
   existe en disco.
4. **Given** un hallazgo cuya caja quedaría parcialmente fuera del recorte, **When**
   se consolida, **Then** la fila queda marcada como recortada y conserva tanto la
   caja original como la caja intersectada.

---

### User Story 4 - Aislar, diagnosticar y reintentar los fallos (Priority: P4)

Como responsable del pipeline, quiero que las imágenes que fallan no bloqueen la
ejecución, que su DICOM quede en cuarentena con la traza del error, y poder
reintentar sólo esas imágenes con un comando, para no volver a descargar 300 GB por
culpa de treinta casos raros.

**Why this priority**: mejora la operación pero el sistema entrega valor sin ella.

**Independent Test**: inyectar un DICOM corrupto en la cola, comprobar que el
pipeline continúa, que el fichero acaba en `images/quarantine/` con su registro de
error, y que `mammo-etl retry --failed` lo vuelve a intentar.

**Acceptance Scenarios**:

1. **Given** un DICOM que no puede decodificarse, **When** se procesa, **Then** el
   pipeline registra el fallo con su tipo y traza, conserva el DICOM en cuarentena y
   continúa con el resto del lote.
2. **Given** una ejecución terminada con fallos, **When** se consulta el resumen,
   **Then** se muestran los recuentos por categoría de error (red, decodificación,
   segmentación, escritura) y la ruta del registro de fallos.
3. **Given** fallos de red transitorios ya resueltos, **When** se ejecuta el
   reintento selectivo, **Then** sólo se procesan las imágenes marcadas como
   fallidas y sus registros previos se sustituyen por el resultado nuevo.

---

### Edge Cases

- **Sesión de PhysioNet caducada a mitad de ejecución**: las respuestas pasan a ser
  HTML o 403. El pipeline debe detectarlo por el tipo de contenido, detener la
  ejecución de forma ordenada (vaciando lo que ya está en cola) e indicar cómo
  renovar la cookie, en lugar de acumular 18.000 fallos.
- **DICOM comprimido con una sintaxis de transferencia sin decodificador
  instalado**: debe fallar con un mensaje que nombre la sintaxis y el paquete que
  falta, no con un `AttributeError` opaco.
- **Imagen sin mama detectable** (máscara de Otsu degenerada, imagen casi uniforme,
  componente conexo mayor que ocupa toda la imagen): se registra como fallo de
  segmentación, no se escribe un PNG con el marco completo disfrazado de recorte.
- **Etiquetas de orientación, marcadores metálicos y texto quemado** adheridos al
  tejido: pueden inflar la caja. Se registra la fracción de área del recorte respecto
  a la imagen original para poder auditarlo después.
- **Fila de hallazgo sin coordenadas**: es el caso mayoritario del CSV de hallazgos
  (18.232 de 20.486 filas). No es un error ni un dato ausente: significa que la
  imagen no tiene lesión anotada, y confundirlo con un fallo de lectura poblaría el
  catálogo de hallazgos con filas nulas.
- **Imagen presente en `metadata.csv` pero ausente de `breast-level_annotations.csv`**
  o viceversa: no afecta al manifiesto, que se construye exclusivamente a partir de
  `breast-level_annotations.csv` (FR-001). Al arrancar, el sistema compara el
  `image_id` de `breast-level_annotations.csv` con el `SOP Instance UID` de
  `metadata.csv` (misma imagen, distinto nombre de columna), registra el recuento
  de discrepancias como aviso y continúa sin abortar la ejecución (FR-032).
- **Dos ejecuciones simultáneas sobre el mismo directorio de salida**: debe impedirse
  mediante un bloqueo, o la escritura incremental quedará entrelazada.
- **Disco lleno durante la escritura de un PNG**: el fichero parcial no debe quedar
  con su nombre definitivo.
- **Estudio con un número de imágenes distinto de cuatro** (vistas repetidas o
  ausentes): el lote por estudio no puede asumir un tamaño fijo.

## Requirements *(mandatory)*

### Manifiesto y selección

- **FR-001**: El sistema DEBE construir el manifiesto de trabajo a partir de
  `data/vindr-mammo/csv/breast-level_annotations.csv`, produciendo pares únicos
  `(study_id, image_id)` con su `split`, y DEBE fallar con un mensaje explícito si
  faltan columnas requeridas.
- **FR-002**: El sistema DEBE permitir acotar el trabajo por split
  (`training`/`test`), por número de estudios y por lista explícita de estudios, y
  por defecto DEBE abarcar el dataset completo.
- **FR-003**: El sistema DEBE agrupar el trabajo en lotes por estudio, sin asumir un
  número fijo de imágenes por estudio.
- **FR-004**: El sistema DEBE persistir el manifiesto efectivo de cada ejecución,
  incluyendo los parámetros con los que se generó.
- **FR-032**: Al construir el manifiesto, el sistema DEBE comparar el conjunto de
  `SOP Instance UID` de `metadata.csv` (columna repetida en la cabecera; se usa una
  sola vez, desduplicada) con el conjunto de `image_id` de
  `breast-level_annotations.csv` —ambos identifican la misma imagen—, registrar el
  recuento de discrepancias como aviso (imágenes presentes en uno y ausentes en el
  otro) y continuar la ejecución sin abortar por esta causa.

### Descarga

- **FR-005**: El sistema DEBE descargar los DICOM de forma asíncrona y concurrente,
  con el número de descargas simultáneas acotado por configuración.
- **FR-006**: El sistema DEBE autenticarse mediante la cookie de sesión de PhysioNet
  leída del entorno, y DEBE validar el acceso al dataset con una petición ligera
  antes de iniciar el lote.
- **FR-007**: El sistema DEBE escribir cada descarga de forma atómica (fichero
  temporal y renombrado final), de modo que un fichero con nombre definitivo sea
  siempre un fichero completo.
- **FR-008**: El sistema DEBE reintentar los fallos transitorios de red con retroceso
  exponencial y jitter, respetando `Retry-After`, y DEBE distinguir los fallos
  transitorios de los permanentes (401, 403, 404, respuesta HTML).
- **FR-009**: El sistema DEBE abortar la ejecución completa, de forma ordenada, ante
  la evidencia de sesión inválida o caducada.

### Procesado de imagen

- **FR-010**: El sistema DEBE decodificar el DICOM, aplicar la ventana de
  visualización declarada en la cabecera cuando esté presente e invertir las imágenes
  MONOCHROME1, de modo que todas las salidas compartan la misma convención de
  intensidad.
- **FR-011**: El sistema DEBE normalizar a 8 bits mediante percentiles robustos,
  registrando los valores de corte empleados.
- **FR-012**: El sistema DEBE detectar el campo mamario mediante umbralización de
  Otsu, cierre morfológico y selección del mayor componente conexo, y DEBE derivar de
  él una bounding box rectangular con un margen configurable.
- **FR-013**: El sistema DEBE recortar la imagen a esa bounding box **sin
  redimensionar**, conservando la resolución nativa.
- **FR-014**: El sistema DEBE guardar el recorte como PNG de 8 bits sin pérdida en
  `data/vindr-mammo/images/processed/<study_id>/<image_id>.png`.
- **FR-015**: El sistema DEBE rechazar como fallo de segmentación los casos en que no
  se obtenga ningún componente conexo válido, y DEBE registrar siempre la fracción de
  área del recorte respecto a la imagen original para permitir auditoría posterior.

### Streaming y ciclo de vida de los ficheros

- **FR-016**: El sistema DEBE solapar descarga y procesado mediante un esquema
  productor/consumidor con cola acotada, de forma que la red no espere a la CPU ni a
  la inversa.
- **FR-017**: El sistema DEBE eliminar cada DICOM inmediatamente después de que su
  PNG y su registro estén confirmados en disco, sin esperar al final del lote.
- **FR-018**: El sistema DEBE conservar el DICOM en un directorio de cuarentena
  cuando su procesado falle, para permitir el diagnóstico sin volver a descargarlo.
- **FR-019**: El consumo máximo de disco intermedio DEBE ser una función conocida de
  los parámetros de concurrencia y no del tamaño del dataset.

### Persistencia, catálogo y trazabilidad

- **FR-020**: El sistema DEBE escribir un registro incremental en formato JSONL, una
  línea por imagen resuelta, inmediatamente después de resolverla.
- **FR-021**: Cada registro DEBE incluir: identificadores (`study_id`, `series_id`,
  `image_id`), estado, ruta y tamaño del PNG, dimensiones originales y de salida, la
  caja de recorte, el margen, el umbral de Otsu, los percentiles de normalización, la
  interpretación fotométrica, la sintaxis de transferencia, la versión del pipeline y
  la marca temporal.
- **FR-022**: Cada registro DEBE incluir las etiquetas clínicas oficiales:
  `split`, `laterality`, `view_position`, `breast_birads`, `breast_density`.
- **FR-023**: El sistema DEBE trasladar las bounding boxes de
  `finding_annotations.csv` desde el sistema de coordenadas del DICOM original al
  sistema de coordenadas del recorte, conservando ambas representaciones. Sólo las
  filas con coordenadas generan hallazgos; las filas sin caja (imágenes sin lesión
  anotada) se descartan del catálogo de hallazgos y su recuento se informa.
- **FR-024**: El sistema DEBE marcar explícitamente los hallazgos cuya caja quede
  parcial o totalmente fuera del recorte.
- **FR-025**: El sistema DEBE ofrecer un comando de consolidación que compacte los
  registros incrementales en `catalog/images.parquet` y `catalog/findings.parquet`.
- **FR-026**: La consolidación DEBE ser idempotente y DEBE resolver los registros
  repetidos de una misma imagen quedándose con el más reciente.

### Reanudación y operación

- **FR-027**: Al arrancar, el sistema DEBE determinar el trabajo pendiente
  descontando del manifiesto las imágenes ya resueltas con éxito, verificando que su
  PNG existe realmente.
- **FR-028**: El sistema DEBE impedir dos ejecuciones simultáneas sobre el mismo
  directorio de salida.
- **FR-029**: El sistema DEBE emitir progreso legible durante la ejecución (imágenes
  resueltas, tasa, fallos acumulados, estimación de tiempo restante) y un resumen
  final con los recuentos por categoría de error.
- **FR-030**: El sistema DEBE permitir reintentar selectivamente las imágenes
  marcadas como fallidas.
- **FR-031**: El sistema NO DEBE registrar credenciales ni cookies en la salida, en
  los mensajes de error ni en los artefactos generados.

### Key Entities

- **ImagenFuente**: una mamografía del dataset, identificada por `image_id` dentro de
  `study_id`; tiene lateralidad, proyección, dimensiones nativas y split oficial.
- **TareaDeDescarga**: la intención de obtener una ImagenFuente; su URL derivada, su
  destino temporal y su estado.
- **RecorteMamario**: el resultado geométrico de la detección: caja en coordenadas de
  la imagen original, margen aplicado, umbral y dimensiones resultantes. Es la pieza
  que hace reversible la transformación.
- **RegistroDeImagen**: la fila del catálogo que une ImagenFuente, RecorteMamario,
  etiquetas clínicas, procedencia técnica y resultado.
- **Hallazgo**: una anotación de lesión con su categoría, su BI-RADS y su caja,
  expresada simultáneamente en coordenadas originales y de recorte.
- **EstadoDeEjecución**: el conjunto de registros incrementales que permite responder
  qué falta por hacer.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Con 5 estudios de muestra, el 100 % de las imágenes procesadas produce
  un recorte en el que la mama aparece completa y sin más de un 20 % de área de fondo
  fuera del margen configurado, verificado por inspección visual.
- **SC-002**: Durante una ejecución sobre 500 estudios, el espacio ocupado por DICOM
  intermedios nunca supera los 3 GB con los parámetros de concurrencia por defecto.
- **SC-003**: El corpus final de PNG ocupa menos del 15 % del tamaño del dataset
  DICOM original.
- **SC-004**: Tras una interrupción y un relanzamiento, el número de descargas
  repetidas es cero y el catálogo consolidado contiene exactamente una fila por
  imagen procesada.
- **SC-005**: Menos del 0,5 % de las imágenes del dataset termina en cuarentena por
  fallo de segmentación o decodificación.
- **SC-006**: El 100 % de los hallazgos remapeados verificados por muestreo (al menos
  30 casos) cae sobre la lesión al dibujarlos sobre el PNG recortado.
- **SC-007**: La lógica pura del proyecto se ejecuta bajo pytest sin red, sin
  credenciales y sin ficheros DICOM reales, en menos de 30 segundos.
- **SC-008**: El rendimiento sostenido alcanza al menos el 80 % del ancho de banda
  medido por el benchmark de descarga pura sobre un único fichero.

## Assumptions

- Las credenciales de PhysioNet y una cookie de sesión válida están disponibles en
  `.env`; renovarla manualmente cuando caduque es parte del procedimiento operativo
  aceptado.
- Los tres CSV de anotaciones ya están descargados en `data/vindr-mammo/csv/` y no
  necesitan obtenerse desde la red.
- `breast-level_annotations.csv` es la fuente de verdad del inventario de imágenes y
  de las etiquetas a nivel de mama. Verificado sobre los ficheros presentes: 20.000
  filas, 20.000 `image_id` únicos, 5.000 estudios, exactamente 4 imágenes por estudio
  (L/R × CC/MLO), 16.000 en `training` y 4.000 en `test`. Todos los `image_id` de
  `finding_annotations.csv` existen en este inventario. Aun así el código no codifica
  el número 4 como invariante.
- El destino del corpus es el preentrenamiento auto-supervisado, por lo que se
  prioriza la fidelidad de la señal (PNG sin pérdida, resolución nativa) sobre el
  ahorro de disco; el redimensionado se hará en el `DataLoader`.
- La máquina de ejecución dispone de varios núcleos y de espacio suficiente para el
  corpus de salida, aunque no para el dataset DICOM completo.
- El entrenamiento del modelo, el `Dataset` de PyTorch y cualquier aumento de datos
  quedan fuera del alcance de esta feature.
- La segmentación fina de la glándula y la eliminación del músculo pectoral quedan
  fuera del alcance: aquí sólo se extrae la caja del campo mamario.
