# Feature Specification: VinDr-Mammo como conjunto de evaluación downstream

**Feature Branch**: `004-vindr-downstream`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "VinDr-Mammo deja de ser el corpus de preentrenamiento —ese papel es hoy de
Mammo-Bench (feature 002)— y pasa a ser el conjunto de evaluación downstream: de sus
20.000 imágenes, 2.254 tienen bounding box de hallazgo con categoría y BI-RADS, y es
lo único en todo el proyecto que permite evaluar localización. Descargar, recortar y
catalogar VinDr con sus cajas de hallazgo remapeadas y sus particiones por estudio,
dejando el conjunto listo para una evaluación futura del encoder preentrenado con
LeJEPA sobre Mammo-Bench (feature 003)."

## Contexto verificado sobre los datos

- El código de la extinta feature 001 (ETL de VinDr-Mammo hacia PhysioNet) existe
  íntegro en el historial de git de este repositorio (commit `0de1617`); su baja del
  árbol de trabajo está sólo en el índice, sin confirmar. Esta feature reconstruye
  ese ETL como un subpaquete aislado (`vindr/`), adaptado al modelo de datos actual
  del repositorio (feature 002 reescribió `models.py`/`config.py`/`manifest.py`/
  `catalog.py`/`worker.py` para Mammo-Bench; el código de VinDr no puede volver a su
  ubicación original sin romper ambos paquetes).
- `breast-level_annotations.csv` (verificado sobre los ficheros originales,
  spec.md de 001): 20.000 filas, 20.000 `image_id` únicos, 5.000 estudios,
  exactamente 4 imágenes por estudio (L/R × CC/MLO). Trae un split oficial ya
  resuelto: 16.000 imágenes (4.000 estudios) en `training`, 4.000 imágenes
  (1.000 estudios) en `test`.
- `finding_annotations.csv`: 20.486 filas: sólo 2.254 traen coordenadas de caja de
  hallazgo con categoría y BI-RADS — el resto son imágenes sin lesión anotada. Es la
  única fuente de cajas de hallazgo de todo el proyecto: Mammo-Bench (002) no tiene
  ninguna.
- Los tres CSV de anotaciones no están en el repositorio (`data/` está excluido de
  git) y deben volver a descargarse de PhysioNet. Las credenciales de PhysioNet
  están disponibles en el entorno del proyecto, pero la cookie de sesión que
  autoriza la descarga caduca independientemente del usuario/contraseña y
  normalmente hay que renovarla desde el navegador antes de poder operar.
- El release público de VinDr-Mammo no expone un identificador de paciente distinto
  del estudio: la unidad de partición y de agrupación clínica es el `study_id`.

## Clarifications

### Session 2026-09-16

- **Q: ¿A qué resolución se guarda el recorte de la región mamaria, dado que el
  encoder que se evaluará (feature 003) entrena a 128×128 pero este conjunto existe
  precisamente para medir localización de hallazgos pequeños? → A**: resolución
  nativa (la del DICOM de origen, menos el fondo descartado). El recorte NUNCA se
  reescala a 128×128 ni a ninguna otra resolución fija al construir este conjunto.
  Justificación: una microcalcificación puede ocupar unos pocos milímetros; a
  128×128 podría quedar en un puñado de píxeles o menos, destruyendo la señal de
  localización que es la razón de ser de este dataset. Guardar a resolución nativa
  conserva toda la información; cualquier reescalado que un protocolo de evaluación
  concreto necesite es, a partir del recorte nativo y de sus cajas de hallazgo ya
  remapeadas al espacio del recorte, un factor de escala lineal trivial de aplicar
  en ese momento — no una decisión que deba tomarse, y menos de forma irreversible,
  al construir el corpus. Ver Edge Cases y FR-010.
- **Q: ¿Cuál es la unidad de partición train/test? → A**: el `study_id`, reutilizando
  el split oficial de `breast-level_annotations.csv` en vez de calcular uno nuevo.
  El release público no distingue paciente de estudio, así que el estudio es la
  unidad de agrupación más fina disponible y la que ya usa el split oficial. Esta
  feature verifica —no inventa— que ningún estudio aparece partido entre `training`
  y `test`, y que ninguna imagen queda fuera de ambos.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reconstruir el corpus de recortes de VinDr (Priority: P1)

Quien vaya a evaluar el encoder necesita, sin más pasos que ejecutar un comando, un
recorte PNG de la región mamaria por cada imagen de VinDr-Mammo que se pudo descargar
y procesar, con sus etiquetas clínicas oficiales (split, lateralidad, proyección,
BI-RADS y densidad mamarios) y trazabilidad completa de qué falló y por qué.

**Why this priority**: sin el corpus reconstruido no hay nada que evaluar; es el
mínimo producto viable de esta feature, igual que lo fue para el corpus de
preentrenamiento en la feature 002.

**Independent Test**: con las credenciales de PhysioNet vigentes, ejecutar el
comando de construcción acotado a un puñado de estudios y comprobar que produce un
PNG por imagen descargada con éxito, un registro incremental legible y un resumen
final sin ambigüedad sobre qué se procesó y qué falló.

**Acceptance Scenarios**:

1. **Given** credenciales de PhysioNet vigentes y los tres CSV de anotaciones ya
   descargados, **When** se ejecuta la construcción acotada a un número reducido de
   estudios, **Then** se obtiene un PNG por imagen procesada con éxito y un registro
   incremental con un resultado (éxito o fallo con categoría) por imagen.
2. **Given** una ejecución interrumpida a mitad de camino, **When** se relanza el
   mismo comando, **Then** continúa únicamente con las imágenes pendientes, sin
   volver a descargar ni reprocesar las que ya tienen su PNG y su registro `ok`.
3. **Given** una cookie de sesión caducada, **When** se intenta construir el corpus,
   **Then** el sistema lo señala de forma inequívoca, sin perder el trabajo ya
   completado, y explica cómo renovarla.

---

### User Story 2 - Cajas de hallazgo en el espacio del recorte (Priority: P2)

Quien vaya a evaluar localización necesita las cajas de hallazgo de
`finding_annotations.csv` —categoría, BI-RADS y coordenadas— trasladadas al sistema
de coordenadas del recorte final, no al del DICOM original, para poder comparar
directamente contra lo que produzca el encoder sobre el PNG.

**Why this priority**: es lo que distingue a VinDr de Mammo-Bench y la razón de ser
de esta feature; sin esto, VinDr sólo serviría para clasificación, que Mammo-Bench ya
cubre mejor en volumen.

**Independent Test**: sobre un corpus ya construido, ejecutar la consolidación y
comprobar que cada hallazgo con coordenadas en el CSV original de una imagen
procesada con éxito aparece en el catálogo de hallazgos con sus coordenadas en el
espacio del recorte, marcado si quedó parcial o totalmente fuera de él.

**Acceptance Scenarios**:

1. **Given** una imagen procesada con éxito con uno o más hallazgos anotados,
   **When** se consolida el catálogo, **Then** cada hallazgo aparece con sus
   coordenadas originales, sus coordenadas en el espacio del recorte, y un
   indicador de si quedó total o parcialmente fuera del recorte.
2. **Given** una fila de `finding_annotations.csv` sin coordenadas (imagen sin
   lesión anotada), **When** se consolida, **Then** no genera ninguna fila en el
   catálogo de hallazgos y su recuento se informa aparte, nunca como un error.
3. **Given** un hallazgo cuya imagen no se procesó con éxito, **When** se consolida,
   **Then** ese hallazgo no aparece en el catálogo y la discrepancia queda
   contabilizada.

---

### User Story 3 - Particiones por estudio verificadas (Priority: P3)

Quien vaya a evaluar el encoder necesita la certeza operativa —no sólo documental—
de que ninguna imagen de `test` comparte estudio con una imagen de `training`, para
que un resultado de evaluación sea honesto (constitución, principio VI).

**Why this priority**: una fuga de estudio entre particiones invalidaría en
silencio cualquier métrica de evaluación futura; es una puerta de calidad, no una
funcionalidad de cara al usuario, así que va después de tener datos que particionar.

**Independent Test**: sobre un catálogo ya consolidado, ejecutar la verificación de
particiones y comprobar que señala con éxito la ausencia de fuga, o que falla de
forma explícita y nombrando el estudio en conflicto si se introduce una fuga
deliberada.

**Acceptance Scenarios**:

1. **Given** el catálogo consolidado con la columna de partición ya incorporada,
   **When** se ejecuta la verificación, **Then** confirma que la intersección de
   `study_id` entre `training` y `test` es exactamente vacía.
2. **Given** un catálogo manipulado en el que una imagen de un estudio aparece con
   una partición distinta a la de las demás imágenes del mismo estudio, **When** se
   ejecuta la verificación, **Then** falla nombrando el `study_id` en conflicto.

---

### Edge Cases

- **Cookie de sesión caducada**: distinguible de una credencial de usuario/contraseña
  ausente o incorrecta; el diagnóstico y el mensaje de error deben decir cuál de los
  dos problemas es, porque se resuelven de formas distintas (renovar la cookie desde
  el navegador vs. corregir `.env`).
- **CSV de anotaciones ausentes**: la construcción del corpus, y el diagnóstico del
  entorno, deben decirlo de forma clara y accionable antes de intentar nada más, sin
  llegar a un error de red o de sesión que oscurezca la causa real.
- **Decodificador de imagen ausente para una sintaxis de transferencia DICOM
  concreta**: se traduce a un fallo categorizado y explicado por imagen, nunca a una
  excepción no controlada que interrumpa el resto del lote.
- **Discrepancia entre `metadata.csv` y `breast-level_annotations.csv`**: un
  `image_id` presente en uno y ausente en el otro se informa como aviso cuantificado,
  no bloquea la construcción del resto del corpus.
- **Hallazgo cuya caja cae parcial o totalmente fuera del recorte** (posible si el
  recorte de la región mamaria excluye el borde de la imagen donde cae la anotación):
  se conserva marcado, nunca se descarta en silencio — quien evalúe localización
  necesita saber cuántos hallazgos están en este caso.
- **Resolución nativa variable entre fabricantes de equipo**: el mismo hallazgo
  físico (p. ej. una microcalcificación de un tamaño en milímetros dado) ocupa un
  número de píxeles distinto según la resolución nativa del DICOM de origen, que
  varía por fabricante. Cualquier evaluación de localización que consuma este
  catálogo deberá poder desagregar sus métricas por esa procedencia en vez de
  reportar un número agregado que la oculte — coherente con cómo la constitución ya
  exige desagregar por fuente en Mammo-Bench. Esta feature deja constancia del
  riesgo y de los campos que lo permiten desagregar; no diseña esa evaluación.
- **Re-ejecución tras ampliar el conjunto de estudios solicitado**: una ejecución
  posterior con un filtro más amplio que incluya estudios ya procesados no debe
  volver a descargarlos ni reprocesarlos.

## Requirements *(mandatory)*

### Alcance y procedencia

- **FR-001**: El sistema NO DEBE reintroducir código muerto de la extinta feature
  001 fuera del subpaquete aislado que esta feature crea: ningún módulo compartido
  del corpus de Mammo-Bench (feature 002) debe modificarse para dar cabida a tipos o
  comportamiento específicos de VinDr.
- **FR-002**: El sistema DEBE poder construirse, probarse y ejecutarse de forma
  completamente independiente del corpus de Mammo-Bench: ningún comando ni módulo de
  esta feature debe requerir que el corpus de la feature 002 exista.

### Manifiesto y selección

- **FR-003**: El sistema DEBE construir el manifiesto de trabajo a partir de
  `breast-level_annotations.csv`, verificando primero que trae las columnas
  requeridas y nombrando la que falte si no es así.
- **FR-004**: El sistema DEBE permitir acotar el trabajo por split oficial
  (`training`/`test`), por lista explícita de `study_id`, y por número de estudios,
  combinables entre sí.
- **FR-005**: La selección por número de estudios DEBE ser determinista (los
  primeros de un orden estable), nunca una muestra aleatoria sin semilla.
- **FR-006**: El sistema DEBE agrupar el trabajo por estudio, sin asumir un número
  fijo de imágenes por estudio aunque en los datos verificados sean siempre 4.

### Descarga y decodificación

- **FR-007**: El sistema DEBE descargar cada imagen de PhysioNet de forma atómica
  (fichero temporal renombrado al terminar): un fichero con nombre definitivo DEBE
  ser siempre un fichero completo.
- **FR-008**: El sistema DEBE reintentar los fallos de red transitorios con
  retroceso exponencial y respetar `Retry-After` cuando el servidor lo indique, y
  DEBE distinguir sin reintentar los fallos permanentes de autenticación (cookie
  caducada o sin acceso) de los fallos de red.
- **FR-009**: El sistema DEBE decodificar el DICOM descargado a píxeles en su tipo
  nativo, y ante un fallo de decodificación DEBE nombrar la sintaxis de
  transferencia y el decodificador ausente en vez de propagar un error opaco.

### Recorte de la región mamaria

- **FR-010**: El sistema DEBE recortar la región mamaria de cada imagen a su
  resolución nativa (la del DICOM de origen menos el fondo descartado). El sistema
  NO DEBE reescalar el recorte a la resolución de preentrenamiento del encoder
  (128×128) ni a ninguna otra resolución fija al construirlo (Clarifications
  2026-09-16): esa decisión es de quien diseñe la evaluación, no de esta feature.
- **FR-011**: El sistema DEBE aplicar la ventana de visualización DICOM cuando el
  DICOM la declare, y DEBE normalizar el resultado a 8 bits mediante percentiles
  robustos para no depender de valores extremos aislados.
- **FR-012**: El sistema DEBE detectar la región mamaria por segmentación de
  intensidad (Otsu) sobre la imagen normalizada; VinDr no trae máscara de
  segmentación propia, a diferencia de Mammo-Bench.
- **FR-013**: El sistema DEBE guardar el recorte como PNG de 8 bits sin pérdida,
  de forma atómica.

### Catálogo, hallazgos y trazabilidad

- **FR-014**: El sistema DEBE escribir un registro incremental en formato JSONL,
  una línea por imagen procesada, con vaciado del búfer del sistema operativo antes
  de continuar.
- **FR-015**: Cada registro DEBE incluir identificadores (`study_id`, `series_id`,
  `image_id`), resultado (`ok`/`failed` con categoría y mensaje si falla), las
  etiquetas clínicas oficiales (`split`, lateralidad, proyección, BI-RADS y
  densidad mamarios) y la procedencia técnica necesaria para auditar el
  preprocesado (parámetros de ventana, normalización y recorte).
- **FR-016**: El sistema DEBE trasladar las cajas de hallazgo de
  `finding_annotations.csv` desde el sistema de coordenadas del DICOM original al
  del recorte, conservando ambas representaciones. Sólo las filas con coordenadas y
  cuya imagen se procesó con éxito generan una fila en el catálogo de hallazgos; el
  resto se cuenta aparte, nunca como error.
- **FR-017**: El sistema DEBE marcar explícitamente los hallazgos cuya caja quede
  parcial o totalmente fuera del recorte, sin descartarlos.
- **FR-018**: El sistema DEBE ofrecer una consolidación idempotente de los
  registros incrementales a un catálogo de imágenes y a un catálogo de hallazgos,
  resolviendo los registros repetidos de una misma imagen quedándose con el más
  reciente.
- **FR-019**: El sistema DEBE validar la coherencia del catálogo consolidado: toda
  fila `ok` debe tener su PNG en disco con el tamaño registrado, y toda caja de
  hallazgo remapeada debe caer dentro de las dimensiones del recorte que declara.

### Particiones

- **FR-020**: El catálogo DEBE incluir la partición oficial (`training`/`test`) de
  `breast-level_annotations.csv` por imagen.
- **FR-021**: El sistema DEBE verificar y dejar constancia de que la intersección de
  `study_id` entre `training` y `test` es exactamente vacía, nunca a nivel de
  imagen (Clarifications 2026-09-16).

### Operación

- **FR-022**: El sistema DEBE diagnosticar el entorno sin descargar ninguna imagen,
  distinguiendo explícitamente al menos estos dos estados: CSV de anotaciones
  ausentes, y cookie de sesión de PhysioNet caducada o sin acceso — nunca deben
  presentarse como el mismo error genérico.
- **FR-023**: El sistema DEBE ser reanudable: al arrancar, debe descontar del
  trabajo pendiente las imágenes ya resueltas con éxito cuyo PNG existe físicamente.
- **FR-024**: El sistema DEBE permitir reintentar selectivamente sólo las imágenes
  marcadas como fallidas, opcionalmente acotado a una categoría de fallo.
- **FR-025**: El sistema DEBE impedir dos ejecuciones simultáneas sobre el mismo
  directorio de salida.
- **FR-026**: El sistema NO DEBE imprimir ni registrar en ningún artefacto
  generado el usuario, la contraseña o la cookie de sesión de PhysioNet, ni
  siquiera dentro de un mensaje de error de una librería externa.
- **FR-027**: El sistema DEBE emitir progreso legible durante la ejecución y un
  resumen final con el recuento de éxitos y fallos por categoría.
- **FR-028**: El sistema DEBE ofrecer una verificación visual de control: una
  muestra de imágenes con su recorte y sus hallazgos remapeados dibujados encima,
  para confirmar visualmente que el remapeado de coordenadas es correcto antes de
  fiarse del catálogo a escala completa.

### Key Entities *(include if feature involves data)*

- **ImageTask**: unidad de trabajo derivada del manifiesto — identificadores de
  estudio/serie/imagen, split oficial, etiquetas clínicas y dimensiones esperadas
  de la imagen. No cambia durante la ejecución.
- **Caja mamaria**: región rectangular detectada por segmentación de intensidad,
  con su margen, el umbral usado, las dimensiones nativas de la imagen de origen y
  la fracción de área que ocupa. Es el mismo concepto y la misma representación que
  usa el corpus de Mammo-Bench (feature 002), reutilizada tal cual para no duplicar
  la lógica de recorte y traslado de coordenadas.
- **Resultado de procesado**: lo que produce el procesado de una imagen —éxito o
  fallo con categoría—, la caja mamaria si tuvo éxito, y la procedencia técnica
  (ventana, normalización, fabricante del equipo).
- **Registro de imagen**: una fila del catálogo — une la unidad de trabajo, su
  resultado, la ejecución que la produjo y su partición oficial.
- **Hallazgo**: una lesión anotada en `finding_annotations.csv` con coordenadas —
  categoría, BI-RADS, coordenadas originales y coordenadas en el espacio del
  recorte, e indicador de si quedó fuera de él total o parcialmente.
- **Resumen de ejecución**: una ejecución completa — configuración efectiva,
  recuentos de éxito/fallo por categoría, y cómo terminó (completa o interrumpida).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: El corpus contiene un recorte por cada una de las 20.000 imágenes de
  VinDr-Mammo que se pudo descargar y decodificar con éxito, o bien el catálogo
  explica cada ausencia con una categoría de fallo.
- **SC-002**: El catálogo de hallazgos contiene una fila por cada uno de los 2.254
  hallazgos con coordenadas cuya imagen se procesó con éxito, con sus coordenadas
  ya trasladadas al espacio del recorte.
- **SC-003**: La intersección de `study_id` entre las particiones `training` y
  `test` es exactamente vacía, verificado automáticamente, no sólo documentado.
- **SC-004**: En una muestra visual de al menos 20 recortes con hallazgo,
  inspeccionada manualmente, el 100 % muestra la caja del hallazgo remapeada en una
  posición visualmente correcta sobre el recorte.
- **SC-005**: Ante una cookie de sesión caducada, el diagnóstico del entorno lo
  distingue de un CSV de anotaciones ausente en menos de lo que tarda un intento de
  descarga real, sin necesidad de leer un mensaje de error de red.
- **SC-006**: Tras una interrupción y un relanzamiento, ninguna imagen se descarga
  ni se procesa dos veces, y el catálogo consolidado no tiene filas duplicadas por
  imagen.

## Assumptions

- El release público de VinDr-Mammo no permite vincular más de un estudio al mismo
  paciente; el `study_id` es la unidad de partición y de agrupación clínica
  disponible más fina. Si en el futuro se dispusiera de un identificador de
  paciente real que agrupe varios estudios, la verificación de partición debería
  repetirse a ese nivel.
- Las credenciales de PhysioNet (usuario, contraseña y cookie de sesión) se
  gestionan fuera de esta feature, en el entorno del proyecto; esta feature asume
  que existen pero que la cookie puede haber caducado, y debe decir cuál de los dos
  problemas es.
- El protocolo de evaluación del encoder (sonda lineal, cabeza de detección o
  ajuste fino) es una feature futura, no incluida aquí; esta feature sólo deja el
  dataset —recortes, catálogo de imágenes, catálogo de hallazgos y partición
  verificada— listo para que esa feature futura lo consuma.
- El margen de recorte, los percentiles de normalización y el resto de parámetros
  de preprocesado heredan los valores ya validados por la extinta feature 001 salvo
  que esta feature documente explícitamente un cambio.
