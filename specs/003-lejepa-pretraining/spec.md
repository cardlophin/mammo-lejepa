# Feature Specification: Preentrenamiento auto-supervisado LeJEPA sobre el corpus mamario

**Feature Branch**: `003-lejepa-pretraining`

**Created**: 2026-09-15

**Status**: Ready for planning

**Depends on**: `002-mammobench-corpus` (catálogo con particiones por paciente)

**Depended on by**: `004-vindr-downstream` (catálogo de VinDr-Mammo con hallazgos
remapeados, resolución nativa) espera el checkpoint que esta feature produce para
evaluar y afinar el encoder con cajas de localización — esa evaluación se
especifica aparte, como feature propia, una vez exista el checkpoint.

**Input**: Implementar el objetivo de LeJEPA (Balestriero y LeCun, arXiv:2511.08544)
de forma propia y agnóstica de la arquitectura, validarlo contra la implementación
oficial, y usarlo para preentrenar un encoder sobre los recortes mamarios de
Mammo-Bench, con evaluación por sonda lineal y diagnóstico de colapso.

## Contexto y expectativas

LeJEPA sustituye el "caldo de heurísticas" del SSL habitual —stop-gradient,
teacher-student, EMA, programadores complejos— por un único objetivo con un solo
hiperparámetro de compromiso:

```
pérdida = λ · SIGReg(embeddings) + (1 − λ) · invarianza(vistas)
```

SIGReg (*Sketched Isotropic Gaussian Regularization*) empuja la distribución de los
embeddings hacia una gaussiana isótropa mediante un contraste estadístico —test de
Epps-Pulley sobre proyecciones aleatorias unidimensionales— con coste lineal en el
tamaño del lote. Es lo que impide el colapso sin necesidad de asimetrías
arquitectónicas.

**Calibración de expectativas, por escrito**: el corpus son 19.731 imágenes de 5.860
pacientes. Es pequeño para SSL comparado con ImageNet, aunque LeJEPA se presenta
precisamente como método apto para dominios especializados y de tamaño moderado. Por
eso esta feature exige **líneas base** desde el principio: un número de sonda lineal
sin nada con lo que compararlo no informa de nada.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Un objetivo LeJEPA propio y verificado (Priority: P1)

Como ingeniero que quiere entender el algoritmo y no sólo usarlo, quiero implementar
SIGReg y el término de invarianza desde cero, con una interfaz que no sepa nada de la
arquitectura del encoder, y comprobar numéricamente que mi implementación es correcta
—tanto contra propiedades estadísticas conocidas como contra el paquete oficial.

**Why this priority**: es el corazón de la feature y lo que pediste. Además es lo
único que se puede verificar sin gastar cómputo en entrenamientos largos: si el
objetivo está mal, todo lo demás es tiempo perdido.

**Independent Test**: ejecutar la suite de tests del objetivo sin GPU y sin datos
reales, y comprobar que la pérdida propia coincide con la del paquete oficial sobre
los mismos tensores dentro de la tolerancia declarada.

**Acceptance Scenarios**:

1. **Given** un lote de embeddings muestreados de una gaussiana isótropa, **When** se
   evalúa SIGReg, **Then** el estadístico es bajo y se mantiene estable al variar el
   número de proyecciones.
2. **Given** embeddings colapsados en un punto, o concentrados en un subespacio de
   rango bajo, **When** se evalúa SIGReg, **Then** el estadístico es marcadamente
   mayor que en el caso isótropo.
3. **Given** los mismos tensores, **When** se evalúan la implementación propia y la
   del paquete `lejepa` oficial, **Then** ambas pérdidas coinciden dentro de la
   tolerancia declarada, y la comparación queda como test permanente.
4. **Given** cualquier módulo de PyTorch que produzca un tensor `[B, D]`, **When** se
   le aplica el objetivo, **Then** funciona sin que el objetivo conozca su
   arquitectura: la única dependencia es la forma del tensor.
5. **Given** un lote con `V` vistas de `B` imágenes, **When** se calcula la
   invarianza, **Then** vale cero si todas las vistas de cada imagen dan el mismo
   embedding y crece con su dispersión.
6. **Given** la pérdida total, **When** se retropropaga, **Then** todos los
   parámetros del encoder reciben gradiente finito, sin `NaN` ni `Inf`.

---

### User Story 2 - Preentrenar sobre el corpus, reanudable y en cualquier dispositivo (Priority: P2)

Como ingeniero que trabaja en un portátil y alquila GPU cuando hace falta, quiero
lanzar el preentrenamiento sobre el corpus de la feature 002 con el mismo código en
CPU, MPS y CUDA, y poder interrumpirlo y reanudarlo sin perder el trabajo hecho.

**Why this priority**: sin esto el objetivo verificado no produce ningún modelo.

**Independent Test**: entrenar 3 épocas sobre 500 imágenes en el portátil,
interrumpir, reanudar y comprobar que la pérdida continúa la curva en lugar de
reiniciarse.

**Acceptance Scenarios**:

1. **Given** el catálogo de la feature 002, **When** se lanza el entrenamiento,
   **Then** carga únicamente la partición `train`, genera `V` vistas por imagen y
   registra la pérdida descompuesta en sus dos términos.
2. **Given** el mismo comando y la misma semilla en CPU y en MPS, **When** se
   comparan las primeras iteraciones, **Then** la pérdida coincide dentro de la
   tolerancia numérica del dispositivo, sin ramas de código distintas.
3. **Given** un entrenamiento interrumpido en la época 7, **When** se reanuda,
   **Then** continúa desde la época 7 con el estado del optimizador, del
   programador y del generador aleatorio restaurados.
4. **Given** una ejecución terminada, **When** se consulta su directorio, **Then**
   contiene la configuración efectiva completa, la semilla, el historial de pérdidas
   y los checkpoints, suficientes para reproducirla.

---

### User Story 3 - Saber si la representación sirve (Priority: P3)

Como ingeniero que tiene que decidir si el preentrenamiento aporta algo, quiero una
evaluación por sonda lineal sobre el encoder congelado, comparada contra líneas base
honestas, y un diagnóstico de la geometría de los embeddings que verifique que SIGReg
está haciendo lo que promete.

**Why this priority**: es lo que convierte un entrenamiento en un resultado. Sin
líneas base, el número de la sonda no significa nada.

**Independent Test**: ejecutar la evaluación sobre un checkpoint y obtener una tabla
con la exactitud de la sonda para el modelo preentrenado y para las líneas base, más
las métricas de isotropía.

**Acceptance Scenarios**:

1. **Given** un checkpoint, **When** se ejecuta la sonda lineal sobre
   `classification`, **Then** se obtiene la métrica en `test`, desagregada además por
   `source_dataset`.
2. **Given** el mismo checkpoint, **When** se ejecutan las sondas de `density` y
   `BIRADS`, **Then** se reportan sobre las filas que tienen esa etiqueta, con el
   número de muestras a la vista.
3. **Given** el protocolo de evaluación, **When** se ejecuta sobre las líneas base
   —encoder con pesos aleatorios congelados, y encoder preentrenado en ImageNet
   congelado—, **Then** se obtienen los mismos números para las tres condiciones,
   comparables entre sí.
4. **Given** un entrenamiento en curso, **When** se consulta el diagnóstico de
   colapso, **Then** se dispone del rango efectivo, la entropía del espectro de
   valores singulares y la desviación de la isotropía a lo largo de las épocas.
5. **Given** cualquier evaluación, **When** se construyen sus conjuntos, **Then**
   proceden de las particiones por paciente del catálogo, sin reparticionar por
   imagen.

---

### User Story 4 - Cambiar de arquitectura sin tocar el objetivo (Priority: P4)

Como autor de una implementación que se dice agnóstica de la arquitectura, quiero
demostrarlo entrenando un segundo encoder de familia distinta cambiando sólo la
configuración.

**Why this priority**: es la prueba de que el diseño cumple lo que promete; el valor
científico añadido es secundario.

**Independent Test**: lanzar el mismo comando con `--encoder vit_small` y comprobar
que entrena sin modificar ni una línea del módulo del objetivo.

**Acceptance Scenarios**:

1. **Given** la configuración de ResNet-50, **When** se cambia a un ViT pequeño,
   **Then** el entrenamiento arranca sin tocar `objective.py` ni `trainer.py`.
2. **Given** ambos encoders entrenados, **When** se comparan sus sondas lineales,
   **Then** la tabla de resultados los sitúa en las mismas condiciones.

---

### Edge Cases

- **Lote pequeño y SIGReg**: el estadístico estima una distribución a partir del
  lote. Con lotes muy pequeños la estimación es ruidosa y puede desestabilizar el
  entrenamiento. El sistema debe advertir por debajo de un tamaño mínimo configurado.
- **Aumentaciones heredadas de imagen natural**: *color jitter*, conversión a escala
  de grises y *solarize* carecen de sentido —o son destructivos— en mamografía de un
  solo canal. Sustituirlos sin pensar por sus equivalentes de brillo y contraste
  también tiene consecuencias: la intensidad en mamografía correlaciona con densidad
  del tejido, que es una de las etiquetas a predecir.
- **Volteo horizontal y lateralidad**: voltear una mamografía izquierda la convierte
  en el aspecto de una derecha. Como el corpus contiene ambas, la aumentación es
  admisible, pero destruye la lateralidad como señal y debe declararse.
- **Recorte aleatorio sobre un recorte ya ajustado**: el corpus ya está recortado a la
  mama, así que la escala de `RandomResizedCrop` pensada para ImageNet (0,08-1,0)
  produce vistas que pueden no contener tejido. El rango debe reconsiderarse.
- **DDSM a 165 px escalado a 128**: casi no hay reescalado; las demás fuentes bajan
  desde 2.000-4.700 px. El modelo puede aprender a distinguir la fuente por la
  nitidez en lugar de aprender anatomía. El diagnóstico debe incluir una sonda lineal
  que intente predecir `source_dataset`: si acierta demasiado, el encoder está
  modelando el equipo de adquisición.
- **MPS y precisión mixta**: `bfloat16` no está disponible igual que en CUDA. El
  código debe degradar a `float32` sin ramas separadas.
- **Checkpoint escrito a medias por una interrupción**: no debe poder confundirse con
  uno válido.

## Requirements *(mandatory)*

### Objetivo

- **FR-001**: El sistema DEBE implementar el estadístico de Epps-Pulley sobre
  proyecciones aleatorias unidimensionales de los embeddings, con el número de
  proyecciones y de nodos de cuadratura configurables.
- **FR-002**: El sistema DEBE implementar el término de invarianza entre las `V`
  vistas de cada imagen.
- **FR-003**: La pérdida total DEBE ser `λ · SIGReg + (1 − λ) · invarianza`, con `λ`
  como único hiperparámetro de compromiso.
- **FR-004**: El objetivo NO DEBE conocer la arquitectura del encoder: su única
  entrada son tensores de forma `[V, B, D]` o `[V·B, D]` con el agrupamiento
  declarado.
- **FR-005**: El objetivo NO DEBE requerir stop-gradient, teacher-student, EMA ni
  predictor asimétrico.
- **FR-006**: El sistema DEBE incluir un test que compare la pérdida propia con la
  del paquete `lejepa` oficial sobre los mismos tensores, con una tolerancia
  declarada y justificada.
- **FR-007**: El coste en tiempo y memoria del objetivo DEBE ser lineal en el tamaño
  del lote, y existir una prueba que lo evidencie empíricamente.

### Datos y vistas

- **FR-008**: El `Dataset` DEBE construirse desde `catalog.parquet` de la feature 002
  y respetar la columna `split`.
- **FR-009**: El sistema DEBE generar `V` vistas por imagen con una canalización de
  aumentaciones **declarada explícitamente para imagen mamográfica**, no heredada sin
  revisión de las recetas de imagen natural.
- **FR-010**: La configuración de aumentaciones DEBE quedar registrada en la
  ejecución, con sus parámetros exactos.
- **FR-011**: El sistema DEBE permitir excluir del entrenamiento las imágenes
  marcadas como sospechosas en el catálogo, y por defecto DEBE incluirlas
  informando de cuántas son.
- **FR-012**: El sistema DEBE permitir filtrar por `source_dataset` para poder
  ablacionar con y sin DDSM sin reconstruir el corpus.

### Entrenamiento

- **FR-013**: El mismo código DEBE ejecutarse en CPU, MPS y CUDA, con el dispositivo
  como parámetro y detección automática por defecto.
- **FR-014**: El sistema DEBE soportar AdamW con calentamiento lineal y descenso
  coseno, y valores por defecto distintos de decaimiento de peso según la familia del
  encoder.
- **FR-015**: El sistema DEBE guardar checkpoints de forma atómica, con estado del
  modelo, del optimizador, del programador, de los generadores aleatorios y la época.
- **FR-016**: El sistema DEBE reanudar desde el último checkpoint válido restaurando
  todo ese estado.
- **FR-017**: El sistema DEBE registrar por época: pérdida total y sus dos términos
  por separado, tasa de aprendizaje, tiempo y métricas de geometría de los
  embeddings.
- **FR-018**: El sistema DEBE advertir cuando el tamaño de lote efectivo esté por
  debajo del mínimo recomendado para la estimación de SIGReg.

### Evaluación

- **FR-019**: El sistema DEBE implementar una sonda lineal sobre el encoder
  **congelado**, con verificación explícita de que no se propagan gradientes al
  encoder.
- **FR-020**: La sonda DEBE evaluarse sobre `classification`, `density` y `BIRADS`,
  informando del número de muestras de cada una.
- **FR-021**: El sistema DEBE reportar las métricas desagregadas por
  `source_dataset` además del agregado.
- **FR-022**: El sistema DEBE ejecutar el mismo protocolo sobre al menos dos líneas
  base: encoder con pesos aleatorios congelado, y encoder preentrenado en ImageNet
  congelado.
- **FR-023**: El sistema DEBE incluir una sonda de control que intente predecir
  `source_dataset` a partir de los embeddings, como medida de cuánto está modelando
  el equipo de adquisición en lugar de la anatomía.
- **FR-024**: Las particiones DEBEN proceder del catálogo; el sistema NO DEBE
  reparticionar por imagen bajo ninguna circunstancia.

### Diagnóstico de la representación

- **FR-025**: El sistema DEBE calcular sobre un lote de validación: rango efectivo,
  entropía normalizada del espectro de valores singulares, norma media de los
  embeddings y desviación respecto a la covarianza isótropa.
- **FR-026**: Estas métricas DEBEN registrarse periódicamente durante el
  entrenamiento, no sólo al final.
- **FR-027**: El sistema DEBE avisar cuando el rango efectivo caiga por debajo de un
  umbral configurado, señal de colapso incipiente.

### Key Entities

- **Vista**: una transformación aleatoria de un recorte; `V` vistas comparten
  identidad de imagen.
- **Encoder**: cualquier módulo que transforme un lote de imágenes en `[B, D]`. El
  sistema sólo conoce esa firma.
- **Proyector**: MLP opcional entre encoder y objetivo; la sonda lineal evalúa el
  encoder, no el proyector.
- **ObjetivoLeJEPA**: SIGReg más invarianza, con `λ`.
- **Ejecución**: configuración efectiva, semilla, historial, checkpoints y resultados
  de evaluación de un entrenamiento concreto.

## Success Criteria *(mandatory)*

- **SC-001**: Sobre embeddings gaussianos isótropos sintéticos, SIGReg propio y
  oficial coinciden con error relativo por debajo de la tolerancia declarada.
- **SC-002**: SIGReg distingue con claridad —al menos un orden de magnitud— entre
  embeddings isótropos y embeddings colapsados.
- **SC-003**: El objetivo se ejecuta sobre dos familias de encoder distintas sin
  ningún cambio en su código.
- **SC-004**: Tiempo y memoria del objetivo crecen de forma lineal con el tamaño del
  lote, medido en al menos cuatro tamaños.
- **SC-005**: El entrenamiento reanudado continúa la curva de pérdida en lugar de
  reiniciarla.
- **SC-006**: La sonda lineal sobre `classification` supera de forma clara a la línea
  base de pesos aleatorios. *(Si no ocurre, es un resultado y se registra como tal;
  el criterio es que la comparación exista y sea concluyente, no que salga
  favorable.)*
- **SC-007**: El rango efectivo de los embeddings al final del entrenamiento se
  mantiene por encima del umbral configurado, confirmando la ausencia de colapso.
- **SC-008**: La sonda de control sobre `source_dataset` se reporta siempre junto a
  las demás, para que cualquier lectura del resultado tenga a la vista cuánto se
  explica por el equipo de adquisición.
- **SC-009**: Todos los tests del objetivo pasan en CPU en menos de 60 segundos.

## Assumptions

- El corpus de la feature 002 está construido, consolidado y particionado.
- La resolución de entrada es 128×128, decidida por la heterogeneidad del corpus
  (DDSM a 165 px de lado corto).
- El primer encoder es ResNet-50; el segundo, un ViT pequeño, sirve para demostrar
  el agnosticismo arquitectónico.
- El paquete `lejepa` oficial se instala como **dependencia de desarrollo**, sólo
  para el test de validación cruzada; el código de producción no lo importa.
- El desarrollo y las pruebas se hacen en el portátil; los entrenamientos largos, en
  GPU alquilada. El código no debe distinguir entre ambos casos.
- El ajuste fino supervisado, la detección de lesiones y cualquier uso clínico quedan
  explícitamente fuera del alcance.
