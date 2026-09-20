# Phase 0 — Investigación y decisiones técnicas

**Feature**: 002-mammobench-corpus | **Fecha**: 2026-09-15

Cada entrada recoge la decisión adoptada, su justificación y las alternativas
descartadas. Lo que aquí queda fijado no vuelve a discutirse durante la
implementación.

## D-01 — La máscara manda, Otsu respalda

**Decisión**: la caja sale de la máscara umbralizada a `mask_threshold` (128 por
defecto), quedándose con el mayor componente conexo y aplicando el mismo margen que
en la feature 001. Se recurre a Otsu cuando la máscara falta, cuando su fracción de
área queda fuera de `[0,02, 0,99]` o cuando la caja resultante es degenerada.

**Justificación**: las máscaras de Mammo-Bench incluyen eliminación del músculo
pectoral, que Otsu no sabe hacer —el pectoral tiene una intensidad parecida al tejido
y queda dentro del mismo componente conexo—. Ignorarlas sería reimplementar peor algo
ya resuelto.

**Alternativas descartadas**: usar Otsu como método único (pierde la eliminación de
pectoral que sí tienen las máscaras); usar la máscara sin respaldo (rompería en las
imágenes sin máscara válida, que las hay).

## D-02 — El IoU máscara-Otsu se calcula siempre

**Decisión**: aunque la caja usada sea la de la máscara, se calcula también la de
Otsu y se guarda el IoU, siempre que haya una máscara válida sobre la que calcularlo.

**Justificación**: cuesta una umbralización más por imagen y convierte la confianza
en las máscaras en un número por fuente. Sin esta métrica, un fallo sistemático de
segmentación en una fuente entera sería indetectable hasta ver los resultados del
modelo.

**Alternativas descartadas**: calcular el IoU sólo bajo demanda (en el informe de
calidad) — obligaría a releer y reprocesar las imágenes ya recortadas, en vez de
capturar el dato una sola vez cuando ya se tienen ambas cajas en memoria.

## D-03 — Recorte rectangular, nunca máscara aplicada

**Decisión**: el derivado es el rectángulo de la imagen; no se multiplica por la
máscara ni se anula el fondo.

**Justificación**: anular el fondo introduce un borde artificial de alto contraste
que las aumentaciones de recorte aleatorio convertirían en una señal trivial que el
modelo aprendería en lugar de la textura del tejido.

**Alternativas descartadas**: aplicar la máscara como multiplicación (fondo a cero) —
descartado por el motivo anterior; recortar a la envolvente convexa de la máscara en
vez de a su bounding box — añade complejidad geométrica sin beneficio claro para un
recorte que ya va a pasar por aumentaciones de recorte aleatorio en el entrenamiento.

## D-04 — Entrada desde `Preprocessed_Dataset`

**Decisión**: `Preprocessed_Dataset` es la entrada de referencia para el recorte;
`Original_Dataset` se conserva sin tocar, como respaldo de auditoría.

**Justificación**: es la versión que el propio Mammo-Bench declara como preprocesada
y es la que referencian las máscaras. Verificado: ambas coinciden en dimensiones
salvo en `ddsm`, donde la preprocesada ya viene recortada.

**Alternativas descartadas**: partir de `Original_Dataset` y aplicar el
preprocesado de Mammo-Bench de nuevo — redundante, y arriesga divergir del
preprocesado que las máscaras ya asumen.

## D-05 — Particiones por paciente con clave compuesta

**Decisión**: `source_subject_id` sólo es único dentro de su fuente, así que la
clave de paciente es el par `(source_dataset, source_subject_id)`. Se ordenan las
claves, se baraja con la semilla y se asignan por proporciones respetando la
estratificación por fuente y por clase.

**Justificación**: sin la clave compuesta, un `source_subject_id` que coincida por
azar entre dos fuentes distintas (p. ej. `"1"` en `ddsm` y en `cmmd`) se trataría
como el mismo paciente y podría forzar sus imágenes a la misma partición sin motivo,
o —peor— dos pacientes reales distintos podrían fusionarse en el cálculo de
proporciones.

**Alternativas descartadas**: usar `source_subject_id` a secas como clave —
descartado por la colisión entre fuentes descrita arriba.

**Cerrado en Clarifications (spec.md, sesión 2026-09-15)**: `classification` es un
atributo por imagen, no por paciente, y 535 de 5.860 pacientes (9,1 %, verificado
sobre `mammo-bench.csv`) tienen imágenes con más de un valor. Antes de estratificar,
`assign_splits` reduce las clasificaciones de un paciente a una sola quedándose con
la más severa, orden `Malignant` > `Suspicious Malignant` > `Benign` > `Normal`.

## D-06 — Sospechoso se marca, no se elimina

**Decisión**: un recorte con área inverosímil se marca con `suspect = true` y
permanece en el corpus.

**Justificación**: filtrar es una decisión del entrenamiento, configurable; borrar
en el preprocesado es irreversible y oculta el problema.

**Alternativas descartadas**: excluir automáticamente los casos sospechosos del
corpus — irreversible y oculta silenciosamente un posible fallo sistemático de una
fuente entera hasta que alguien lo note en los resultados del modelo.

## D-07 — La máscara se realinea a la resolución de la imagen, nunca se descarta por eso

**Decisión**: si `mask.shape != image.shape`, la máscara se reescala a la resolución
de la imagen con el vecino más próximo (`boxing.align_mask_to_image`) antes de
umbralizar. No se trata como máscara ausente ni dispara Otsu.

**Justificación**: verificado sobre datos reales de `ddsm` (Fase 5, T033-T034): 99 de
100 pares imagen/máscara muestreados difieren de tamaño — es la norma en esta fuente
(10.400 de 19.731 imágenes del corpus), no una excepción. La primera implementación
trataba cualquier discrepancia de tamaño como máscara inválida y forzaba el respaldo
a Otsu, lo que habría descartado la máscara real —la que sí excluye el pectoral,
D-01— en más de la mitad del corpus. Redimensionar con vecino más próximo (la
máscara es casi binaria; interpolar introduciría grises espurios) mantiene la
máscara como fuente autoritativa en el caso normal.

**Alternativas descartadas**: (a) tratar la discrepancia de resolución como
`fallback_reason` y usar Otsu — statu quo inicial, descartado por lo anterior; (b)
exigir que manifest.py rechace estos pares — perdería la mayoría de `ddsm` sin
necesidad, ya que la máscara reescalada sigue siendo una fuente de verdad válida
sobre la silueta de la mama.

**Verificación**: `test_mismatched_mask_resolution_is_realigned_not_rejected`
(`tests/unit/test_boxing.py`) prueba que una máscara a mitad de resolución sigue
excluyendo el pectoral tras realinear. Contra datos reales: reconstruir 100 imágenes
de `ddsm` pasó de 96 ok/4 fallidas (con un `UNKNOWN` en `cv2.imencode` por recorte
vacío, p. ej. `ddsm_10046`: imagen 354x84 vs máscara 360x240) a 100 ok/0 fallidas,
0 respaldos a Otsu.
