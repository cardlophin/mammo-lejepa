# mammo-lejepa Constitution

Proyecto: corpus de recortes mamarios a partir de Mammo-Bench y preentrenamiento
auto-supervisado con LeJEPA (SIGReg) sobre imagen mamaria.

## Core Principles

### I. Motor puro separado de la E/S (NO NEGOCIABLE)

Toda la lógica de decisión —umbralización, derivación de la caja mamaria a partir de
una máscara, recorte, transformación de coordenadas, construcción del manifiesto,
particiones, y el propio objetivo de entrenamiento— vive en funciones puras que
reciben y devuelven datos en memoria (`np.ndarray`, `torch.Tensor`, dataclasses,
`pl.DataFrame`). Estas funciones no abren sockets, no leen del disco, no escriben
ficheros, no llaman a `print` y no dependen de variables de entorno.

La E/S (lectura de imágenes, escritura de derivados, JSONL, Parquet, checkpoints) se
concentra en módulos de frontera explícitos. Consecuencia práctica: cualquier regla
de negocio, y cualquier término de la función de pérdida, debe ser testeable sin
disco y sin GPU.

### II. Reproducibilidad exigible

Cada capacidad se expone como subcomando de una CLI única, con parámetros explícitos
y valores por defecto declarados en un único objeto de configuración. Nada se
configura editando constantes en el código fuente.

Toda ejecución —de preprocesado o de entrenamiento— deja constancia de su identidad:
versión del código, configuración efectiva completa, semillas, dispositivo, marca
temporal y resultados. Dos ejecuciones con la misma configuración y la misma semilla
producen artefactos equivalentes. Un experimento cuyo resultado no puede reproducirse
desde su registro no es un resultado.

### III. Tests con pytest sobre la lógica crítica

La suite es obligatoria en: derivación de la caja mamaria, transformación de
coordenadas, construcción del manifiesto y las particiones, y **el objetivo de
entrenamiento**. Un estimador estadístico se prueba contra propiedades conocidas
—valor esperado sobre muestras gaussianas, invarianzas, gradientes finitos— y, cuando
existe una implementación de referencia, contra ella sobre los mismos tensores.

Los tests no dependen de la red ni de GPU: se usan imágenes sintéticas de geometría
conocida y tensores deterministas. Todo defecto reproducible se convierte primero en
un test que falla y después en una corrección.

### IV. Cómputo acotado, reanudable y agnóstico del dispositivo

El mismo código se ejecuta en CPU, MPS y CUDA sin ramas paralelas: el dispositivo es
un parámetro, no una variante del programa. Ninguna operación asume la existencia de
GPU, de `bfloat16` ni de más de un proceso.

Todo trabajo largo —preprocesado del corpus o entrenamiento— es interrumpible y
reanudable desde su último estado confirmado, y su consumo de memoria es función de
los parámetros de configuración, no del tamaño del corpus. Un entrenamiento que hay
que relanzar desde cero tras un corte es un defecto, no un inconveniente.

### V. Trazabilidad y reversibilidad del dato

Ningún derivado se considera válido sin su procedencia: identificador original,
dataset de origen, parámetros geométricos exactos de la transformación aplicada y
versión del código que la produjo. Toda coordenada almacenada declara su sistema de
referencia.

Cualquier anotación expresada en píxeles de la imagen original debe poder
recalcularse en el espacio del recorte, y viceversa. Las transformaciones
irreversibles se documentan explícitamente como tales.

### VI. Evaluación honesta

Una métrica que no se puede defender no se publica en el registro del experimento.
En concreto: las particiones se hacen de forma que ninguna imagen del mismo paciente
aparezca a ambos lados de una frontera de evaluación; toda comparación entre
configuraciones fija cuanto no se está comparando; y las garantías que el método
afirma dar —isotropía de las representaciones, ausencia de colapso— se miden, no se
asumen.

Un resultado negativo registrado vale más que uno positivo no reproducible.

## Restricciones del dominio

**Datos clínicos y licencia**: Mammo-Bench agrega seis datasets públicos (INbreast,
DDSM, CMMD, CDD-CESM, DMID, KAU-BCMD), cada uno con sus propias condiciones de uso.
Las imágenes y sus derivados no se redistribuyen ni se suben a ningún servicio de
terceros. `data/` y los checkpoints quedan fuera del control de versiones. Cualquier
publicación de resultados cita las seis fuentes originales además de Mammo-Bench.

**Credenciales y secretos**: se leen exclusivamente del entorno y nunca aparecen en
el código, en los registros ni en los artefactos generados.

**Heterogeneidad del corpus, declarada**: las seis fuentes difieren en resolución
nativa hasta en dos órdenes de magnitud. El `source_dataset` acompaña a cada imagen
en todo el recorrido, desde el manifiesto hasta el registro de evaluación, para que
cualquier métrica pueda desagregarse por fuente. Un resultado agregado que oculte una
diferencia grande entre fuentes está incompleto.

**Fidelidad radiológica**: el recorte es la única pérdida geométrica admitida sobre
la imagen almacenada, y queda registrada. El redimensionado y las aumentaciones
ocurren en tiempo de entrenamiento, nunca escritas sobre el corpus.

## Flujo de desarrollo y puertas de calidad

1. Los módulos de lógica pura se implementan y se prueban antes que los módulos de
   E/S que los consumen.
2. El código pasa `ruff format` y `ruff check` antes de considerarse terminado.
3. Anotaciones de tipo obligatorias en todas las funciones públicas.
4. Una historia de usuario está terminada cuando su criterio de aceptación es
   verificable ejecutando la CLI sobre una muestra reducida y reproducible.
5. Las dependencias se gestionan con `uv`; se añaden con justificación explícita.
6. El código de una feature retirada se elimina del árbol de trabajo en la misma
   entrega que introduce su sustituto; no se deja código muerto "por si acaso".

## Governance

Esta constitución prevalece sobre cualquier otra práctica del repositorio. Las
desviaciones se documentan en la sección *Complexity Tracking* del plan
correspondiente, con la alternativa más simple que se ha descartado y el motivo.

Las enmiendas requieren justificación escrita en el plan de la feature que las motiva
y una revisión de los artefactos afectados. La versión sigue MAJOR.MINOR.PATCH: MAJOR
para la retirada o redefinición de un principio, MINOR para la incorporación de un
principio o sección, PATCH para aclaraciones de redacción.

**Version**: 2.0.0 | **Ratified**: 2026-09-14 | **Last Amended**: 2026-09-15

<!--
Registro de enmiendas
2.0.0 (2026-09-15): el proyecto abandona la descarga en streaming desde PhysioNet y
pasa a operar sobre un corpus local (Mammo-Bench), añadiendo una fase de
entrenamiento auto-supervisado. Se retira el principio IV original ("Streaming con
techo de recursos acotado"), sustituido por "Cómputo acotado, reanudable y agnóstico
del dispositivo". Se añade el principio VI ("Evaluación honesta"), sin el cual una
fase de entrenamiento no tiene puertas de calidad. Se reescriben las restricciones
del dominio: la licencia de PhysioNet deja de aplicar y aparece la heterogeneidad de
resolución entre las seis fuentes como restricción de primer orden.
1.0.0 (2026-09-14): versión inicial para el ETL de VinDr-Mammo.
-->
