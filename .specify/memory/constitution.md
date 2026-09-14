# mammo-lejepa Constitution

Proyecto: pipeline de extracción, recorte y catalogación de VinDr-Mammo
(PhysioNet v1.0.0) para preentrenamiento auto-supervisado sobre imagen mamaria.

## Core Principles

### I. Motor puro separado de la E/S (NO NEGOCIABLE)

Toda la lógica de decisión —umbralización, detección de la bounding box mamaria,
recorte, transformación de coordenadas, construcción del manifiesto, política de
reanudación— vive en funciones puras que reciben y devuelven datos en memoria
(`np.ndarray`, dataclasses, `pl.DataFrame`). Estas funciones no abren sockets, no
leen del disco, no escriben ficheros, no llaman a `print` y no dependen de
variables de entorno.

La E/S (HTTP, lectura de DICOM, escritura de PNG, JSONL, Parquet) se concentra en
módulos de frontera explícitos. Consecuencia práctica: cualquier regla de negocio
debe ser testeable sin red, sin disco y sin credenciales.

### II. Interfaz de línea de comandos y reproducibilidad

Cada capacidad se expone como subcomando de una CLI única (`mammo-etl`), con
parámetros explícitos y valores por defecto declarados en un único objeto de
configuración. Nada se configura editando constantes en el código fuente.

Toda ejecución deja constancia de su propia identidad: versión del pipeline,
parámetros efectivos, marca temporal y recuento de resultados. Dos ejecuciones con
los mismos parámetros sobre los mismos datos producen artefactos equivalentes; el
pipeline es idempotente y reanudable por construcción.

### III. Tests con pytest sobre la lógica crítica

La suite de tests es obligatoria en: detección de la bounding box mamaria,
transformación de coordenadas entre sistemas de referencia, construcción y filtrado
del manifiesto, lógica de reanudación y clasificación de errores.

Los tests no dependen de la red ni de las credenciales de PhysioNet: se usan DICOM
sintéticos generados en el propio test y un servidor HTTP local para las pruebas de
integración. Un test que necesita descargar de PhysioNet para pasar es un test roto.

Corolario de regresión: todo defecto reproducible se convierte primero en un test
que falla y después en una corrección.

### IV. Streaming con techo de recursos acotado

El volumen de origen (~300 GB) excede deliberadamente el almacenamiento disponible.
El pipeline nunca materializa el dataset completo: opera sobre una ventana acotada
mediante colas con contrapresión, y el consumo máximo de disco es una función
conocida de los parámetros de concurrencia, no del tamaño del dataset.

Cada DICOM se elimina en cuanto su derivado y su registro están confirmados en
disco. Un incremento de escala se resuelve alargando el tiempo de ejecución, nunca
elevando el pico de memoria o de disco.

### V. Trazabilidad y reversibilidad del dato

Ningún derivado se considera válido sin su procedencia: identificador original,
parámetros geométricos exactos de la transformación aplicada y versión del código
que la produjo. Toda coordenada almacenada declara el sistema de referencia al que
pertenece.

Cualquier anotación expresada en píxeles del DICOM original debe poder recalcularse
en el espacio del recorte, y viceversa, sin volver a descargar la imagen. Las
transformaciones irreversibles se documentan explícitamente como tales.

## Restricciones del dominio

**Datos clínicos y licencia**: VinDr-Mammo se distribuye bajo el acuerdo de uso de
PhysioNet. Las imágenes, los derivados y los CSV de anotaciones no se redistribuyen
ni se suben a ningún servicio de terceros. `data/` está fuera del control de
versiones.

**Credenciales**: usuario, contraseña y cookie de sesión se leen exclusivamente del
entorno (`.env`, ignorado por git). Nunca aparecen en el código, en los registros,
en los mensajes de error ni en los artefactos generados.

**Cortesía con el servidor de origen**: la concurrencia de descarga está acotada por
configuración con un límite conservador por defecto; se respeta `Retry-After` y se
aplica retroceso exponencial con jitter. Una sesión caducada aborta la ejecución de
forma inmediata en lugar de generar miles de peticiones fallidas.

**Fidelidad radiológica**: la conversión de 12/14 bits a 8 bits y el recorte son las
únicas pérdidas admitidas, y ambas quedan registradas. No se aplica ninguna
transformación que altere la geometría de la anatomía (deformación de aspecto,
rotación, espejado) sin dejar constancia de ella en los metadatos.

## Flujo de desarrollo y puertas de calidad

1. Los módulos de lógica pura se implementan y se prueban antes que los módulos de
   E/S que los consumen.
2. El código pasa `ruff format` y `ruff check` antes de considerarse terminado.
3. Anotaciones de tipo obligatorias en todas las funciones públicas; el estilo sigue
   el ya presente en `design/` (`from __future__ import annotations`, constantes
   `Final`, nombres descriptivos en castellano en la documentación y en inglés en el
   código).
4. Una historia de usuario está terminada cuando su criterio de aceptación es
   verificable ejecutando la CLI sobre una muestra reducida y reproducible.
5. Las dependencias se gestionan con `uv`; se añaden con justificación explícita.

## Governance

Esta constitución prevalece sobre cualquier otra práctica del repositorio. Las
desviaciones se documentan en la sección *Complexity Tracking* del plan
correspondiente, con la alternativa más simple que se ha descartado y el motivo.

Las enmiendas requieren una justificación escrita en el plan de la feature que las
motiva y una revisión de los artefactos afectados. La versión sigue el esquema
MAJOR.MINOR.PATCH: MAJOR para la retirada o redefinición de un principio, MINOR para
la incorporación de un principio o sección, PATCH para aclaraciones de redacción.

**Version**: 1.0.0 | **Ratified**: 2026-09-14 | **Last Amended**: 2026-09-14
