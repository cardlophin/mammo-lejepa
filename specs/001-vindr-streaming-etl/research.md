# Phase 0 — Investigación y decisiones técnicas

**Feature**: 001-vindr-streaming-etl | **Fecha**: 2026-09-14

Cada entrada recoge la decisión adoptada, su justificación y las alternativas
descartadas. Lo que aquí queda fijado no vuelve a discutirse durante la
implementación; lo que queda marcado como *a verificar* tiene una tarea asociada en
`tasks.md`.

## D-01 — Decodificación de los píxeles DICOM

**Decisión**: `pydicom` como lector de cabeceras y de píxeles, con un *plugin* de
decodificación explícito instalado como dependencia (`pylibjpeg` +
`pylibjpeg-openjpeg`, con `python-gdcm` como alternativa).

**Justificación**: los DICOM de mamografía de PhysioNet se distribuyen comprimidos.
`pydicom` por sí solo no decodifica JPEG 2000 ni JPEG-LS: sin *plugin* falla al
acceder a `pixel_array` con un error que no nombra la causa real. Declararlo como
dependencia desde el principio evita un fallo masivo en la primera ejecución larga.

**Cerrado (T059)**: verificado contra 4 imágenes reales descargadas de PhysioNet
(`mammo-etl run --n-studies 1 --split training`, 2026-09-14): las cuatro son
SIEMENS Mammomat Inspiration con `TransferSyntaxUID` `1.2.840.10008.1.2.1`
(Explicit VR Little Endian, sin comprimir) — `pylibjpeg`/`pylibjpeg-openjpeg` no
llegaron a ejercitarse con estos cuatro ejemplares, pero el decodificador JPEG 2000
se validó por separado y con éxito en `mammo-etl doctor` (comprime y decodifica un
DICOM sintético JPEG2000Lossless en memoria). `crop_area_ratio` observado:
0.19–0.25 en las cuatro, sin cajas sospechosamente infladas. Queda pendiente repetir
esta observación sobre una muestra mucho mayor (T058, no ejecutado en esta sesión
por el ancho de banda disponible) para detectar mezclas de sintaxis de transferencia
entre fabricantes y la cola de la distribución de `crop_area_ratio`.

**Alternativas descartadas**: `SimpleITK` o `GDCM` como lector principal —añaden una
dependencia binaria pesada para una funcionalidad que `pydicom` ya cubre; conversión
previa con `dcmj2pnm` —introduce un proceso externo y un formato intermedio.

## D-02 — Modelo de concurrencia

**Decisión**: bucle `asyncio` único en el proceso principal para toda la E/S de red y
un `ProcessPoolExecutor` para el procesado de imagen, acoplados mediante una
`asyncio.Queue` acotada.

**Justificación**: las dos fases tienen naturalezas opuestas. La descarga es E/S pura
y escala con corrutinas; la decodificación JPEG 2000 de una imagen de 3518×2800 más
el Otsu y la morfología de OpenCV son CPU intensiva y liberan el GIL sólo de forma
parcial, por lo que un `ThreadPoolExecutor` dejaría la CPU infrautilizada y
ralentizaría el bucle de eventos. La cola acotada aporta la contrapresión que
garantiza el techo de disco exigido por el principio IV de la constitución.

**Alternativas descartadas**: lotes estrictos descargar→procesar→borrar —más simples
pero dejan la CPU ociosa durante toda la fase de red, con una pérdida estimada del
30-40 % del rendimiento sobre 20.000 imágenes; `ThreadPoolExecutor` para todo, como
en `design/a.py` —válido para el prototipo de 5 estudios, insuficiente a escala;
`multiprocessing` puro con descargas síncronas por proceso —multiplica las sesiones
HTTP y complica el control de cortesía con el servidor.

## D-03 — Cálculo del techo de disco

**Decisión**: el disco intermedio máximo es
`(descargas_simultáneas + tamaño_cola + workers_de_proceso) × tamaño_máximo_DICOM`.
Con los valores por defecto (6 descargas, cola de 12, 4 workers) y un máximo
observado de ~35 MB por DICOM, el techo es de aproximadamente 780 MB.

**Justificación**: hace del consumo de disco un parámetro de diseño verificable en un
test, y no una consecuencia emergente. El comando `run` imprime este cálculo al
arrancar para que el operador lo contraste con su espacio libre.

## D-04 — Autenticación contra PhysioNet

**Decisión**: cookie `sessionid` del navegador, cargada del entorno, aplicada a un
`aiohttp.CookieJar` restringido al dominio de PhysioNet, tal y como ya funciona en
`design/b.py`. Validación previa del acceso con una petición al CSV de anotaciones
antes de encolar descargas.

**Justificación**: es el mecanismo ya probado en el repositorio. La validación previa
convierte un fallo de 20.000 peticiones en un fallo de una.

**Riesgo asumido**: la cookie caduca. Se mitiga con detección temprana
(`Content-Type: text/html` o estado 401/403 ⇒ aborto ordenado) y con la reanudación
de US2, que hace que renovar la cookie y relanzar cueste sólo el tiempo perdido.

## D-05 — Formato y resolución de salida

**Decisión**: PNG de 8 bits, sin pérdida, a resolución nativa del recorte.

**Justificación**: el destino es preentrenamiento auto-supervisado; introducir
artefactos de compresión en el corpus es una pérdida irreversible que contaminaría
cualquier experimento posterior, y guardar la resolución nativa deja abierta la
elección del tamaño de entrada en el `DataLoader`. El recorte por sí solo elimina ya
la mayor parte del fondo, que es donde se concentra el volumen del DICOM original.

**Consecuencia**: el campo `scale` del catálogo vale 1.0 en esta versión, pero existe
desde el principio para que un futuro pipeline con redimensionado no rompa el
esquema.

**Alternativas descartadas**: JPEG calidad 95 —tres a cinco veces más ligero pero con
pérdida irreversible; PNG de 16 bits —preserva el rango radiométrico completo a costa
de multiplicar el disco y de complicar el *loader*, y la ventana DICOM ya se aplica
en esta etapa.

## D-06 — Detección del campo mamario

**Decisión**: reutilizar el algoritmo ya validado en `design/c.py`: desenfoque
gaussiano, umbral de Otsu, cierre morfológico con un núcleo proporcional al tamaño de
la imagen, selección del mayor componente conexo y `boundingRect` con margen.

**Justificación**: es simple, determinista, sin modelo que entrenar ni pesos que
distribuir, y el mayor componente conexo elimina de forma natural las etiquetas de
orientación y el texto quemado. Para extraer una caja —no una segmentación
anatómica— es suficiente.

**Límite conocido**: un marcador metálico en contacto con el tejido, o un artefacto
que conecte la mama con el borde, inflan la caja. No se rechaza automáticamente por
esa causa; se registra `crop_area_ratio` en el catálogo para poder auditar la cola de
la distribución después de la ejecución.

**Alternativas descartadas**: umbral fijo —frágil ante cambios de fabricante y de
ventana; segmentación con red neuronal —desproporcionada para obtener un rectángulo e
introduce una dependencia de pesos externos.

## D-07 — Persistencia del estado

**Decisión**: manifiesto inmutable en Parquet por ejecución, registro incremental
JSONL de una línea por imagen resuelta escrito por un único escritor (el proceso
principal) y consolidación posterior a Parquet con Polars.

**Justificación**: el JSONL es *append-only*, resistente a una interrupción a mitad de
línea (la línea incompleta se descarta al releer) y no exige transacciones ni un
servidor. Con un único escritor no hay carrera posible entre workers. Polars ya es
dependencia del proyecto.

**Alternativas descartadas**: SQLite —más preciso para consultas de estado, pero
introduce bloqueos y una segunda tecnología de persistencia para 20.000 filas; sólo
sistema de ficheros —no distingue «no intentado» de «fallo permanente».

## D-08 — Orden de escritura y borrado del DICOM

**Decisión**: secuencia estricta por imagen: escribir el PNG en fichero temporal →
`fsync` → renombrar al nombre definitivo → escribir la línea JSONL y vaciar el
búfer → borrar el DICOM.

**Justificación**: hace imposible el estado «DICOM borrado y derivado inexistente».
El estado inverso —PNG presente sin línea en el catálogo— es recuperable: la
reanudación verifica la existencia del PNG antes de dar una imagen por resuelta y,
ante la duda, la reprocesa.

## D-09 — Transformación de coordenadas

**Decisión**: una única función pura que traslada cualquier caja del espacio del
DICOM original al espacio del recorte, y su inversa, ambas con la caja de recorte
como único parámetro de contexto. Las cajas que no quedan contenidas se intersectan y
se marcan.

**Justificación**: es el punto del sistema donde un error silencioso resulta más caro
—un desplazamiento sistemático de coordenadas invalidaría cualquier evaluación de
detección y no se detecta a simple vista—. Al ser pura, se prueba de forma exhaustiva
con casos límite y con una comprobación de ida y vuelta.

## D-10 — Reproducibilidad de los tests

**Decisión**: generar DICOM sintéticos en los propios tests (una elipse clara sobre
fondo oscuro, con variantes MONOCHROME1/MONOCHROME2, distintas profundidades de bit y
una etiqueta de orientación simulada en una esquina) y levantar un servidor
`aiohttp` local para las pruebas de integración de la descarga.

**Justificación**: la constitución prohíbe tests que dependan de la red o de las
credenciales. Un DICOM sintético con la geometría conocida de antemano permite
además verificar numéricamente la caja detectada, algo imposible con una imagen real
sin anotación de referencia.
