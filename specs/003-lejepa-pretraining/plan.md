# Implementation Plan: Preentrenamiento LeJEPA sobre el corpus mamario

**Branch**: `003-lejepa-pretraining` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

## Summary

Implementar el objetivo de LeJEPA —SIGReg más invarianza entre vistas— como un módulo
puro de PyTorch que sólo conoce tensores `[V, B, D]`, validarlo contra el paquete
oficial, y construir a su alrededor el mínimo aparato necesario para preentrenar un
ResNet-50 sobre los recortes de Mammo-Bench y medir si la representación resultante
vale para algo.

El orden importa: primero el objetivo verificado numéricamente sin gastar cómputo,
después el bucle de entrenamiento, después la evaluación con líneas base. Invertir
ese orden lleva a entrenar durante horas con un estadístico mal implementado y a
atribuir el mal resultado al método.

## Technical Context

**Language/Version**: Python 3.12, `uv`

**Primary Dependencies**: `torch` (≥2.4, con soporte MPS), `torchvision` (ResNet-50 y
transformaciones), `timm` (ViT para la historia 4), `polars` (catálogo), `numpy`,
`typer`, `rich`. Desarrollo: `pytest`, `lejepa` (sólo para el test de validación
cruzada)

**Storage**: checkpoints y registros bajo `runs/<run_id>/`; el corpus es de sólo
lectura

**Testing**: `pytest` sobre tensores deterministas en CPU

**Target Platform**: macOS con MPS para desarrollo; Linux con CUDA para
entrenamientos largos

**Project Type**: biblioteca con CLI

**Performance Goals**: el objetivo no debe suponer más del 15 % del tiempo de paso;
coste lineal en el tamaño de lote

**Constraints**: memoria acotada por lote y número de vistas; sin dependencias
específicas de CUDA en el camino principal; un entrenamiento debe poder cortarse en
cualquier momento

**Scale/Scope**: ~15.800 imágenes de entrenamiento (partición `train` del corpus),
128×128, V=4 vistas

## Constitution Check

| Principio | Cumplimiento |
|---|---|
| I. Motor puro | `objective.py`, `diagnostics.py` y `schedules.py` son funciones de tensores sin E/S. El disco vive en `data.py`, `checkpoint.py` y `reporting.py`. |
| II. Reproducibilidad | Configuración efectiva completa, semilla y versión en `runs/<run_id>/config.json`; semillas fijadas para Python, NumPy y Torch; orden del `DataLoader` determinista dada la semilla. |
| III. Tests | El objetivo se prueba contra propiedades estadísticas conocidas **y** contra la implementación oficial. Es la exigencia del principio III aplicada a un estimador. |
| IV. Cómputo acotado y reanudable | Dispositivo como parámetro, sin ramas; checkpoints atómicos con estado completo; reanudación verificada por test. |
| V. Trazabilidad | Cada resultado de evaluación referencia el checkpoint, la configuración y el `run_id` que lo produjo. |
| VI. Evaluación honesta | Particiones por paciente heredadas del catálogo, líneas base obligatorias, sonda de control sobre `source_dataset`, métricas desagregadas por fuente. |

**Resultado**: sin violaciones.

## Project Structure

```text
src/mammo_lejepa/ssl/
├── __init__.py
├── config.py          # TrainConfig, EvalConfig, AugmentConfig; serialización completa
├── objective.py       # PURO: epps_pulley, sigreg, invariance, lejepa_loss
├── diagnostics.py     # PURO: rango efectivo, entropía espectral, desviación isótropa
├── schedules.py       # PURO: calentamiento lineal + coseno
├── encoders.py        # registro nombre -> (constructor, dim de salida); ResNet-50, ViT
├── projector.py       # MLP de proyección
├── augment.py         # canalización de vistas para mamografía
├── data.py            # E/S: Dataset sobre catalog.parquet, collate de V vistas
├── checkpoint.py      # E/S: guardado atómico, carga, reanudación
├── trainer.py         # bucle de entrenamiento, agnóstico de dispositivo
├── probe.py           # sonda lineal sobre encoder congelado
├── baselines.py       # pesos aleatorios e ImageNet bajo el mismo protocolo
├── reporting.py       # E/S: tabla de resultados, historial, registro de la ejecución
└── cli.py             # pretrain / probe / diagnose / compare

tests/ssl/
├── test_objective.py       # propiedades estadísticas, gradientes, formas
├── test_objective_vs_official.py   # validación cruzada contra el paquete lejepa
├── test_scaling.py         # linealidad en tiempo y memoria
├── test_diagnostics.py     # rango efectivo sobre casos conocidos
├── test_schedules.py
├── test_augment.py         # invariantes de la canalización de vistas
├── test_data.py            # respeto de la partición, V vistas por imagen
├── test_checkpoint.py      # reanudación exacta
└── test_encoder_agnostic.py  # el objetivo corre con ResNet y con ViT sin cambios
```

**Structure Decision**: subpaquete `ssl/` dentro del paquete existente. El corpus
(feature 002) y el preentrenamiento (003) comparten repositorio y configuración pero
no comparten código: la única superficie de contacto es `catalog.parquet`. Esa
frontera estrecha es deliberada — permite reconstruir el corpus o cambiar de método
de SSL sin tocar el otro lado.

## Decisiones técnicas

### D-01 — SIGReg: qué se implementa exactamente

El estadístico de Epps-Pulley contrasta la función característica empírica de una
muestra unidimensional contra la de una normal estándar, integrando la diferencia
contra una función de peso. SIGReg lo aplica sobre `num_slices` proyecciones
aleatorias de los embeddings, con cuadratura de `num_points` nodos (17 por defecto en
la referencia, sobre el intervalo [0, 3]).

La referencia oficial expone:

```python
univariate_test = lejepa.univariate.EppsPulley(num_points=17)
loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test, num_slices=1024
)
loss = loss_fn(embeddings)  # embeddings: [num_samples, num_dims]
```

**Cerrado en T002 (2026-09-16)**, leyendo el código instalado
(`lejepa==0.0.1`, commit `c293d29`, instalado vía
`git+https://github.com/rbalestr-lab/lejepa` — **el paquete no está publicado en
PyPI pese a que su propio README dice `pip install lejepa`**; verificado con un 404
directo contra `pypi.org/pypi/lejepa/json`, así que esta feature lo declara como
dependencia de git, no de PyPI) y `MINIMAL.md` del mismo repositorio (el "ejemplo
mínimo oficial" al que se refiere esta sección y D-02/D-03):

- **Cuadratura**: trapezoidal, `t` linealmente espaciado en `[0, 3]` con 17 nodos
  (impar), peso `2·dt` salvo medio peso (`dt`) en los dos extremos, multiplicado
  además por `φ(t) = exp(-t²/2)` (la función característica teórica de la normal
  estándar, real por simetría). No hay variante Gauss-Hermite activa por defecto.
- **Direcciones aleatorias**: `torch.randn` sobre `(D, num_slices)`, normalizadas a
  norma 2 unitaria por columna (`A /= A.norm(p=2, dim=0)`). **Se remuestrean en cada
  llamada**, nunca se fijan tras la primera.
- **Sin estandarización previa**: los embeddings (en la práctica, la salida del
  proyector, no del encoder — D-04) se proyectan tal cual; SIGReg no resta la media
  ni divide por la desviación típica antes de proyectar. Es el propio objetivo el
  que empuja al proyector a producir algo ya isótropo.
- **`num_slices`**: sin un único valor "oficial" — el snippet aislado del README usa
  1024, el ejemplo de entrenamiento completo de `MINIMAL.md` usa 256. Esta feature
  mantiene 1024 como valor por defecto de `sigreg()` (contracts/module-contracts.md),
  por ser el que aparece en la presentación aislada del estadístico.
- **Forma de entrada real**: en `MINIMAL.md`, `sigreg(proj)` recibe `proj` con forma
  `[V, B, proj_dim]` (proyector, no encoder) y el broadcasting de PyTorch hace que el
  estadístico se calcule **por vista** y se promedie al final (`.mean()` sobre vista
  y sobre `slices`), no sobre las `V·B` muestras mezcladas. `lejepa_loss` debe
  replicar esto explícitamente, no asumir que aplanar `[V,B,D] -> [V·B,D]` da el
  mismo resultado (no lo da: mezclaría vistas de imágenes distintas en la misma
  proyección, cambiando el estadístico).

**Hallazgo colateral, documentado y descartado**: una lectura previa (vía extracción
HTML del PDF de arXiv, con artefactos de codificación) sugería `λ=0,05` y un esquema
"multi-crop" asimétrico (`Vg=2` vistas globales de 224×224, `Vl` vistas locales de
96×96, ancla de invarianza sólo en la media de las globales). **Se descarta para esta
feature**: es la configuración de un experimento a gran escala descrito en el cuerpo
del paper (multi-resolución, imagen natural), no la de `MINIMAL.md`. `MINIMAL.md` usa
`λ=0,02`, `V=4`, todas las vistas a la misma resolución y el ancla es la media
simétrica de las `V` vistas — exactamente lo que D-02/D-03 y `plan.md` (Scale/Scope)
ya tenían escrito. Confirmado, no corregido. Adoptar el esquema multi-crop no tendría
sentido aquí: el corpus ya es pequeño y de resolución fija (128×128) por la propia
heterogeneidad de Mammo-Bench, sin la variedad de escalas nativas de ImageNet que
motiva las vistas locales/globales del paper.

### D-02 — Invarianza sin predictor

El término de invarianza es la dispersión de las `V` vistas de una misma imagen
respecto a su media: el ejemplo mínimo oficial lo describe como la desviación
respecto a la proyección media. No hay predictor asimétrico ni stop-gradient, que es
justamente la simplificación que LeJEPA reivindica. El gradiente fluye por todas las
vistas.

**Cerrado en T002**: `MINIMAL.md` lo implementa literalmente como
`(proj.mean(0) - proj).square().mean()` sobre `proj` de forma `[V, B, D]` —la media
es sobre el eje `V`, simétrica entre todas las vistas, sin distinguir vistas
"globales" de "locales" (D-01 descarta esa variante para esta feature)—, y opera
sobre la salida del **proyector**, no del encoder (D-04).

### D-03 — `λ` es el único hiperparámetro de compromiso

Valor por defecto 0,02, el del ejemplo mínimo oficial (confirmado en T002 contra
`MINIMAL.md`: `python mnist.py +lamb=0.02 +V=4 ...`). Se expone en la configuración y
se registra en cada ejecución. Cualquier barrido sobre `λ` deja fijo todo lo demás.

### D-04 — Interfaz de encoder

Un registro `nombre -> (constructor, dim_salida)`. Un encoder es cualquier
`nn.Module` cuyo `forward` devuelva `[B, D]`. El objetivo no lo importa, no lo
inspecciona y no lo instancia: lo recibe ya evaluado. Así, añadir una arquitectura es
añadir una entrada al registro.

La sonda lineal evalúa la salida del **encoder**, no del proyector: el proyector se
descarta tras el preentrenamiento, como es estándar en SSL.

### D-05 — Agnosticismo de dispositivo sin ramas

`device` es un parámetro con detección automática por defecto (`cuda` → `mps` →
`cpu`). La precisión mixta se activa mediante un gestor de contexto que degrada a
`float32` cuando el dispositivo no soporta el tipo pedido, en un solo punto del
código. No hay `if device == "mps"` repartido por el bucle.

### D-06 — Aumentaciones pensadas para mamografía, no heredadas

La receta de LeJEPA para imagen natural incluye *color jitter*, conversión a escala
de grises y *solarize*. En mamografía de un canal, los dos primeros son inertes o
absurdos, y el tercero destruye la relación de intensidad con la densidad del tejido,
que es una de las etiquetas a predecir.

Punto de partida propuesto, a fijar en T-012 y registrado en cada ejecución:

- recorte aleatorio con escala **revisada** respecto al 0,08-1,0 de ImageNet, porque
  el corpus ya viene recortado a la mama y una vista del 8 % puede no contener tejido
- volteo horizontal, admisible porque el corpus contiene mamas izquierdas y derechas,
  declarando que destruye la lateralidad como señal
- jitter suave de brillo y contraste, con rango conservador y justificado
- desenfoque gaussiano
- sin *solarize*, sin conversión de color, sin rotaciones grandes (la orientación en
  mamografía es canónica)

### D-07 — Checkpoints atómicos con estado completo

Escritura a temporal, `fsync`, renombrado. Contienen modelo, proyector, optimizador,
programador, estado de los generadores aleatorios, época y configuración. La
reanudación que no restaura el estado del generador no es reanudación: cambia la
secuencia de aumentaciones y la curva no continúa.

### D-08 — Validación cruzada como test permanente

El paquete oficial entra como dependencia **de desarrollo**. El test compara ambas
pérdidas sobre los mismos tensores con semilla fijada. Si la implementación oficial
cambia y deja de coincidir, el test lo detecta. El código de producción nunca lo
importa: la implementación propia es la que se usa.

### D-09 — Líneas base obligatorias

Tres condiciones bajo idéntico protocolo de sonda: encoder aleatorio congelado,
encoder ImageNet congelado y encoder LeJEPA congelado. La primera acota por abajo, la
segunda dice si el preentrenamiento en el dominio aporta algo sobre transferir de
imagen natural. Sin ellas, la exactitud de la sonda es un número sin escala.

### D-10 — Sonda de control sobre la fuente

Una sonda lineal que intente predecir `source_dataset` desde los embeddings. Con
DDSM a 165 px y DMID a 4.748, distinguir la fuente es trivial por nitidez; si esa
sonda acierta casi siempre mientras la de patología va mal, el encoder está modelando
el equipo de adquisición. Se reporta siempre, junto a las demás.

## Riesgos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Error sutil en el estadístico de Epps-Pulley | Alto: entrenamientos inútiles sin señal de alarma | Validación cruzada contra el paquete oficial como test permanente (D-08) |
| El corpus es pequeño para SSL | Medio | Expectativas calibradas por escrito; líneas base obligatorias; un resultado negativo bien medido es un resultado |
| El encoder aprende la fuente en lugar de la anatomía | Alto y fácil de pasar por alto | Sonda de control sobre `source_dataset` (D-10) y ablación con y sin DDSM |
| Colapso de representaciones pese a SIGReg | Alto | Diagnóstico de rango efectivo registrado por época con umbral de aviso |
| Lote pequeño en el portátil degrada la estimación de SIGReg | Medio | Aviso explícito por debajo del mínimo; entrenamientos serios en GPU |
| Deriva entre CPU/MPS/CUDA | Medio | Test que compara las primeras iteraciones entre dispositivos disponibles |

## Complexity Tracking

Sin violaciones de la constitución. No procede.
