# Implementation Plan: Corpus de recortes mamarios desde Mammo-Bench

**Branch**: `002-mammobench-corpus` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

## Summary

Convertir las 19.731 imágenes de Mammo-Bench en un corpus local de recortes de la
región mamaria en PNG sin pérdida, con catálogo Parquet, particiones por paciente e
informe de calidad, y retirar del árbol de trabajo todo el código de descarga y
decodificación DICOM de la feature 001.

El cambio de origen simplifica radicalmente la arquitectura: desaparecen la red, la
autenticación, la contrapresión y el ciclo de vida de ficheros efímeros. Lo que queda
es un problema vergonzosamente paralelo sobre ficheros locales —leer JPEG, derivar
una caja, recortar, escribir— que resuelve un `ProcessPoolExecutor` con un escritor
único de JSONL.

La novedad técnica no es la orquestación sino la **doble fuente de la caja**: la
máscara provista por Mammo-Bench manda, Otsu queda de respaldo, y el IoU entre ambas
se calcula siempre para tener una medida objetiva de cuánto difieren. Esa métrica es
lo que convierte "confío en las máscaras" en una afirmación verificable.

## Technical Context

**Language/Version**: Python 3.12, `uv`

**Primary Dependencies**: `opencv-python` (umbralización, componentes conexos,
escritura PNG), `numpy`, `polars` (CSV, Parquet), `typer`, `rich`, `matplotlib`
(lámina de control), `pytest`

**Storage**: JPEG de entrada, PNG de salida, JSONL incremental, Parquet consolidado

**Testing**: `pytest` con imágenes y máscaras sintéticas de geometría conocida

**Target Platform**: macOS y Linux, ejecución local

**Project Type**: aplicación de línea de comandos con núcleo de biblioteca

**Performance Goals**: 19.731 imágenes en menos de 60 minutos con 8 procesos

**Constraints**: memoria acotada por el número de procesos, no por el corpus; la
imagen más grande del corpus es de 4.754×6.000 px (`dmid`), unos 85 MB descomprimida
en escala de grises

**Scale/Scope**: 19.731 imágenes, 5.860 pacientes, 6 fuentes, ~11 GB de entrada

## Constitution Check

*GATE: debe pasarse antes de la fase 0 y volver a comprobarse tras la fase 1.*

| Principio | Cumplimiento |
|---|---|
| I. Motor puro | `manifest`, `boxing`, `geometry`, `segmentation`, `splits`, `quality` son puros. La E/S queda en `image_io`, `storage` y `catalog`. `tests/unit/test_purity.py` lo verifica por AST. |
| II. Reproducibilidad | Subcomandos `build`, `consolidate`, `split`, `qc`, `inspect`. Configuración completa y semilla registradas en `runs.jsonl`. Particiones deterministas. |
| III. Tests | Máscaras sintéticas de geometría conocida permiten verificar numéricamente la caja; el IoU máscara-Otsu tiene test propio. |
| IV. Cómputo acotado y reanudable | Pool de procesos con número configurable, reanudación por existencia de fichero más registro, memoria función del número de procesos. No hay GPU en esta feature. |
| V. Trazabilidad | `box_source`, umbral, margen, dimensiones nativas y de recorte, IoU y versión en cada fila. |
| VI. Evaluación honesta | Las particiones son por paciente y la ausencia de fuga se verifica automáticamente, no se confía en la construcción. |

**Resultado**: sin violaciones.

## Project Structure

### Documentation (this feature)

```text
specs/002-mammobench-corpus/
├── spec.md
├── plan.md              # este fichero
├── research.md          # decisiones D-01…D-06
├── data-model.md        # entidades y esquemas de los artefactos
├── quickstart.md        # puesta en marcha y verificación visual
├── migration.md         # retirada de 001-vindr-streaming-etl
├── contracts/
│   └── module-contracts.md   # firmas y contratos de los módulos puros
└── tasks.md
```

### Source Code (repository root)

```text
src/mammo_lejepa/
├── config.py            # CorpusConfig: rutas, umbral de máscara, margen, procesos, semilla
├── models.py            # ImagenMammoBench, CajaMamaria, RegistroDeRecorte, BoxSource
├── manifest.py          # PURO: manifiesto desde mammo-bench.csv, filtros, verificación de rutas
├── boxing.py            # PURO: caja desde máscara, política máscara→Otsu, IoU entre cajas
├── segmentation.py      # PURO: Otsu + mayor componente conexo (conservado de 001)
├── geometry.py          # PURO: recorte y traslación de coordenadas (conservado de 001)
├── splits.py            # PURO: particiones por paciente, estratificadas y deterministas
├── quality.py           # PURO: estadísticos del informe, criterio de sospecha
├── resume.py            # PURO: trabajo pendiente (conservado de 001)
├── image_io.py          # E/S: lectura de JPEG e imagen/máscara en escala de grises
├── storage.py           # E/S: PNG atómico, JSONL, bloqueo (conservado de 001)
├── catalog.py           # E/S: consolidación a Parquet, unión con etiquetas
├── worker.py            # proceso hijo: ruta -> recorte en disco + registro
├── runner.py            # pool de procesos, escritor único de JSONL, progreso
└── cli.py               # build / consolidate / split / qc / inspect

tests/
├── conftest.py
├── unit/            test_manifest, test_boxing, test_segmentation, test_geometry,
│                    test_splits, test_quality, test_resume, test_purity
├── integration/     test_build_end_to_end, test_resume_build, test_catalog, test_splits_integrity
└── fixtures/        synthetic_mammogram.py   # imagen + máscara de geometría conocida
```

**Structure Decision**: se mantiene el paquete único de la feature 001 y su frontera
puro/E-S. `pipeline.py` desaparece y su lugar lo ocupa `runner.py`, mucho más simple:
sin corrutinas, sin cola, sin contrapresión. El módulo nuevo con peso propio es
`boxing.py`, donde vive la política de doble fuente de la caja.

## Decisiones técnicas

Ver [research.md](./research.md) para las decisiones D-01 a D-07 (política
máscara→Otsu, cálculo del IoU, recorte rectangular sin aplicar la máscara, entrada
desde `Preprocessed_Dataset`, clave de paciente compuesta, política de casos
sospechosos, y realineado de máscara a la resolución de la imagen), con su
justificación y las alternativas descartadas.

## Riesgos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Las máscaras no eliminan el pectoral como dice la documentación | Alto: el corpus entero llevaría pectoral | Verificación visual obligatoria por fuente antes de dar el corpus por bueno (T‑QC, US2) — **cerrado**, ver Verificación abajo |
| `cdd-cesm` con máscaras que cubren el 80 % de la imagen | Medio | El informe por fuente lo cuantifica; decisión explícita documentada, no silenciosa — **cerrado** |
| DDSM a 165 px domina el corpus y sesga el preentrenamiento | Medio | `source_dataset` y resolución nativa en cada fila permiten ablación en la feature 003 |
| Fuga de paciente entre particiones | Alto y silencioso | Clave compuesta y test de intersección vacía como criterio de aceptación — **cerrado**, 0 pacientes con más de una partición sobre las 19.731 imágenes reales |

## Verificación (T041-T042): ejecución real completa, 19.731 imágenes

Ejecutada el 2026-09-15 con `mammo-corpus build --workers 8` sobre el portátil de
trabajo (8 núcleos asignados, 10 disponibles), seguida de `consolidate` y
`split --seed 0`.

**Rendimiento (SC-006)**: 2 min 40 s de principio a fin — muy por debajo del límite
de 60 minutos.

**Resultado (SC-001)**: 19.724 imágenes `ok`, 7 fallidas (`SEGMENTATION`: "no se
detectó ningún componente conexo de mama" en ninguna de las dos fuentes, máscara ni
Otsu). Verificado con `cv2`: las 7 (`ddsm_1636`, `ddsm_3474`, `ddsm_37`,
`ddsm_4590`, `ddsm_6371`, `ddsm_7347`, `ddsm_7394`) son ficheros de imagen y de
máscara enteramente negros (min=max=0) en el propio Mammo-Bench — un defecto de la
fuente, no del pipeline. El catálogo explica cada ausencia mediante
`failure_category`/`error_message`, satisfaciendo SC-001.

**Respaldo a Otsu (SC-002)**: 1 de 19.731 imágenes (0,005 %), muy por debajo del
2 % exigido.

**IoU máscara-Otsu por fuente, mediana observada (SC-003)**:

| Fuente | Mediana IoU | ¿≥ 0,80? |
|---|---|---|
| inbreast | 0,998 | sí |
| dmid | 0,978 | sí |
| kau-bcmd | 0,944 | sí |
| cdd-cesm | 0,925 | sí |
| ddsm | 0,919 | sí |
| cmmd | 0,737 | **no — excepción documentada** |

`cmmd` es la única fuente por debajo de 0,80. Investigado visualmente (ver
`data/corpus/qc/findings.md`, sección "Hallazgos de la ejecución completa"): sus
imágenes nativas de alta resolución (2294×1914, sin recortar) tienen tejido que se
atenúa gradualmente hacia el borde; Otsu subestima sistemáticamente esa extensión,
mientras que el recorte final de la máscara —inspeccionado directamente— es correcto
y limpio. Es la limitación de Otsu que D-01 anticipa como motivo para no usarlo por
defecto, no un defecto de la máscara ni del recorte. SC-003 se da por satisfecho con
esta excepción documentada, en los términos que el propio criterio prevé.

**Particiones (SC-005)**: `train` 15.799 (80,1 %) / `val` 1.975 (10,0 %) / `test`
1.957 (9,9 %) imágenes, sobre 5.860 pacientes. Verificado directamente sobre
`catalog.parquet`: 0 pacientes en más de una partición, 0 filas con `split` nulo.

**Casos sospechosos**: 4.035 de 19.731 (20,5 %), concentrados en tres fuentes cuyas
imágenes ya vienen recortadas casi sin fondo por el proveedor original —`cdd-cesm`
(959/1.003), `ddsm` (2.948/10.393, sobre todo por `area_ratio`≈1,0 corroborado de
forma independiente por Otsu) y en menor medida `kau-bcmd` (15/2.206)—, más 111 en
`cmmd` y 2 en `dmid`. Ninguno se filtra automáticamente (FR-023); quedan marcados en
`catalog.parquet` para que el entrenamiento decida.

**Conclusión D-01/D-02**: la política "máscara manda, Otsu respalda" queda verificada
con datos reales a escala completa, no sólo con el fixture sintético. La única
desviación de los criterios de éxito (SC-003 en `cmmd`) tiene una causa identificada
y documentada, no un fallo silencioso.

## Complexity Tracking

Sin violaciones de la constitución. No procede.
