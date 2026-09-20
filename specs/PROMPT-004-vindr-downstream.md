# Prompt para el agente: recuperar el ETL de VinDr como conjunto de evaluación

Redactado el 2026-09-15 tras verificar el estado del repositorio. Pegar tal cual en
una sesión nueva del agente.

## Estado verificado (para referencia, no forma parte del prompt)

- Rama `002-mammobench-corpus`; corpus construido en `data/corpus/` (19.724 ok / 7 fallos).
- Commit `0de1617` conserva íntegro el ETL de VinDr; las bajas están sólo en el índice.
- Ficheros recuperables: `download.py`, `dicom_io.py`, `windowing.py`, `findings.py`,
  `pipeline.py`, `design/{a,b,c}.py`, `tests/fixtures/synthetic_dicom.py`,
  `tests/unit/test_{windowing,findings,redaction}.py`,
  `tests/integration/test_{download_local_server,pipeline_end_to_end,pipeline_resume,backpressure,failure_isolation,retry_failed,catalog_consolidation}.py`.
- Los CSV de anotaciones NO están en git (`data/` ignorado). `.env` conserva las tres
  variables de PhysioNet.
- Símbolos que el código restaurado espera y que ya no existen: `PipelineConfig`,
  `PhysioNetCredentials`, `ImageTask`, `BreastCrop`, `ProcessOutcome`, `ImageRecord`,
  `DicomPayload`, `DownloadResult`. `BoundingBox` sí sobrevive.

---

## Prompt

Contexto verificado del repo (no lo des por supuesto, compruébalo):

- Rama actual: 002-mammobench-corpus. El corpus de Mammo-Bench está construido en
  data/corpus/ (19.724 recortes ok de 19.731, catalog.parquet, splits.parquet, qc/).
- El commit 0de1617 ("mammo-etl") contiene íntegro el ETL de VinDr-Mammo de la
  feature 001. Las bajas de esos ficheros están sólo en el índice, sin confirmar:
  nada se ha perdido.
- Los CSV de anotaciones de VinDr NO están en git (data/ está en .gitignore): hay que
  volver a descargarlos de PhysioNet a data/vindr-mammo/csv/.
- .env conserva PHYSIONET_USERNAME, PHYSIONET_PASSWORD y PHYSIONET_SESSIONID. La
  cookie estará caducada y habrá que renovarla desde el navegador.

Lo que quiero:

VinDr-Mammo vuelve al proyecto con un papel distinto. Ya no es el corpus de
preentrenamiento — eso es Mammo-Bench. VinDr pasa a ser el conjunto de evaluación
downstream: de sus 20.000 imágenes, 2.254 tienen bounding box de hallazgo con
categoría y BI-RADS, y es lo único en todo el proyecto que permite evaluar
localización. El plan es preentrenar con LeJEPA sobre Mammo-Bench (feature 003) y
después evaluar y afinar ese encoder sobre VinDr usando sus cajas.

Tareas:

1. Escribe con spec-kit la feature 004-vindr-downstream (spec.md, plan.md, tasks.md,
   contracts/). Respeta la constitución v2.0.0, incluido el principio VI.

2. Recupera el código de la feature 001 desde el commit 0de1617, pero NO a su
   ubicación original. Punto de partida:

   git restore --source=0de1617 --worktree --staged \
     src/mammo_lejepa/download.py src/mammo_lejepa/dicom_io.py \
     src/mammo_lejepa/windowing.py src/mammo_lejepa/findings.py \
     src/mammo_lejepa/pipeline.py tests/fixtures/synthetic_dicom.py \
     tests/unit/test_windowing.py tests/unit/test_findings.py \
     tests/integration/test_download_local_server.py \
     tests/integration/test_pipeline_end_to_end.py \
     tests/integration/test_pipeline_resume.py

   Y a continuación muévelos a un subpaquete aislado src/mammo_lejepa/vindr/ con sus
   propios models.py y config.py, igual que el subpaquete ssl/ de la feature 003.

   Motivo, compruébalo antes de empezar: models.py, config.py, manifest.py,
   catalog.py y worker.py se reescribieron para Mammo-Bench. El código de VinDr
   importa PipelineConfig, PhysioNetCredentials, ImageTask, BreastCrop,
   ProcessOutcome, ImageRecord, DicomPayload y DownloadResult, que ya no existen
   (BreastCrop es hoy CajaMamaria, con otros campos). Restaurar en su sitio rompe el
   paquete.

   Reutiliza sin duplicar sólo lo genérico y estable: geometry.py, segmentation.py,
   storage.py, resume.py, errors.py.

3. Restaura en pyproject.toml las dependencias retiradas: aiohttp, aiofiles,
   physionet, pydicom, pylibjpeg, pylibjpeg-openjpeg, requests, dotenv, y
   pytest-asyncio con asyncio_mode. Mantén el script mammo-corpus y AÑADE mammo-etl
   como segundo script; no lo sustituyas.

4. "mammo-etl doctor" debe volver a funcionar y diagnosticar claramente los dos
   estados esperables: cookie caducada y CSV de anotaciones ausentes.

5. El catálogo de VinDr debe salir con la misma convención de coordenadas que el
   corpus: cajas de hallazgo remapeadas al espacio del recorte (requisitos FR-023 y
   FR-024 de la feature 001, cuyo código está en findings.py), y particiones por
   paciente/estudio, nunca por imagen.

6. Alcance de la 004: descargar, recortar y catalogar VinDr con sus cajas, y dejar el
   conjunto listo. La evaluación del encoder (sonda lineal, detección, ajuste fino) se
   especificará aparte cuando la 003 produzca un checkpoint.

7. Resuelve explícitamente en la spec esta tensión y no la dejes implícita: la feature
   001 guardaba los recortes a resolución nativa, pero el encoder de LeJEPA trabaja a
   128x128. Una lesión pequeña puede quedar en pocos píxeles a esa escala. Decide y
   justifica qué resolución tiene sentido para un conjunto de evaluación de
   localización, y si hace falta una resolución distinta de la del preentrenamiento.

No toques: data/corpus/, la feature 002 ya terminada, ni las specs 001, 002 y 003
salvo para añadir referencias cruzadas.

Al terminar dime qué quedó pendiente de decisión y qué requisitos de la 001 dejaron
de tener sentido bajo el papel nuevo.
