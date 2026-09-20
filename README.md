# mammo-lejepa

Self-supervised pretraining of mammography encoders with **LeJEPA**, plus the
data pipeline and the evaluation protocol needed to tell whether it actually
worked.

The project does three things, end to end:

1. **Builds a corpus.** Turns [Mammo-Bench](#data-sources-and-licenses) (six public
   mammography datasets, 19,731 images) into breast-region PNG crops, catalogued
   with clinical labels and split **by patient** into train / val / test.
2. **Pretrains an encoder.** Trains a ResNet-50 (a ViT-S is planned) on those crops with the
   LeJEPA objective: no labels, no teacher network, no stop-gradient, and no
   momentum encoder.
3. **Evaluates honestly.** Probes the frozen encoder against a random-weights
   floor and an ImageNet baseline under one identical protocol, and adds a control
   probe that measures how much the encoder simply learned *which dataset an image
   came from*.

A fourth component prepares [VinDr-Mammo](#vindr-mammo-downstream-set) as a
separate downstream evaluation set, with native-resolution crops and finding boxes
remapped into crop space.

> **Status.** The corpus pipeline (002) and the VinDr pipeline (004) are complete
> and verified on real data. The pretraining and evaluation code (003, phases 1 to
> 4) is implemented and tested, and has been smoke-tested on real data with a
> 3-epoch run. **No full-length pretraining run has been evaluated yet**, so this
> repository does not yet report any result on whether LeJEPA beats the baselines.
> See [Project status](#project-status).

## The pipeline at a glance

![Pipeline overview](docs/img/pipeline.png)

Each box is a command you can run. The three `mammo-*` entry points
(`mammo-corpus`, `mammo-etl`, `mammo-ssl`) map one-to-one onto the stages above.
Interactive versions of every diagram (pan, zoom, trace a path) live in
[`docs/`](docs/): [`pipeline.html`](docs/pipeline.html),
[`training-step.html`](docs/training-step.html) and
[`evaluation.html`](docs/evaluation.html).

## Why LeJEPA

Most self-supervised methods stay away from representation collapse with a stack
of heuristics: a momentum teacher, stop-gradient, a predictor head, carefully
tuned schedules. LeJEPA replaces them with one principled regularizer. If the
embeddings are pushed towards an isotropic Gaussian, they cannot collapse, and
the pretext task reduces to view invariance.

```
loss = lambda * SIGReg(projections) + (1 - lambda) * invariance(projections)
```

- **SIGReg** compares the distribution of the embeddings, projected on many random
  1-D directions, with a standard Gaussian using the Epps-Pulley statistic. The
  directions are redrawn at every call, so the model cannot overfit to a fixed set.
- **Invariance** pulls the `V` augmented views of each image towards their mean
  embedding.
- Defaults: `lambda = 0.02`, `V = 4` views, 1024 slices, 17 quadrature nodes.

For mammography this is attractive for a practical reason: there is no reliable
teacher to distil from, the images are large and heavily structured, and a
collapse alarm that does not depend on labels is exactly what you want when
labels are scarce.

The objective is implemented from the paper and **verified against the official
`lejepa` package**: the core statistic matches bit for bit, and the full pipeline
matches within sampling noise. The official package is installed only as a
dev-dependency for that cross-check; training does not import it.

### One training step

![One LeJEPA training step](docs/img/training-step.png)

A batch of breast crops becomes `V` random views. The encoder produces embeddings,
a projector maps them to `[V, B, D]`, and the two loss terms are computed on the
projections. In parallel, **detached** embeddings feed a diagnostics module
(effective rank, spectral entropy, isotropy deviation). If the effective rank
falls below a threshold, the trainer emits a collapse warning during training
instead of letting it surface in the final metrics. Every epoch ends with a
checkpoint holding weights, optimizer state and every RNG state.

## Evaluation you can trust

![Evaluation protocol](docs/img/evaluation.png)

A linear-probe accuracy means nothing on its own. The protocol is built to make
that number interpretable:

- **Three conditions, one recipe.** Random frozen weights (the floor), ImageNet
  frozen weights (does in-domain pretraining beat transfer?), and the LeJEPA
  checkpoint. Same transform, same probe, same seed, same partitions.
- **Splits come from the catalog, never re-drawn.** The unit of partition is the
  patient, so no patient appears in more than one split.
- **Majority-class baseline and `n` in every row.** Accuracy, balanced accuracy,
  the majority-class rate and the sample counts are always reported together.
- **Per-source breakdown.** Each result is broken down by `source_dataset`.
- **A source-identification control.** A probe that predicts `source_dataset`.
  If it is nearly perfect, part of any clinical accuracy may be a shortcut.
- **Negative results are recorded.** A run that does not beat the baselines is
  written up the same way as one that does.

Targets: `classification`, `density`, `BIRADS` (each only on rows that carry the
label), plus the source control.

## Data sources and licenses

Mammo-Bench aggregates six public datasets, each with its own terms of use. The
exact terms are not listed here because they change and are the responsibility of
whoever redistributes: check the original source before using the corpus outside
this repository.

| Source | Images | Patients |
|---|---|---|
| DDSM | 10,400 | 2,280 |
| CMMD | 5,202 | 1,856 |
| KAU-BCMD | 2,206 | 478 |
| CDD-CESM | 1,003 | 326 |
| DMID | 510 | 510 |
| INbreast | 410 | 410 |

19,731 images and 5,860 patients in total. Any publication of results should cite
the six original sources in addition to Mammo-Bench.

## Clinical data: do not redistribute

Images, their derivatives (PNG crops, the Parquet catalog) and the annotation CSV
are **not redistributed and must not be uploaded to any third-party service**. The
whole of `data/` is outside version control (`.gitignore`), as are `runs/` and
checkpoints.

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Mammo-Bench is a local corpus and needs no credentials. `data/Mammo_Bench_v2/`
must exist with `CSV_Files/mammo-bench.csv` and, per source, the
`Preprocessed_Dataset/`, `Masks/` and `Original_Dataset/` directories.

### 1. Build the corpus (`mammo-corpus`)

```bash
uv run mammo-corpus build --source inbreast --limit 8    # small sample from one source
uv run mammo-corpus inspect --source inbreast --n 8      # visual contact sheet: check before scaling up
uv run mammo-corpus build --workers 8                    # full corpus, interruptible with Ctrl-C
uv run mammo-corpus qc                                   # quality report + percentiles per source
uv run mammo-corpus consolidate                          # records.jsonl -> catalog.parquet
uv run mammo-corpus split --seed 0 --ratios 0.8 0.1 0.1  # patient-level split -> splits.parquet
```

Re-running `build` with the same command resumes where it stopped and does not
reprocess finished work. The breast bounding box comes from the dataset mask when
one exists and falls back to an Otsu threshold otherwise; the source of every box
is recorded, and implausible crops are flagged `suspect` rather than silently
dropped. See `specs/002-mammobench-corpus/quickstart.md` for the full walkthrough,
including the mandatory visual check before the full run.

### 2. Pretrain (`mammo-ssl pretrain`)

```bash
uv run mammo-ssl pretrain --epochs 100 --batch-size 256            # ResNet-50, new run in runs/run_<id>
uv run mammo-ssl pretrain --source ddsm --source cmmd              # restrict to some sources
```

- **Interruptible and resumable.** `Ctrl-C` (or `SIGTERM`) saves a checkpoint
  before exiting. Point `--run-dir` at the same directory to continue; no separate
  resume flag exists. On CPU the resumed run reproduces an uninterrupted one bit
  for bit.
- **Device-agnostic.** CUDA, Apple MPS or CPU are picked automatically (`--device`
  to override). Mixed precision is `bfloat16` on CUDA only, in a single code path.
  MPS is not bit-reproducible against CPU.
- **Reproducible.** The seed, the full config (`config.json`), the per-epoch
  history (`history.jsonl`) and all RNG states are stored with the run.

### 3. Evaluate (`mammo-ssl probe | diagnose | compare`)

```bash
uv run mammo-ssl probe    --run-dir runs/<run_id>     # random / ImageNet / LeJEPA probes + source control
uv run mammo-ssl probe    --run-dir runs/<run_id> --max-samples 300 --no-imagenet   # quick check
uv run mammo-ssl diagnose --run-dir runs/<run_id>     # rank / isotropy per epoch + on the test split
uv run mammo-ssl compare  --run-dir runs/a --run-dir runs/b   # side-by-side table
```

`probe` writes `runs/<run_id>/eval/results.json` and `results.md`, along with the
evaluation config and the checkpoint that produced them. The ImageNet baseline
downloads its weights the first time it is used.

### VinDr-Mammo downstream set

`mammo-etl` streams VinDr-Mammo from PhysioNet, crops each DICOM to the breast
region **at native resolution** (never resized at build time, so the evaluation can
choose its own resolution), and remaps every finding box into crop coordinates. It
partitions by study using the dataset's official split.

```bash
uv run mammo-etl doctor                                  # env check, downloads nothing
uv run mammo-etl run --split training --n-studies 5      # bounded sample
uv run mammo-etl resume                                  # continue the last run
uv run mammo-etl retry                                   # retry only failed images
```

It needs PhysioNet credentials in a git-ignored `.env`. `doctor` tells an expired
session cookie apart from missing annotation CSVs. See
`specs/004-vindr-downstream/quickstart.md`.

## Catalog schema

### `catalog.parquet`: one row per image

| Group | Fields |
|---|---|
| Identity | `image_id`, `source_dataset`, `source_subject_id`, `patient_key`, `laterality`, `view` |
| Outcome | `status` (`ok`\|`failed`), `failure_category`, `error_message`, `crop_path`, `crop_bytes` |
| Geometry | `image_height`, `image_width`, `crop_x0`, `crop_y0`, `crop_x1`, `crop_y1`, `crop_height`, `crop_width`, `crop_margin_px`, `crop_area_ratio` |
| Quality | `box_source` (`MASK`\|`OTSU`), `fallback_reason`, `threshold`, `mask_area_ratio`, `mask_otsu_iou`, `suspect`, `suspect_reason` |
| Labels | `classification`, `density`, `birads`, `abnormality`, `molecular_subtype`, `subject_age` |
| Provenance | `run_id`, `code_version`, `processed_at`, `process_seconds` |
| Partition | `split` (`train`\|`val`\|`test`, added by `split`) |

Uniqueness key: `image_id`. `suspect` flags a crop with an implausible area or
aspect ratio, or a low mask/Otsu IoU. It is **not filtered automatically**: what to
do with it is a training decision (`--include-suspect` / `--no-include-suspect`).

### `splits.parquet`: one row per `(patient_key, split)`

`patient_key`, `source_dataset`, `split`, `n_images`, `seed`. The partition unit is
the patient (`source_dataset` + `source_subject_id`), never the image: no
`patient_key` appears in more than one split. A patient with images of more than
one `classification` is stratified by the most severe present (`Malignant` >
`Suspicious Malignant` > `Benign` > `Normal`).

### `quality_by_source.parquet` and `qc/quality_report.md`

Percentiles of `crop_area_ratio` and `mask_otsu_iou` per source, the count of Otsu
fallbacks, and the list of suspect cases by path (`mammo-corpus qc`).

## Known data-quality notes

- **`cdd-cesm`**: images already ship tightly cropped to the breast by the original
  provider, so `crop_area_ratio` is close to 1.0 for the whole source and the
  suspect check flags it systematically on area. That is not a cropping defect;
  see `specs/002-mammobench-corpus/research.md` (D-01, D-07).
- **`ddsm`**: mask and image almost never share native resolution; they are
  realigned with nearest-neighbour before the box is computed (D-07).
- **`kau-bcmd`**: manual inspection found masks systematically narrower than the
  visible tissue in several samples; watch for low `mask_area_ratio` in the full
  quality report.

## Project layout

```
src/mammo_lejepa/
  geometry.py segmentation.py boxing.py quality.py splits.py   pure logic (no I/O)
  image_io.py storage.py catalog.py manifest.py resume.py      I/O and persistence
  runner.py worker.py cli.py                                    mammo-corpus
  ssl/                                                          mammo-ssl
    objective.py diagnostics.py schedules.py augment.py         pure: loss, metrics, LR, views
    encoders.py projector.py data.py checkpoint.py trainer.py   model, data, training loop
    probe.py baselines.py reporting.py cli.py                   evaluation and CLI
  vindr/                                                        mammo-etl (VinDr-Mammo)
specs/    spec-kit artifacts per feature (spec, plan, research, contracts, tasks)
docs/     interactive diagrams (HTML) and their PNG exports (img/)
tests/    mirrors src/, plus tests/ssl and tests/vindr
```

The code follows a strict split: **pure modules** (geometry, objective, metrics,
schedules) never touch disk, network or the clock, so they are unit-tested in
isolation; **I/O modules** wrap them. This is the first principle of the project
constitution (`.specify/memory/constitution.md`), alongside reproducibility,
bounded and resumable execution, traceability, and honest evaluation.

## Tests

```bash
uv run pytest                     # whole suite, no real data and no GPU required
uv run pytest tests/ssl -q        # pretraining and evaluation
uv run pytest -m "not slow"       # skip the tests that train tiny models
```

## Project status

| Feature | What | State |
|---|---|---|
| 001 | VinDr streaming ETL (original) | Superseded by 004 |
| 002 | Mammo-Bench corpus | Complete, verified on real data |
| 003 | LeJEPA pretraining and evaluation | Phases 1 to 4 implemented; **T031 (critical reading of a real run) pending** |
| 004 | VinDr-Mammo downstream set | Complete; verified on 3 real studies |

Still to do on 003: a full-length pretraining run and its written analysis
(`runs/<run_id>/findings.md`), the ViT-S encoder and its architecture-agnostic test
(phase 5), and an ablation on DDSM (phase 6). Wiring the VinDr set into the probe
is out of scope for now.

Every feature is specified under `specs/<nnn>-<name>/` with `spec.md`, `plan.md`,
`research.md`, `data-model.md`, `contracts/`, `quickstart.md` and `tasks.md`.
