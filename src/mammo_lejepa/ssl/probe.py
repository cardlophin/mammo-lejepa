from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import polars as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mammo_lejepa.ssl.augment import build_eval_transform
from mammo_lejepa.ssl.config import AugmentConfig, EvalConfig
from mammo_lejepa.ssl.data import CropDataset, collate_views

# `EvalConfig.targets` usa los nombres de columna del catálogo (`BIRADS`);
# `CropDataset` devuelve las etiquetas en minúsculas.
_LABEL_KEY = {"BIRADS": "birads"}


def _label_key(target: str) -> str:
    return _LABEL_KEY.get(target, target)


@dataclass(frozen=True, slots=True)
class Embeddings:
    """Embeddings del encoder congelado para una partición, con las etiquetas
    de cada fila. Se extraen una sola vez y sirven para todas las sondas."""

    features: Tensor
    labels: dict[str, list[Any]]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Resultado de una sonda. Incluye la línea base trivial de la clase
    mayoritaria y el número de muestras: una exactitud sin ellas no dice
    nada (constitución, principio VI)."""

    target: str
    n_train: int
    n_test: int
    n_classes: int
    accuracy: float
    balanced_accuracy: float
    majority_baseline: float
    accuracy_by_source: dict[str, float]
    n_test_by_source: dict[str, int]
    class_counts_train: dict[str, int]
    class_counts_test: dict[str, int]


def assert_frozen(encoder: nn.Module) -> None:
    """FR-019: ningún parámetro del encoder ha recibido gradiente."""
    leaked = [
        name for name, param in encoder.named_parameters() if param.grad is not None
    ]
    if leaked:
        raise AssertionError(
            f"El encoder no está congelado: {len(leaked)} parámetro(s) con "
            f"gradiente (p. ej. {leaked[0]})."
        )


def extract_embeddings(
    encoder: nn.Module,
    catalog: pl.DataFrame,
    *,
    split: str,
    config: EvalConfig,
) -> Embeddings:
    """Pasa las imágenes de `split` (y sólo de `split`: nunca se reparticiona,
    FR-024) por el encoder en modo evaluación y sin gradientes, con la
    transformación determinista de evaluación (D-09: la misma para todas las
    condiciones)."""
    subset = catalog.filter((pl.col("split") == split) & (pl.col("status") == "ok"))
    if config.max_samples_per_split is not None:
        n = min(config.max_samples_per_split, subset.height)
        subset = subset.sample(n=n, seed=config.seed)

    dataset = CropDataset(
        subset,
        split=split,
        views=1,
        transform=build_eval_transform(AugmentConfig(resolution=config.resolution)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_views,
    )

    device = config.resolved_device()
    encoder.to(device)
    encoder.eval()

    chunks: list[Tensor] = []
    labels: dict[str, list[Any]] = {}
    with torch.no_grad():
        for views, batch_labels in loader:
            batch = views[0].to(device)
            chunks.append(encoder(batch).float().cpu())
            for key, values in batch_labels.items():
                labels.setdefault(key, []).extend(values)

    assert_frozen(encoder)
    if not chunks:
        return Embeddings(features=torch.empty(0, 0), labels=labels)
    return Embeddings(features=torch.cat(chunks), labels=labels)


def _labelled(
    embeddings: Embeddings, target: str
) -> tuple[Tensor, list[str], list[str]]:
    key = _label_key(target)
    values = embeddings.labels.get(key, [])
    sources = embeddings.labels.get("source_dataset", [])
    keep = [i for i, value in enumerate(values) if value is not None]
    features = embeddings.features[keep]
    return (
        features,
        [str(values[i]) for i in keep],
        [str(sources[i]) for i in keep],
    )


def fit_probe(
    train: Embeddings,
    test: Embeddings,
    *,
    target: str,
    config: EvalConfig,
) -> ProbeResult:
    """Sonda lineal (regresión logística) entrenada sobre los embeddings
    congelados de `train` y evaluada en `test`. Sólo se usan las filas que
    tienen la etiqueta (`density`/`BIRADS` no están en todas las fuentes);
    el recuento de muestras va en el resultado. Una clase de `test` que no
    aparece en `train` cuenta como error."""
    x_train, y_train, _ = _labelled(train, target)
    x_test, y_test, s_test = _labelled(test, target)

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError(
            f"Sin muestras etiquetadas para {target!r} "
            f"(train={len(y_train)}, test={len(y_test)})."
        )

    classes = sorted(set(y_train))
    class_index = {name: i for i, name in enumerate(classes)}
    t_train = torch.tensor([class_index[y] for y in y_train])

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        linear = nn.Linear(x_train.shape[1], len(classes))
        optimizer = torch.optim.Adam(
            linear.parameters(),
            lr=config.probe_lr,
            weight_decay=config.probe_weight_decay,
        )
        for _ in range(config.probe_epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(linear(x_train), t_train)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        predicted = [classes[i] for i in linear(x_test).argmax(dim=1).tolist()]

    correct = [p == y for p, y in zip(predicted, y_test, strict=True)]
    counts_test = Counter(y_test)

    recalls = []
    for name in counts_test:
        hits = sum(c for c, y in zip(correct, y_test, strict=True) if y == name)
        recalls.append(hits / counts_test[name])

    by_source: dict[str, list[bool]] = {}
    for source, ok in zip(s_test, correct, strict=True):
        by_source.setdefault(source, []).append(ok)

    return ProbeResult(
        target=target,
        n_train=len(y_train),
        n_test=len(y_test),
        n_classes=len(classes),
        accuracy=sum(correct) / len(correct),
        balanced_accuracy=sum(recalls) / len(recalls),
        majority_baseline=max(counts_test.values()) / len(y_test),
        accuracy_by_source={s: sum(v) / len(v) for s, v in sorted(by_source.items())},
        n_test_by_source={s: len(v) for s, v in sorted(by_source.items())},
        class_counts_train=dict(Counter(y_train)),
        class_counts_test=dict(counts_test),
    )


def linear_probe(
    encoder: nn.Module,
    catalog: pl.DataFrame,
    *,
    target: str,
    config: EvalConfig,
) -> ProbeResult:
    train = extract_embeddings(encoder, catalog, split="train", config=config)
    test = extract_embeddings(encoder, catalog, split="test", config=config)
    return fit_probe(train, test, target=target, config=config)


def run_probes(
    encoder: nn.Module, catalog: pl.DataFrame, *, config: EvalConfig
) -> dict[str, ProbeResult]:
    """Todas las sondas del protocolo —las etiquetas clínicas y la de control
    sobre `source_dataset` (D-10, FR-023)— con una única extracción de
    embeddings. La sonda de control se ejecuta siempre: no es opcional."""
    train = extract_embeddings(encoder, catalog, split="train", config=config)
    test = extract_embeddings(encoder, catalog, split="test", config=config)

    results: dict[str, ProbeResult] = {}
    for target in (*config.targets, config.control_target):
        results[target] = fit_probe(train, test, target=target, config=config)
    return results
