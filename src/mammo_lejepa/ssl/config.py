from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def resolve_device(explicit: str | None = None) -> str:
    """Detección automática `cuda` -> `mps` -> `cpu` (D-05); un valor
    explícito la evita. Único punto del código que decide el dispositivo por
    defecto."""
    if explicit is not None:
        return explicit

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_json_dict(value: object) -> dict[str, object]:
    def default(item: object) -> object:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"No serializable a JSON: {type(item)!r}")

    return json.loads(json.dumps(asdict(value), default=default))  # type: ignore[call-overload]


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Canalización de vistas para mamografía (D-06, T012): serializable por
    completo — una aumentación que no aparece aquí no puede estar en la
    canalización (contracts.md, `augment.describe`)."""

    resolution: int = 128
    crop_scale_min: float = 0.4
    crop_scale_max: float = 1.0
    horizontal_flip_prob: float = 0.5
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    gaussian_blur_prob: float = 0.5
    gaussian_blur_sigma_min: float = 0.1
    gaussian_blur_sigma_max: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Configuración efectiva de un preentrenamiento (constitución, principio
    II): nada se edita en el código fuente."""

    catalog_path: Path
    run_dir: Path
    seed: int = 0
    device: str | None = None
    encoder_name: str = "resnet50"
    proj_dim: int = 128
    views: int = 4
    lamb: float = 0.02
    num_slices: int = 1024
    num_points: int = 17
    batch_size: int = 256
    epochs: int = 100
    lr: float = 2e-3
    weight_decay: float | None = None  # None -> resuelto por familia de encoder
    warmup_epochs: int = 1
    final_lr_ratio: float = 1e-3
    sources: tuple[str, ...] | None = None
    include_suspect: bool = True
    min_batch_size: int = 128
    min_effective_rank: float = 10.0
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def resolved_device(self) -> str:
        return resolve_device(self.device)

    def to_json(self) -> str:
        return json.dumps(_to_json_dict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> TrainConfig:
        data = json.loads(text)
        data["catalog_path"] = Path(data["catalog_path"])
        data["run_dir"] = Path(data["run_dir"])
        if data.get("sources") is not None:
            data["sources"] = tuple(data["sources"])
        if isinstance(data.get("augment"), dict):
            data["augment"] = AugmentConfig(**data["augment"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """El mismo `EvalConfig` se usa para las líneas base y para el encoder
    preentrenado (contracts.md, `probe.py`): si los protocolos difieren, la
    comparación no vale (D-09)."""

    catalog_path: Path
    seed: int = 0
    device: str | None = None
    batch_size: int = 256
    targets: tuple[str, ...] = ("classification", "density", "BIRADS")
    control_target: str = "source_dataset"
    resolution: int = 128
    max_samples_per_split: int | None = None
    probe_epochs: int = 300
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4

    def resolved_device(self) -> str:
        return resolve_device(self.device)

    def to_json(self) -> str:
        return json.dumps(_to_json_dict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> EvalConfig:
        data = json.loads(text)
        data["catalog_path"] = Path(data["catalog_path"])
        data["targets"] = tuple(data["targets"])
        return cls(**data)
