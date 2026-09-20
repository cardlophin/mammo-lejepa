from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset


class CropDataset(Dataset):
    """Lee exclusivamente filas del `split` pedido; nunca reparticiona
    (FR-008, FR-024). Devuelve `V` vistas por imagen más su identidad y sus
    etiquetas. Si `include_suspect` es falso, `n_excluded_suspect` informa de
    cuántas filas excluyó — nunca lo hace en silencio (FR-011)."""

    def __init__(
        self,
        catalog: pl.DataFrame,
        *,
        split: str,
        views: int,
        transform: Callable[[Image.Image], torch.Tensor],
        include_suspect: bool = True,
        sources: Sequence[str] | None = None,
    ) -> None:
        filtered = catalog.filter(
            (pl.col("split") == split) & (pl.col("status") == "ok")
        )

        if sources is not None:
            filtered = filtered.filter(pl.col("source_dataset").is_in(list(sources)))

        self.n_excluded_suspect = 0
        if not include_suspect:
            total_before = filtered.height
            filtered = filtered.filter(~pl.col("suspect"))
            self.n_excluded_suspect = total_before - filtered.height

        self.split = split
        self.views = views
        self.transform = transform
        self._rows: list[dict[str, Any]] = filtered.to_dicts()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        row = self._rows[index]
        image = Image.open(row["crop_path"]).convert("RGB")
        views = torch.stack([self.transform(image) for _ in range(self.views)])

        labels = {
            "image_id": row["image_id"],
            "patient_key": row.get("patient_key"),
            "source_dataset": row.get("source_dataset"),
            "classification": row.get("classification"),
            "density": row.get("density"),
            "birads": row.get("birads"),
            "suspect": row.get("suspect"),
        }
        return views, labels


def collate_views(
    batch: Sequence[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    """Agrupa un lote de `(vistas[V,C,H,W], etiquetas)` en `(vistas[V,B,C,H,W],
    etiquetas)` (contracts.md): `V` primero, para que se pueda pasar
    directamente a `objective.lejepa_loss` tras el encoder y el proyector."""
    stacked_views = torch.stack([item[0] for item in batch])  # [B, V, C, H, W]
    views = stacked_views.transpose(0, 1).contiguous()  # [V, B, C, H, W]

    label_keys = batch[0][1].keys()
    labels = {key: [item[1][key] for item in batch] for key in label_keys}

    return views, labels
