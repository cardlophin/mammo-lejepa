from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image

from mammo_lejepa.ssl.config import TrainConfig
from mammo_lejepa.ssl.trainer import train

pytestmark = pytest.mark.slow

# CPU y MPS no garantizan el mismo stream de números aleatorios ni la misma
# implementación de cada kernel (verificado empíricamente: la pérdida difiere
# hasta ~3% entre dispositivos sobre datos sintéticos idénticos con la misma
# semilla). La tolerancia se fija por encima de esa desviación observada,
# generosa a propósito: el objetivo de este test es detectar una rama de
# código específica de un dispositivo que rompa el entrenamiento (D-05), no
# exigir reproducibilidad bit a bit entre backends distintos.
_RELATIVE_TOLERANCE = 0.15


@pytest.fixture
def tiny_catalog(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)

    rows = []
    for i in range(20):
        png_path = images_dir / f"img_{i}.png"
        pixels = (rng.random((200, 150)) * 255).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(png_path)
        rows.append(
            {
                "image_id": f"img_{i}",
                "patient_key": f"src:p{i}",
                "source_dataset": "src",
                "split": "train",
                "status": "ok",
                "crop_path": str(png_path),
                "classification": "Normal",
                "density": "DENSITY B",
                "birads": "2",
                "suspect": False,
            }
        )

    catalog_path = tmp_path / "catalog.parquet"
    pl.DataFrame(rows).write_parquet(catalog_path)
    return catalog_path


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS no disponible")
def test_first_epoch_agrees_between_cpu_and_mps(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    """No hay ninguna rama `if device == "mps"` en `trainer.py`/`objective.py`
    (D-05): el mismo código, con la misma configuración y semilla, debe
    producir una pérdida del mismo orden en CPU y en MPS."""
    results = {}
    for device in ("cpu", "mps"):
        config = TrainConfig(
            catalog_path=tiny_catalog,
            run_dir=tmp_path / f"run_{device}",
            seed=0,
            device=device,
            views=2,
            batch_size=4,
            epochs=1,
            num_slices=64,
            min_batch_size=1,
        )
        summary = train(config)
        results[device] = summary.history[0]

    assert results["cpu"].loss_total == pytest.approx(
        results["mps"].loss_total, rel=_RELATIVE_TOLERANCE
    )
    assert results["cpu"].loss_invariance == pytest.approx(
        results["mps"].loss_invariance, rel=_RELATIVE_TOLERANCE
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS no disponible")
def test_resume_works_on_a_non_cpu_device(tiny_catalog: Path, tmp_path: Path) -> None:
    """Regresión: `load(checkpoint_path, map_location=device)` movía TODO el
    payload —incluido el estado de los generadores aleatorios, que debe
    seguir siendo un `ByteTensor` de CPU— al dispositivo, y
    `torch.set_rng_state` fallaba con `TypeError` al reanudar en MPS.
    Encontrado en T023 (ejecución de humo sobre datos reales), reproducido
    aquí con datos sintéticos para que no vuelva a colarse."""
    run_dir = tmp_path / "run"
    config = TrainConfig(
        catalog_path=tiny_catalog,
        run_dir=run_dir,
        seed=0,
        device="mps",
        views=2,
        batch_size=4,
        epochs=2,
        num_slices=32,
        min_batch_size=1,
    )
    train(config)  # primera "mitad": deja un checkpoint válido en run_dir

    resumed_config = TrainConfig(
        catalog_path=tiny_catalog,
        run_dir=run_dir,
        seed=0,
        device="mps",
        views=2,
        batch_size=4,
        epochs=3,
        num_slices=32,
        min_batch_size=1,
    )
    summary = train(resumed_config)  # no debe lanzar TypeError

    assert summary.epochs_completed == 3
    assert [m.epoch for m in summary.history] == [2]
