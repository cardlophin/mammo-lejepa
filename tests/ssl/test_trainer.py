from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mammo_lejepa.ssl.config import TrainConfig
from mammo_lejepa.ssl.trainer import train

pytestmark = pytest.mark.slow


def _config(catalog_path: Path, run_dir: Path, **overrides: object) -> TrainConfig:
    defaults = dict(
        catalog_path=catalog_path,
        run_dir=run_dir,
        seed=0,
        device="cpu",
        views=2,
        batch_size=4,
        epochs=2,
        num_slices=32,
        min_batch_size=1,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)  # type: ignore[arg-type]


def test_train_produces_expected_summary_shape(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    config = _config(tiny_catalog, tmp_path / "run")

    summary = train(config)

    assert summary.exit_reason == "completed"
    assert summary.epochs_completed == 2
    assert len(summary.history) == 2
    assert [m.epoch for m in summary.history] == [0, 1]
    for metrics in summary.history:
        assert metrics.loss_total >= 0
        assert metrics.effective_rank > 0


def test_train_writes_config_and_history_to_run_dir(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    config = _config(tiny_catalog, run_dir)

    train(config)

    assert (run_dir / "config.json").exists()
    history_lines = (run_dir / "history.jsonl").read_text().splitlines()
    assert len(history_lines) == 2

    restored_config = TrainConfig.from_json((run_dir / "config.json").read_text())
    assert restored_config == config


def test_train_warns_below_minimum_batch_size(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    config = _config(tiny_catalog, tmp_path / "run", batch_size=2, min_batch_size=128)

    with pytest.warns(UserWarning, match="FR-018"):
        train(config)


def test_train_raises_on_empty_dataset_after_filters(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    config = _config(
        tiny_catalog, tmp_path / "run", sources=("this-source-does-not-exist",)
    )

    with pytest.raises(ValueError, match="vacío"):
        train(config)


def test_resume_reproduces_the_exact_continuation(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    """T018: entrenar hasta el final sin interrupción, y por separado
    simular una interrupción tras la época 1 (truncando checkpoints e
    historial, no cambiando `epochs`: el horizonte del programador de tasa de
    aprendizaje depende de `epochs` desde el primer paso, así que cambiarlo
    entre la primera mitad y la reanudación compararía dos entrenamientos
    distintos, no una interrupción real). El paso siguiente a la reanudación
    debe ser idéntico al que se habría dado sin cortar — incluida la
    secuencia de aumentaciones, restaurada vía el estado del generador
    aleatorio (D-07)."""
    run_full = tmp_path / "run_full"
    config_full = _config(tiny_catalog, run_full, epochs=4)
    summary_full = train(config_full)

    run_resumed = tmp_path / "run_resumed"
    shutil.copytree(run_full, run_resumed)
    for epoch in (2, 3):
        (run_resumed / "checkpoints" / f"epoch_{epoch:04d}.pt").unlink()
    kept_lines = (run_resumed / "history.jsonl").read_text().splitlines()[:2]
    (run_resumed / "history.jsonl").write_text("\n".join(kept_lines) + "\n")

    config_resume = _config(tiny_catalog, run_resumed, epochs=4)
    summary_resumed = train(config_resume)

    assert [m.epoch for m in summary_resumed.history] == [2, 3]
    for expected, actual in zip(
        summary_full.history[2:], summary_resumed.history, strict=True
    ):
        assert actual.loss_total == pytest.approx(expected.loss_total, rel=1e-6)
        assert actual.loss_sigreg == pytest.approx(expected.loss_sigreg, rel=1e-6)
        assert actual.loss_invariance == pytest.approx(
            expected.loss_invariance, rel=1e-6
        )


def test_resuming_a_fully_completed_run_does_nothing(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    config = _config(tiny_catalog, run_dir, epochs=2)
    train(config)

    summary = train(config)

    assert summary.epochs_completed == 2
    assert summary.history == ()


# Nota: se consideró un test con `os.kill(os.getpid(), signal.SIGINT)` real
# desde un hilo temporizador, pero se descartó — la entrega de la señal puede
# retrasarse hasta que el intérprete vuelve a bytecode puro (tras una llamada
# C de PyTorch en curso), y si eso ocurre después de que `train()` ya haya
# restaurado los manejadores originales, la señal aterriza en un test
# completamente distinto y lo hace fallar con un `KeyboardInterrupt` no
# relacionado — se comprobó en la práctica. El mecanismo de manejo de señales
# es idéntico, campo por campo, al ya usado y probado en `runner.py` (feature
# 002) y `vindr/pipeline.py` (feature 004); lo que sí es propio de esta
# feature —que la reanudación reproduce la continuación exacta— ya lo cubre
# `test_resume_reproduces_the_exact_continuation` arriba, sin depender de
# temporización de señales del sistema operativo.


def test_low_effective_rank_emits_a_collapse_warning(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    config = _config(tiny_catalog, tmp_path / "run", epochs=1, min_effective_rank=1e9)

    with pytest.warns(UserWarning, match="colapso incipiente"):
        train(config)


def test_no_collapse_warning_when_threshold_is_met(
    tiny_catalog: Path, tmp_path: Path
) -> None:
    import warnings

    config = _config(tiny_catalog, tmp_path / "run", epochs=1, min_effective_rank=0.0)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*colapso incipiente.*")
        train(config)
