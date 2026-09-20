from __future__ import annotations

from pathlib import Path

from mammo_lejepa.ssl.config import EvalConfig
from mammo_lejepa.ssl.probe import ProbeResult
from mammo_lejepa.ssl.reporting import (
    load_eval,
    read_history,
    results_from_dict,
    results_to_dict,
    save_eval,
    to_markdown,
)


def _result(target: str, accuracy: float) -> ProbeResult:
    return ProbeResult(
        target=target,
        n_train=100,
        n_test=40,
        n_classes=2,
        accuracy=accuracy,
        balanced_accuracy=accuracy - 0.05,
        majority_baseline=0.5,
        accuracy_by_source={"a": accuracy, "b": accuracy - 0.1},
        n_test_by_source={"a": 20, "b": 20},
        class_counts_train={"x": 50, "y": 50},
        class_counts_test={"x": 20, "y": 20},
    )


def _results() -> dict[str, dict[str, ProbeResult]]:
    return {
        "random": {
            "classification": _result("classification", 0.6),
            "source_dataset": _result("source_dataset", 0.9),
        },
        "lejepa": {
            "classification": _result("classification", 0.8),
            "source_dataset": _result("source_dataset", 0.95),
        },
    }


def test_dict_round_trip_is_lossless() -> None:
    results = _results()

    assert results_from_dict(results_to_dict(results)) == results


def test_markdown_shows_baseline_sample_sizes_and_conditions() -> None:
    text = to_markdown(_results(), control_target="source_dataset")

    assert "## classification" in text
    assert "## source_dataset (sonda de control)" in text
    assert "| random | 0.600 |" in text
    assert "| lejepa | 0.800 |" in text
    assert "0.500" in text
    assert "| 100 | 40 |" in text


def test_markdown_breaks_down_by_source_except_for_the_control() -> None:
    text = to_markdown(_results(), control_target="source_dataset")

    assert text.count("Exactitud por `source_dataset`") == 1
    assert "0.800 (20)" in text


def test_save_and_load_eval(tmp_path: Path) -> None:
    config = EvalConfig(catalog_path=tmp_path / "c.parquet")
    checkpoint = tmp_path / "checkpoints" / "epoch_0000.pt"

    eval_dir = save_eval(tmp_path, _results(), config, checkpoint=checkpoint)

    assert (eval_dir / "results.md").exists()
    assert "epoch_0000.pt" in (eval_dir / "results.json").read_text()
    assert load_eval(tmp_path) == _results()


def test_read_history_handles_missing_and_present_files(tmp_path: Path) -> None:
    assert read_history(tmp_path) == []

    (tmp_path / "history.jsonl").write_text('{"epoch": 0}\n\n{"epoch": 1}\n')

    assert [row["epoch"] for row in read_history(tmp_path)] == [0, 1]
