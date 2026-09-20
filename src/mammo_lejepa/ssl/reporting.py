from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mammo_lejepa.ssl.config import EvalConfig
from mammo_lejepa.ssl.probe import ProbeResult

# condición -> objetivo -> resultado
ConditionResults = dict[str, dict[str, ProbeResult]]


def results_to_dict(results: ConditionResults) -> dict[str, Any]:
    return {
        condition: {target: asdict(result) for target, result in targets.items()}
        for condition, targets in results.items()
    }


def results_from_dict(data: dict[str, Any]) -> ConditionResults:
    return {
        condition: {
            target: ProbeResult(**payload) for target, payload in targets.items()
        }
        for condition, targets in data.items()
    }


def _ordered_targets(results: ConditionResults) -> list[str]:
    seen: dict[str, None] = {}
    for targets in results.values():
        for target in targets:
            seen.setdefault(target, None)
    return list(seen)


def to_markdown(results: ConditionResults, *, control_target: str) -> str:
    """Tabla comparativa de las condiciones bajo el mismo protocolo, una
    sección por tarea, con la línea base de la clase mayoritaria y el número
    de muestras siempre a la vista (FR-020, FR-021, FR-022). La sonda de
    control sobre la fuente se reporta junto a las demás (FR-023)."""
    lines: list[str] = []
    for target in _ordered_targets(results):
        title = f"{target} (sonda de control)" if target == control_target else target
        lines += [
            f"## {title}",
            "",
            "| condición | exactitud | exactitud equilibrada | clase mayoritaria "
            "| n train | n test | clases |",
            "|---|---|---|---|---|---|---|",
        ]
        for condition, targets in results.items():
            r = targets.get(target)
            if r is None:
                continue
            lines.append(
                f"| {condition} | {r.accuracy:.3f} | {r.balanced_accuracy:.3f} "
                f"| {r.majority_baseline:.3f} | {r.n_train} | {r.n_test} "
                f"| {r.n_classes} |"
            )

        sources = sorted(
            {
                s
                for targets in results.values()
                if target in targets
                for s in targets[target].accuracy_by_source
            }
        )
        if sources and target != control_target:
            lines += [
                "",
                "Exactitud por `source_dataset` (n de test entre paréntesis):",
                "",
                "| condición | " + " | ".join(sources) + " |",
                "|---|" + "---|" * len(sources),
            ]
            for condition, targets in results.items():
                r = targets.get(target)
                if r is None:
                    continue
                cells = [
                    f"{r.accuracy_by_source[s]:.3f} ({r.n_test_by_source[s]})"
                    if s in r.accuracy_by_source
                    else "—"
                    for s in sources
                ]
                lines.append(f"| {condition} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def save_eval(
    run_dir: Path,
    results: ConditionResults,
    config: EvalConfig,
    *,
    checkpoint: Path | None,
) -> Path:
    """Guarda resultados, configuración de evaluación y referencia al
    checkpoint que los produjo (constitución, principio V)."""
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "eval_config": json.loads(config.to_json()),
        "results": results_to_dict(results),
    }
    (eval_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (eval_dir / "results.md").write_text(
        to_markdown(results, control_target=config.control_target), encoding="utf-8"
    )
    return eval_dir


def load_eval(run_dir: Path) -> ConditionResults:
    payload = json.loads((run_dir / "eval" / "results.json").read_text())
    return results_from_dict(payload["results"])


def read_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "history.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
