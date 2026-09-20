from __future__ import annotations

import math


def warmup_cosine(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    final_lr_ratio: float = 1e-3,
) -> float:
    """Calentamiento lineal desde 0 hasta `base_lr` en `warmup_steps`, seguido
    de descenso coseno hasta `base_lr * final_lr_ratio` en los pasos
    restantes. Función pura del paso (contracts.md): la reanudación en el
    paso `k` reproduce exactamente la misma tasa sin depender de cuántas
    veces se haya llamado antes — no hay estado que restaurar aparte de
    `step` mismo."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    final_lr = base_lr * final_lr_ratio
    remaining_steps = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / remaining_steps, 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    return final_lr + (base_lr - final_lr) * cosine_factor
