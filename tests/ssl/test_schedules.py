from __future__ import annotations

from itertools import pairwise

import pytest

from mammo_lejepa.ssl.schedules import warmup_cosine


def test_warmup_increases_linearly_to_base_lr() -> None:
    base_lr = 1e-3
    warmup_steps = 10

    lrs = [
        warmup_cosine(step, total_steps=100, warmup_steps=warmup_steps, base_lr=base_lr)
        for step in range(warmup_steps)
    ]

    assert lrs == sorted(lrs)
    assert lrs[0] > 0
    assert lrs[-1] == pytest.approx(base_lr, rel=1e-6)


def test_lr_at_end_of_warmup_reaches_base_lr() -> None:
    base_lr = 1e-3
    warmup_steps = 10

    lr = warmup_cosine(
        warmup_steps, total_steps=100, warmup_steps=warmup_steps, base_lr=base_lr
    )

    assert lr == pytest.approx(base_lr, rel=1e-6)


def test_lr_decays_to_final_ratio_at_last_step() -> None:
    base_lr = 1e-3
    total_steps = 100
    warmup_steps = 10
    final_lr_ratio = 1e-3

    lr = warmup_cosine(
        total_steps,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        base_lr=base_lr,
        final_lr_ratio=final_lr_ratio,
    )

    assert lr == pytest.approx(base_lr * final_lr_ratio, rel=1e-6)


def test_lr_is_monotonically_decreasing_after_warmup() -> None:
    total_steps = 100
    warmup_steps = 10
    lrs = [
        warmup_cosine(
            step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=1e-3
        )
        for step in range(warmup_steps, total_steps + 1)
    ]

    assert all(a >= b for a, b in pairwise(lrs))


def test_schedule_is_a_pure_function_of_step() -> None:
    """No depende de cuántas veces se haya llamado antes: reanudar en el paso
    k da la misma tasa que llegar a k de corrido."""
    kwargs = dict(total_steps=50, warmup_steps=5, base_lr=1e-3)

    for _ in range(3):
        warmup_cosine(0, **kwargs)
        warmup_cosine(10, **kwargs)

    fresh = warmup_cosine(30, **kwargs)
    assert fresh == warmup_cosine(30, **kwargs)


def test_zero_warmup_steps_starts_at_cosine_decay() -> None:
    lr = warmup_cosine(0, total_steps=100, warmup_steps=0, base_lr=1e-3)

    assert lr == pytest.approx(1e-3, rel=1e-6)
