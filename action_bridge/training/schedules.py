"""Small scalar schedules used by the pilot losses."""

from __future__ import annotations


def linear_warmup(step: int, start: float, end: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(end)
    mix = min(1.0, max(0.0, float(step) / float(warmup_steps)))
    return float(start) + mix * (float(end) - float(start))
