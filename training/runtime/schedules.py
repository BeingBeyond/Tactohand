"""Learning-rate schedules shared by supported tasks."""
from __future__ import annotations
import math

def warmup_cosine_multiplier(
    step: int, total_steps: int, warmup_steps: int, min_ratio: float
) -> float:
    """Linear warmup to 1, then cosine decay to ``min_ratio``."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
