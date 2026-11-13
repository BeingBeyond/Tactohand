"""Robust per-component wrench scale shared by training and preflight stacks.

Extracted from ``experiments/tactohand_preflight/data.py`` so the production
``training/tactohand_wrench`` task no longer imports from ``experiments/``.
"""
from __future__ import annotations

import numpy as np


def robust_wrench_scale(wrench: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    values = np.asarray(wrench, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826
    rms = np.sqrt(np.mean(values * values, axis=0))
    return np.maximum(np.maximum(mad, 0.1 * rms), floor)
