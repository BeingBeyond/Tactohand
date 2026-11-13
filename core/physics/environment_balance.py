"""Environment-owned wrench handling outside hand-contact intervals."""
from __future__ import annotations

import numpy as np


def direct_environment_balance(
    target_wrench: np.ndarray,
    dynamics_valid: np.ndarray,
    hand_force_active: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign the complete target wrench to the hand-free environment."""
    target = np.asarray(target_wrench, np.float64)
    dynamics = np.asarray(dynamics_valid, bool)
    hand = np.asarray(hand_force_active, bool)
    if target.ndim != 2 or target.shape[1] != 6:
        raise ValueError("target_wrench must have shape (T, 6)")
    if dynamics.shape != (len(target),) or hand.shape != (len(target),):
        raise ValueError("dynamics_valid and hand_force_active must have shape (T,)")

    direct = dynamics & ~hand
    environment_wrench = np.zeros_like(target)
    environment_wrench[direct] = target[direct]
    return {
        "active": direct,
        "environment_wrench": environment_wrench,
        "environment_force": environment_wrench[:, :3].copy(),
        # An unsupported transition for this object is neutral when validity
        # is intersected across multiple object tracks.
        "valid": ~dynamics | direct,
        "qp_active": dynamics & hand,
    }
