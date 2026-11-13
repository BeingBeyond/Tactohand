"""Wrench target parameterizations shared by data and training code.

The physical label is the applied/contact wrench at the object COM.  For
learning, ``net_residual`` removes the known gravity contribution from the
force channels while leaving the COM torque unchanged.  Keeping this
conversion in one module prevents data scaling, loss computation, metrics,
and rollout from silently using different conventions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


TARGET_MODES = frozenset({"applied", "net_residual"})


def validate_target_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in TARGET_MODES:
        raise ValueError(
            f"unsupported wrench target_mode={mode!r}; "
            f"expected one of {sorted(TARGET_MODES)}"
        )
    return value


def _gravity_force_torch(mass: Any, gravity_camera: Any, reference: torch.Tensor) -> torch.Tensor:
    mass_tensor = torch.as_tensor(mass, device=reference.device, dtype=reference.dtype)
    gravity_tensor = torch.as_tensor(
        gravity_camera, device=reference.device, dtype=reference.dtype
    )
    while mass_tensor.ndim < reference.ndim:
        mass_tensor = mass_tensor.unsqueeze(-1)
    while gravity_tensor.ndim < reference.ndim:
        gravity_tensor = gravity_tensor.unsqueeze(-2)
    return mass_tensor * gravity_tensor


def _gravity_force_numpy(mass: Any, gravity_camera: Any, reference: np.ndarray) -> np.ndarray:
    mass_array = np.asarray(mass, dtype=reference.dtype)
    gravity_array = np.asarray(gravity_camera, dtype=reference.dtype)
    while mass_array.ndim < reference.ndim:
        mass_array = np.expand_dims(mass_array, axis=-1)
    while gravity_array.ndim < reference.ndim:
        gravity_array = np.expand_dims(gravity_array, axis=-2)
    return mass_array * gravity_array


def to_learning_target(
    wrench_applied: torch.Tensor | np.ndarray,
    mass: Any,
    gravity_camera: Any,
    mode: str,
) -> torch.Tensor | np.ndarray:
    """Convert physical applied wrench labels to the configured head target."""
    target_mode = validate_target_mode(mode)
    if target_mode == "applied":
        return wrench_applied
    if isinstance(wrench_applied, torch.Tensor):
        result = wrench_applied.clone()
        result[..., :3] = result[..., :3] + _gravity_force_torch(
            mass, gravity_camera, result
        )
        return result
    result = np.array(wrench_applied, dtype=np.asarray(wrench_applied).dtype, copy=True)
    result[..., :3] = result[..., :3] + _gravity_force_numpy(
        mass, gravity_camera, result
    )
    return result


def to_applied_wrench(
    learning_target: torch.Tensor | np.ndarray,
    mass: Any,
    gravity_camera: Any,
    mode: str,
) -> torch.Tensor | np.ndarray:
    """Reconstruct the physical applied wrench from a head prediction."""
    target_mode = validate_target_mode(mode)
    if target_mode == "applied":
        return learning_target
    if isinstance(learning_target, torch.Tensor):
        result = learning_target.clone()
        result[..., :3] = result[..., :3] - _gravity_force_torch(
            mass, gravity_camera, result
        )
        return result
    result = np.array(learning_target, dtype=np.asarray(learning_target).dtype, copy=True)
    result[..., :3] = result[..., :3] - _gravity_force_numpy(
        mass, gravity_camera, result
    )
    return result
