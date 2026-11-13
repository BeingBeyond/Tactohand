"""Direct-wrench and open-loop rollout metrics for the formal experiment."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from core.physics.backends import BatchedTorchBackend, RigidState
from core.physics.math3d import matrix_to_quat, quat_to_matrix, rotation_geodesic
from core.physics.wrench import gravity_wrench


@dataclass(frozen=True)
class PoseLossWindows:
    horizon: int
    mask: torch.Tensor
    count: int
    position_scale: float
    rotation_scale: float


def window_subset(mask: torch.Tensor) -> str:
    """Classify a rollout window by whether its transitions are real/padded."""
    values = mask.bool()
    if bool(values.all()):
        return "real_only"
    if bool((~values).all()):
        return "padding_only"
    return "boundary_crossing"


def rollout_metrics(
    prediction_camera: torch.Tensor,
    batch: dict[str, Any],
    horizons: list[int],
    subsets: list[str],
    explosion_linear_speed: float,
    explosion_angular_speed: float,
    integrator: str = "implicit_midpoint",
    implicit_iterations: int = 6,
    *,
    midpoint_implementation: str = "reference",
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate camera-inertial rollouts, initialized from camera-frame GT."""
    metrics, _ = rollout_metrics_and_pose_loss(
        prediction_camera, batch, horizons, subsets, explosion_linear_speed,
        explosion_angular_speed, integrator, implicit_iterations,
        midpoint_implementation=midpoint_implementation,
    )
    return metrics


@torch.no_grad()
def rollout_metrics_and_pose_loss(
    prediction_camera: torch.Tensor,
    batch: dict[str, Any],
    horizons: list[int],
    subsets: list[str],
    explosion_linear_speed: float,
    explosion_angular_speed: float,
    integrator: str = "implicit_midpoint",
    implicit_iterations: int = 6,
    *,
    midpoint_implementation: str = "reference",
    pose_loss: PoseLossWindows | None = None,
    window_observer: Callable[[int, str, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], None] | None = None,
) -> tuple[dict[str, dict[str, dict[str, float]]], tuple[torch.Tensor, torch.Tensor] | None]:
    """Evaluate metrics and an optional validation objective in one rollout.

    An optional observer receives the selected and finite non-divergent
    window masks and unreduced metrics before the batch is pooled.
    """
    device, dtype = prediction_camera.device, prediction_camera.dtype
    if prediction_camera.ndim != 3 or prediction_camera.shape[-1] != 6:
        raise ValueError("prediction_camera must have shape [batch, transitions, 6]")
    if not horizons or min(horizons) < 1 or max(horizons) > prediction_camera.shape[1]:
        raise ValueError("rollout horizons must be within the available transitions")
    transition_count = prediction_camera.shape[1]
    pose_starts = 0
    if pose_loss is not None:
        pose_starts = transition_count - pose_loss.horizon + 1
        if (not 1 <= pose_loss.horizon <= transition_count or pose_loss.count <= 0
                or pose_loss.mask.shape != (prediction_camera.shape[0], pose_starts)):
            raise ValueError("invalid validation rollout loss windows")
    physics = batch["physics"]
    backend = BatchedTorchBackend(
        physics, prediction_camera, integrator=integrator,
        implicit_iterations=implicit_iterations, midpoint_implementation=midpoint_implementation,
    )
    net = prediction_camera + gravity_wrench(
        physics, prediction_camera, batch.get("gravity"), mass=backend.mass)
    pos = batch["position"].to(device=device, dtype=dtype)
    rot = batch["rotation"].to(device=device, dtype=dtype)
    vel = batch["velocity"].to(device=device, dtype=dtype)
    mask = batch["real_mask"].to(device=device).bool()
    prediction_mask = batch.get("prediction_mask")
    if prediction_mask is not None:
        prediction_mask = prediction_mask.to(device=device).bool()
    current = RigidState(
        pos[:, :transition_count], matrix_to_quat(rot[:, :transition_count]),
        vel[:, :transition_count, :3], vel[:, :transition_count, 3:],
    )
    step_errors: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "position", "rotation", "velocity", "omega", "explosion"
        )
    }
    metric_horizon = max(horizons)
    horizon = max(metric_horizon, pose_loss.horizon) if pose_loss is not None else metric_horizon
    position_terms, rotation_terms = [], []
    for step in range(horizon):
        active = transition_count - step
        state = RigidState(
            current.x_com[:, :active], current.quat_xyzw[:, :active],
            current.v_world[:, :active], current.omega_world[:, :active],
        )
        current = backend.rollout_interval(state, net[:, step:])
        target_slice = slice(step + 1, transition_count + 1)
        position_error = (current.x_com - pos[:, target_slice]).norm(dim=-1)
        predicted_rotation = quat_to_matrix(current.quat_xyzw)
        target_rotation = rot[:, target_slice]
        if pose_loss is not None and step < pose_loss.horizon:
            position_terms.append((position_error[:, :pose_starts] / pose_loss.position_scale).square())
            rotation_terms.append(0.5 * (
                (predicted_rotation[:, :pose_starts] - target_rotation[:, :pose_starts])
                / pose_loss.rotation_scale).square().sum((-1, -2)))
        if step >= metric_horizon:
            continue
        step_errors["position"].append(position_error)
        step_errors["rotation"].append(rotation_geodesic(
            target_rotation, predicted_rotation
        ))
        step_errors["velocity"].append(
            (current.v_world - vel[:, target_slice, :3]).norm(dim=-1)
        )
        step_errors["omega"].append(
            (current.omega_world - vel[:, target_slice, 3:]).norm(dim=-1)
        )
        step_errors["explosion"].append(
            (current.v_world.norm(dim=-1) > explosion_linear_speed)
            | (current.omega_world.norm(dim=-1) > explosion_angular_speed)
            | ~torch.isfinite(current.x_com).all(dim=-1)
            | ~torch.isfinite(current.quat_xyzw).all(dim=-1)
            | ~torch.isfinite(current.v_world).all(dim=-1)
            | ~torch.isfinite(current.omega_world).all(dim=-1)
        )
    backend.check_solve_errors()
    losses = None
    if pose_loss is not None:
        loss_mask = pose_loss.mask.to(device=device, dtype=torch.bool).unsqueeze(-1)
        denominator = pose_loss.count * pose_loss.horizon
        losses = (
            torch.where(loss_mask, torch.stack(position_terms, -1), 0.0).sum() / denominator,
            torch.where(loss_mask, torch.stack(rotation_terms, -1), 0.0).sum() / denominator,
        )
    result: dict[str, dict[str, dict[str, float]]] = {}
    packed: list[torch.Tensor] = []
    destinations: list[tuple[str, str, list[str]]] = []
    for horizon in horizons:
        starts = transition_count - horizon + 1
        position = torch.stack([value[:, :starts] for value in step_errors["position"][:horizon]], -1)
        rotation = torch.stack([value[:, :starts] for value in step_errors["rotation"][:horizon]], -1)
        velocity = torch.stack([value[:, :starts] for value in step_errors["velocity"][:horizon]], -1)
        omega = torch.stack([value[:, :starts] for value in step_errors["omega"][:horizon]], -1)
        exploded = torch.stack([value[:, :starts] for value in step_errors["explosion"][:horizon]], -1).any(-1)
        window_mask = mask.unfold(1, horizon, 1)
        real_only = window_mask.all(-1)
        padding_only = (~window_mask).all(-1)
        selections = {
            "all": torch.ones_like(real_only),
            "real_only": real_only,
            "padding_only": padding_only,
            "boundary_crossing": ~(real_only | padding_only),
        }
        values = {
            "position_ade_m": position.mean(-1),
            "position_fde_m": position[..., -1],
            "rotation_ade_rad": rotation.mean(-1),
            "rotation_ade_deg": rotation.mean(-1) * 57.29577951308232,
            "rotation_fde_rad": rotation[..., -1],
            "velocity_ade_m_s": velocity.mean(-1),
            "omega_ade_rad_s": omega.mean(-1),
            "explosion": exploded.to(dtype),
        }
        result[str(horizon)] = {}
        for subset in subsets:
            if subset not in selections:
                raise ValueError(f"unknown rollout subset: {subset}")
            selected = selections[subset]
            if prediction_mask is not None:
                selected = selected & prediction_mask.unfold(1, horizon, 1).any(-1)
            count = selected.sum()
            valid = selected & ~exploded
            for value in values.values():
                if value.shape == valid.shape:
                    valid &= torch.isfinite(value)
            if window_observer is not None:
                window_observer(horizon, subset, selected, valid, values)
            valid_count = valid.sum()
            explosion_rate = (selected & exploded).sum().to(dtype) / count.clamp_min(1)
            keys = [key for key in values if key != "explosion"]
            means = [torch.where(valid, values[key], 0.0).sum() / valid_count.clamp_min(1)
                     for key in keys]
            packed.append(torch.stack([value.double() for value in (count, valid_count, explosion_rate, *means)]))
            destinations.append((str(horizon), subset, keys))
    # Each statistic uses a fixed-shape reduction. Transfer all horizons and
    # subsets together, instead of synchronizing once per mask and statistic.
    if packed:
        for (horizon_key, subset, keys), row in zip(destinations, torch.stack(packed).cpu().tolist()):
            count, valid_count = int(row[0]), int(row[1])
            metrics: dict[str, float] = {"window_count": count, "valid_window_count": valid_count}
            if count:
                metrics.update(explosion_rate=row[2], explosion=row[2], valid_window_fraction=valid_count / count)
                metrics.update({key: value if valid_count else float("nan")
                                for key, value in zip(keys, row[3:])})
            result[horizon_key][subset] = metrics
    return result, losses
