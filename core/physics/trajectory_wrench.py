"""Offline SE(3) differentiation and rigid-body inverse dynamics.

This module deliberately does not roll out or optimize a wrench.  It estimates
derivatives from a fixed, symmetric pose window and applies the Newton--Euler
equations once at every requested frame.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .backends import Physics
from .discrete_kinematics import (
    cayley_angular_velocity, estimate_discrete_gaussian_kinematics,
    gaussian_coefficients as _gaussian_coefficients,
    project_to_so3, smooth_pose_gaussian, smooth_rotations_gaussian,
)
from .math3d import matrix_to_quat, quat_to_matrix


GAUSSIAN_SIGMA_S = 0.075
GAUSSIAN_HALF_WINDOW_S = 0.30
def gaussian_radius_for_fps(fps: float, half_window_s: float = GAUSSIAN_HALF_WINDOW_S) -> int:
    """Convert the shared physical Gaussian half-window to dataset frames."""
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(half_window_s) or half_window_s <= 0:
        raise ValueError("fps and half_window_s must be finite and positive")
    return max(1, int(round(float(fps) * float(half_window_s))))


def gaussian_radius_for_sigma(fps: float, sigma_s: float) -> int:
    """Kernel support for one channel: legacy half-window, widened to 3 sigma.

    The legacy fixed half-window (4 sigma at the original 0.075 s) is kept as a
    floor so unchanged sigmas reproduce historical labels exactly; wider sigmas
    grow the support to avoid truncation ringing.
    """
    if not np.isfinite(sigma_s) or sigma_s <= 0:
        raise ValueError("sigma_s must be finite and positive")
    return max(gaussian_radius_for_fps(fps), int(np.ceil(3.0 * float(sigma_s) * float(fps))))


def estimate_rigid_discrete_gaussian_wrench(
    position_world: np.ndarray, quat_xyzw: np.ndarray, physics: Physics,
    state_centers: np.ndarray, *, fps: float = 30.0,
    sigma_s: float = GAUSSIAN_SIGMA_S, sigma_rot_s: float | None = None,
    radius: int | None = None, radius_rot: int | None = None,
    gravity_world: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Integrator-consistent rigid wrench between Gaussian-smoothed states."""
    centers = np.asarray(state_centers, dtype=np.int64)
    effective_sigma_rot = sigma_s if sigma_rot_s is None else float(sigma_rot_s)
    if radius is None:
        radius = gaussian_radius_for_sigma(fps, sigma_s)
    effective_radius_rot = (
        gaussian_radius_for_sigma(fps, effective_sigma_rot)
        if radius_rot is None else int(radius_rot))
    rotations = quat_to_matrix(torch.as_tensor(quat_xyzw, dtype=torch.float64)).numpy()
    kinematics = estimate_discrete_gaussian_kinematics(
        position_world, rotations, centers, fps=fps, sigma_s=sigma_s,
        radius=radius, sigma_rot_s=effective_sigma_rot,
        radius_rot=effective_radius_rot,
        angular_velocity_map=cayley_angular_velocity)
    position = kinematics["smoothed_position_world"]
    rotation = kinematics["smoothed_rotation_world"]
    next_velocity = kinematics["next_generalized_velocity"]
    incoming_velocity = kinematics["generalized_velocity"]
    acceleration = kinematics["generalized_acceleration"]
    inertia = np.asarray(physics.inertia_body, dtype=np.float64)
    omega_body = np.einsum("nji,nj->ni", rotation[:-1], incoming_velocity[:, 3:])
    alpha_body = np.einsum("nji,nj->ni", rotation[:-1], acceleration[:, 3:])
    torque_body = np.einsum("ij,nj->ni", inertia, alpha_body) + np.cross(
        omega_body, np.einsum("ij,nj->ni", inertia, omega_body))
    net = np.column_stack((
        physics.mass * acceleration[:, :3],
        np.einsum("nij,nj->ni", rotation[:-1], torque_body)))
    gravity = np.zeros_like(net)
    gravity_values = (np.asarray(physics.gravity_world, np.float64)
                      if gravity_world is None else np.asarray(gravity_world, np.float64))
    if gravity_values.shape == (3,):
        gravity_values = np.broadcast_to(gravity_values, (len(net), 3))
    if gravity_values.shape != (len(net), 3) or not np.isfinite(gravity_values).all():
        raise ValueError("gravity_world must have shape (3,) or (T,3) and be finite")
    gravity[:, :3] = physics.mass * gravity_values
    return {
        "smoothed_position_world": position,
        "smoothed_rotation_world": rotation,
        "smoothed_quat_xyzw": matrix_to_quat(torch.as_tensor(rotation)).numpy(),
        "initial_generalized_velocity": kinematics["initial_generalized_velocity"],
        "generalized_velocity": incoming_velocity,
        "next_generalized_velocity": next_velocity,
        "generalized_acceleration": acceleration,
        "net_wrench_world": net,
        "gravity_wrench_world": gravity,
        "contact_wrench_world": net - gravity,
    }
