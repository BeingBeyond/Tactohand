"""Dataset-independent Gaussian smoothing and discrete minimal-coordinate kinematics."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .math3d import so3_log
AngularVelocityMap = Callable[[np.ndarray, float], np.ndarray]


def gaussian_coefficients(dt: float, sigma_s: float, radius: int) -> tuple[np.ndarray, ...]:
    """Sample truncated Gaussian kernels with repaired discrete moments."""
    if dt <= 0 or sigma_s <= 0 or radius < 1:
        raise ValueError("dt, sigma_s and radius must be positive")
    t = np.arange(-radius, radius + 1, dtype=np.float64) * dt
    g = np.exp(-0.5 * (t / sigma_s) ** 2)
    c0 = g / g.sum()
    c1 = t * g / sigma_s**2
    c1 /= np.dot(c1, t)
    c2 = (t**2 / sigma_s**4 - 1.0 / sigma_s**2) * g
    c2 -= c2.mean()
    c2 *= 2.0 / np.dot(c2, t**2)
    return c0, c1, c2


def project_to_so3(rotations: np.ndarray,device: torch.device | str = "cpu") -> np.ndarray:
    value = torch.as_tensor(rotations,dtype=torch.float64,device=device)
    u, _, vh = torch.linalg.svd(value)
    projected = u @ vh
    reflected = torch.linalg.det(projected) < 0
    if bool(reflected.any()):
        u = u.clone()
        u[reflected, :, -1] *= -1
        projected = u @ vh
    return projected.cpu().numpy()


def smooth_rotations_gaussian(
    rotations: np.ndarray, centers: np.ndarray, weights: np.ndarray,
    radius: int, *, iterations: int = 8,device: torch.device | str = "cpu",
) -> np.ndarray:
    source=torch.as_tensor(project_to_so3(rotations,device),dtype=torch.float64,device=device)
    center=torch.as_tensor(np.asarray(centers,np.int64),device=device)
    offsets=torch.arange(-radius,radius+1,device=device)
    weight=torch.as_tensor(weights,dtype=torch.float64,device=device)
    indices=(center[:,None]+offsets).clamp(0,len(source)-1)
    neighborhood=source[indices]
    mean=source[center.clamp(0,len(source)-1)].clone()
    for _ in range(iterations):
        relative=mean.transpose(-1,-2)[:,None]@neighborhood
        tangent=torch.einsum("j,njk->nk",weight,so3_log(relative))
        mean=mean@torch.matrix_exp(_skew(tangent))
    return mean.cpu().numpy()


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x,y,z=vector.unbind(-1)
    zero=torch.zeros_like(x)
    return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*vector.shape[:-1],3,3)


def smooth_pose_gaussian(
    position: np.ndarray, rotations: np.ndarray, centers: np.ndarray,
    *, fps: float, sigma_s: float, radius: int,
    sigma_rot_s: float | None = None, radius_rot: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth translation and rotation with independent Gaussian bandwidths.

    ``sigma_s``/``radius`` govern the translation kernel.  ``sigma_rot_s``/
    ``radius_rot`` govern the SO(3) kernel and default to the translation
    values, which reproduces the historical single-sigma behavior bit for bit.
    """
    centers = np.asarray(centers, np.int64)
    if len(centers) < 2 or centers.min() < 0 or centers.max() > len(position):
        raise ValueError("Gaussian centers must lie in [0, sequence_length]")
    weights, _, _ = gaussian_coefficients(1.0 / fps, sigma_s, radius)
    offsets = np.arange(-radius, radius + 1)
    indices = np.clip(centers[:, None] + offsets, 0, len(position) - 1)
    smoothed = np.einsum("j,njk->nk", weights, np.asarray(position)[indices])
    effective_sigma_rot = sigma_s if sigma_rot_s is None else float(sigma_rot_s)
    effective_radius_rot = radius if radius_rot is None else int(radius_rot)
    weights_rot, _, _ = gaussian_coefficients(
        1.0 / fps, effective_sigma_rot, effective_radius_rot)
    return smoothed, smooth_rotations_gaussian(
        rotations, centers, weights_rot, effective_radius_rot, device=device)


def exponential_angular_velocity(relative_rotation: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity for an exponential-map rotation update."""
    return so3_log(torch.as_tensor(relative_rotation, dtype=torch.float64)).numpy() / dt


def cayley_angular_velocity(relative_rotation: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity matching the semi-implicit quaternion Euler update."""
    rotvec = so3_log(torch.as_tensor(relative_rotation, dtype=torch.float64)).numpy()
    angle = float(np.linalg.norm(rotvec))
    scale = 2.0 * np.tan(0.5 * angle) / (dt * angle) if angle > 1e-12 else 1.0 / dt
    return rotvec * scale


def estimate_discrete_gaussian_kinematics(
    position_world: np.ndarray,
    rotation_world: np.ndarray,
    state_centers: np.ndarray,
    *,
    fps: float,
    sigma_s: float,
    radius: int,
    sigma_rot_s: float | None = None,
    radius_rot: int | None = None,
    extra_coordinates: np.ndarray | None = None,
    angular_velocity_map: AngularVelocityMap = exponential_angular_velocity,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Smooth minimal coordinates and return integrator-consistent derivatives.

    The first six axes are always world translation and world rotation. Optional
    scalar coordinates are unwrapped, Gaussian-smoothed, and appended. Dataset
    adapters choose only the angular update convention required by their backend.
    Rotation may use a wider kernel than translation via ``sigma_rot_s``/
    ``radius_rot``; both default to the translation values.
    """
    centers = np.asarray(state_centers, np.int64)
    if len(centers) < 2 or np.any(np.diff(centers) != 1):
        raise ValueError("state_centers must contain at least two contiguous frames")
    if centers.min() < 0 or centers.max() > len(position_world):
        raise ValueError("state_centers must lie in [0, sequence_length]")
    dt = 1.0 / fps
    position, rotation = smooth_pose_gaussian(
        position_world, rotation_world, centers, fps=fps, sigma_s=sigma_s,
        radius=radius, sigma_rot_s=sigma_rot_s, radius_rot=radius_rot,
        device=device)
    angular_next = np.stack([
        angular_velocity_map(rotation[index + 1] @ rotation[index].T, dt)
        for index in range(len(rotation) - 1)
    ])
    columns = [np.diff(position, axis=0) / dt, angular_next]
    result: dict[str, np.ndarray] = {
        "smoothed_position_world": position,
        "smoothed_rotation_world": rotation,
    }
    if extra_coordinates is not None:
        coordinates = np.asarray(extra_coordinates, np.float64)
        if coordinates.ndim == 1:
            coordinates = coordinates[:, None]
        if coordinates.ndim != 2 or len(coordinates) != len(position_world):
            raise ValueError("extra_coordinates must have shape (T,) or (T,K)")
        coefficients, _, _ = gaussian_coefficients(dt, sigma_s, radius)
        offsets = np.arange(-radius, radius + 1)
        unwrapped = np.unwrap(coordinates, axis=0)
        indices = np.clip(centers[:, None] + offsets, 0, len(coordinates) - 1)
        smoothed = np.einsum("j,njk->nk", coefficients, unwrapped[indices])
        columns.append(np.diff(smoothed, axis=0) / dt)
        result["smoothed_extra_coordinates"] = smoothed
    next_velocity = np.concatenate(columns, axis=1)
    incoming_velocity = np.vstack((next_velocity[0], next_velocity[:-1]))
    result.update({
        "initial_generalized_velocity": incoming_velocity[0],
        "generalized_velocity": incoming_velocity,
        "next_generalized_velocity": next_velocity,
        "generalized_acceleration": (next_velocity - incoming_velocity) / dt,
    })
    return result
