"""Symmetric Gaussian differentiation for ARCTIC minimal coordinates."""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .parameters import NormalizedPartPhysics
from .articulated_torch import ArticulatedState, ArticulatedTorchBackend
from .discrete_kinematics import estimate_discrete_gaussian_kinematics
from .trajectory_wrench import (
    GAUSSIAN_SIGMA_S,
    gaussian_radius_for_sigma,
)


def estimate_articulated_discrete_gaussian_wrench(
    base_pose_world: np.ndarray,
    joint_angle: np.ndarray,
    bodies: list[NormalizedPartPhysics],
    child_zero_transform: np.ndarray,
    state_centers: np.ndarray,
    *,
    fps: float = 25.0,
    sigma_s: float = GAUSSIAN_SIGMA_S,
    sigma_rot_s: float | None = None,
    radius: int | None = None,
    radius_rot: int | None = None,
    gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.80665),
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Return integrator-consistent wrench between Gaussian-smoothed states.

    There are ``S-1`` output intervals for ``S`` contiguous state centers.
    The returned wrench exactly realizes each smoothed state transition under
    one 25 Hz semi-implicit step of :class:`ArticulatedTorchBackend`.
    """
    poses = np.asarray(base_pose_world, np.float64)
    centers = np.asarray(state_centers, np.int64)
    effective_sigma_rot = sigma_s if sigma_rot_s is None else float(sigma_rot_s)
    if radius is None:
        radius = gaussian_radius_for_sigma(fps, sigma_s)
    if radius_rot is None:
        radius_rot = gaussian_radius_for_sigma(fps, effective_sigma_rot)
    kinematics = estimate_discrete_gaussian_kinematics(
        poses[:, :3, 3], poses[:, :3, :3], centers,
        fps=fps, sigma_s=sigma_s, radius=radius,
        sigma_rot_s=effective_sigma_rot, radius_rot=radius_rot,
        extra_coordinates=joint_angle, device=device)
    position = kinematics["smoothed_position_world"]
    rotation = kinematics["smoothed_rotation_world"]
    angle = kinematics["smoothed_extra_coordinates"][:, 0]
    # Build boundary velocities for Newton's discrete FREE-joint convention.
    # Translation is integrated at the base-link COM and orientation uses the
    # normalized quaternion Euler update, not the SO(3) exponential map used
    # by the generic rigid smoother.
    dt = 1.0 / float(fps)
    base_com = position + np.einsum("tij,j->ti", rotation, bodies[0].com_part)
    boundary_velocity = np.empty((len(position), 7), dtype=np.float64)
    initial = np.asarray(kinematics["initial_generalized_velocity"], np.float64)
    initial_offset = rotation[0] @ np.asarray(bodies[0].com_part, np.float64)
    boundary_velocity[0, :3] = initial[:3] + np.cross(initial[3:6], initial_offset)
    boundary_velocity[0, 3:] = initial[3:]
    boundary_velocity[1:, :3] = np.diff(base_com, axis=0) / dt
    relative = rotation[1:] @ np.swapaxes(rotation[:-1], 1, 2)
    rotation_vector = Rotation.from_matrix(relative).as_rotvec()
    angle_step = np.linalg.norm(rotation_vector, axis=1)
    scale = np.ones_like(angle_step)
    nonzero = angle_step > 1e-12
    scale[nonzero] = 2.0 * np.tan(0.5 * angle_step[nonzero]) / angle_step[nonzero]
    boundary_velocity[1:, 3:6] = rotation_vector * (scale / dt)[:, None]
    boundary_velocity[1:, 6] = np.diff(angle) / dt
    incoming_velocity = boundary_velocity[:-1]
    next_velocity = boundary_velocity[1:]
    acceleration = (next_velocity - incoming_velocity) / dt
    backend = ArticulatedTorchBackend(
        bodies, child_zero_transform, fps=fps, substeps=1,
        gravity_world=(0.0, 0.0, 0.0),device=device)
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64,device=device)
    positions=tensor(position[:-1]);rotations=tensor(rotation[:-1]);angles=tensor(angle[:-1])
    velocities=tensor(incoming_velocity);accelerations=tensor(acceleration)
    def inverse_one(p,r,a,v,q):
        return backend.inverse_dynamics(ArticulatedState(p,r,a,v),q)
    net=torch.vmap(inverse_one)(positions,rotations,angles,velocities,accelerations)
    gravity_vector=tensor(gravity_world)
    def gravity_one(p,r,a,v):
        state=ArticulatedState(p,r,a,v)
        geometry = backend.geometry(state)
        child_lever = geometry["part_com_world"][1] - geometry["pivot_world"]
        joint_gravity = torch.dot(
            torch.linalg.cross(geometry["axis_world"], child_lever),
            backend.mass[1] * gravity_vector)
        return torch.cat((backend.mass.sum()*gravity_vector,
                          torch.zeros(3,dtype=torch.float64,device=device),
                          joint_gravity[None]))
    gravity=torch.vmap(gravity_one)(positions,rotations,angles,velocities)
    net=net.cpu().numpy();gravity=gravity.cpu().numpy()
    return {
        "smoothed_base_position_world": position,
        "smoothed_base_rotation_world": rotation,
        "smoothed_joint_angle": angle,
        "initial_generalized_velocity": boundary_velocity[0],
        "generalized_velocity": incoming_velocity,
        "next_generalized_velocity": next_velocity,
        "generalized_acceleration": acceleration,
        "generalized_net_wrench": net,
        "generalized_gravity_wrench": gravity,
        "generalized_contact_wrench": net - gravity,
    }
