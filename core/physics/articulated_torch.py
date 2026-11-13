"""Differentiable two-link floating-base articulated dynamics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .parameters import NormalizedPartPhysics


@dataclass(frozen=True)
class ArticulatedState:
    """Pose plus Newton-compatible generalized velocity.

    ``base_position_world`` is the authored base-link origin, while the first
    three velocity entries are the world velocity of the base-link COM.  The
    remaining entries are world angular velocity and hinge rate.
    """
    base_position_world: torch.Tensor
    base_rotation_world: torch.Tensor
    joint_angle: torch.Tensor
    generalized_velocity: torch.Tensor


def skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    x, y, z = vector.unbind()
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)


def so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues map with stable coefficients at zero."""
    theta = torch.linalg.vector_norm(rotation_vector)
    generator = skew(rotation_vector)
    theta2 = theta * theta
    a = torch.sinc(theta / torch.pi)
    b = torch.where(theta > 1e-4,
                    (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
                    0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + a * generator + b * (generator @ generator)


def newton_quaternion_rotation_step(rotation: torch.Tensor,
                                    rotation_vector: torch.Tensor) -> torch.Tensor:
    """Match Newton's ``normalize(q + .5*[w*dt,0]*q)`` update."""
    vector = 0.5 * rotation_vector
    scalar = torch.ones((), dtype=rotation.dtype, device=rotation.device)
    norm = torch.sqrt(scalar + torch.dot(vector, vector))
    x, y, z = vector / norm
    w = scalar / norm
    delta = torch.stack((
        scalar - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
        2.0 * (x * z + y * w), 2.0 * (x * y + z * w),
        scalar - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
        scalar - 2.0 * (x * x + y * y),
    )).reshape(3, 3)
    return delta @ rotation


class ArticulatedTorchBackend:
    """Minimal-coordinate dynamics for one free base and one revolute child."""

    def __init__(self, bodies: list[NormalizedPartPhysics], child_zero_transform: np.ndarray,
                 *, fps: float = 25.0, substeps: int = 4,
                 gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.80665),
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | str = "cpu") -> None:
        if len(bodies) != 2 or bodies[1].joint_type != "revolute":
            raise ValueError("expected a fixed base body and one revolute child")
        if fps <= 0 or substeps < 1:
            raise ValueError("fps and substeps must be positive")
        zero = np.asarray(child_zero_transform, np.float64)
        if zero.shape != (4, 4):
            raise ValueError("child_zero_transform must have shape (4,4)")
        tensor = lambda value: torch.as_tensor(value, dtype=dtype, device=device)
        self.mass = tensor([bodies[0].mass, bodies[1].mass])
        self.com_local = torch.stack((tensor(bodies[0].com_part), tensor(bodies[1].com_part)))
        self.inertia_local = torch.stack((tensor(bodies[0].inertia_com_part),
                                          tensor(bodies[1].inertia_com_part)))
        self.joint_axis_base = tensor(bodies[1].joint_axis_parent)
        self.joint_axis_base = self.joint_axis_base / torch.linalg.vector_norm(self.joint_axis_base)
        self.joint_origin_base = tensor(bodies[1].joint_origin_parent)
        self.child_zero_rotation = tensor(zero[:3, :3])
        self.child_zero_translation = tensor(zero[:3, 3])
        self.gravity_world = tensor(gravity_world)
        self.dt = 1.0 / (float(fps) * int(substeps))
        self.substeps = int(substeps)

    def geometry(self, state: ArticulatedState) -> dict[str, torch.Tensor]:
        position, rotation, angle = (state.base_position_world,
                                     state.base_rotation_world, state.joint_angle)
        axis_world = rotation @ self.joint_axis_base
        pivot_world = position + rotation @ self.joint_origin_base
        joint_rotation_base = so3_exp(self.joint_axis_base * angle)
        child_rotation = rotation @ joint_rotation_base @ self.child_zero_rotation
        child_origin_base = (self.joint_origin_base + joint_rotation_base
                             @ (self.child_zero_translation - self.joint_origin_base))
        child_origin_world = position + rotation @ child_origin_base
        com0 = position + rotation @ self.com_local[0]
        com1 = child_origin_world + child_rotation @ self.com_local[1]
        system_com = (self.mass[0] * com0 + self.mass[1] * com1) / self.mass.sum()
        return {"axis_world": axis_world, "pivot_world": pivot_world,
                "child_rotation_world": child_rotation,
                "part_com_world": torch.stack((com0, com1)),
                "system_com_world": system_com}

    def acceleration(self, state: ArticulatedState, wrench: torch.Tensor) -> torch.Tensor:
        """Return world-frame [base linear, base angular, joint] acceleration."""
        if wrench.shape != (7,) or state.generalized_velocity.shape != (7,):
            raise ValueError("wrench and generalized velocity must have shape (7,)")
        geometry = self.geometry(state)
        position = state.base_position_world
        omega = state.generalized_velocity[3:6]
        joint_rate = state.generalized_velocity[6]
        axis, pivot = geometry["axis_world"], geometry["pivot_world"]
        com = geometry["part_com_world"]
        r1, s1 = com[1] - com[0], com[1] - pivot
        joint_linear = torch.linalg.cross(axis, s1)
        zero3 = torch.zeros((3, 3), dtype=wrench.dtype, device=wrench.device)
        identity = torch.eye(3, dtype=wrench.dtype, device=wrench.device)
        zero_column = torch.zeros((3, 1), dtype=wrench.dtype, device=wrench.device)
        jv0 = torch.cat((identity, zero3, zero_column), dim=1)
        jv1 = torch.cat((identity, -skew(r1), joint_linear[:, None]), dim=1)
        jw0 = torch.cat((zero3, identity, zero_column), dim=1)
        jw1 = torch.cat((zero3, identity, axis[:, None]), dim=1)
        base_rotation = state.base_rotation_world
        child_rotation = geometry["child_rotation_world"]
        inertia0 = base_rotation @ self.inertia_local[0] @ base_rotation.T
        inertia1 = child_rotation @ self.inertia_local[1] @ child_rotation.T
        mass_matrix = (self.mass[0] * (jv0.T @ jv0) + jw0.T @ inertia0 @ jw0
                       + self.mass[1] * (jv1.T @ jv1) + jw1.T @ inertia1 @ jw1)

        omega1 = omega + axis * joint_rate
        acceleration_bias0 = torch.zeros(3, dtype=wrench.dtype, device=wrench.device)
        joint_velocity = joint_linear * joint_rate
        acceleration_bias1 = (torch.linalg.cross(omega, torch.linalg.cross(omega, r1))
                              + 2.0 * torch.linalg.cross(omega, joint_velocity)
                              + torch.linalg.cross(axis, torch.linalg.cross(axis, s1))
                              * joint_rate.square())
        angular_bias1 = torch.linalg.cross(omega, axis) * joint_rate
        force0 = self.mass[0] * (acceleration_bias0 - self.gravity_world)
        force1 = self.mass[1] * (acceleration_bias1 - self.gravity_world)
        torque0 = torch.linalg.cross(omega, inertia0 @ omega)
        torque1 = (inertia1 @ angular_bias1
                   + torch.linalg.cross(omega1, inertia1 @ omega1))
        bias = jv0.T @ force0 + jw0.T @ torque0 + jv1.T @ force1 + jw1.T @ torque1

        force_world = wrench[:3]
        torque_base_com = (wrench[3:6]
                           + torch.linalg.cross(geometry["system_com_world"] - com[0],
                                                force_world))
        generalized_wrench = torch.cat((force_world, torque_base_com, wrench[6:7]))
        return torch.linalg.solve(mass_matrix, generalized_wrench - bias)

    def inverse_dynamics(self, state: ArticulatedState,
                         generalized_acceleration: torch.Tensor) -> torch.Tensor:
        """Return the 7D wrench producing a requested coordinate acceleration.

        ``acceleration`` is affine in the public wrench convention (force and
        torque about system COM, followed by joint torque).  Recovering that
        affine map here keeps forward and inverse dynamics exactly aligned and
        avoids duplicating the floating-base origin/COM conversion.
        """
        if generalized_acceleration.shape != (7,):
            raise ValueError("generalized_acceleration must have shape (7,)")
        zero = torch.zeros(7, dtype=generalized_acceleration.dtype,
                           device=generalized_acceleration.device)
        bias_acceleration = self.acceleration(state, zero)
        columns = []
        for index in range(7):
            unit = zero.clone(); unit[index] = 1.0
            columns.append(self.acceleration(state, unit) - bias_acceleration)
        inverse_mass_map = torch.stack(columns, dim=1)
        return torch.linalg.solve(
            inverse_mass_map, generalized_acceleration - bias_acceleration)

    def step(self, state: ArticulatedState, wrench: torch.Tensor) -> ArticulatedState:
        velocity = state.generalized_velocity + self.dt * self.acceleration(state, wrench)
        old_com = state.base_position_world + state.base_rotation_world @ self.com_local[0]
        rotation = newton_quaternion_rotation_step(
            state.base_rotation_world, self.dt * velocity[3:6])
        new_com = old_com + self.dt * velocity[:3]
        position = new_com - rotation @ self.com_local[0]
        angle = state.joint_angle + self.dt * velocity[6]
        return ArticulatedState(position, rotation, angle, velocity)

    def rollout_interval(self, state: ArticulatedState, wrench: torch.Tensor) -> ArticulatedState:
        for _ in range(self.substeps):
            state = self.step(state, wrench)
        return state
