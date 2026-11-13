"""Interchangeable 4-substep free-rigid-body forward backends.

Both backends consume a *net* world-frame wrench about the COM.  The Torch
path directly integrates net force; the Newton path subtracts ``m*g`` before
writing the force accumulator because Newton adds gravity internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch

from .math3d import (
    normalize_quat,
    quat_mul,
    quat_to_matrix,
    rotation_geodesic,
    rotation_vector_to_quat,
)


DEFAULT_MAX_ROTATION_STEP_RAD = 0.1
DEFAULT_MAX_OMEGA_STEP_RAD_S = 10.0
DEFAULT_MAX_ADAPTIVE_SUBSTEPS = 64
DEFAULT_INTEGRATOR = "implicit_midpoint"
DEFAULT_IMPLICIT_ITERATIONS = 6


def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def _implicit_midpoint_angular_velocity(
    omega_body: torch.Tensor,
    torque_body: torch.Tensor,
    inertia: torch.Tensor,
    inertia_inv: torch.Tensor,
    dt: torch.Tensor | float,
    iterations: int,
) -> torch.Tensor:
    """Solve one body-frame Euler step with fixed differentiable iterations."""
    dt_vector = torch.as_tensor(
        dt, dtype=omega_body.dtype, device=omega_body.device
    )
    while dt_vector.ndim < omega_body.ndim:
        dt_vector = dt_vector.unsqueeze(-1)
    dt_matrix = dt_vector.unsqueeze(-1)
    identity = torch.eye(
        3, dtype=omega_body.dtype, device=omega_body.device
    )
    initial = omega_body
    updated = initial
    for _ in range(iterations):
        midpoint = 0.5 * (initial + updated)
        angular_momentum = (
            inertia @ midpoint.unsqueeze(-1)
        ).squeeze(-1)
        gyroscopic = torch.cross(midpoint, angular_momentum, dim=-1)
        residual = updated - initial - dt_vector * (
            inertia_inv @ (torque_body - gyroscopic).unsqueeze(-1)
        ).squeeze(-1)
        gyroscopic_jacobian = (
            _skew_matrix(midpoint) @ inertia
            - _skew_matrix(angular_momentum)
        )
        jacobian = identity + 0.5 * dt_matrix * (
            inertia_inv @ gyroscopic_jacobian
        )
        correction = torch.linalg.solve(
            jacobian, residual.unsqueeze(-1)
        ).squeeze(-1)
        updated = updated - correction
    return updated


def _adaptive_substep_count(
    omega: torch.Tensor,
    angular_acceleration: torch.Tensor,
    dt: torch.Tensor | float,
    max_rotation_step_rad: float,
    max_omega_step_rad_s: float,
    max_adaptive_substeps: int,
) -> torch.Tensor:
    """Choose an independent microstep count for every rollout trajectory."""
    with torch.no_grad():
        dt_tensor = torch.as_tensor(dt, dtype=omega.dtype, device=omega.device)
        while dt_tensor.ndim > omega.ndim - 1:
            dt_tensor = dt_tensor.squeeze(-1)
        rotation_ratio = dt_tensor * omega.norm(dim=-1) / max_rotation_step_rad
        omega_ratio = (
            dt_tensor * angular_acceleration.norm(dim=-1)
            / max_omega_step_rad_s
        )
        required = torch.maximum(rotation_ratio, omega_ratio)
        required = torch.nan_to_num(
            required, nan=float(max_adaptive_substeps),
            posinf=float(max_adaptive_substeps), neginf=1.0,
        )
        return torch.ceil(required).clamp(1, max_adaptive_substeps).to(torch.int64)


@dataclass(frozen=True)
class Physics:
    mass: float
    com_body: tuple[float, float, float]
    inertia_body: tuple[float, float, float] | np.ndarray
    gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.80665)
    dt: float = 1.0 / 120.0
    substeps: int = 4

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        inertia = np.asarray(self.inertia_body, dtype=np.float64)
        if inertia.shape == (3,):
            inertia = np.diag(inertia)
        if inertia.shape != (3, 3) or np.any(np.linalg.eigvalsh(inertia) <= 0):
            raise ValueError("inertia_body must be a positive 3-vector or SPD 3x3 matrix")
        object.__setattr__(self, "inertia_body", inertia)


@dataclass
class RigidState:
    x_com: torch.Tensor
    quat_xyzw: torch.Tensor
    v_world: torch.Tensor
    omega_world: torch.Tensor

    def clone(self) -> "RigidState":
        return RigidState(*(x.clone() for x in (self.x_com, self.quat_xyzw, self.v_world, self.omega_world)))


class Backend(Protocol):
    def rollout_interval(self, state: RigidState, wrench_net_world: torch.Tensor) -> RigidState: ...


class TorchBackend:
    """Differentiable no-contact replica of Newton's `integrate_rigid_body`."""
    def __init__(
        self,
        physics: Physics,
        *,
        max_rotation_step_rad: float = DEFAULT_MAX_ROTATION_STEP_RAD,
        max_omega_step_rad_s: float = DEFAULT_MAX_OMEGA_STEP_RAD_S,
        max_adaptive_substeps: int = DEFAULT_MAX_ADAPTIVE_SUBSTEPS,
        integrator: str = DEFAULT_INTEGRATOR,
        implicit_iterations: int = DEFAULT_IMPLICIT_ITERATIONS,
    ):
        self.physics = physics
        self.max_rotation_step_rad = float(max_rotation_step_rad)
        self.max_omega_step_rad_s = float(max_omega_step_rad_s)
        self.max_adaptive_substeps = int(max_adaptive_substeps)
        self.integrator = str(integrator).strip().lower()
        self.implicit_iterations = int(implicit_iterations)
        if self.integrator not in {"implicit_midpoint", "adaptive_explicit"}:
            raise ValueError(
                "integrator must be implicit_midpoint or adaptive_explicit"
            )
        if self.implicit_iterations < 1:
            raise ValueError("implicit_iterations must be positive")
        if self.max_rotation_step_rad <= 0 or self.max_omega_step_rad_s <= 0:
            raise ValueError("adaptive integration limits must be positive")
        if self.max_adaptive_substeps < 1:
            raise ValueError("max_adaptive_substeps must be positive")
        self._inertia_cache: dict[tuple[torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _inertia(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache the episode-constant inertia and its inverse per dtype/device."""
        key = (reference.dtype, reference.device)
        cached = self._inertia_cache.get(key)
        if cached is None:
            inertia = torch.as_tensor(
                self.physics.inertia_body, dtype=reference.dtype, device=reference.device)
            cached = (inertia, torch.linalg.inv(inertia))
            self._inertia_cache[key] = cached
        return cached

    def rollout_interval(self, state: RigidState, wrench_net_world: torch.Tensor) -> RigidState:
        if wrench_net_world.shape[-1] != 6:
            raise ValueError("wrench_net_world must end in 6 values [force, torque]")
        x, q, v, omega = state.x_com, normalize_quat(state.quat_xyzw), state.v_world, state.omega_world
        F, tau = wrench_net_world[..., :3], wrench_net_world[..., 3:]
        I, I_inv = self._inertia(x)
        h = self.physics.dt
        for _ in range(self.physics.substeps):
            R = quat_to_matrix(q)
            wb = (R.transpose(-1, -2) @ omega.unsqueeze(-1)).squeeze(-1)
            tb = (R.transpose(-1, -2) @ tau.unsqueeze(-1)).squeeze(-1) - torch.cross(wb, (I @ wb.unsqueeze(-1)).squeeze(-1), dim=-1)
            if self.integrator == "implicit_midpoint":
                v = v + h * F / self.physics.mass
                x = x + h * v
                torque_body = (R.transpose(-1, -2) @ tau.unsqueeze(-1)).squeeze(-1)
                wb = _implicit_midpoint_angular_velocity(
                    wb, torque_body, I, I_inv, h, self.implicit_iterations
                )
                omega = (R @ wb.unsqueeze(-1)).squeeze(-1)
                delta_q = rotation_vector_to_quat(omega * h)
                q = normalize_quat(quat_mul(delta_q, q))
                continue
            alpha_body = (I_inv @ tb.unsqueeze(-1)).squeeze(-1)
            adaptive_substeps = _adaptive_substep_count(
                omega, alpha_body, h, self.max_rotation_step_rad,
                self.max_omega_step_rad_s, self.max_adaptive_substeps,
            )
            micro_h = h / adaptive_substeps.to(omega.dtype)
            micro_h_vector = micro_h.unsqueeze(-1)
            for microstep in range(int(adaptive_substeps.max().item())):
                active = microstep < adaptive_substeps
                candidate_v = v + micro_h_vector * F / self.physics.mass
                candidate_x = x + micro_h_vector * candidate_v
                v = torch.where(active.unsqueeze(-1), candidate_v, v)
                x = torch.where(active.unsqueeze(-1), candidate_x, x)
                R = quat_to_matrix(q)
                wb = (R.transpose(-1, -2) @ omega.unsqueeze(-1)).squeeze(-1)
                tb = (
                    (R.transpose(-1, -2) @ tau.unsqueeze(-1)).squeeze(-1)
                    - torch.cross(
                        wb, (I @ wb.unsqueeze(-1)).squeeze(-1), dim=-1
                    )
                )
                wb = wb + micro_h_vector * (I_inv @ tb.unsqueeze(-1)).squeeze(-1)
                candidate_omega = (R @ wb.unsqueeze(-1)).squeeze(-1)
                omega = torch.where(active.unsqueeze(-1), candidate_omega, omega)
                delta_q = rotation_vector_to_quat(omega * micro_h_vector)
                candidate_q = normalize_quat(quat_mul(delta_q, q))
                q = torch.where(active.unsqueeze(-1), candidate_q, q)
        return RigidState(x, q, v, omega)


class BatchedTorchBackend:
    """Vectorized rigid-body integration for a batch of heterogeneous bodies.

    The first tensor dimension selects the body/episode.  Any dimensions
    between it and the final coordinate dimension (for example rollout-window
    starts) are processed in parallel.
    """

    def __init__(
        self,
        physics: Sequence[Physics],
        reference: torch.Tensor,
        *,
        max_rotation_step_rad: float = DEFAULT_MAX_ROTATION_STEP_RAD,
        max_omega_step_rad_s: float = DEFAULT_MAX_OMEGA_STEP_RAD_S,
        max_adaptive_substeps: int = DEFAULT_MAX_ADAPTIVE_SUBSTEPS,
        integrator: str = DEFAULT_INTEGRATOR,
        implicit_iterations: int = DEFAULT_IMPLICIT_ITERATIONS,
        midpoint_implementation: str = "reference",
    ):
        if not physics:
            raise ValueError("physics must contain at least one body")
        substeps = {value.substeps for value in physics}
        if len(substeps) != 1:
            raise ValueError("all batched bodies must use the same number of substeps")
        device, dtype = reference.device, reference.dtype
        self.substeps = next(iter(substeps))
        self.max_rotation_step_rad = float(max_rotation_step_rad)
        self.max_omega_step_rad_s = float(max_omega_step_rad_s)
        self.max_adaptive_substeps = int(max_adaptive_substeps)
        self.integrator = str(integrator).strip().lower()
        self.implicit_iterations = int(implicit_iterations)
        self.midpoint_implementation = str(midpoint_implementation)
        if self.midpoint_implementation not in {
                "reference", "deferred", "compiled", "compiled_substep", "compiled_solve3", "compiled_fused3"}:
            raise ValueError(f"Unknown midpoint implementation: {self.midpoint_implementation}")
        self._solve_failed = torch.zeros((), dtype=torch.bool, device=device)
        self._identity = torch.eye(3, device=device, dtype=dtype)
        if self.integrator not in {"implicit_midpoint", "adaptive_explicit"}:
            raise ValueError(
                "integrator must be implicit_midpoint or adaptive_explicit"
            )
        if self.implicit_iterations < 1:
            raise ValueError("implicit_iterations must be positive")
        if self.max_rotation_step_rad <= 0 or self.max_omega_step_rad_s <= 0:
            raise ValueError("adaptive integration limits must be positive")
        if self.max_adaptive_substeps < 1:
            raise ValueError("max_adaptive_substeps must be positive")
        if self.midpoint_implementation == "compiled_fused3" and device.type == "cuda":
            # These constants start on the host. Submit one copy for the
            # whole scene instead of a blocking copy for each body's inertia.
            count = len(physics)
            packed = np.concatenate((
                [value.mass for value in physics],
                [value.dt for value in physics],
                np.stack([value.inertia_body for value in physics]).reshape(-1),
            ))
            constants = torch.from_numpy(packed).to(device, dtype, non_blocking=True)
            self.mass = constants[:count].view(-1, 1, 1)
            self.dt = constants[count:2 * count].view(-1, 1, 1)
            self.inertia = constants[2 * count:].view(-1, 1, 3, 3)
            self.inertia_inv, info = torch.linalg.inv_ex(self.inertia, check_errors=False)
            self._solve_failed = (info != 0).any()
        else:
            self.mass = torch.as_tensor(
                [value.mass for value in physics], device=device, dtype=dtype
            ).view(-1, 1, 1)
            self.dt = torch.as_tensor(
                [value.dt for value in physics], device=device, dtype=dtype
            ).view(-1, 1, 1)
            self.inertia = torch.stack([
                torch.as_tensor(value.inertia_body, device=device, dtype=dtype)
                for value in physics
            ]).unsqueeze(1)
            self.inertia_inv = torch.linalg.inv(self.inertia)

    def check_solve_errors(self, additional_failure: torch.Tensor | None = None) -> None:
        """Call once after all rollout intervals and BEFORE loss/backward.

        Failed solve graphs are discarded by the training caller, never masked
        into the loss (zero times a NaN gradient is still NaN).
        """
        if self.midpoint_implementation != "reference":
            failed = self._solve_failed
            if additional_failure is not None:
                failed = failed | additional_failure
            if bool(failed):
                raise torch._C._LinAlgError("rollout midpoint solve or loss check failed")

    def rollout_interval(self, state: RigidState, wrench_net_world: torch.Tensor) -> RigidState:
        if wrench_net_world.shape[-1] != 6:
            raise ValueError("wrench_net_world must end in 6 values [force, torque]")
        if state.x_com.shape[0] != self.mass.shape[0]:
            raise ValueError("state batch dimension must match physics")
        x, q = state.x_com, normalize_quat(state.quat_xyzw)
        v, omega = state.v_world, state.omega_world
        force, torque = wrench_net_world[..., :3], wrench_net_world[..., 3:]
        if (self.integrator == "implicit_midpoint"
                and self.midpoint_implementation in {"compiled_substep", "compiled_solve3", "compiled_fused3"}):
            from .rollout_fast import compiled_substep, dense_rollout_input

            fused = self.midpoint_implementation == "compiled_fused3"
            if fused:
                # Slices of longer trajectories should use the same
                # compiler specialization as standalone dense tensors.
                x, q, v, omega, force, torque = (
                    dense_rollout_input(value) for value in (x, q, v, omega, force, torque))
            step = (compiled_substep(inference=True)
                    if fused and not torch.is_grad_enabled() else compiled_substep())
            for _ in range(self.substeps):
                x, q, v, omega, failed = step(
                    x, q, v, omega, force, torque, self.mass, self.inertia,
                    self.inertia_inv, self.dt, self._identity, self.implicit_iterations,
                    specialize_solve=self.midpoint_implementation != "compiled_substep",
                    specialize_products=fused)
                self._solve_failed = self._solve_failed | failed
            return RigidState(x, q, v, omega)
        for _ in range(self.substeps):
            rotation = quat_to_matrix(q)
            rotation_t = rotation.transpose(-1, -2)
            omega_body = (rotation_t @ omega.unsqueeze(-1)).squeeze(-1)
            if self.integrator == "implicit_midpoint":
                v = v + self.dt * force / self.mass
                x = x + self.dt * v
                applied_torque_body = (
                    rotation_t @ torque.unsqueeze(-1)
                ).squeeze(-1)
                if self.midpoint_implementation == "reference":
                    omega_body = _implicit_midpoint_angular_velocity(
                        omega_body, applied_torque_body,
                        self.inertia, self.inertia_inv, self.dt,
                        self.implicit_iterations)
                else:
                    from .rollout_fast import midpoint_ex, compiled_midpoint

                    solver = (compiled_midpoint() if self.midpoint_implementation == "compiled"
                              else midpoint_ex)
                    omega_body, failed = solver(
                        omega_body, applied_torque_body, self.inertia,
                        self.inertia_inv, self.dt, self._identity,
                        self.implicit_iterations)
                    self._solve_failed = self._solve_failed | failed
                omega = (
                    rotation @ omega_body.unsqueeze(-1)
                ).squeeze(-1)
                delta_q = rotation_vector_to_quat(omega * self.dt)
                q = normalize_quat(quat_mul(delta_q, q))
                continue
            torque_body = (
                (rotation_t @ torque.unsqueeze(-1)).squeeze(-1)
                - torch.cross(omega_body,
                              (self.inertia @ omega_body.unsqueeze(-1)).squeeze(-1),
                              dim=-1))
            alpha_body = (
                self.inertia_inv @ torque_body.unsqueeze(-1)
            ).squeeze(-1)
            adaptive_substeps = _adaptive_substep_count(
                omega, alpha_body, self.dt, self.max_rotation_step_rad,
                self.max_omega_step_rad_s, self.max_adaptive_substeps,
            )
            micro_dt = self.dt / adaptive_substeps.to(omega.dtype).unsqueeze(-1)
            for microstep in range(int(adaptive_substeps.max().item())):
                active = microstep < adaptive_substeps
                candidate_v = v + micro_dt * force / self.mass
                candidate_x = x + micro_dt * candidate_v
                v = torch.where(active.unsqueeze(-1), candidate_v, v)
                x = torch.where(active.unsqueeze(-1), candidate_x, x)
                rotation = quat_to_matrix(q)
                rotation_t = rotation.transpose(-1, -2)
                omega_body = (rotation_t @ omega.unsqueeze(-1)).squeeze(-1)
                torque_body = (
                    (rotation_t @ torque.unsqueeze(-1)).squeeze(-1)
                    - torch.cross(
                        omega_body,
                        (self.inertia @ omega_body.unsqueeze(-1)).squeeze(-1),
                        dim=-1,
                    )
                )
                omega_body = omega_body + micro_dt * (
                    self.inertia_inv @ torque_body.unsqueeze(-1)
                ).squeeze(-1)
                candidate_omega = (
                    rotation @ omega_body.unsqueeze(-1)
                ).squeeze(-1)
                omega = torch.where(active.unsqueeze(-1), candidate_omega, omega)
                delta_q = rotation_vector_to_quat(omega * micro_dt)
                candidate_q = normalize_quat(quat_mul(delta_q, q))
                q = torch.where(active.unsqueeze(-1), candidate_q, q)
        return RigidState(x, q, v, omega)


class NewtonBackend:
    """Reference `newton.solvers.SolverSemiImplicit` backend (CPU/GPU selectable)."""
    def __init__(self, physics: Physics, device: str = "cpu"):
        self.physics = physics
        try:
            import newton
            import warp as wp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("NewtonBackend requires the newton and warp packages") from exc
        self.newton, self.wp = newton, wp
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder(gravity=tuple(physics.gravity_world))
        builder.add_body(
            com=wp.vec3(*physics.com_body), inertia=wp.mat33(np.asarray(physics.inertia_body, dtype=np.float32)),
            mass=float(physics.mass), lock_inertia=True,
        )
        self.model = builder.finalize(device=device)
        self.solver = newton.solvers.SolverSemiImplicit(
            self.model, angular_damping=0.0, enable_tri_contact=False,
        )

    def rollout_interval(self, state: RigidState, wrench_net_world: torch.Tensor) -> RigidState:
        if state.x_com.ndim != 1:
            raise ValueError("NewtonBackend currently accepts one unbatched state; use TorchBackend for training batches")
        p = self.physics
        R = quat_to_matrix(normalize_quat(state.quat_xyzw)).detach().cpu().numpy()
        origin = state.x_com.detach().cpu().numpy() - R @ np.asarray(p.com_body)
        q = state.quat_xyzw.detach().cpu().numpy()
        q0 = np.concatenate((origin, q)).astype(np.float32)
        qd0 = np.concatenate((state.v_world.detach().cpu().numpy(), state.omega_world.detach().cpu().numpy())).astype(np.float32)
        s_in, s_out = self.model.state(), self.model.state()
        s_in.body_q.assign(q0)
        s_in.body_qd.assign(qd0)
        applied = wrench_net_world.detach().cpu().numpy().astype(np.float32).copy()
        applied[:3] -= p.mass * np.asarray(p.gravity_world, dtype=np.float32)
        for _ in range(p.substeps):
            s_in.clear_forces()
            s_in.body_f.assign(applied)
            self.solver.step(s_in, s_out, self.model.control(), None, p.dt)
            s_in, s_out = s_out, s_in
        raw_q = s_in.body_q.numpy()[0]
        raw_qd = s_in.body_qd.numpy()[0]
        q_out = torch.as_tensor(raw_q[3:], dtype=state.quat_xyzw.dtype, device=state.quat_xyzw.device)
        R_out = quat_to_matrix(q_out).detach().cpu().numpy()
        x_out = raw_q[:3] + R_out @ np.asarray(p.com_body)
        return RigidState(
            torch.as_tensor(x_out, dtype=state.x_com.dtype, device=state.x_com.device), normalize_quat(q_out),
            torch.as_tensor(raw_qd[:3], dtype=state.v_world.dtype, device=state.v_world.device),
            torch.as_tensor(raw_qd[3:], dtype=state.omega_world.dtype, device=state.omega_world.device),
        )


def replay_sequence(initial: RigidState, wrench: torch.Tensor,
                    backend: Backend) -> list[RigidState]:
    """Replay a wrench sequence and retain every interval boundary state."""
    states = [initial.clone()]
    current = initial
    for value in wrench:
        current = backend.rollout_interval(current, value)
        states.append(current)
    return states


def summarize_replay(states: list[RigidState], x_target: torch.Tensor,
                     q_target: torch.Tensor) -> dict[str, float]:
    """Summarize position and orientation errors at replay boundaries."""
    x = torch.stack([state.x_com for state in states])
    q = torch.stack([state.quat_xyzw for state in states])
    position_error = (x - x_target).norm(dim=-1)
    rotation_error = rotation_geodesic(quat_to_matrix(q_target), quat_to_matrix(q))
    return {
        "position_mean_m": float(position_error.mean()),
        "position_max_m": float(position_error.max()),
        "rotation_mean_rad": float(rotation_error.mean()),
        "rotation_max_rad": float(rotation_error.max()),
    }
