"""Newton/Featherstone reference backend for a free base plus one hinge."""
from __future__ import annotations

import numpy as np
import torch

from .parameters import NormalizedPartPhysics
from .articulated_torch import ArticulatedState


class ArticulatedNewtonBackend:
    """Independent Newton reference for :class:`ArticulatedTorchBackend`.

    The public seven-dimensional wrench is identical to the Torch backend:
    world force and torque about the *system* COM, followed by hinge torque.
    Newton receives the equivalent wrench on the base-link COM plus the hinge
    control.  As in :class:`ArticulatedTorchBackend`, this wrench is the
    non-gravitational external wrench; both implementations add gravity in the
    equations of motion.
    """

    def __init__(self, bodies: list[NormalizedPartPhysics], child_zero_transform: np.ndarray,
                 *, fps: float = 25.0, substeps: int = 4,
                 gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.80665),
                 device: str = "cpu", requires_grad: bool = False) -> None:
        if len(bodies) != 2 or bodies[1].joint_type != "revolute":
            raise ValueError("expected a fixed base body and one revolute child")
        if fps <= 0 or substeps < 1:
            raise ValueError("fps and substeps must be positive")
        zero = np.asarray(child_zero_transform, dtype=np.float64)
        if zero.shape != (4, 4):
            raise ValueError("child_zero_transform must have shape (4,4)")
        try:
            import newton
            import warp as wp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("ArticulatedNewtonBackend requires newton and warp") from exc

        self.newton, self.wp = newton, wp
        self.bodies = bodies
        self.zero = zero
        self.gravity_world = np.asarray(gravity_world, dtype=np.float32)
        self.total_mass = float(sum(body.mass for body in bodies))
        self.dt = 1.0 / (float(fps) * int(substeps))
        self.substeps = int(substeps)
        newton.use_coord_layout_targets = True

        builder = newton.ModelBuilder(gravity=tuple(gravity_world))
        links = []
        for body in bodies:
            links.append(builder.add_link(
                com=wp.vec3(*np.asarray(body.com_part, dtype=np.float32)),
                inertia=wp.mat33(np.asarray(body.inertia_com_part, dtype=np.float32)),
                mass=float(body.mass), lock_inertia=True, label=body.name,
            ))
        free_joint = builder.add_joint_free(child=links[0], label="floating_base")

        pivot = np.asarray(bodies[1].joint_origin_parent, dtype=np.float64)
        parent_anchor = np.eye(4, dtype=np.float64)
        parent_anchor[:3, 3] = pivot
        child_anchor = np.linalg.inv(zero) @ parent_anchor
        revolute_joint = builder.add_joint_revolute(
            parent=links[0], child=links[1],
            parent_xform=self._wp_transform(parent_anchor),
            child_xform=self._wp_transform(child_anchor),
            axis=wp.vec3(*np.asarray(bodies[1].joint_axis_parent, dtype=np.float32)),
            damping=0.0, label="hinge",
        )
        builder.add_articulation([free_joint, revolute_joint], label="articulated_object")
        self.model = builder.finalize(device=device, requires_grad=requires_grad)
        self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=0.0)
        self.base_body, self.child_body = links
        self.hinge_dof = int(self.model.joint_qd_start.numpy()[revolute_joint])
        self.hinge_q = int(self.model.joint_q_start.numpy()[revolute_joint])

    def _wp_transform(self, matrix: np.ndarray):
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat().astype(np.float32)
        return self.wp.transform(self.wp.vec3(*matrix[:3, 3].astype(np.float32)), self.wp.quat(*q))

    @staticmethod
    def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(rotation).as_quat().astype(np.float32)

    def _initialize_state(self, state: ArticulatedState):
        position = state.base_position_world.detach().cpu().numpy().astype(np.float64)
        rotation = state.base_rotation_world.detach().cpu().numpy().astype(np.float64)
        velocity = state.generalized_velocity.detach().cpu().numpy().astype(np.float64)
        q = self.model.joint_q.numpy().copy()
        q[:3] = position.astype(np.float32)
        q[3:7] = self._matrix_to_quaternion(rotation)
        q[self.hinge_q] = float(state.joint_angle.detach().cpu())

        # The public articulated state follows Newton's base-link COM twist.
        qd = self.model.joint_qd.numpy().copy()
        qd[:3] = velocity[:3].astype(np.float32)
        qd[3:6] = velocity[3:6].astype(np.float32)
        qd[self.hinge_dof] = float(velocity[6])
        result = self.model.state()
        result.joint_q.assign(q)
        result.joint_qd.assign(qd)
        self.newton.eval_fk(self.model, result.joint_q, result.joint_qd, result)
        return result

    def rollout_interval(self, state: ArticulatedState, wrench: torch.Tensor) -> ArticulatedState:
        if wrench.shape != (7,) or state.generalized_velocity.shape != (7,):
            raise ValueError("wrench and generalized velocity must have shape (7,)")
        state_in = self._initialize_state(state)
        state_out = self.model.state()
        control = self.model.control()
        applied = wrench.detach().cpu().numpy().astype(np.float32)

        body_q = state_in.body_q.numpy()
        base_origin = body_q[self.base_body, :3]
        base_rotation = body_q[self.base_body, 3:7]
        qx, qy, qz, qw = base_rotation
        rotation = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
        ], dtype=np.float64)
        base_com = base_origin + rotation @ np.asarray(self.bodies[0].com_part)

        # Compute the current system COM from Newton FK.
        coms = []
        for index, body in enumerate(self.bodies):
            raw = body_q[index]
            qx, qy, qz, qw = raw[3:7]
            r = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
            ])
            coms.append(raw[:3] + r @ np.asarray(body.com_part))
        system_com = sum(body.mass * com for body, com in zip(self.bodies, coms)) / self.total_mass

        force = applied[:3]
        torque_base_com = applied[3:6] + np.cross(system_com - base_com, applied[:3])
        body_force = np.zeros((self.model.body_count, 6), dtype=np.float32)
        body_force[self.base_body, :3] = force
        body_force[self.base_body, 3:6] = torque_base_com
        joint_force = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        joint_force[self.hinge_dof] = applied[6]
        control.joint_f.assign(joint_force)

        for _ in range(self.substeps):
            state_in.clear_forces()
            state_in.body_f.assign(body_force)
            self.solver.step(state_in, state_out, control, None, self.dt)
            state_in, state_out = state_out, state_in

        q = state_in.joint_q.numpy()
        qd = state_in.joint_qd.numpy()
        body_q = state_in.body_q.numpy()[self.base_body]
        body_qd = state_in.body_qd.numpy()[self.base_body]
        quat = body_q[3:7]
        qx, qy, qz, qw = quat
        rotation = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
        ])
        reference = state.base_position_world
        tensor = lambda value: torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        return ArticulatedState(
            tensor(body_q[:3]), tensor(rotation), tensor(q[self.hinge_q]),
            tensor(np.concatenate((body_qd[:3], body_qd[3:6], qd[self.hinge_dof:self.hinge_dof+1]))),
        )
