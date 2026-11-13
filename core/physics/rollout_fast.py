"""Tensor-only midpoint solver with deferred error checking.

The caller MUST check the returned failure flag before using this graph for
backward. This preserves the legacy whole-batch singular-solve rejection while
avoiding a CUDA-to-host error check at every Newton iteration.
"""
from __future__ import annotations

from functools import lru_cache, partial

import torch


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        vector.shape[:-1] + (3, 3))


def _matvec(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _matvec3(matrix, vector):
    col0, col1, col2 = matrix.unbind(-1)
    x, y, z = vector.unbind(-1)
    return col0 * x[..., None] + col1 * y[..., None] + col2 * z[..., None]


def _matmul3(left, right):
    col0, col1, col2 = left.unbind(-1)
    row0, row1, row2 = right.unbind(-2)
    return (col0[..., :, None] * row0[..., None, :]
            + col1[..., :, None] * row1[..., None, :]
            + col2[..., :, None] * row2[..., None, :])


class _Solve3(torch.autograd.Function):
    """Fixed-size partial-pivot LU with the first-order solve adjoint."""

    @staticmethod
    def forward(ctx, matrix, rhs):
        row0, row1, row2 = matrix.unbind(-2)
        abs0, abs1, abs2 = (row[..., 0].abs() for row in (row0, row1, row2))
        swap01 = (abs1 > abs0) & (abs1 >= abs2)
        swap02 = (abs2 > abs0) & (abs2 > abs1)
        first = torch.where(swap01[..., None], row1, row0)
        first = torch.where(swap02[..., None], row2, first)
        u00, u01, u02 = first.unbind(-1)
        a10, a11, a12 = torch.where(swap01[..., None], row0, row1).unbind(-1)
        a20, a21, a22 = torch.where(swap02[..., None], row0, row2).unbind(-1)

        # A failed lane is rejected by the rollout caller. A safe denominator
        # lets the rest of the batched kernel report failures without a sync.
        bad0 = u00 == 0
        u00 = torch.where(bad0, 1.0, u00)
        l10, l20 = a10 / u00, a20 / u00
        a11, a12 = a11 - l10 * u01, a12 - l10 * u02
        a21, a22 = a21 - l20 * u01, a22 - l20 * u02
        swap12 = a21.abs() > a11.abs()
        l10, l20 = torch.where(swap12, l20, l10), torch.where(swap12, l10, l20)
        u11, a21 = torch.where(swap12, a21, a11), torch.where(swap12, a11, a21)
        u12, a22 = torch.where(swap12, a22, a12), torch.where(swap12, a12, a22)
        bad1 = u11 == 0
        u11 = torch.where(bad1, 1.0, u11)
        l21 = a21 / u11
        u22 = a22 - l21 * u12
        bad2 = u22 == 0
        u22 = torch.where(bad2, 1.0, u22)

        b0, b1, b2 = rhs.unbind(-1)
        y0 = torch.where(swap02, b2, torch.where(swap01, b1, b0))
        b1, b2 = torch.where(swap01, b0, b1), torch.where(swap02, b0, b2)
        y1 = torch.where(swap12, b2, b1) - l10 * y0
        y2 = torch.where(swap12, b1, b2) - l20 * y0 - l21 * y1
        x2 = y2 / u22
        x1 = (y1 - u12 * x2) / u11
        x0 = (y0 - u01 * x1 - u02 * x2) / u00
        solution = torch.stack((x0, x1, x2), dim=-1)
        failed = bad0 | bad1 | bad2
        ctx.save_for_backward(solution, u00, u01, u02, u11, u12, u22,
                              l10, l20, l21, swap01, swap02, swap12)
        ctx.mark_non_differentiable(failed)
        return solution, failed

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient, _failed):  # pyright: ignore[reportIncompatibleMethodOverride]
        (solution, u00, u01, u02, u11, u12, u22,
         l10, l20, l21, swap01, swap02, swap12) = ctx.saved_tensors
        # Solve A.T @ adjoint = gradient with the saved factors and apply
        # the inverse row permutation after the two triangular solves.
        g0, g1, g2 = gradient.unbind(-1)
        y0 = g0 / u00
        y1 = (g1 - u01 * y0) / u11
        y2 = (g2 - u02 * y0 - u12 * y1) / u22
        z1 = y1 - l21 * y2
        z0 = y0 - l10 * z1 - l20 * y2
        z1, z2 = torch.where(swap12, y2, z1), torch.where(swap12, z1, y2)
        adjoint = torch.stack((torch.where(swap02, z2, torch.where(swap01, z1, z0)),
                               torch.where(swap01, z0, z1),
                               torch.where(swap02, z0, z2)), dim=-1)
        return -adjoint.unsqueeze(-1) * solution.unsqueeze(-2), adjoint


def midpoint_ex(omega, torque, inertia, inertia_inv, dt, identity, iterations,
                specialize_solve=False, specialize_products=False):
    """Same fixed Newton updates as the reference, including FP64 arithmetic."""
    initial = omega
    updated = initial
    failed = torch.zeros((), dtype=torch.bool, device=omega.device)
    dt_matrix = dt.unsqueeze(-1)
    matvec = _matvec3 if specialize_products else _matvec
    matmul = _matmul3 if specialize_products else torch.matmul
    for _ in range(iterations):
        midpoint = 0.5 * (initial + updated)
        momentum = matvec(inertia, midpoint)
        gyroscopic = torch.cross(midpoint, momentum, dim=-1)
        residual = updated - initial - dt * matvec(inertia_inv, torque - gyroscopic)
        jacobian = identity + 0.5 * dt_matrix * matmul(
            inertia_inv, matmul(_skew(midpoint), inertia) - _skew(momentum))
        if specialize_solve:
            correction, singular = _Solve3.apply(jacobian, residual)  # pyright: ignore[reportGeneralTypeIssues]
        else:
            correction, info = torch.linalg.solve_ex(
                jacobian, residual.unsqueeze(-1), check_errors=False)
            correction, singular = correction.squeeze(-1), info != 0
        failed = failed | singular.any()
        updated = updated - correction
    return updated, failed


@lru_cache(maxsize=1)
def compiled_midpoint():
    # No CUDA graphs: parameters/state are newly allocated for every batch.
    # Compile only the small fixed-iteration tensor function, not the model or
    # Python Physics objects. Lazy construction avoids touching compiler state
    # in reader workers or reference-only processes.
    return torch.compile(midpoint_ex, backend="inductor", fullgraph=True,
                         dynamic=True, mode="default")


def rigid_substep(x, q, v, omega, force, torque, mass, inertia, inertia_inv,
                  dt, identity, iterations, specialize_solve=False, specialize_products=False):
    """Production implicit substep, including world/body transforms and pose update."""
    from .math3d import quat_to_matrix, rotation_vector_to_quat, normalize_quat, quat_mul
    rotation = quat_to_matrix(q)
    rotation_t = rotation.transpose(-1, -2)
    matvec = _matvec3 if specialize_products else _matvec
    omega_body = matvec(rotation_t, omega)
    v = v + dt * force / mass
    x = x + dt * v
    torque_body = matvec(rotation_t, torque)
    omega_body, failed = midpoint_ex(omega_body, torque_body, inertia,
                                     inertia_inv, dt, identity, iterations,
                                     specialize_solve, specialize_products)
    omega = matvec(rotation, omega_body)
    q = normalize_quat(quat_mul(rotation_vector_to_quat(omega * dt), q))
    return x, q, v, omega, failed


def _inference_rigid_substep(*args, **kwargs):
    return rigid_substep(*args, **kwargs)


@lru_cache(maxsize=2)
def compiled_substep(*, inference: bool = False):
    # Validation uses a separate Dynamo frame from the captured/autograd
    # training kernel. Unseen validation layouts can fall back to eager
    # execution if they exhaust that separate cache.
    step = _inference_rigid_substep if inference else rigid_substep
    return torch.compile(step, backend="inductor", fullgraph=not inference,
                         dynamic=True, mode="default")


def dense_rollout_input(tensor: torch.Tensor) -> torch.Tensor:
    # contiguous() alone can keep a parent stride on size-one axes.
    return tensor.contiguous().view(-1).view(tensor.shape)


def _flat_rollout_input(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.flatten(0, 1).contiguous()


def _rigid_pose_errors(net, position, rotation, velocity, mass, inertia, inertia_inv,
                       dt, identity, mask, *, horizon, substeps, iterations,
                       position_scale, rotation_scale, flat_substep=False):
    from .math3d import matrix_to_quat, normalize_quat, quat_to_matrix

    batch = net.shape[0]
    starts = net.shape[1] - horizon + 1
    layout = _flat_rollout_input if flat_substep else dense_rollout_input
    x = layout(position[:, :starts])
    q = matrix_to_quat(rotation[:, :starts])
    v = layout(velocity[:, :starts, :3])
    omega = layout(velocity[:, :starts, 3:])
    if flat_substep:
        q = layout(q)
        # Keep the first substep's gradient guards consistent with those of
        # later substeps. The initial state comes from fixed supervision.
        if torch.is_grad_enabled():
            for state in (x, q, v, omega):
                state.requires_grad_(True)
        mass, inertia, inertia_inv, dt = (
            layout(value.expand(batch, starts, *value.shape[2:]))
            for value in (mass, inertia, inertia_inv, dt))
    failed = torch.zeros((), dtype=torch.bool, device=net.device)
    position_terms, rotation_terms = [], []
    substep = compiled_substep()
    for step in range(horizon):
        force, torque = (
            layout(component)
            for component in net[:, step:step + starts].split(3, dim=-1))
        q = normalize_quat(q)
        if not flat_substep:
            q = dense_rollout_input(q)
        for _ in range(substeps):
            x, q, v, omega, singular = substep(
                x, q, v, omega, force, torque, mass, inertia, inertia_inv,
                dt, identity, iterations, specialize_solve=True, specialize_products=True)
            failed = failed | singular
        target = slice(step + 1, step + 1 + starts)
        predicted_position, predicted_rotation = x, quat_to_matrix(q)
        if flat_substep:
            predicted_position = x.reshape(batch, starts, 3)
            predicted_rotation = predicted_rotation.reshape(batch, starts, 3, 3)
        position_terms.append(((predicted_position - position[:, target]).norm(dim=-1) / position_scale).square())
        rotation_terms.append(0.5 * ((predicted_rotation - rotation[:, target])
                                    / rotation_scale).square().sum((-1, -2)))
    weight = mask.unsqueeze(-1)
    denominator = mask.sum().clamp_min(1) * horizon
    position_loss = torch.where(weight, torch.stack(position_terms, -1), 0.0).sum() / denominator
    rotation_loss = torch.where(weight, torch.stack(rotation_terms, -1), 0.0).sum() / denominator
    return position_loss, rotation_loss, failed


class RigidRolloutGraph:
    """Own one training graph. The caller uses the eager path for other shapes."""

    def __init__(self, *, flat_substep: bool = False):
        self._flat_substep = flat_substep
        self._key = None
        self._run = None

    def run(self, backend, net, position, rotation, velocity, mask, *,
            horizon, position_scale, rotation_scale):
        args = (net, position, rotation, velocity, backend.mass, backend.inertia,
                backend.inertia_inv, backend.dt, backend._identity, mask)
        settings = (horizon, backend.substeps, backend.implicit_iterations,
                    position_scale, rotation_scale)
        layouts = tuple((arg.device, arg.dtype, arg.shape, arg.stride(), arg.requires_grad)
                        for arg in args)
        key = settings, layouts
        if self._key is not None and self._key != key:
            return None
        if self._run is None:
            pose_errors = partial(_rigid_pose_errors, flat_substep=True) if self._flat_substep else _rigid_pose_errors
            rollout = partial(pose_errors, horizon=horizon, substeps=backend.substeps,
                              iterations=backend.implicit_iterations,
                              position_scale=position_scale, rotation_scale=rotation_scale)
            # The capture writes subsequent inputs into this storage. Never
            # let it take ownership of the caller's batch tensors.
            samples = tuple(arg.detach().clone().requires_grad_(arg.requires_grad) for arg in args)
            self._run = torch.cuda.make_graphed_callables(rollout, samples, num_warmup_iters=3)
            self._key = key
            print(f"[rollout] captured CUDA graph batch={net.shape[0]} transitions={net.shape[1]} "
                  f"substeps={backend.substeps}", flush=True)
        return self._run(*args)
