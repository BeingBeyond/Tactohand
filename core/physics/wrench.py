"""Differentiable camera-frame contact-force aggregation (SI units)."""
from __future__ import annotations

import torch


def gravity_wrench(physics, reference: torch.Tensor,
                   gravity: torch.Tensor | None = None,
                   *, mass: torch.Tensor | None = None) -> torch.Tensor:
    """Gravity about each body's COM, in the reference wrench's frame.

    ``reference`` is [B,L,6]; optional acceleration is [B,L,3] in m/s².
    Physics.gravity_world remains the constant-gravity compatibility fallback.
    Callers with materialized physics can reuse their [B,1,1] mass tensor.
    """
    if reference.ndim != 3 or reference.shape[-1] != 6:
        raise ValueError('gravity reference must be [B,L,6]')
    if len(physics) != reference.shape[0]:
        raise ValueError('one Physics record is required per batch item')
    if gravity is None:
        acceleration = reference.new_tensor([p.gravity_world for p in physics])[:, None, :]
        acceleration = acceleration.expand(*reference.shape[:-1], 3)
    else:
        acceleration = gravity.to(reference)
        if acceleration.shape != reference.shape[:-1] + (3,):
            raise ValueError('gravity must match the full wrench transition axis')
    if mass is None:
        mass = reference.new_tensor([p.mass for p in physics])[:, None, None]
    elif mass.shape != (reference.shape[0], 1, 1):
        raise ValueError('mass must match the reference wrench batch')
    force = mass * acceleration
    return torch.cat((force, torch.zeros_like(force)), dim=-1)


def aggregate_vertex_forces(force: torch.Tensor, lever: torch.Tensor) -> torch.Tensor:
    """Sum forces and r x f torques about the GT object COM.

    Both inputs are [..., vertices, 3], in N and m respectively. This returns
    hand-contact wrench only; gravity and environment forces are separate.
    Accumulate in float32 even under mixed precision.
    """
    if force.shape != lever.shape or force.ndim < 2 or force.shape[-1] != 3:
        raise ValueError("force and lever must have matching [..., vertices, 3] shapes")
    dtype = torch.float64 if force.dtype == torch.float64 else torch.float32
    force, lever = force.to(dtype), lever.to(dtype)
    return torch.cat((force.sum(-2), torch.linalg.cross(lever, force).sum(-2)), -1)


def aggregate_body_forces(force: torch.Tensor, points: torch.Tensor,
                          centers: torch.Tensor, owner: torch.Tensor, *,
                          owner_range_validated: bool = False) -> torch.Tensor:
    """Scatter vertex forces into independent COM-referenced body wrenches.

    Shapes: force/points [B,L,V,3], centers [B,L,O,3], owner [B,L,V].
    Ownership is geometry-derived reference data, independent of predictions;
    -1 marks vertices outside reference contact support. No body receives a
    copy of another body's wrench.
    """
    if force.shape != points.shape or owner.shape != force.shape[:-1]:
        raise ValueError('body force attribution shapes disagree')
    if centers.shape[:2] != force.shape[:2] or centers.shape[-1] != 3:
        raise ValueError('body centers must be [B,L,O,3]')
    count = centers.shape[-2]
    if not owner_range_validated and bool(((owner < -1) | (owner >= count)).any()):
        raise ValueError('vertex owner lies outside physical body slots')
    index = owner.long().clamp_min(0)
    center = torch.gather(centers.to(force), 2, index[..., None].expand(-1, -1, -1, 3))
    active = (owner >= 0).unsqueeze(-1)
    selected = force * active
    wrench = torch.cat((selected, torch.linalg.cross(points.to(force) - center, selected)), -1)
    output = force.new_zeros(force.shape[:2] + (count, 6))
    return output.scatter_add(2, index[..., None].expand(-1, -1, -1, 6), wrench)
