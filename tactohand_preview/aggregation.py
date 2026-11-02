"""Differentiable camera-frame contact-force aggregation (SI units)."""
from __future__ import annotations

import torch


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
