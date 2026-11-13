"""Generalized contact geometry for ARCTIC's two-link objects."""
from __future__ import annotations

import numpy as np

from .parameters import NormalizedPartPhysics
from .contact_constraints import point_force_generalized_map


def articulated_contact_geometry(
    part_to_world: np.ndarray,
    physics: list[NormalizedPartPhysics],
) -> dict:
    """Return geometry needed to map contacts into seven generalized axes."""
    if part_to_world.ndim != 4 or part_to_world.shape[1:] != (2, 4, 4):
        raise ValueError("part_to_world must have shape (T,2,4,4)")
    if len(physics) != 2 or physics[1].joint_type != "revolute":
        raise ValueError("expected fixed base plus one revolute child")
    rotations = part_to_world[:, :, :3, :3]
    translations = part_to_world[:, :, :3, 3]
    local_com = np.stack([x.com_part for x in physics])
    com = translations + np.einsum("tpij,pj->tpi", rotations, local_com)
    masses = np.array([x.mass for x in physics])
    system_com = np.einsum("p,tpi->ti", masses, com) / masses.sum()
    base_world = part_to_world[:, 0]
    pivot = (base_world[:, :3, :3] @ physics[1].joint_origin_parent[:, None]).squeeze(-1)
    pivot += base_world[:, :3, 3]
    axis = np.einsum("tij,j->ti", base_world[:, :3, :3], physics[1].joint_axis_parent)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    return {
        "system_com_world": system_com,
        "part_com_world": com,
        "joint_axis_world": axis,
        "joint_pivot_world": pivot,
    }


def contact_generalized_map(
    point_world: np.ndarray, part_id: int, system_com_world: np.ndarray,
    joint_axis_world: np.ndarray, joint_pivot_world: np.ndarray,
) -> np.ndarray:
    """Map one 3D contact force to [F, torque@system-COM, joint torque]."""
    if part_id not in (0, 1):
        raise ValueError("part_id must be 0 or 1")
    return point_force_generalized_map(
        point_world, system_com_world,
        joint_axes_world=joint_axis_world,
        joint_pivots_world=joint_pivot_world,
        joint_active=np.array([part_id == 1]),
    )
