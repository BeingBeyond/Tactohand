"""Shared contact-cone geometry for every unified wrench topology."""
from __future__ import annotations

import numpy as np


def cross_matrix(offset: np.ndarray) -> np.ndarray:
    """Matrix ``C`` satisfying ``C @ force == offset x force``."""
    x, y, z = np.asarray(offset, np.float64)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def point_force_generalized_map(
    point_world: np.ndarray,
    reference_world: np.ndarray,
    *,
    joint_axes_world: np.ndarray | None = None,
    joint_pivots_world: np.ndarray | None = None,
    joint_active: np.ndarray | None = None,
) -> np.ndarray:
    """Map a point force to force, reference torque, and optional joint torques."""
    point = np.asarray(point_world, np.float64)
    reference = np.asarray(reference_world, np.float64)
    if point.shape != (3,) or reference.shape != (3,):
        raise ValueError("point_world and reference_world must have shape (3,)")
    if joint_axes_world is None:
        joint_count = 0
        axes = pivots = np.empty((0, 3), np.float64)
    else:
        axes = np.asarray(joint_axes_world, np.float64)
        pivots = np.asarray(joint_pivots_world, np.float64)
        if axes.ndim == 1:
            axes = axes[None]
        if pivots.ndim == 1:
            pivots = pivots[None]
        if axes.shape != pivots.shape or axes.ndim != 2 or axes.shape[1] != 3:
            raise ValueError("joint axes and pivots must have matching shape (J,3)")
        joint_count = len(axes)
    active = (np.ones(joint_count, bool) if joint_active is None
              else np.asarray(joint_active, bool))
    if active.shape != (joint_count,):
        raise ValueError("joint_active must have shape (J,)")
    mapping = np.zeros((6 + joint_count, 3), np.float64)
    mapping[:3] = np.eye(3)
    mapping[3:6] = cross_matrix(point - reference)
    for joint in np.flatnonzero(active):
        mapping[6 + joint] = np.cross(axes[joint], point - pivots[joint])
    return mapping


def support_force_world(
    support_n: np.ndarray,
    support_direction_world: np.ndarray,
    *,
    contact_active: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a nonnegative scalar support force in newtons to world vectors."""
    values = np.asarray(support_n, np.float64).copy()
    direction = np.asarray(support_direction_world, np.float64)
    if values.ndim != 1 or direction.shape != (3,):
        raise ValueError("support_n and support direction must have shape (T,) and (3,)")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(values).all() or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("support force and direction must be finite and valid")
    if contact_active is not None:
        active = np.asarray(contact_active, bool)
        if active.shape != values.shape:
            raise ValueError("contact_active must have shape (T,)")
        values[~active] = 0.0
    return values[:, None] * (direction / norm)


def normalize_candidate_normals(normals: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Validate and normalize contact normals without changing inactive rows."""
    values = np.asarray(normals, np.float64).copy()
    mask = np.asarray(candidate, bool)
    finite = np.isfinite(values).all(axis=-1)
    if np.any(mask & ~finite):
        raise ValueError("candidate vertex normals must be finite")
    # Tangent transport iterates over the complete time/vertex grid.  Replace
    # inactive invalid normals with a deterministic unit vector so missing hand
    # geometry cannot contaminate a later active candidate through time.
    values[~mask & ~finite] = [0.0, 0.0, 1.0]
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(mask & (lengths[..., 0] < 1e-8)):
        raise ValueError("candidate vertex normals must be nonzero")
    return values / np.maximum(lengths, 1e-12)


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal tangent basis for a unit contact normal."""
    normal = np.asarray(normal, np.float64)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    first = axis - normal * (normal @ axis)
    first /= max(np.linalg.norm(first), 1e-12)
    return first, np.cross(normal, first)


def independent_tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized deterministic basis equivalent to ``tangent_basis`` per row."""
    values = np.asarray(normals, np.float64)
    axes = np.eye(3, dtype=np.float64)[np.argmin(np.abs(values), axis=-1)]
    first = axes - values * np.sum(values * axes, axis=-1, keepdims=True)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    return first, np.cross(values, first)


def friction_facets(normal: np.ndarray, mu: float, sides: int) -> np.ndarray:
    """Rows ``a`` for the shared polyhedral constraint ``a @ force <= 0``."""
    if sides < 3 or mu < 0:
        raise ValueError("expected mu >= 0 and at least three friction sides")
    first, second = tangent_basis(normal)
    angles = (np.arange(sides, dtype=np.float64) + .5) * (2 * np.pi / sides)
    tangents = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    return tangents - (mu * np.cos(np.pi / sides)) * np.asarray(normal, np.float64)


def transported_tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time-continuous tangent orientation used by sequence-level rigid QPs."""
    values = np.asarray(normals, np.float64)
    first = np.empty_like(values)
    for vertex in range(values.shape[1]):
        first[0, vertex] = tangent_basis(values[0, vertex])[0]
        for frame in range(1, values.shape[0]):
            normal = values[frame, vertex]
            tangent = first[frame - 1, vertex] - normal * (normal @ first[frame - 1, vertex])
            if np.linalg.norm(tangent) < 1e-8:
                tangent = tangent_basis(normal)[0]
            first[frame, vertex] = tangent / max(np.linalg.norm(tangent), 1e-12)
    return first, np.cross(values, first)
