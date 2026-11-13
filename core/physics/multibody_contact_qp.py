"""Sparse sequence contact allocation for an arbitrary HOI4D kinematic tree."""
from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse

from .contact_constraints import friction_facets, normalize_candidate_normals
from .multibody import contact_generalized_map


def solve_multibody_contact_qp(
    *, target: np.ndarray, system_com: np.ndarray,
    joint_parts: np.ndarray, physics: list, joint_axis: np.ndarray,
    joint_pivot: np.ndarray, points: np.ndarray, normals: np.ndarray,
    priority: np.ndarray, candidate: np.ndarray, contact_part: np.ndarray,
    environment_active: np.ndarray, support_direction: np.ndarray,
    force_regularizer: float = 1.0, temporal_regularizer: float = 1.0,
    internal_regularizer: float = 100.0, mu: float = 1.2,
    friction_sides: int = 6, eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
) -> dict:
    target = np.asarray(target, np.float64); T, D = target.shape
    J = D - 6; expected = (T, 2, 778)
    if points.shape != (*expected, 3) or normals.shape != (*expected, 3):
        raise ValueError("invalid multibody contact geometry")
    if candidate.shape != expected or contact_part.shape != expected:
        raise ValueError("invalid multibody contact masks")
    if len(joint_parts) != J or joint_axis.shape != (T, J, 3):
        raise ValueError("generalized target and joint topology disagree")
    normals = normalize_candidate_normals(normals, candidate)
    active: list[list[tuple[int, int, int]]] = []
    for frame in range(T):
        ids = np.argwhere(candidate[frame] & (contact_part[frame] >= 0))
        active.append([(int(hand), int(vertex), int(contact_part[frame, hand, vertex]))
                       for hand, vertex in ids])
    offsets = np.zeros(T + 1, np.int64)
    for frame, values in enumerate(active):
        offsets[frame + 1] = offsets[frame] + 3 * len(values)
    support_offset = int(offsets[-1]); internal_offset = support_offset + T
    nvar = internal_offset + T * J
    p_row: list[int] = []; p_col: list[int] = []; p_val: list[float] = []

    def quadratic(left: int, right: int, value: float) -> None:
        if left > right: left, right = right, left
        p_row.append(left); p_col.append(right); p_val.append(value)

    lookup: list[dict[int, int]] = []
    total = max(sum(map(len, active)), 1)
    for frame, values in enumerate(active):
        current = {}
        for local, (hand, vertex, part) in enumerate(values):
            identity = part * 1556 + hand * 778 + vertex
            current[identity] = local
            weight = 1.0 / (float(priority[frame, hand, vertex]) ** 2 + 1e-4)
            for component in range(3):
                index = int(offsets[frame] + 3 * local + component)
                quadratic(index, index, 2 * force_regularizer * weight / total)
        lookup.append(current)
    pairs = max(sum(len(set(lookup[t]) | set(lookup[t + 1])) for t in range(T - 1)), 1)
    for frame in range(T - 1):
        for identity in set(lookup[frame]) | set(lookup[frame + 1]):
            for component in range(3):
                terms = []
                if identity in lookup[frame]:
                    terms.append((int(offsets[frame] + 3 * lookup[frame][identity] + component), -1.0))
                if identity in lookup[frame + 1]:
                    terms.append((int(offsets[frame + 1] + 3 * lookup[frame + 1][identity] + component), 1.0))
                for i, (left, a) in enumerate(terms):
                    for right, b in terms[i:]:
                        quadratic(left, right, 2 * temporal_regularizer * a * b / pairs)
    for frame in range(T):
        quadratic(support_offset + frame, support_offset + frame, 2e-4 / T)
        for joint in range(J):
            index = internal_offset + frame * J + joint
            quadratic(index, index, 2 * internal_regularizer / max(T * J, 1))
    P = sparse.coo_matrix((p_val, (p_row, p_col)), shape=(nvar, nvar)).tocsc()
    P.sum_duplicates()
    rows: list[int] = []; cols: list[int] = []; values: list[float] = []
    lower: list[float] = []; upper: list[float] = []

    def constraint(columns, coefficients, lo, hi) -> None:
        row = len(lower); rows.extend([row] * len(columns)); cols.extend(columns)
        values.extend(coefficients); lower.append(lo); upper.append(hi)

    scales = np.concatenate((np.ones(3), np.full(3, .1), np.full(J, .1)))
    for frame, contacts in enumerate(active):
        support = support_offset + frame
        constraint([support], [1.0], 0.0, np.inf if environment_active[frame] else 0.0)
        maps = [contact_generalized_map(
            points[frame, hand, vertex], part, system_com[frame], joint_parts,
            physics, joint_axis[frame], joint_pivot[frame],
        ) for hand, vertex, part in contacts]
        support_map = np.zeros(D); support_map[:3] = support_direction[frame]
        for component in range(D):
            columns = []; coefficients = []
            for local, mapping in enumerate(maps):
                for xyz in range(3):
                    value = float(mapping[component, xyz] / scales[component])
                    if value:
                        columns.append(int(offsets[frame] + 3 * local + xyz)); coefficients.append(value)
            value = float(support_map[component] / scales[component])
            if value:
                columns.append(support); coefficients.append(value)
            if component >= 6:
                columns.append(internal_offset + frame * J + component - 6)
                coefficients.append(1.0 / scales[component])
            rhs = float(target[frame, component] / scales[component])
            constraint(columns, coefficients, rhs, rhs)
        for local, (hand, vertex, _) in enumerate(contacts):
            base = int(offsets[frame] + 3 * local)
            columns = [base, base + 1, base + 2]
            normal = normals[frame, hand, vertex]
            constraint(columns, (-normal).tolist(), -np.inf, 0.0)
            for facet in friction_facets(normal, mu, friction_sides):
                constraint(columns, facet.tolist(), -np.inf, 0.0)
    A = sparse.coo_matrix((values, (rows, cols)), shape=(len(lower), nvar)).tocsc()
    solver = osqp.OSQP(); solver.setup(
        P=P, q=np.zeros(nvar), A=A, l=np.asarray(lower), u=np.asarray(upper),
        eps_abs=eps_abs, eps_rel=eps_rel, max_iter=200000, polishing=True,
        verbose=False,
    )
    result = solver.solve()
    if str(result.info.status).lower() not in ("solved", "solved inaccurate"):
        raise RuntimeError(f"multibody contact QP failed: {result.info.status}")
    solution = np.asarray(result.x)
    force = np.zeros((T, 2, 778, 3)); fitted = np.zeros_like(target)
    support_force = np.zeros((T, 3)); internal = solution[internal_offset:].reshape(T, J)
    for frame, contacts in enumerate(active):
        support_force[frame] = max(solution[support_offset + frame], 0.0) * support_direction[frame]
        fitted[frame, :3] += support_force[frame]
        fitted[frame, 6:] += internal[frame]
        for local, (hand, vertex, part) in enumerate(contacts):
            value = solution[offsets[frame] + 3 * local:offsets[frame] + 3 * local + 3]
            force[frame, hand, vertex] = value
            fitted[frame] += contact_generalized_map(
                points[frame, hand, vertex], part, system_com[frame], joint_parts,
                physics, joint_axis[frame], joint_pivot[frame],
            ) @ value
    return {
        "vertex_force_world": force, "environment_force_world": support_force,
        "joint_internal_torque": internal, "fitted_generalized_wrench": fitted,
        "residual": fitted - target, "status": str(result.info.status),
        "iterations": int(result.info.iter), "objective": float(result.info.obj_val),
    }
