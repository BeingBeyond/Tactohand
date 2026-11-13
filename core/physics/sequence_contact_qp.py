"""Sequence-level strict contact-wrench allocation with proximity priority."""
from __future__ import annotations

from collections import OrderedDict
import cvxpy as cp
import hashlib
import numpy as np
import osqp
from scipy import sparse
import time

from .contact_constraints import point_force_generalized_map, support_force_world
from .contact_constraints import normalize_candidate_normals, transported_tangent_basis


# OSQP factorization workspaces are process-local and bounded. Reuse is only
# attempted when the exact CSC sparsity pattern and solver settings match; all
# numerical matrix values and bounds are updated before every solve.
_OSQP_WORKSPACE_CACHE_MAX = 16
_OSQP_WORKSPACE_CACHE: OrderedDict[bytes, tuple[osqp.OSQP, int, int]] = OrderedDict()


def _workspace_signature(P: sparse.csc_matrix, A: sparse.csc_matrix,
                         settings: tuple[object, ...]) -> bytes:
    digest = hashlib.blake2b(digest_size=20)
    for value in (P.shape, A.shape, settings):
        digest.update(repr(value).encode("utf-8"))
    for array in (P.indptr, P.indices, A.indptr, A.indices):
        digest.update(np.asarray(array, dtype=np.int64).tobytes())
    return digest.digest()


def _solve_direct_osqp(
    target: np.ndarray,
    com: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    priority: np.ndarray,
    active_ids: list[np.ndarray],
    lookups: list[dict[int, int]],
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    *,
    mu: float,
    friction_sides: int,
    force_regularizer: float,
    temporal_regularizer: float,
    priority_epsilon: float,
    table_direction: np.ndarray,
    table_support_point: np.ndarray,
    table_contact_active: np.ndarray,
    table_support_regularizer: float,
    table_temporal_regularizer: float,
    torque_scale_nm: float,
    eps_abs: float,
    eps_rel: float,
    time_limit_s: float | None,
    initial_vertex_force: np.ndarray | None,
    initial_table_force: np.ndarray | None,
) -> dict:
    """Assemble the strict sequence QP directly in OSQP standard form."""
    begin = time.perf_counter()
    T, V = priority.shape
    force_offsets = np.zeros(T + 1, dtype=np.int64)
    for t, ids in enumerate(active_ids):
        force_offsets[t + 1] = force_offsets[t] + 3 * len(ids)
    force_size = int(force_offsets[-1])
    support_offset = force_size
    base_variable_count = force_size + T

    # Keep the canonical QP sparse: affine smoothness residuals are lifted into
    # auxiliary z variables with z=D x and diagonal quadratic cost on z.
    base_diagonal = np.zeros(base_variable_count, dtype=np.float64)
    base_diagonal[support_offset:support_offset + T] = (
        2.0 * table_support_regularizer / max(T, 1))
    total_active = max(sum(map(len, active_ids)), 1)
    quadratic_groups: list[tuple[sparse.csr_matrix, float]] = []

    def add_least_squares(
        row_ids: list[int], col_ids: list[int], values: list[float],
        row_count: int, coefficient: float,
    ) -> None:
        if row_count == 0 or coefficient == 0:
            return
        D = sparse.coo_matrix(
            (values, (row_ids, col_ids)), shape=(row_count, base_variable_count)).tocsr()
        quadratic_groups.append((D, coefficient))

    # Match CVXPY's QP canonicalization exactly: even the diagonal weighted
    # force magnitude is represented by z=Wf instead of being placed directly
    # on the original force variables. This scaling materially improves ADMM.
    rows = list(range(force_size)); cols = list(range(force_size)); values = []
    for t, ids in enumerate(active_ids):
        values.extend(np.repeat(
            1.0 / np.sqrt(priority[t, ids] ** 2 + priority_epsilon), 3).tolist())
    add_least_squares(rows, cols, values, force_size, force_regularizer / total_active)

    # First temporal force differences.
    rows = []; cols = []; values = []
    temporal_count = 0; scalar_row = 0
    for t in range(T - 1):
        union = np.union1d(active_ids[t], active_ids[t + 1])
        for vertex in union:
            left = lookups[t].get(int(vertex)); right = lookups[t + 1].get(int(vertex))
            for component in range(3):
                if left is not None:
                    rows.append(scalar_row); cols.append(int(force_offsets[t] + 3 * left + component)); values.append(-1.0)
                if right is not None:
                    rows.append(scalar_row); cols.append(int(force_offsets[t + 1] + 3 * right + component)); values.append(1.0)
                scalar_row += 1
        temporal_count += len(union)
    add_least_squares(rows, cols, values, scalar_row,
                      temporal_regularizer / temporal_count if temporal_count else 0.0)

    # Table support first differences.
    rows = []; cols = []; values = []
    for t in range(T - 1):
        rows.extend((t, t)); cols.extend((support_offset + t, support_offset + t + 1)); values.extend((-1.0, 1.0))
    add_least_squares(rows, cols, values, max(T - 1, 0),
                      table_temporal_regularizer / max(T - 1, 1))
    auxiliary_count = sum(D.shape[0] for D, _ in quadratic_groups)
    variable_count = base_variable_count + auxiliary_count
    diagonal = np.concatenate((
        base_diagonal,
        *[np.full(D.shape[0], 2.0 * coefficient, dtype=np.float64)
          for D, coefficient in quadratic_groups],
    ))
    P = sparse.diags(diagonal, format="csc")
    q = np.zeros(variable_count, dtype=np.float64)

    # A/l/u: unilateral support bounds, strict wrench equalities, and friction facets.
    a_rows: list[int] = []; a_cols: list[int] = []; a_values: list[float] = []
    lower: list[float] = []; upper: list[float] = []

    def append_row(columns: list[int], row_values: list[float], lo: float, hi: float) -> None:
        row = len(lower)
        a_rows.extend([row] * len(columns)); a_cols.extend(columns); a_values.extend(row_values)
        lower.append(lo); upper.append(hi)

    for t in range(T):
        upper_support = np.inf if table_contact_active[t] else 0.0
        append_row([support_offset + t], [1.0], 0.0, upper_support)

    for t, ids in enumerate(active_ids):
        for component in range(3):
            columns = [int(force_offsets[t] + 3 * local + component) for local in range(len(ids))]
            row_values = [1.0] * len(ids)
            columns.append(support_offset + t); row_values.append(float(table_direction[t, component]))
            append_row(columns, row_values, float(target[t, component]), float(target[t, component]))
        for component in range(3):
            columns = []; row_values = []
            for local, vertex in enumerate(ids):
                mapping = point_force_generalized_map(points[t, vertex], com[t])
                for force_component in range(3):
                    value = float(mapping[3 + component, force_component] / torque_scale_nm)
                    if value:
                        columns.append(int(force_offsets[t] + 3 * local + force_component)); row_values.append(value)
            table_map = point_force_generalized_map(table_support_point[t], com[t])
            table_value = float(
                table_map[3 + component] @ table_direction[t] / torque_scale_nm)
            if table_value:
                columns.append(support_offset + t)
                row_values.append(table_value)
            rhs = float(target[t, 3 + component] / torque_scale_nm)
            append_row(columns, row_values, rhs, rhs)

        facet_scale = mu * np.cos(np.pi / friction_sides)
        for local, vertex in enumerate(ids):
            base = int(force_offsets[t] + 3 * local)
            append_row([base, base + 1, base + 2], (-normals[t, vertex]).tolist(), -np.inf, 0.0)
            for side in range(friction_sides):
                angle = (side + 0.5) * 2 * np.pi / friction_sides
                facet = (np.cos(angle) * tangent1[t, vertex]
                         + np.sin(angle) * tangent2[t, vertex])
                vector = facet - facet_scale * normals[t, vertex]
                append_row([base, base + 1, base + 2], vector.tolist(), -np.inf, 0.0)

    auxiliary_offset = base_variable_count
    for D, _ in quadratic_groups:
        for row_id in range(D.shape[0]):
            begin_id, end_id = D.indptr[row_id], D.indptr[row_id + 1]
            columns = D.indices[begin_id:end_id].astype(np.int64).tolist()
            row_values = D.data[begin_id:end_id].tolist()
            columns.append(auxiliary_offset + row_id); row_values.append(-1.0)
            append_row(columns, row_values, 0.0, 0.0)
        auxiliary_offset += D.shape[0]

    A = sparse.coo_matrix(
        (a_values, (a_rows, a_cols)), shape=(len(lower), variable_count)).tocsc()
    A.eliminate_zeros()
    assembly_end = time.perf_counter()
    setup_kwargs = dict(
        eps_abs=eps_abs, eps_rel=eps_rel, max_iter=200000,
        polishing=False, verbose=False,
    )
    if time_limit_s is not None:
        setup_kwargs["time_limit"] = float(time_limit_s)
    lower_array = np.asarray(lower)
    upper_array = np.asarray(upper)
    signature = _workspace_signature(
        P, A, (eps_abs, eps_rel, time_limit_s, setup_kwargs["max_iter"],
               setup_kwargs["polishing"]),
    )
    cached = _OSQP_WORKSPACE_CACHE.pop(signature, None)
    workspace_reused = cached is not None
    if cached is None:
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A, l=lower_array, u=upper_array,
                     **setup_kwargs)
    else:
        solver, cached_variables, cached_constraints = cached
        if cached_variables != P.shape[0] or cached_constraints != A.shape[0]:
            raise RuntimeError("invalid cached OSQP workspace dimensions")
        solver.update(Px=P.data, Ax=A.data, q=q, l=lower_array, u=upper_array)
    _OSQP_WORKSPACE_CACHE[signature] = (solver, P.shape[0], A.shape[0])
    while len(_OSQP_WORKSPACE_CACHE) > _OSQP_WORKSPACE_CACHE_MAX:
        _OSQP_WORKSPACE_CACHE.popitem(last=False)
    setup_end = time.perf_counter()
    warm_start_used = initial_vertex_force is not None or initial_table_force is not None
    if warm_start_used:
        base_initial = np.zeros(base_variable_count, dtype=np.float64)
        if initial_vertex_force is not None:
            for t, ids in enumerate(active_ids):
                if len(ids):
                    base_initial[force_offsets[t]:force_offsets[t + 1]] = (
                        initial_vertex_force[t, ids].reshape(-1))
        if initial_table_force is not None:
            base_initial[support_offset:support_offset + T] = np.einsum(
                "ti,ti->t", initial_table_force, table_direction)
        auxiliary_initial = [D @ base_initial for D, _ in quadratic_groups]
        solver.warm_start(x=np.concatenate((base_initial, *auxiliary_initial)),
                          y=np.zeros(A.shape[0], dtype=np.float64))
    elif workspace_reused:
        # Reusing setup must not silently change a cold solve into an implicit
        # previous-problem warm start.
        solver.warm_start(x=np.zeros(P.shape[0], dtype=np.float64),
                          y=np.zeros(A.shape[0], dtype=np.float64))
    result = solver.solve()
    solve_end = time.perf_counter()
    status = str(result.info.status).lower()
    if status not in ("solved", "solved inaccurate"):
        raise RuntimeError(f"sequence contact QP failed: {result.info.status}")
    return {
        "solution": np.asarray(result.x, dtype=np.float64),
        "force_offsets": force_offsets,
        "support_offset": support_offset,
        "status": "optimal" if status == "solved" else "optimal_inaccurate",
        "objective": float(result.info.obj_val),
        "timing_s": {
            "build": float(assembly_end - begin),
            "canonicalize": 0.0,
            "setup": float(setup_end - assembly_end),
            "solve": float(solve_end - setup_end),
            "total": float(solve_end - begin),
        },
        "osqp_iterations": int(result.info.iter),
        "osqp_rho_updates": int(result.info.rho_updates),
        "osqp_rho_estimate": float(result.info.rho_estimate),
        "warm_start_used": bool(warm_start_used),
        "workspace_reused": bool(workspace_reused),
        "matrix_stats": {"variables": int(P.shape[0]), "constraints": int(A.shape[0]),
                         "P_nnz": int(P.nnz), "A_nnz": int(A.nnz)},
    }


def solve_sequence_contact_qp(
    target_contact_wrench: np.ndarray,
    com_world: np.ndarray,
    vertex_points_world: np.ndarray,
    vertex_normals_world: np.ndarray,
    priority_score: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    mu: float = 1.2,
    friction_sides: int = 6,
    force_regularizer: float = 1.0,
    temporal_regularizer: float = 1.0,
    priority_epsilon: float = 1.0e-4,
    table_normal: np.ndarray,
    table_contact_active: np.ndarray | None = None,
    table_support_point_world: np.ndarray | None = None,
    table_support_regularizer: float = 1.0e-4,
    table_temporal_regularizer: float = 1.0,
    torque_scale_nm: float = 0.05,
    allow_wrench_slack: bool = False,
    wrench_slack_regularizer: float = 100.0,
    force_slack_scale_n: float = 1.0,
    torque_slack_scale_nm: float = 0.05,
    max_force_slack_n: float = 0.1,
    max_torque_slack_nm: float = 0.005,
    cumulative_force_slack_regularizer: float = 0.0,
    cumulative_torque_slack_regularizer: float = 0.0,
    integrated_force_slack_regularizer: float = 0.0,
    eps_abs: float = 1.0e-6,
    eps_rel: float = 1.0e-6,
    solver_backend: str = "auto",
    time_limit_s: float | None = None,
    initial_vertex_force_world: np.ndarray | None = None,
    initial_table_force_world: np.ndarray | None = None,
    prepared_contact_normals_world: np.ndarray | None = None,
    prepared_tangent_basis_world: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Project zero force onto exact sequence wrench and contact constraints."""
    build_begin = time.perf_counter()
    target = np.asarray(target_contact_wrench, dtype=np.float64)
    com = np.asarray(com_world, dtype=np.float64)
    points = np.asarray(vertex_points_world, dtype=np.float64)
    normals = np.asarray(vertex_normals_world, dtype=np.float64)
    priority = np.asarray(priority_score, dtype=np.float64)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if priority.ndim != 2:
        raise ValueError("priority_score must have shape (T,V)")
    T, V = priority.shape
    if table_contact_active is None:
        table_active = np.ones(T, dtype=bool)
    else:
        table_active = np.asarray(table_contact_active, dtype=bool)
        if table_active.shape != (T,):
            raise ValueError("table_contact_active must have shape (T,)")
    if (target.shape != (T, 6) or com.shape != (T, 3) or
            points.shape != (T, V, 3) or normals.shape != (T, V, 3) or
            candidate.shape != (T, V)):
        raise ValueError("inconsistent sequence contact QP input shapes")
    initial_force = (None if initial_vertex_force_world is None else
                     np.asarray(initial_vertex_force_world, dtype=np.float64))
    initial_table = (None if initial_table_force_world is None else
                     np.asarray(initial_table_force_world, dtype=np.float64))
    if initial_force is not None and (
            initial_force.shape != (T, V, 3) or not np.isfinite(initial_force).all()):
        raise ValueError("initial_vertex_force_world must have finite shape (T,V,3)")
    if initial_table is not None and (
            initial_table.shape != (T, 3) or not np.isfinite(initial_table).all()):
        raise ValueError("initial_table_force_world must have finite shape (T,3)")
    if not np.isfinite(priority).all() or np.any((priority < 0) | (priority > 2)):
        raise ValueError("priority_score must be finite and lie in [0,2]")
    if np.any(candidate & (priority <= 0)):
        raise ValueError("candidate vertices must have positive priority")
    if np.any(candidate & ~np.isfinite(points).all(axis=-1)):
        raise ValueError("candidate vertex positions must be finite")
    if mu < 0 or friction_sides < 3 or priority_epsilon <= 0:
        raise ValueError("invalid cone, priority, or temporal parameter")
    if min(force_regularizer, temporal_regularizer, table_support_regularizer,
           table_temporal_regularizer,
           wrench_slack_regularizer, cumulative_force_slack_regularizer,
           cumulative_torque_slack_regularizer,
           integrated_force_slack_regularizer) < 0:
        raise ValueError("regularizers must be nonnegative")
    if min(force_slack_scale_n, torque_slack_scale_nm) <= 0:
        raise ValueError("wrench slack scales must be positive")
    if min(max_force_slack_n, max_torque_slack_nm) < 0:
        raise ValueError("wrench slack bounds must be nonnegative")
    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive when provided")
    table_direction = np.asarray(table_normal, dtype=np.float64)
    if table_direction.shape == (3,):
        table_direction = np.broadcast_to(table_direction, (T, 3)).copy()
    if (table_direction.shape != (T, 3) or not np.isfinite(table_direction).all() or
            np.any(np.linalg.norm(table_direction, axis=1) < 1e-12)):
        raise ValueError("table_normal must have shape (3,) or (T,3) and be finite")
    table_direction /= np.linalg.norm(table_direction, axis=1, keepdims=True)
    table_point = (com.copy() if table_support_point_world is None
                   else np.asarray(table_support_point_world, dtype=np.float64))
    if table_point.shape != (T, 3) or not np.isfinite(table_point).all():
        raise ValueError("table_support_point_world must have shape (T,3) and be finite")
    if prepared_contact_normals_world is None:
        normals = normalize_candidate_normals(normals, candidate)
    else:
        prepared_normals = np.asarray(prepared_contact_normals_world, np.float64)
        if prepared_normals.shape != normals.shape or not np.isfinite(
                prepared_normals[candidate]).all():
            raise ValueError("invalid prepared_contact_normals_world")
        normals = prepared_normals
    if prepared_tangent_basis_world is None:
        tangent1, tangent2 = transported_tangent_basis(normals)
    else:
        tangent1 = np.asarray(prepared_tangent_basis_world[0], np.float64)
        tangent2 = np.asarray(prepared_tangent_basis_world[1], np.float64)
        if tangent1.shape != normals.shape or tangent2.shape != normals.shape:
            raise ValueError("prepared tangent bases must match contact normals")
    if solver_backend not in ("auto", "direct", "cvxpy"):
        raise ValueError("solver_backend must be auto, direct, or cvxpy")
    if solver_backend == "direct" and allow_wrench_slack:
        raise ValueError("direct OSQP currently supports strict wrench mode only")
    selected_backend = ("cvxpy" if allow_wrench_slack else "direct"
                        if solver_backend == "auto" else solver_backend)

    active_ids = [np.flatnonzero(candidate[frame]) for frame in range(T)]
    lookups = [{int(vertex): i for i, vertex in enumerate(ids)} for ids in active_ids]
    if selected_backend == "direct":
        direct = _solve_direct_osqp(
            target, com, points, normals, priority, active_ids, lookups,
            tangent1, tangent2, mu=mu, friction_sides=friction_sides,
            force_regularizer=force_regularizer, temporal_regularizer=temporal_regularizer,
            priority_epsilon=priority_epsilon,
            table_direction=table_direction,
            table_support_point=table_point,
            table_contact_active=table_active,
            table_support_regularizer=table_support_regularizer,
            table_temporal_regularizer=table_temporal_regularizer,
            torque_scale_nm=torque_scale_nm,
            eps_abs=eps_abs, eps_rel=eps_rel,
            time_limit_s=time_limit_s,
            initial_vertex_force=initial_force,
            initial_table_force=initial_table,
        )
        vertex_force = np.zeros((T, V, 3), dtype=np.float64)
        solution = direct["solution"]
        offsets = direct["force_offsets"]
        for t, ids in enumerate(active_ids):
            if len(ids):
                vertex_force[t, ids] = solution[offsets[t]:offsets[t + 1]].reshape(-1, 3)
        support_n = solution[direct["support_offset"]:direct["support_offset"] + T]
        support_n[~table_active] = 0.0
        table_force = support_n[:, None] * table_direction
        table_force[~table_active] = 0.0
        fitted = np.zeros((T, 6), np.float64)
        for t, ids in enumerate(active_ids):
            for vertex in ids:
                fitted[t] += point_force_generalized_map(points[t, vertex], com[t]) @ vertex_force[t, vertex]
            fitted[t] += point_force_generalized_map(table_point[t], com[t]) @ table_force[t]
        return {
            "vertex_force_world": vertex_force,
            "table_support_n": support_n,
            "table_force_world": table_force,
            "environment_force_world": table_force,
            "fitted_contact_wrench": fitted,
            "target_contact_wrench": target,
            "force_slack_world": np.zeros((T, 3), dtype=np.float64),
            "torque_slack_world": np.zeros((T, 3), dtype=np.float64),
            "allow_wrench_slack": False,
            "force_residual_norm": np.linalg.norm(fitted[:, :3] - target[:, :3], axis=1),
            "torque_residual_norm": np.linalg.norm(fitted[:, 3:] - target[:, 3:], axis=1),
            "priority_score": priority,
            "candidate_mask": candidate,
            "table_contact_active": table_active,
            "active_count": candidate.sum(1),
            "friction_sides": int(friction_sides),
            "is_dcp": True,
            "status": direct["status"],
            "objective": direct["objective"],
            "timing_s": direct["timing_s"],
            "solver_backend": "direct",
            "osqp_iterations": direct["osqp_iterations"],
            "osqp_rho_updates": direct["osqp_rho_updates"],
            "osqp_rho_estimate": direct["osqp_rho_estimate"],
            "warm_start_used": direct["warm_start_used"],
            "workspace_reused": direct["workspace_reused"],
            "matrix_stats": direct["matrix_stats"],
        }
    forces = [cp.Variable((len(ids), 3), name=f"contact_force_{t}") if len(ids) else None
              for t, ids in enumerate(active_ids)]
    support_n = cp.Variable(T, name="table_support_n")
    force_slack = cp.Variable((T, 3), name="force_slack") if allow_wrench_slack else None
    torque_slack = cp.Variable((T, 3), name="torque_slack") if allow_wrench_slack else None
    constraints: list = []
    magnitude: list = []
    temporal: list = []
    temporal_count = 0
    table_temporal: list = []

    for t in range(T):
        constraints.append(support_n[t] >= 0.)
        if not table_active[t]:
            constraints.append(support_n[t] == 0.)

    for t, ids in enumerate(active_ids):
        variable = forces[t]
        if variable is None:
            flat_force = None
            hand_force = cp.Constant(np.zeros(3))
            torque = cp.Constant(np.zeros(3))
        else:
            flat_force = cp.reshape(variable, (3 * len(ids),), order="C")
            grasp = np.concatenate([
                point_force_generalized_map(points[t, vertex], com[t])
                for vertex in ids
            ], axis=1)
            hand_force = grasp[:3] @ flat_force
            torque = grasp[3:] @ flat_force
        table_torque = cp.multiply(
            support_n[t], np.cross(table_point[t] - com[t], table_direction[t]))
        if allow_wrench_slack:
            constraints.extend((
                hand_force + support_n[t] * table_direction[t]
                    + force_slack[t] == target[t, :3],
                torque + table_torque + torque_slack[t] == target[t, 3:],
                force_slack[t] <= max_force_slack_n,
                force_slack[t] >= -max_force_slack_n,
                torque_slack[t] <= max_torque_slack_nm,
                torque_slack[t] >= -max_torque_slack_nm,
            ))
        else:
            constraints.extend((
                hand_force + support_n[t] * table_direction[t] == target[t, :3],
                (torque + table_torque) / torque_scale_nm
                    == target[t, 3:] / torque_scale_nm,
            ))
        if variable is None:
            continue
        cone_blocks = []
        facet_scale = mu * np.cos(np.pi / friction_sides)
        for vertex in ids:
            normal = normals[t, vertex]
            block = [-normal]
            for side in range(friction_sides):
                angle = (side + .5) * 2 * np.pi / friction_sides
                facet = (np.cos(angle) * tangent1[t, vertex] +
                         np.sin(angle) * tangent2[t, vertex])
                block.append(facet - facet_scale * normal)
            cone_blocks.append(np.stack(block))
        cone = sparse.block_diag(cone_blocks, format="csr")
        constraints.append(cone @ flat_force <= 0.)
        magnitude_weight = np.repeat(
            1. / np.sqrt(priority[t, ids] ** 2 + priority_epsilon), 3)
        magnitude.append(cp.sum_squares(cp.multiply(magnitude_weight, flat_force)))
    for t in range(T - 1):
        union = np.union1d(active_ids[t], active_ids[t + 1])
        if not len(union):
            table_temporal.append(cp.square(support_n[t + 1] - support_n[t]))
            continue
        rows_left, cols_left, rows_right, cols_right, weights = [], [], [], [], []
        for union_id, vertex in enumerate(union):
            left_id, right_id = lookups[t].get(int(vertex)), lookups[t + 1].get(int(vertex))
            weights.extend((1.0,) * 3)
            for component in range(3):
                row = 3 * union_id + component
                if left_id is not None:
                    rows_left.append(row); cols_left.append(3 * left_id + component)
                if right_id is not None:
                    rows_right.append(row); cols_right.append(3 * right_id + component)
        left_selection = sparse.coo_matrix(
            (np.ones(len(rows_left)), (rows_left, cols_left)),
            shape=(3 * len(union), 3 * len(active_ids[t]))).tocsr()
        right_selection = sparse.coo_matrix(
            (np.ones(len(rows_right)), (rows_right, cols_right)),
            shape=(3 * len(union), 3 * len(active_ids[t + 1]))).tocsr()
        left_value = (np.zeros(3 * len(union)) if forces[t] is None else
                      left_selection @ cp.reshape(forces[t], (3 * len(active_ids[t]),), order="C"))
        right_value = (np.zeros(3 * len(union)) if forces[t + 1] is None else
                       right_selection @ cp.reshape(
                           forces[t + 1], (3 * len(active_ids[t + 1]),), order="C"))
        temporal.append(cp.sum_squares(
            cp.multiply(np.asarray(weights), right_value - left_value)))
        temporal_count += len(union)
        table_temporal.append(cp.square(support_n[t + 1] - support_n[t]))

    objective = force_regularizer * cp.sum(magnitude) / max(sum(map(len, active_ids)), 1)
    objective += table_support_regularizer * cp.sum_squares(support_n) / T
    if temporal:
        objective += temporal_regularizer * cp.sum(temporal) / temporal_count
    if table_temporal:
        objective += table_temporal_regularizer * cp.sum(table_temporal) / len(table_temporal)
    if allow_wrench_slack:
        objective += wrench_slack_regularizer * (
            cp.sum_squares(force_slack / force_slack_scale_n)
            + cp.sum_squares(torque_slack / torque_slack_scale_nm)
        ) / T
        integration = np.tril(np.ones((T, T), dtype=np.float64))
        cumulative_force = integration @ force_slack
        cumulative_torque = integration @ torque_slack
        if cumulative_force_slack_regularizer:
            objective += cumulative_force_slack_regularizer * cp.sum_squares(
                cumulative_force / force_slack_scale_n) / T
        if cumulative_torque_slack_regularizer:
            objective += cumulative_torque_slack_regularizer * cp.sum_squares(
                cumulative_torque / torque_slack_scale_nm) / T
        if integrated_force_slack_regularizer:
            objective += integrated_force_slack_regularizer * cp.sum_squares(
                integration @ cumulative_force / force_slack_scale_n) / T
    problem = cp.Problem(cp.Minimize(objective), constraints)
    if not problem.is_dcp():
        raise RuntimeError("sequence contact problem is not DCP")
    canonicalize_begin = time.perf_counter()
    problem_data, _, _ = problem.get_problem_data(cp.OSQP)
    solve_begin = time.perf_counter()
    solve_kwargs = dict(
        solver=cp.OSQP, eps_abs=eps_abs, eps_rel=eps_rel,
        max_iter=200000, polishing=True,
    )
    if time_limit_s is not None:
        solve_kwargs["time_limit"] = float(time_limit_s)
    problem.solve(**solve_kwargs)
    solve_end = time.perf_counter()
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"sequence contact QP failed: {problem.status}")

    vertex_force = np.zeros((T, V, 3), dtype=np.float64)
    for t, ids in enumerate(active_ids):
        if len(ids):
            vertex_force[t, ids] = np.asarray(forces[t].value)
    support_value_n = np.asarray(support_n.value, dtype=np.float64)
    support_value_n[~table_active] = 0.0
    table_force = support_value_n[:, None] * table_direction
    table_force[~table_active] = 0.0
    force_slack_value = (np.asarray(force_slack.value, dtype=np.float64)
                         if allow_wrench_slack else np.zeros((T, 3), dtype=np.float64))
    torque_slack_value = (np.asarray(torque_slack.value, dtype=np.float64)
                          if allow_wrench_slack else np.zeros((T, 3), dtype=np.float64))
    fitted = np.zeros((T, 6), np.float64)
    for t, ids in enumerate(active_ids):
        for vertex in ids:
            fitted[t] += point_force_generalized_map(points[t, vertex], com[t]) @ vertex_force[t, vertex]
        fitted[t] += point_force_generalized_map(table_point[t], com[t]) @ table_force[t]
    return {
        "vertex_force_world": vertex_force,
        "table_support_n": support_value_n,
        "table_force_world": table_force,
        "environment_force_world": table_force,
        "fitted_contact_wrench": fitted,
        "target_contact_wrench": target,
        "force_slack_world": force_slack_value,
        "torque_slack_world": torque_slack_value,
        "allow_wrench_slack": bool(allow_wrench_slack),
        "force_residual_norm": np.linalg.norm(fitted[:, :3] - target[:, :3], axis=1),
        "torque_residual_norm": np.linalg.norm(fitted[:, 3:] - target[:, 3:], axis=1),
        "priority_score": priority,
        "candidate_mask": candidate,
        "table_contact_active": table_active,
        "active_count": candidate.sum(1),
        "friction_sides": int(friction_sides),
        "is_dcp": bool(problem.is_dcp()),
        "status": str(problem.status),
        "objective": float(problem.value),
        "solver_backend": "cvxpy",
        "osqp_iterations": int(problem.solver_stats.num_iters),
        "osqp_rho_updates": int(problem.solver_stats.extra_stats.info.rho_updates),
        "osqp_rho_estimate": float(problem.solver_stats.extra_stats.info.rho_estimate),
        "matrix_stats": {
            "variables": int(problem_data["P"].shape[0]),
            "constraints": int(problem_data["A"].shape[0] + problem_data["F"].shape[0]),
            "P_nnz": int(problem_data["P"].nnz),
            "A_nnz": int(problem_data["A"].nnz + problem_data["F"].nnz),
        },
        "timing_s": {
            "build": float(canonicalize_begin - build_begin),
            "canonicalize": float(solve_begin - canonicalize_begin),
            "solve": float(solve_end - solve_begin),
            "total": float(solve_end - build_begin),
        },
    }
