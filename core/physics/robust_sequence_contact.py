"""Guaranteed-return strict per-frame allocation with optional window refinement."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .contact_constraints import independent_tangent_basis, normalize_candidate_normals
from .sequence_contact_qp import solve_sequence_contact_qp


LogFunction = Callable[[dict], None]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.pad(np.asarray(mask, np.int8), (1, 1))))
    return [(int(begin), int(end)) for begin, end in edges.reshape(-1, 2)]


def _fitted_wrench(
    points: np.ndarray,
    com: np.ndarray,
    force: np.ndarray,
    table_force: np.ndarray,
    table_support_point: np.ndarray | None = None,
) -> np.ndarray:
    fitted = np.zeros((len(force), 6), np.float64)
    fitted[:, :3] = force.sum(axis=1) + table_force
    fitted[:, 3:] = np.cross(points - com[:, None, :], force).sum(axis=1)
    if table_support_point is not None:
        fitted[:, 3:] += np.cross(np.asarray(table_support_point) - com, table_force)
    return fitted


def _strictly_matches(
    fitted: np.ndarray,
    target: np.ndarray,
    *,
    force_tolerance_n: float,
    torque_tolerance_nm: float,
) -> np.ndarray:
    return (
        np.isfinite(fitted).all(axis=1)
        & (np.linalg.norm(fitted[:, :3] - target[:, :3], axis=1) <= force_tolerance_n)
        & (np.linalg.norm(fitted[:, 3:] - target[:, 3:], axis=1) <= torque_tolerance_nm)
    )


def solve_per_frame_then_refine(
    *,
    wrench: np.ndarray,
    com: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    priority: np.ndarray,
    candidate: np.ndarray,
    table_normal: np.ndarray,
    table_contact_active: np.ndarray,
    table_support_point_world: np.ndarray | None = None,
    active: np.ndarray,
    temporal_regularizer: float,
    window_size: int = 64,
    overlap: int = 16,
    per_frame_time_limit_s: float = 30.0,
    window_time_limit_s: float = 60.0,
    refinement_warm_start: bool = True,
    force_tolerance_n: float = 5.0e-5,
    torque_tolerance_nm: float = 5.0e-6,
    mu: float = 1.2,
    log: LogFunction | None = None,
) -> dict:
    """Solve every active frame first, then improve smoothness opportunistically.

    The per-frame result is authoritative.  A temporal window is accepted only
    when it returns within its budget and independently reconstructs the exact
    target wrench.  Consequently, a difficult refinement can never invalidate
    or block an already feasible frame.
    """
    target = np.asarray(wrench, np.float64)
    T, V = np.asarray(candidate).shape
    active = np.asarray(active, bool)
    if active.shape != (T,):
        raise ValueError("active must have shape (T,)")
    if window_size < 1 or overlap < 0 or overlap >= window_size:
        raise ValueError("require window_size >= 1 and 0 <= overlap < window_size")

    baseline_force = np.zeros((T, V, 3), np.float64)
    baseline_table = np.zeros((T, 3), np.float64)
    valid = ~active
    failures: list[dict] = []
    frame_timings = np.zeros(T, np.float64)

    # Baseline frames are independent QPs. Normalize their complete contact
    # geometry and construct all deterministic per-frame tangent bases in one
    # vectorized pass, then slice the prepared arrays for each strict solve.
    baseline_normals = normalize_candidate_normals(normals, candidate)
    baseline_tangent1, baseline_tangent2 = independent_tangent_basis(baseline_normals)

    for frame in np.flatnonzero(active):
        begin = time.perf_counter()
        if log is not None:
            log({
                "event": "stage2_baseline_frame_start",
                "frame": int(frame),
                "candidates": int(candidate[frame].sum()),
            })
        try:
            result = solve_sequence_contact_qp(
                target[frame:frame + 1], com[frame:frame + 1],
                points[frame:frame + 1], normals[frame:frame + 1],
                priority[frame:frame + 1], candidate[frame:frame + 1],
                mu=mu, friction_sides=6, temporal_regularizer=0.0,
                table_normal=(table_normal[frame:frame + 1]
                              if np.asarray(table_normal).ndim == 2 else table_normal),
                table_contact_active=table_contact_active[frame:frame + 1],
                table_support_point_world=(None if table_support_point_world is None else
                                           table_support_point_world[frame:frame + 1]),
                table_temporal_regularizer=0.0, allow_wrench_slack=False,
                time_limit_s=per_frame_time_limit_s,
                prepared_contact_normals_world=baseline_normals[frame:frame + 1],
                prepared_tangent_basis_world=(
                    baseline_tangent1[frame:frame + 1],
                    baseline_tangent2[frame:frame + 1],
                ),
            )
            baseline_force[frame] = result["vertex_force_world"][0]
            baseline_table[frame] = result["table_force_world"][0]
            fitted = _fitted_wrench(
                points[frame:frame + 1], com[frame:frame + 1],
                baseline_force[frame:frame + 1], baseline_table[frame:frame + 1],
                None if table_support_point_world is None else
                table_support_point_world[frame:frame + 1],
            )
            valid[frame] = bool(_strictly_matches(
                fitted, target[frame:frame + 1],
                force_tolerance_n=force_tolerance_n,
                torque_tolerance_nm=torque_tolerance_nm,
            )[0])
            if not valid[frame]:
                raise RuntimeError("per-frame solution failed independent wrench verification")
            error = None
            solver_timing = result.get("timing_s", {})
            solver_iterations = result.get("osqp_iterations")
            workspace_reused = bool(result.get("workspace_reused", False))
        except RuntimeError as exc:
            valid[frame] = False
            baseline_force[frame] = 0.0
            baseline_table[frame] = 0.0
            error = str(exc)
            solver_timing = {}
            solver_iterations = None
            workspace_reused = False
            failures.append({"frame_interval": [int(frame), int(frame + 1)], "error": error})
        frame_timings[frame] = time.perf_counter() - begin
        if log is not None:
            log({
                "event": "stage2_baseline_frame_end",
                "frame": int(frame),
                "valid": bool(valid[frame]),
                "error": error,
                "wall_s": float(frame_timings[frame]),
                "solver_timing_s": solver_timing,
                "osqp_iterations": solver_iterations,
                "workspace_reused": workspace_reused,
            })

    output_force = baseline_force.copy()
    output_table = baseline_table.copy()
    refined_sum_force = np.zeros_like(output_force)
    refined_sum_table = np.zeros_like(output_table)
    refined_weight = np.zeros(T, np.float64)
    refinement_records: list[dict] = []
    stride = window_size - overlap

    for run_begin, run_end in _runs(active & valid):
        starts = list(range(run_begin, run_end, stride))
        for start in starts:
            stop = min(start + window_size, run_end)
            if stop - start < 2:
                continue
            begin = time.perf_counter()
            record = {"begin": int(start), "end": int(stop), "frames": int(stop - start)}
            if log is not None:
                log({"event": "stage2_refine_window_start", **record,
                     "candidates": int(candidate[start:stop].sum())})
            try:
                result = solve_sequence_contact_qp(
                    target[start:stop], com[start:stop], points[start:stop], normals[start:stop],
                    priority[start:stop], candidate[start:stop],
                    mu=mu, friction_sides=6,
                    temporal_regularizer=temporal_regularizer,
                    table_normal=(table_normal[start:stop]
                                  if np.asarray(table_normal).ndim == 2 else table_normal),
                    table_contact_active=table_contact_active[start:stop],
                    table_support_point_world=(None if table_support_point_world is None else
                                               table_support_point_world[start:stop]),
                    table_temporal_regularizer=1.0, allow_wrench_slack=False,
                    time_limit_s=window_time_limit_s,
                    initial_vertex_force_world=(baseline_force[start:stop]
                                                if refinement_warm_start else None),
                    initial_table_force_world=(baseline_table[start:stop]
                                               if refinement_warm_start else None),
                )
                proposed_force = result["vertex_force_world"]
                proposed_table = result["table_force_world"]
                fitted = _fitted_wrench(
                    points[start:stop], com[start:stop], proposed_force, proposed_table,
                    None if table_support_point_world is None else
                    table_support_point_world[start:stop],
                )
                accepted = bool(_strictly_matches(
                    fitted, target[start:stop],
                    force_tolerance_n=force_tolerance_n,
                    torque_tolerance_nm=torque_tolerance_nm,
                ).all())
                error = None if accepted else "window failed independent wrench verification"
                solver_timing = result.get("timing_s", {})
                solver_iterations = result.get("osqp_iterations")
            except RuntimeError as exc:
                accepted = False
                error = str(exc)
                solver_timing = {}
                solver_iterations = None
            if accepted:
                length = stop - start
                weight = np.minimum(np.arange(1, length + 1), np.arange(length, 0, -1)).astype(np.float64)
                refined_sum_force[start:stop] += proposed_force * weight[:, None, None]
                refined_sum_table[start:stop] += proposed_table * weight[:, None]
                refined_weight[start:stop] += weight
            record.update({
                "accepted": bool(accepted), "error": error,
                "wall_s": float(time.perf_counter() - begin),
                "solver_timing_s": solver_timing,
                "osqp_iterations": solver_iterations,
                "warm_start_used": bool(result.get("warm_start_used", False))
                    if solver_timing else False,
                "workspace_reused": bool(result.get("workspace_reused", False))
                    if solver_timing else False,
            })
            refinement_records.append(record)
            if log is not None:
                log({"event": "stage2_refine_window_end", **record})
            if stop == run_end:
                break

    refined = refined_weight > 0
    output_force[refined] = refined_sum_force[refined] / refined_weight[refined, None, None]
    output_table[refined] = refined_sum_table[refined] / refined_weight[refined, None]
    fitted = _fitted_wrench(points, com, output_force, output_table,
                            table_support_point_world)
    verified = _strictly_matches(
        fitted, target,
        force_tolerance_n=force_tolerance_n,
        torque_tolerance_nm=torque_tolerance_nm,
    )
    fallback = active & valid & ~verified
    output_force[fallback] = baseline_force[fallback]
    output_table[fallback] = baseline_table[fallback]
    fitted = _fitted_wrench(points, com, output_force, output_table,
                            table_support_point_world)

    return {
        "vertex_force_world": output_force,
        "table_force_world": output_table,
        "fitted_contact_wrench": fitted,
        "force_residual_norm": np.linalg.norm(fitted[:, :3] - target[:, :3], axis=1),
        "torque_residual_norm": np.linalg.norm(fitted[:, 3:] - target[:, 3:], axis=1),
        "force_label_valid": valid,
        "failures": failures,
        "refined_frame_mask": refined,
        "frame_timing_s": frame_timings,
        "refinement_records": refinement_records,
    }
