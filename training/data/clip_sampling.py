"""Native RGB clip sampling with independent geometry and force masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceClipPlan:
    source_frames: np.ndarray
    substep_rows: np.ndarray
    closing_row: int
    geometry_rows: np.ndarray
    geometry_aligned: np.ndarray | None = None
    force_intervals_valid: np.ndarray | None = None
    force_prediction_aligned: np.ndarray | None = None


def native_frame_origins(first, stop, displayed_rows=None) -> np.ndarray:
    """Every decodable source frame in this half-open source interval."""
    if displayed_rows is None:
        return np.arange(first, stop, dtype=np.int64)
    frames = np.unique(displayed_rows)
    return frames[(frames >= first) & (frames < stop)]


def _nearest_label_rows(positions, requested):
    """Find nearest finite A1 source positions; first row wins exact ties."""
    rows = np.flatnonzero(np.isfinite(positions))
    if not len(rows):
        raise ValueError("source clip has no finite label positions")
    values = positions[rows]
    if np.any(np.diff(values) < 0):
        order = np.argsort(values, kind="stable")
        rows, values = rows[order], values[order]
    insertion = np.searchsorted(values, requested, side="left")
    previous = np.maximum(0, insertion - 1)
    following = np.minimum(len(values) - 1, insertion)
    distance_before = np.abs(values[previous] - requested)
    distance_after = np.abs(values[following] - requested)
    nearest = np.where(distance_before < distance_after, previous, following)
    return rows[nearest]


def source_clip_plan(source_positions, force_start, force_stop, origin_frame,
                     sequence_length, source_fps, solver_fps, target_fps,
                     frame_count, displayed_rows=None) -> SourceClipPlan:
    """Generate 10 fps RGB addresses independently of complete force intervals.

    Geometry only supervises frames that have an A1 row at the actual
    displayed time. For force, index the regular solver clock separately and
    mask intervals outside the admitted label run or with an unaligned visual
    prediction. Clamp the masked gathers to a safe existing row.
    """
    if not 0 <= origin_frame < frame_count:
        raise ValueError("source clip origin maps outside the video")
    if not 0 <= force_start <= force_stop < len(source_positions):
        raise ValueError("source clip has an invalid solver-row span")
    if min(source_fps, solver_fps, target_fps) <= 0:
        raise ValueError("source clip clocks must be positive")

    desired = origin_frame + np.arange(sequence_length) * source_fps / target_fps
    frames = np.floor(desired + 0.5).astype(np.int64)
    frames = frames[frames < frame_count]
    displayed = frames if displayed_rows is None else displayed_rows[frames]
    geometry_rows = force_start + _nearest_label_rows(
        source_positions[force_start:force_stop + 1], displayed)
    geometry_aligned = np.abs(source_positions[geometry_rows] - displayed) <= 1e-4

    # Use the closest solver row as the first time, then retain the solver
    # cadence. A 30 Hz solver has three consecutive intervals per visual
    # step. Other source/solver rates are subject to the same label checks.
    anchor = int(geometry_rows[0])
    offset = np.floor(np.arange(len(frames) + 1) * solver_fps / target_fps + 0.5)
    boundaries = anchor + offset.astype(np.int64)
    substeps = np.rint(np.linspace(boundaries[:-1], boundaries[1:], 4, axis=1)[:, :-1])
    substeps = substeps.astype(np.int64)
    force_intervals_valid = (substeps >= force_start) & (substeps < force_stop)
    valid_prediction_row = substeps[:, 0] <= force_stop
    substeps = np.clip(substeps, force_start, force_stop)
    if solver_fps != 30.0:
        # Point force uses the real aligned solver row. These clocks are
        # excluded from the fixed 30 Hz rollout integration path.
        substeps[:, 0] = geometry_rows
        valid_prediction_row = geometry_aligned
    force_prediction_aligned = np.abs(
        source_positions[substeps[:, 0]] - displayed) <= 1e-4
    force_prediction_aligned &= valid_prediction_row
    return SourceClipPlan(
        frames, substeps.reshape(-1), int(np.clip(boundaries[-1], force_start, force_stop)),
        geometry_rows, geometry_aligned, force_intervals_valid, force_prediction_aligned,
    )
