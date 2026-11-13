"""Uniform 30 Hz solver timelines with lossless source-sample provenance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SOLVER_FPS = 30.0


def gaussian_observed_transition_mask(pose_observed: np.ndarray, radius: int) -> np.ndarray:
    """Require observed poses throughout both states' Gaussian support.

    Match generation's endpoint replication; interpolated pose gaps are not
    independent observations and must not create valid dynamics references.
    """
    observed = np.asarray(pose_observed, bool)
    if observed.ndim != 1 or len(observed) < 2:
        raise ValueError('pose_observed must contain at least two states')
    if radius < 1:
        raise ValueError('radius must be positive')
    padded = np.pad((~observed).astype(np.int32), (radius, radius), mode='edge')
    missing = np.convolve(padded, np.ones(2 * radius + 1, np.int32), mode='valid')
    return (missing[:-1] == 0) & (missing[1:] == 0)


def interpolate_pose_samples(frame_index, poses, target_index):
    """Interpolate SE(3), holding endpoints; shared with rigid generation."""
    source = np.asarray(frame_index, np.float64)
    target = np.asarray(target_index, np.float64)
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError('pose interpolation requires ordered unique frames')
    output = np.repeat(np.eye(4, dtype=np.float64)[None], len(target), axis=0)
    for axis in range(3):
        output[:, axis, 3] = np.interp(target, source, poses[:, axis, 3])
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    inside = (target >= source[0]) & (target <= source[-1])
    output[inside, :3, :3] = Slerp(source, rotations)(target[inside]).as_matrix()
    output[target < source[0], :3, :3] = rotations[0].as_matrix()
    output[target > source[-1], :3, :3] = rotations[-1].as_matrix()
    return output


@dataclass(frozen=True)
class SolverTimeline:
    source_fps: float
    solver_fps: float
    source_timestamp_s: np.ndarray
    solver_timestamp_s: np.ndarray
    source_left_index: np.ndarray
    source_right_index: np.ndarray
    interpolation_alpha: np.ndarray


def make_solver_timeline(
    source_count: int, source_fps: float, *, solver_fps: float = SOLVER_FPS,
) -> SolverTimeline:
    """Cover the source clip duration on a regular solver-rate state grid.

    A clip with ``N`` source frames represents ``N/source_fps`` seconds. Solver
    states occur at ``k/solver_fps`` while strictly inside that duration. Near
    the final boundary, interpolation clamps to the last observed source state.
    The original timestamps and bracket mapping are retained in full.
    """
    if source_count < 1:
        raise ValueError("source_count must be positive")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    if not np.isfinite(solver_fps) or solver_fps <= 0:
        raise ValueError("solver_fps must be finite and positive")
    source_time = np.arange(source_count, dtype=np.float64) / float(source_fps)
    solver_count = max(1, int(np.ceil(source_count * float(solver_fps) / float(source_fps))))
    solver_time = np.arange(solver_count, dtype=np.float64) / float(solver_fps)
    right = np.searchsorted(source_time, solver_time, side="left")
    right = np.clip(right, 0, source_count - 1)
    left = np.maximum(right - 1, 0)
    exact = source_time[right] == solver_time
    left[exact] = right[exact]
    denominator = source_time[right] - source_time[left]
    alpha = np.zeros(solver_count, dtype=np.float64)
    np.divide(solver_time - source_time[left], denominator, out=alpha, where=denominator > 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    return SolverTimeline(
        source_fps=float(source_fps), solver_fps=float(solver_fps),
        source_timestamp_s=source_time, solver_timestamp_s=solver_time,
        source_left_index=left.astype(np.int64),
        source_right_index=right.astype(np.int64), interpolation_alpha=alpha,
    )


def interpolate_linear(values: np.ndarray, timeline: SolverTimeline) -> np.ndarray:
    source = np.asarray(values)
    if len(source) != len(timeline.source_timestamp_s):
        raise ValueError("values must cover the source timeline")
    shape = (len(timeline.interpolation_alpha),) + (1,) * (source.ndim - 1)
    alpha = timeline.interpolation_alpha.reshape(shape)
    return ((1.0 - alpha) * source[timeline.source_left_index]
            + alpha * source[timeline.source_right_index])


def interpolate_rotations(rotations: np.ndarray, timeline: SolverTimeline) -> np.ndarray:
    source = np.asarray(rotations, np.float64)
    if source.shape != (len(timeline.source_timestamp_s), 3, 3):
        raise ValueError("rotations must have shape [source_count,3,3]")
    if len(source) == 1:
        return np.repeat(source, len(timeline.solver_timestamp_s), axis=0)
    query = np.clip(timeline.solver_timestamp_s, timeline.source_timestamp_s[0],
                    timeline.source_timestamp_s[-1])
    return Slerp(timeline.source_timestamp_s, Rotation.from_matrix(source))(query).as_matrix()


def interpolate_poses(poses: np.ndarray, timeline: SolverTimeline) -> np.ndarray:
    source = np.asarray(poses, np.float64)
    if source.shape != (len(timeline.source_timestamp_s), 4, 4):
        raise ValueError("poses must have shape [source_count,4,4]")
    output = np.repeat(np.eye(4, dtype=np.float64)[None],
                       len(timeline.solver_timestamp_s), axis=0)
    output[:, :3, :3] = interpolate_rotations(source[:, :3, :3], timeline)
    output[:, :3, 3] = interpolate_linear(source[:, :3, 3], timeline)
    return output


def nearest_source(values: np.ndarray, timeline: SolverTimeline) -> np.ndarray:
    """Resample discrete metadata without inventing fractional labels."""
    source = np.asarray(values)
    choose_right = timeline.interpolation_alpha >= 0.5
    index = np.where(choose_right, timeline.source_right_index, timeline.source_left_index)
    return source[index]


def timeline_payload(timeline: SolverTimeline) -> dict[str, np.ndarray]:
    """Fields stored in every 30 Hz canonical/backend label."""
    return {
        "solver_fps": np.float64(timeline.solver_fps),
        "source_fps": np.float64(timeline.source_fps),
        "solver_timestamp_s": timeline.solver_timestamp_s,
        "source_timestamp_s": timeline.source_timestamp_s,
        "source_frame_left": timeline.source_left_index,
        "source_frame_right": timeline.source_right_index,
        "source_interpolation_alpha": timeline.interpolation_alpha,
        "solver_timeline_policy": np.array("uniform_30hz_source_preserved_v1"),
    }


def transition_timeline_payload(timeline: SolverTimeline) -> dict[str, np.ndarray]:
    """Return the complete provenance for consecutive solver transitions."""
    if len(timeline.solver_timestamp_s) < 2:
        raise ValueError("at least two solver states are required for transitions")
    before = slice(None, -1)
    after = slice(1, None)
    return {
        **timeline_payload(timeline),
        "solver_transition_index": np.arange(
            len(timeline.solver_timestamp_s) - 1, dtype=np.int64),
        "solver_transition_timestamp_s": timeline.solver_timestamp_s[before],
        "solver_next_timestamp_s": timeline.solver_timestamp_s[after],
        "solver_dt_s": np.float64(1.0 / timeline.solver_fps),
        "source_frame_left": timeline.source_left_index[before],
        "source_frame_right": timeline.source_right_index[before],
        "source_interpolation_alpha": timeline.interpolation_alpha[before],
        "next_source_frame_left": timeline.source_left_index[after],
        "next_source_frame_right": timeline.source_right_index[after],
        "next_source_interpolation_alpha": timeline.interpolation_alpha[after],
    }
