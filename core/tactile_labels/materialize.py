"""Shared labels-v5 materialization (spec §5/§7).

Single implementation of the write-time derivations — native solver-row target objects, packed flags, index rows, stream layout, and
statistics — used by the production staging writer
(``label_generation/publish.py``).  All
inputs are plain arrays decoded from the solver's canonical NPZ run tree and
the ``_processed`` multimodal metadata; nothing here reads or imports the
labels-v4 world.

The target-object rule is the §5.5 selection: metadata (role=target) first,
otherwise the loaded-object wrench-energy argmax.  The caller injects it as
``target_of``; it consumes minimal row dicts built by ``oracle_rows`` from
the unmasked per-object force — invalid rows keep solver output until the
A6 zeroing below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.tactile_labels.schema import (
    FLAG_CONTACT_VALID, FLAG_FORCE_VALID, FLAG_LEFT_PRESENT,
    FLAG_PROXIMITY_VALID, FLAG_RIGHT_PRESENT, LABELS_REVISION,
    LABELS_SCHEMA_NAME, LABEL_INDEX_SCHEMA,
    PROXIMITY_CLIP_M, TACTILE_SCHEMA, FORCE_SCHEMA, FORCE_SEMANTICS,
    TOPOLOGY_ARTICULATED, TOPOLOGY_RIGID, TOPOLOGY_STATIC,
)

from core.tactile_labels.native_timeline import native_timeline_from_rows

EPISODES_PER_FILE = 128
DEFAULT_SUBSTEPS = 4


def runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(mask):
        if not bool(mask[index]):
            index += 1
            continue
        stop = index + 1
        while stop < len(mask) and bool(mask[stop]):
            stop += 1
        runs.append((index, stop))
        index = stop
    return runs


def oracle_rows(object_ids: list[str],
                vertex_force_per_object: np.ndarray,
                object_contact_wrench: np.ndarray) -> list[dict[str, Any]]:
    """Minimal v4-shaped rows for the archived ``target_object`` oracle."""
    return [{
        "object_instance_ids": object_ids,
        "object_contact_wrench": object_contact_wrench[t].astype(
            np.float64).tobytes(),
        "object_hand_vertex_force_world_n_per_kg": vertex_force_per_object[t].astype(
            np.float32).tobytes(),
    } for t in range(len(object_contact_wrench))]


def build_episode_labels(
    *,
    episode_index: int,
    canonical_view: str,
    solver_fps: float,
    source_fps: float,
    dt_s: float,
    source_frame_left: np.ndarray,
    source_alpha: np.ndarray,
    hand_present: np.ndarray,
    contact: np.ndarray,
    signed_distance: np.ndarray,
    force_valid: np.ndarray,
    vertex_force_per_object: np.ndarray,
    object_contact_wrench: np.ndarray,
    object_environment_wrench: np.ndarray,
    object_position: np.ndarray,
    object_rotation: np.ndarray,
    object_velocity: np.ndarray,
    object_gravity_world: np.ndarray,
    object_ids: list[str],
    reference_mass_kg: float,
    physics_of: Callable[[str], Any],
    target_of: Callable[[list[dict[str, Any]], tuple[int, int]], tuple[str, int]],
    topology: str = TOPOLOGY_RIGID,
    substeps: int = DEFAULT_SUBSTEPS,
) -> dict[str, Any]:
    """Materialize one episode into the shared pre-write representation.

    Inputs are plain arrays on the solver timeline; payloads follow the
    solver conventions (forces N/kg hand-on-object world, wrenches about
    each object's COM, forces NOT yet zeroed on force-invalid rows).
    Stored force semantics is the union across objects (spec §5.5,
    decision 9): each vertex's force is owned by exactly one object, so
    the per-object sum reproduces the canonical flat per-hand force and
    both hands keep supervision when they touch different objects.
    Stored wrenches are the per-object sums transported to the target
    reference point, keeping the training readout over the stored forces
    consistent with the stored contact − environment wrench (single-object
    episodes are numerically identical to the previous target-resolved
    storage).  ``has_non_target_force`` marks episodes whose rollout must
    be gated (union wrench does not drive the target body alone).  Runs
    the §7 staging assertions A1–A8 inline.
    """
    total = len(force_valid)
    n_objects = len(object_ids)
    shapes = {
        "hand_present": ((total, 2), hand_present),
        "contact": ((total, 2, 778), contact),
        "signed_distance": ((total, 2, 778), signed_distance),
        "vertex_force_per_object": ((total, n_objects, 2, 778, 3),
                                    vertex_force_per_object),
        "object_contact_wrench": ((total, n_objects, 6), object_contact_wrench),
        "object_environment_wrench": ((total, n_objects, 6), object_environment_wrench),
        "object_position": ((total, n_objects, 3), object_position),
        "object_rotation": ((total, n_objects, 3, 3), object_rotation),
        "object_velocity": ((total, n_objects, 6), object_velocity),
    }
    for name, (expected, array) in shapes.items():
        if tuple(np.asarray(array).shape) != expected:
            raise ValueError(
                f"episode {episode_index}: {name} shape {np.asarray(array).shape} "
                f"!= {expected}")
    if len(source_frame_left) != total or len(source_alpha) != total:
        raise ValueError(f"episode {episode_index}: A1 source mapping row count")

    source_positions = (np.asarray(source_frame_left, np.float64)
                        + np.asarray(source_alpha, np.float64))
    # Missing geometry cannot enter force supervision or target selection.
    raw_distance = np.asarray(signed_distance, np.float32).reshape(total, 2, 778)
    geometry_invalid = raw_distance.min(axis=(1, 2)) > 1.0
    force_valid = np.asarray(force_valid, bool) & ~geometry_invalid
    row_slot = np.zeros(total, np.int32)
    if n_objects > 1:
        oracle = oracle_rows(object_ids, vertex_force_per_object, object_contact_wrench)
        for begin, end in runs_from_mask(force_valid):
            _, slot = target_of(oracle, (begin, end))
            if not 0 <= slot < n_objects:
                raise ValueError(f"episode {episode_index}: invalid target index {slot}")
            row_slot[begin:] = slot

    # A8: physics for every object in object_ids (the lists stay aligned,
    # spec §4.3); the solver's reference-mass contract must hold for all.
    physics = [physics_of(object_id) for object_id in object_ids]
    for body in physics:
        if not np.isclose(float(body.mass), float(reference_mass_kg), rtol=0, atol=1e-9):
            raise ValueError(
                f"episode {episode_index}: A8 physics mass {body.mass} != "
                f"reference {reference_mass_kg}")

    present = np.asarray(hand_present, bool)
    tactile_flags = np.zeros(total, np.uint8)
    contact_out = np.zeros((total, 2, 778), np.uint8)
    signed_out = np.full((total, 2, 778), PROXIMITY_CLIP_M, np.float16)
    contact_in = np.asarray(contact).reshape(total, 2, 778).astype(np.uint8)
    distance_in = np.clip(np.asarray(signed_distance, np.float32).reshape(total, 2, 778),
                          -PROXIMITY_CLIP_M, PROXIMITY_CLIP_M)
    contact_in[~present] = 0
    distance_in[~present] = PROXIMITY_CLIP_M
    tactile_flags[:] = ((FLAG_LEFT_PRESENT * present[:, 0].astype(np.uint8))
                        | (FLAG_RIGHT_PRESENT * present[:, 1].astype(np.uint8))
                        | FLAG_CONTACT_VALID * present.any(axis=1).astype(np.uint8)
                        | FLAG_PROXIMITY_VALID * present.any(axis=1).astype(np.uint8))
    contact_out[:] = contact_in.astype(np.uint8)
    signed_out[:] = distance_in.astype(np.float16)
    if geometry_invalid.any():
        # uint8-safe mask: plain ~(...) is a negative Python int, which
        # numpy>=2 refuses to assign into the uint8 flag array.
        tactile_flags[geometry_invalid] &= np.uint8(
            0xFF ^ (FLAG_CONTACT_VALID | FLAG_PROXIMITY_VALID))
    if not np.isfinite(signed_out.astype(np.float32)).all():
        raise ValueError(f"episode {episode_index}: A5 nonfinite distance")
    if contact_out.max(initial=0) > 1:
        raise ValueError(f"episode {episode_index}: A5 contact not binary")

    force_flags = FLAG_FORCE_VALID * np.asarray(force_valid, bool).astype(np.uint8)
    per_object_f32 = np.asarray(vertex_force_per_object, np.float32)
    slot_rows = np.asarray(row_slot, np.int64)
    # Union across objects (spec §5.5, decision 9): each vertex's force is
    # owned by exactly one object, so the sum reproduces the canonical
    # per-vertex force and both hands keep supervision when they touch
    # different objects.  The target slot still drives wrench/state columns.
    vertex_force = per_object_f32.sum(axis=1)
    masked_invalid_rows = int((~force_valid
                               & (np.abs(vertex_force).max(axis=(1, 2, 3)) > 0)).sum())
    vertex_force[~force_valid] = 0  # A6: invalid rows read as all-zero

    # Rollout gate: any force-valid row with nonzero hand force on a
    # non-target object (the union wrench then does not describe the target
    # body's dynamics alone).
    per_object_nonzero = np.abs(per_object_f32).max(axis=(2, 3, 4)) > 0  # (T, O)
    if n_objects > 1:
        non_target = per_object_nonzero.copy()
        non_target[np.arange(total), slot_rows] = False
        has_non_target_force = bool(
            (non_target & np.asarray(force_valid, bool)[:, None]).any())
    else:
        has_non_target_force = False

    # Union wrenches: per-object sums transported to the target reference
    # point, so readout(Σ f, lever vs object_position) == contact −
    # environment holds on the stored columns (single object: unchanged).
    def transported_union(per_object_wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(per_object_wrench, np.float64)            # (T, O, 6)
        arm = (np.asarray(object_position, np.float64)
               - position_all[np.arange(total), slot_rows][:, None, :])
        forces = wrench[:, :, :3]
        taus = wrench[:, :, 3:] + np.cross(arm, forces)
        return np.concatenate([forces.sum(axis=1), taus.sum(axis=1)],
                              axis=1).astype(np.float32)

    position_all = np.asarray(object_position, np.float32)
    contact_wrench = transported_union(object_contact_wrench)
    environment_wrench = transported_union(object_environment_wrench)
    position = position_all[np.arange(total), slot_rows]
    rotation = np.asarray(object_rotation, np.float32)[
        np.arange(total), slot_rows].reshape(total, 3, 3)
    velocity = np.asarray(object_velocity, np.float32)[np.arange(total), slot_rows]

    return {
        "episode_index": episode_index,
        "canonical_view": canonical_view,
        "solver_fps": float(solver_fps),
        "source_fps": float(source_fps),
        "dt_s": float(dt_s),
        "n_rows": total,
        "topology": topology,
        "substeps": int(substeps),
        "object_ids": object_ids,
        "physics": physics,
        "gravity_world": np.asarray(object_gravity_world, np.float64).reshape(3),
        "row_slot": row_slot,
        "source_positions": source_positions.astype(np.float32),
        "tactile_flags": tactile_flags,
        "contact": contact_out,
        "signed": signed_out,
        "force_flags": force_flags,
        "masked_invalid_force_rows": masked_invalid_rows,
        "has_non_target_force": has_non_target_force,
        "vertex_force": vertex_force,
        "contact_wrench": contact_wrench,
        "environment_wrench": environment_wrench,
        "position": position,
        "rotation": rotation,
        "velocity": velocity,
    }


def build_articulated_episode_labels(
    *,
    episode_index: int,
    canonical_view: str,
    solver_fps: float,
    source_fps: float,
    dt_s: float,
    source_frame_left: np.ndarray,
    hand_present: np.ndarray,
    contact: np.ndarray,
    signed_distance: np.ndarray,
    force_valid: np.ndarray,
    vertex_force: np.ndarray,
    generalized_contact_wrench: np.ndarray,
    generalized_environment_wrench: np.ndarray,
    joint_angle: np.ndarray,
    joint_velocity: np.ndarray,
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    base_velocity: np.ndarray,
    gravity_world: np.ndarray,
    object_name: str,
    physics_body,
    substeps: int = DEFAULT_SUBSTEPS,
    generalized_wrench_dim: int = 7,
) -> dict[str, Any]:
    """Articulated (floating-base revolute) twin of ``build_episode_labels``.

    The articulated root has no per-object 6D target: the "object" is the
    whole articulated body, supervision is the ``generalized_wrench``
    (7-dim: force, torque about the reference point, joint torque) plus
    ``joint_state``; ``object_position/rotation/velocity`` carry the root
    link state (spec §4.2/§4.3).  Inputs follow the solver's canonical NPZ
    conventions (N/kg reference mass, forces not yet zeroed on invalid rows).
    """
    total = len(force_valid)
    source_positions = np.asarray(source_frame_left, np.float64)
    present = np.asarray(hand_present, bool)
    tactile_flags = np.zeros(total, np.uint8)
    contact_in = np.asarray(contact).reshape(total, 2, 778).astype(np.uint8)
    distance_in = np.clip(
        np.asarray(signed_distance, np.float32).reshape(total, 2, 778),
        -PROXIMITY_CLIP_M, PROXIMITY_CLIP_M)
    contact_in[~present] = 0
    distance_in[~present] = PROXIMITY_CLIP_M
    tactile_flags[:] = ((FLAG_LEFT_PRESENT * present[:, 0].astype(np.uint8))
                        | (FLAG_RIGHT_PRESENT * present[:, 1].astype(np.uint8))
                        | FLAG_CONTACT_VALID * present.any(axis=1).astype(np.uint8)
                        | FLAG_PROXIMITY_VALID * present.any(axis=1).astype(np.uint8))
    signed_out = distance_in.astype(np.float16)
    if not np.isfinite(signed_out.astype(np.float32)).all():
        raise ValueError(f"episode {episode_index}: A5 nonfinite distance")

    force_flags = FLAG_FORCE_VALID * np.asarray(force_valid, bool).astype(np.uint8)
    force_f32 = np.asarray(vertex_force, np.float32).reshape(total, 2, 778, 3)
    masked_invalid_rows = int((~force_valid
                               & (np.abs(force_f32).max(axis=(1, 2, 3)) > 0)).sum())
    force_f32[~force_valid] = 0  # A6

    generalized = np.zeros((total, 8), np.float32)
    generalized[:, :generalized_wrench_dim] = np.asarray(
        generalized_contact_wrench, np.float32)[:, :generalized_wrench_dim]
    joint_state = np.zeros((total, 4), np.float32)
    joint_state[:, 0] = np.asarray(joint_angle, np.float64)
    joint_state[:, 1] = np.asarray(joint_velocity, np.float64)

    return {
        "episode_index": episode_index,
        "canonical_view": canonical_view,
        "solver_fps": float(solver_fps),
        "source_fps": float(source_fps),
        "dt_s": float(dt_s),
        "n_rows": total,
        "topology": TOPOLOGY_ARTICULATED,
        "substeps": int(substeps),
        "generalized_wrench_dim": int(generalized_wrench_dim),
        "object_ids": [str(object_name)],
        "physics": [physics_body],
        "gravity_world": np.asarray(gravity_world, np.float64).reshape(3),
        "row_slot": np.zeros(total, np.int8),
        "source_positions": source_positions.astype(np.float32),
        "tactile_flags": tactile_flags,
        "contact": contact_in,
        "signed": signed_out,
        "force_flags": force_flags,
        "masked_invalid_force_rows": masked_invalid_rows,
        "has_non_target_force": False,
        "vertex_force": force_f32,
        "contact_wrench": np.asarray(
            generalized_contact_wrench, np.float32)[:, :6].copy(),
        "environment_wrench": np.asarray(
            generalized_environment_wrench, np.float32)[:, :6].copy(),
        "position": np.asarray(base_position, np.float32),
        "rotation": np.asarray(base_rotation, np.float32).reshape(total, 3, 3),
        "velocity": np.asarray(base_velocity, np.float32),
        "generalized_wrench": generalized,
        "joint_state": joint_state,
    }


def build_static_episode_labels(
    *,
    episode_index: int,
    canonical_view: str,
    fps: float,
    source_frame_left: np.ndarray,
    hand_present: np.ndarray,
    contact: np.ndarray,
    signed_distance: np.ndarray,
    force_valid: np.ndarray,
    vertex_force: np.ndarray,
    contact_wrench: np.ndarray,
    object_com: np.ndarray,
    object_ids: list[str],
    object_index_per_row: np.ndarray,
    gravity_world: np.ndarray,
    physics_of,
    substeps: int = DEFAULT_SUBSTEPS,
    object_rotation: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-frame static (independent-sample) twin of ``build_episode_labels``.

    Each row is an independent equilibrium with its own source frame,
    gravity and object slot.
    """
    total = len(force_valid)
    present = np.asarray(hand_present, bool)
    tactile_flags = np.zeros(total, np.uint8)
    contact_in = np.asarray(contact).reshape(total, 2, 778).astype(np.uint8)
    distance_in = np.clip(
        np.asarray(signed_distance, np.float32).reshape(total, 2, 778),
        -PROXIMITY_CLIP_M, PROXIMITY_CLIP_M)
    contact_in[~present] = 0
    distance_in[~present] = PROXIMITY_CLIP_M
    tactile_flags[:] = ((FLAG_LEFT_PRESENT * present[:, 0].astype(np.uint8))
                        | (FLAG_RIGHT_PRESENT * present[:, 1].astype(np.uint8))
                        | FLAG_CONTACT_VALID * present.any(axis=1).astype(np.uint8)
                        | FLAG_PROXIMITY_VALID * present.any(axis=1).astype(np.uint8))
    signed_out = distance_in.astype(np.float16)
    if not np.isfinite(signed_out.astype(np.float32)).all():
        raise ValueError(f"episode {episode_index}: A5 nonfinite distance")

    force_flags = FLAG_FORCE_VALID * np.asarray(force_valid, bool).astype(np.uint8)
    force_f32 = np.asarray(vertex_force, np.float32).reshape(total, 2, 778, 3)
    masked_invalid_rows = int((~force_valid
                               & (np.abs(force_f32).max(axis=(1, 2, 3)) > 0)).sum())
    force_f32[~force_valid] = 0  # A6

    # Row-level object index: static datasets may switch objects per
    # independent frame (obman holds thousands per episode), so this stays
    # wide enough for all object slots.
    row_slot = np.asarray(object_index_per_row, np.int32)
    physics = [physics_of(object_id) for object_id in object_ids]
    gravity_rows = np.asarray(gravity_world, np.float64)
    if gravity_rows.shape == (3,):
        gravity_rows = np.broadcast_to(gravity_rows, (total, 3)).copy()
    if gravity_rows.shape != (total, 3) or not np.isfinite(gravity_rows).all():
        raise ValueError("Static gravity must be a finite vector for every row")
    valid = np.asarray(force_valid, bool)
    if ((row_slot[valid] < 0) | (row_slot[valid] >= len(physics))).any():
        raise ValueError("Static force-valid rows require an exact object index")
    row_mass = np.asarray([body.mass for body in physics])[row_slot[valid]]
    if not np.allclose(np.asarray(contact_wrench)[valid, :3],
                       -row_mass[:, None] * gravity_rows[valid], rtol=1e-5, atol=1e-5):
        raise ValueError("Static per-row gravity disagrees with its equilibrium target")
    if object_rotation is None:
        rotation = np.zeros((total, 3, 3), np.float32)
        rotation[:, 0, 0] = rotation[:, 1, 1] = rotation[:, 2, 2] = 1.0
    else:
        # World-frame per-row orientation of the row's object.  Identity is
        # only a placeholder for rows whose source pose is unavailable; the
        # review overlay poses the mesh with this rotation, so a wrong one
        # floats the silhouette off the real object.
        rotation = np.asarray(object_rotation, np.float32).reshape(total, 3, 3)
        bad = np.abs(np.linalg.det(rotation) - 1.0) > 1e-3
        if bad.any():
            raise ValueError(
                f"episode {episode_index}: object_rotation rows are not "
                f"rotations ({int(bad.sum())} bad of {total})")
    return {
        "episode_index": episode_index,
        "canonical_view": canonical_view,
        "solver_fps": float(fps),
        "source_fps": float(fps),
        "dt_s": 1.0 / float(fps),
        "n_rows": total,
        "topology": TOPOLOGY_STATIC,
        "substeps": int(substeps),
        "generalized_wrench_dim": 0,
        "object_ids": [str(value) for value in object_ids],
        "physics": physics,
        "gravity_world": gravity_rows[0].copy(),
        "static_gravity_world": gravity_rows,
        "static_object_index": row_slot,
        "row_slot": row_slot,
        "source_positions": np.asarray(source_frame_left,
                                       np.float64).astype(np.float32),
        "tactile_flags": tactile_flags,
        "contact": contact_in,
        "signed": signed_out,
        "force_flags": force_flags,
        "masked_invalid_force_rows": masked_invalid_rows,
        "has_non_target_force": False,
        "vertex_force": force_f32,
        "contact_wrench": np.asarray(contact_wrench, np.float32),
        "environment_wrench": np.zeros((total, 6), np.float32),
        "position": np.asarray(object_com, np.float32),
        "rotation": rotation,
        "velocity": np.zeros((total, 6), np.float32),
    }


def build_static_tactile_labels(
    *,
    episode_index: int,
    canonical_view: str,
    fps: float,
    source_frame_left: np.ndarray,
    hand_present: np.ndarray,
    contact: np.ndarray,
    signed_distance: np.ndarray,
    contact_valid: np.ndarray | None = None,
    proximity_valid: np.ndarray | None = None,
    physics_of=None,
    object_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Contact/proximity-only twin for per-frame static datasets.

    Source: the installed per-side tactile streams of single-frame datasets
    (obman / mow), which carry no force payload.  The episode is stored with
    Each row is an independent source frame, with optional object physics.
    """
    total = len(source_frame_left)
    present = np.asarray(hand_present, bool)
    tactile_flags = np.zeros(total, np.uint8)
    contact_in = np.asarray(contact).reshape(total, 2, 778).astype(np.uint8)
    distance_in = np.asarray(signed_distance, np.float32).reshape(total, 2, 778)
    # The v4-era -1.0 m sentinel marks absent hands (never real geometry);
    # mask those rows out before the clip.
    real = np.isfinite(distance_in) & (distance_in > -0.9)
    distance_in = np.where(real, distance_in, PROXIMITY_CLIP_M)
    distance_in = np.clip(distance_in, -PROXIMITY_CLIP_M, PROXIMITY_CLIP_M)
    contact_in[~present] = 0
    distance_in[~present] = PROXIMITY_CLIP_M
    valid_any = (np.asarray(contact_valid, bool).any(axis=1)
                 if contact_valid is not None else present.any(axis=1))
    prox_any = (np.asarray(proximity_valid, bool).any(axis=1)
                if proximity_valid is not None else present.any(axis=1))
    tactile_flags[:] = ((FLAG_LEFT_PRESENT * present[:, 0].astype(np.uint8))
                        | (FLAG_RIGHT_PRESENT * present[:, 1].astype(np.uint8))
                        | FLAG_CONTACT_VALID * valid_any.astype(np.uint8)
                        | FLAG_PROXIMITY_VALID * prox_any.astype(np.uint8))
    signed_out = distance_in.astype(np.float16)
    if not np.isfinite(signed_out.astype(np.float32)).all():
        raise ValueError(f"episode {episode_index}: A5 nonfinite distance")
    return {
        "episode_index": episode_index,
        "canonical_view": canonical_view,
        "solver_fps": float(fps),
        "source_fps": float(fps),
        "dt_s": 1.0 / float(fps),
        "n_rows": total,
        "topology": TOPOLOGY_STATIC,
        "substeps": int(DEFAULT_SUBSTEPS),
        "generalized_wrench_dim": 0,
        "object_ids": [str(v) for v in (object_ids or [])],
        "physics": ([physics_of(oid) for oid in object_ids]
                    if physics_of is not None and object_ids else []),
        "gravity_world": np.asarray((0.0, 0.0, -9.80665)),
        "row_slot": np.zeros(total, np.int32),
        "source_positions": np.asarray(source_frame_left,
                                       np.float64).astype(np.float32),
        "tactile_flags": tactile_flags,
        "contact": contact_in,
        "signed": signed_out,
        "force_flags": np.zeros(total, np.uint8),
        "masked_invalid_force_rows": 0,
        "vertex_force": np.zeros((total, 2, 778, 3), np.float32),
        "contact_wrench": np.zeros((total, 6), np.float32),
        "environment_wrench": np.zeros((total, 6), np.float32),
        "position": np.zeros((total, 3), np.float32),
        "rotation": np.broadcast_to(
            np.eye(3, dtype=np.float32), (total, 3, 3)).copy(),
        "velocity": np.zeros((total, 6), np.float32),
    }


def fixed_list(values: np.ndarray, width: int) -> pa.Array:
    """Efficient FixedSizeListArray from a contiguous numpy payload."""
    flat = np.ascontiguousarray(values).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), width)


def index_row(episode: dict, tactile_path: str, force_path: str,
              group_index: int) -> dict:
    rows = int(episode["n_rows"])
    row_ids = np.arange(rows, dtype=np.int32)
    # A9: content hash over every payload column in schema order; the
    # articulated-only columns (generalized_wrench, joint_state) join the
    # hash when the writer populates them.
    force_bytes = (row_ids.tobytes() + episode["force_flags"].tobytes()
                   + episode["vertex_force"].tobytes()
                   + episode["contact_wrench"].tobytes()
                   + episode["environment_wrench"].tobytes()
                   + episode["position"].tobytes()
                   + episode["rotation"].reshape(rows, 9).tobytes()
                   + episode["velocity"].tobytes())
    if "generalized_wrench" in episode:
        force_bytes += (episode["generalized_wrench"].tobytes()
                        + episode["joint_state"].tobytes())
    if "static_gravity_world" in episode:
        force_bytes += (episode["static_gravity_world"].astype(np.float64).tobytes()
                        + episode["static_object_index"].astype(np.int32).tobytes())
    tactile_sha1 = hashlib.sha1(
        row_ids.tobytes()
        + episode["source_positions"].astype(np.float32).tobytes()
        + episode["tactile_flags"].tobytes()
        + episode["contact"].tobytes() + episode["signed"].tobytes()).hexdigest()
    force_sha1 = hashlib.sha1(force_bytes).hexdigest()
    # object_ids and the physics lists stay aligned (spec §4.3); every
    # object_idx / target_object indexes that one list.
    return {
        "episode_index": episode["episode_index"],
        "solver_fps": episode["solver_fps"],
        "source_fps": episode["source_fps"],
        "n_rows": episode["n_rows"],
        "topology": episode["topology"],
        "has_force": True,
        "has_non_target_force": bool(
            episode.get("has_non_target_force", False)),
        "dt_s": episode["dt_s"],
        "substeps": episode["substeps"],
        "gravity_world": episode["gravity_world"].tolist(),
        "generalized_wrench_dim": int(episode.get("generalized_wrench_dim", 0)),
        "object_ids": [str(value) for value in episode["object_ids"]],
        "object_mass_kg": [float(body.mass) for body in episode["physics"]],
        "object_com_body": [np.asarray(body.com_body, np.float32).tolist()
                            for body in episode["physics"]],
        "object_inertia_body": [np.asarray(
            body.inertia_body, np.float32).reshape(-1).tolist()
            for body in episode["physics"]],
        "tactile_groups": [{"path": tactile_path, "row_group": group_index}],
        "force_groups": [{"path": force_path, "row_group": group_index}],
        "canonical_view": episode["canonical_view"],
        "tactile_sha1": tactile_sha1,
        "force_sha1": force_sha1,
    }


def write_label_streams(output_root: Path, episodes: Iterable[dict],
                        info_declaration: dict[str, Any],
                        source_info: dict[str, Any]) -> None:
    """Stream one episode at a time; retain the existing 128-episode layout."""
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    file_index = -1
    tactile_writer = force_writer = None
    tactile_path = force_path = ""
    (output_root / "timeline_data").mkdir(parents=True, exist_ok=True)
    revision = str(info_declaration.get("solver_run_id", "native_solver_rows"))
    try:
        for ordinal, episode in enumerate(episodes):
            current = ordinal // EPISODES_PER_FILE
            if current != file_index:
                if tactile_writer is not None:
                    tactile_writer.close()
                    if force_writer is not None:
                        force_writer.close()
                file_index = current
                tactile_path = f"tactile_data/chunk-{file_index:05d}/file-{file_index:05d}.parquet"
                force_path = f"force_data/chunk-{file_index:05d}/file-{file_index:05d}.parquet"
                (output_root / tactile_path).parent.mkdir(parents=True, exist_ok=True)
                (output_root / force_path).parent.mkdir(parents=True, exist_ok=True)
                tactile_writer = pq.ParquetWriter(output_root / tactile_path, TACTILE_SCHEMA, compression="zstd")
                force_writer = pq.ParquetWriter(output_root / force_path, FORCE_SCHEMA, compression="zstd")
            assert tactile_writer is not None and force_writer is not None
            rows = episode["n_rows"]
            tactile_writer.write_table(pa.Table.from_arrays([
                pa.array(np.arange(rows, dtype=np.int32)),
                pa.array(episode["source_positions"], pa.float32()),
                pa.array(episode["tactile_flags"], pa.uint8()),
                fixed_list(episode["contact"], 1556),
                fixed_list(episode["signed"], 1556),
            ], schema=TACTILE_SCHEMA), row_group_size=rows)
            force_writer.write_table(pa.Table.from_arrays([
                pa.array(np.arange(rows, dtype=np.int32)),
                pa.array(episode["force_flags"], pa.uint8()),
                fixed_list(episode["vertex_force"], 4668),
                fixed_list(episode["contact_wrench"], 6),
                fixed_list(episode["environment_wrench"], 6),
                fixed_list(episode["position"], 3),
                fixed_list(episode["rotation"].reshape(rows, 9), 9),
                fixed_list(episode["velocity"], 6),
                (fixed_list(episode["generalized_wrench"], 8).cast(FORCE_SCHEMA.field("generalized_wrench").type)
                 if "generalized_wrench" in episode else
                 pa.nulls(rows, FORCE_SCHEMA.field("generalized_wrench").type)),
                (fixed_list(episode["joint_state"], 4).cast(FORCE_SCHEMA.field("joint_state").type)
                 if "joint_state" in episode else
                 pa.nulls(rows, FORCE_SCHEMA.field("joint_state").type)),
                (fixed_list(episode["static_gravity_world"], 3).cast(FORCE_SCHEMA.field("static_gravity_world").type)
                 if "static_gravity_world" in episode else
                 pa.nulls(rows, FORCE_SCHEMA.field("static_gravity_world").type)),
                (pa.array(episode["static_object_index"], pa.int32())
                 if "static_object_index" in episode else pa.nulls(rows, pa.int32())),
            ], schema=FORCE_SCHEMA), row_group_size=rows)

            label = index_row(episode, tactile_path, force_path, ordinal % EPISODES_PER_FILE)
            index_rows.append(label)
            timeline = native_timeline_from_rows(
                label, episode["source_positions"], episode["tactile_flags"],
                (episode["force_flags"] & FLAG_FORCE_VALID) != 0, episode["row_slot"])
            timeline = timeline.replace_schema_metadata(
                (timeline.schema.metadata or {}) | {b"source_revision": revision.encode()})
            filename = f"episode_{episode['episode_index']:06d}.parquet"
            pq.write_table(timeline, output_root / "timeline_data" / filename, compression="zstd")

    finally:
        if tactile_writer is not None:
            tactile_writer.close()
        if force_writer is not None:
            force_writer.close()
    pq.write_table(pa.Table.from_pylist(index_rows, schema=LABEL_INDEX_SCHEMA),
                   output_root / "meta/label_index.parquet", compression="zstd")
    declared = dict(info_declaration)
    declared.setdefault("force_semantics", FORCE_SEMANTICS)
    declared.setdefault("schema", LABELS_SCHEMA_NAME)
    declared.setdefault("revision", LABELS_REVISION)
    declared["timeline"] = "solver_rows_v1"
    declared["tactile_content_hash"] = "solver_rows_sha1_v1"
    declared["timeline_revision"] = revision
    declared.setdefault("proximity_clip_m", PROXIMITY_CLIP_M)
    declared.setdefault("signed_distance_dtype", "float16")
    declared.setdefault("force_dtype", "float32")
    info = {key: value for key, value in source_info.items()
            if key not in {"format_version", "manifest", "revision", "_processed_manifest"}}
    info["beingh"] = dict(info.get("beingh", {}), labels=declared)
    (output_root / "meta/info.json").write_text(json.dumps(info, indent=2))


class LabelStatsAccumulator:
    """Accumulate scales without retaining per-vertex force payloads.

    Exact robust wrench quantiles retain only the six wrench columns; the much
    larger vertex payload is released after each episode has been written.
    """

    def __init__(self):
        self.squared = 0.0
        self.count = 0
        self.wrenches = []
        self.n_rows = self.episodes = 0

    def add(self, episode: dict) -> None:
        masses = np.asarray([float(body.mass) for body in episode["physics"]],
                            np.float64)
        per_row_mass = masses[np.asarray(episode["row_slot"], np.int64)]
        values = episode["vertex_force"].astype(np.float64) \
            * per_row_mass[:, None, None, None]
        self.squared += float(np.square(values).sum())
        self.count += values.size
        self.wrenches.append(episode["contact_wrench"].copy())
        self.n_rows += int(episode["n_rows"])
        self.episodes += 1

    def finalize(self, solver_run_id: str) -> dict[str, Any]:
        from core.physics.scales import robust_wrench_scale
        import datetime
        if not self.episodes:
            raise ValueError("No episodes for label statistics")
        return {
            "point_force_rms_n": max(float(np.sqrt(self.squared / max(self.count, 1))), 1e-6),
            "wrench_scale": robust_wrench_scale(np.concatenate(self.wrenches)).tolist(),
            "n_rows": self.n_rows, "solver_run_id": solver_run_id, "episodes": self.episodes,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }


def compute_label_stats(episodes: Iterable[dict],
                        solver_run_id: str) -> dict[str, Any]:
    accumulator = LabelStatsAccumulator()
    for episode in episodes:
        accumulator.add(episode)
    return accumulator.finalize(solver_run_id)
