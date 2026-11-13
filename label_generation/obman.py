#!/usr/bin/env python3
"""Allocate independent OBMan frame forces from static gravity balance.

OBMan stores unrelated synthetic samples in long LeRobot episodes.  This
backend deliberately treats every row as a separate physical problem: there
is no trajectory Stage 1, environment support, or temporal regularization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_generation.adapters.dexycb_lerobot import _inertia_from_mesh, load_hand_tactile  # noqa: E402
from core.data.mano_geometry import load_mano_world_geometry  # noqa: E402
from label_generation.adapters.part3_rigid_lerobot import (  # noqa: E402
    REFERENCE_TOTAL_MASS_KG, _episode_gravity_directions, _episode_row, _mesh,
    _slice, eligible_observations,
)
from core.physics.proximity import (  # noqa: E402
    contact_gated_force_candidates, signed_distance_candidate_mask,
    resolve_standard_contact_mask, signed_distance_contact_priority,
)
from core.physics.sequence_contact_qp import solve_sequence_contact_qp  # noqa: E402


def atomic_save(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_frames(root: Path, episode_index: int, view_id: str | None,
                 start: int, stop: int | None) -> dict:
    observations = eligible_observations(root, episode_index)
    if observations.empty:
        raise ValueError(f"episode={episode_index} has no eligible observation")
    matches = observations if view_id is None else observations[
        observations.view_id.astype(str) == str(view_id)]
    if matches.empty:
        raise KeyError(f"view={view_id!r} is not eligible")
    observation = matches.iloc[0]
    hmr = _slice(root, observation, "hmr").sort_values("frame_index").reset_index(drop=True)
    objects = _slice(root, observation, "objects")
    if "view_id" in hmr:
        hmr = hmr[hmr.view_id.astype(str) == str(observation.view_id)].reset_index(drop=True)
    if "view_id" in objects:
        objects = objects[objects.view_id.astype(str) == str(observation.view_id)]
    stop = len(hmr) if stop is None else min(stop, len(hmr))
    if not 0 <= start < stop:
        raise ValueError(f"empty frame range [{start}, {stop}) for episode length {len(hmr)}")
    hmr = hmr.iloc[start:stop].reset_index(drop=True)
    frame_index = hmr.frame_index.to_numpy(np.int64)
    visible = objects[objects.visible.fillna(False)].copy()
    visible = visible[visible.frame_index.isin(frame_index)]
    counts = visible.groupby("frame_index").size()
    ambiguous = [int(i) for i in frame_index if int(counts.get(i, 0)) > 1]
    if ambiguous:
        raise ValueError(f"OBMan has multiple visible objects in frames {ambiguous[:8]}")
    by_frame = {int(row.frame_index): row for _, row in visible.iterrows()}
    object_valid = np.asarray([int(index) in by_frame for index in frame_index])

    instances = pq.read_table(root / "objects/instances.parquet").to_pandas()
    assets = pq.read_table(root / "objects/assets.parquet").to_pandas()
    instance_to_asset = dict(zip(instances.object_instance_id.astype(str),
                                 instances.object_asset_id.astype(str)))
    asset_rows = {str(row.object_asset_id): row for _, row in assets.iterrows()}
    c2w = np.stack([np.asarray(value).reshape(4, 4) for value in hmr["camera.c2w"]])
    com_world = np.full((len(hmr), 3), np.nan, np.float64)
    names, instance_ids = [], []
    for local, index in enumerate(frame_index):
        row = by_frame.get(int(index))
        if row is None:
            names.append(""); instance_ids.append("")
            continue
        instance_id = str(row.object_instance_id)
        asset = asset_rows[instance_to_asset[instance_id]]
        vertices, faces = _mesh(str(root.resolve()), str(asset.object_geometry_id))
        com, _ = _inertia_from_mesh(vertices, faces, REFERENCE_TOTAL_MASS_KG)
        o2w = c2w[local] @ np.asarray(row.pose_cam).reshape(4, 4)
        com_world[local] = o2w[:3, 3] + o2w[:3, :3] @ com
        names.append(str(asset["name"])); instance_ids.append(instance_id)
    episode = _episode_row(root, episode_index)
    gravity_direction = _episode_gravity_directions(root, episode, frame_index)
    tactile = {}
    for side in ("left", "right"):
        values = load_hand_tactile(root, episode_index, side=side)
        lookup = {int(value): i for i, value in enumerate(values["frame_index"])}
        missing = [int(value) for value in frame_index if int(value) not in lookup]
        if missing:
            raise ValueError(f"{side} tactile misses frames {missing[:8]}")
        selected = np.asarray([lookup[int(value)] for value in frame_index])
        tactile[side] = {key: (value[selected] if isinstance(value, np.ndarray)
                               and len(value) == len(lookup) else value)
                          for key, value in values.items()}
    return dict(view_id=str(observation.view_id), frame_index=frame_index,
                com_world=com_world, gravity_direction=gravity_direction,
                tactile=tactile, names=names, instance_ids=instance_ids,
                object_valid=object_valid,
                episode_length=int(_episode_row(root, episode_index)["length"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--view-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d-off-m", type=float, default=.020)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    args = parser.parse_args()
    if args.root.name != "obman_lerobot":
        parser.error("this backend is only for obman_lerobot")

    data = _load_frames(args.root, args.episode_index, args.view_id,
                        args.frame_start, args.frame_stop)
    T = len(data["frame_index"])
    stop = args.frame_start + T
    geometry = [load_mano_world_geometry(
        args.root, args.episode_index, data["view_id"], side=side,
        device=args.device, frame_start=args.frame_start, frame_stop=stop,
    ) for side in ("left", "right")]
    points = np.concatenate([value[0] for value in geometry], axis=1)
    normals = np.concatenate([value[1] for value in geometry], axis=1)
    priorities, candidates, contacts = [], [], []
    for side in ("left", "right"):
        tactile = data["tactile"][side]
        signed = tactile["signed_distance_m"]
        valid = tactile["valid"] & np.isfinite(signed).all(axis=1) & (signed > -.5).all(axis=1)
        priorities.append(signed_distance_contact_priority(signed, valid=valid))
        candidates.append(signed_distance_candidate_mask(signed, d_off_m=args.d_off_m, valid=valid))
        contacts.append(resolve_standard_contact_mask(
            signed, valid=valid, stored_contact_mask=tactile["contact_mask"])[0])
    priority = np.concatenate(priorities, axis=1)
    candidate = np.concatenate(candidates, axis=1)
    contact = np.concatenate(contacts, axis=1)
    candidate = contact_gated_force_candidates(candidate, contact)
    wrench = np.zeros((T, 6), np.float64)
    wrench[:, :3] = -data["gravity_direction"] * 9.80665
    force = np.zeros((T, 1556, 3), np.float32)
    valid = np.zeros(T, bool)
    residual_f = np.full(T, np.nan); residual_tau = np.full(T, np.nan)
    failures = [
        {"frame_index": int(index), "error": "missing_object_annotation"}
        for index, available in zip(data["frame_index"], data["object_valid"])
        if not available
    ]
    timings = []
    for frame in range(T):
        begin = time.perf_counter()
        if not data["object_valid"][frame]:
            timings.append(0.0)
            continue
        try:
            result = solve_sequence_contact_qp(
                wrench[frame:frame + 1], data["com_world"][frame:frame + 1],
                points[frame:frame + 1], normals[frame:frame + 1],
                priority[frame:frame + 1], candidate[frame:frame + 1],
                mu=1.2, friction_sides=6, temporal_regularizer=0.0,
                table_normal=-data["gravity_direction"][frame],
                table_contact_active=np.zeros(1, bool), allow_wrench_slack=False,
            )
            force[frame] = result["vertex_force_world"][0]
            residual_f[frame] = result["force_residual_norm"][0]
            residual_tau[frame] = result["torque_residual_norm"][0]
            valid[frame] = True
        except RuntimeError as error:
            failures.append({"frame_index": int(data["frame_index"][frame]), "error": str(error)})
        timings.append(time.perf_counter() - begin)
        if frame == 0 or (frame + 1) % 100 == 0:
            print({"episode": args.episode_index, "done": frame + 1, "total": T,
                   "valid": int(valid.sum())}, flush=True)

    status = "optimal" if valid.all() else "partial"
    atomic_save(
        args.output, format_version=np.int64(4), status=np.array(status),
        dataset=np.array(args.root.name), episode_index=np.int64(args.episode_index),
        object_name=np.array("per_frame_obman_object"), object_index=np.int64(-1),
        object_instance_id=np.asarray(data["instance_ids"]), object_name_per_frame=np.asarray(data["names"]),
        frame_index=data["frame_index"], view_id=np.array(data["view_id"]), hand_side=np.array("both"),
        hand_sides=np.array(["left", "right"]), hand_vertex_count=np.int64(778),
        start=np.int64(args.frame_start), stop=np.int64(stop), episode_length=np.int64(data["episode_length"]),
        normalization_json=np.array(json.dumps({"reference_total_mass_kg": 1.0}, sort_keys=True)),
        stage1_method=np.array("skipped_independent_frame_static_gravity_v1"),
        wrench_source=np.array("static_gravity_balance_v1"),
        force_allocation_policy=np.array("independent_frame_static_gravity_strict_qp_v1"),
        proximity_d_off_m=np.float64(args.d_off_m), force_temporal_regularizer=np.float64(0),
        contact_wrench_world=wrench, vertex_force_world=force,
        table_force_world=np.zeros((T, 3)), table_support_n=np.zeros(T),
        table_contact_active=np.zeros(T, bool), environment_clearance_m=np.full(T, np.nan),
        x_com_world=data["com_world"], force_residual_norm=residual_f,
        signed_distance_vertices_m=np.concatenate([
            data["tactile"][side]["signed_distance_m"] for side in ("left", "right")
        ], axis=1),
        proximity_vertices=priority, hand_contact_mask=contact,
        hand_contact_object=np.where(contact, 0, -1).astype(np.int16),
        contact_candidate_object=np.where(candidate, 0, -1).astype(np.int16),
        contact_geometry_method=np.array("stored_source_signed_distance_v1"),
        torque_residual_norm=residual_tau, contact_candidate_mask=candidate,
        force_label_valid=valid, force_label_source=np.full(T, "static_gravity_independent_qp"),
        direct_table_support_mask=np.zeros(T, bool),
        invalid_reason_json=np.array(json.dumps({"strict_infeasible": failures}, sort_keys=True)),
        solver_backend=np.array("independent_single_frame_direct_sequence_strict"),
        qp_timing_json=np.array(json.dumps(timings)),
    )
    print({"status": status, "episode": args.episode_index, "range": [args.frame_start, stop],
           "valid": int(valid.sum()), "total": T, "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
