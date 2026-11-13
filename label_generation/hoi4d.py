#!/usr/bin/env python3
"""Allocate HOI4D multibody generalized wrench to MANO vertex forces."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_generation.adapters.hoi4d_articulated_lerobot import (  # noqa: E402
    LinkPhysics, contact_generalized_map,
)
from core.physics.multibody_contact_qp import solve_multibody_contact_qp  # noqa: E402


def atomic_save(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream: np.savez_compressed(stream, **values)
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--force-temporal-regularizer", type=float, default=1.0)
    args = parser.parse_args()
    if args.window_size < 1 or args.overlap < 0 or args.overlap >= args.window_size:
        parser.error("expected window-size>=1 and 0<=overlap<window-size")
    source = np.load(args.input); target = np.asarray(source["original_generalized_contact_wrench"])
    T, D = target.shape; J = D - 6
    physics = [LinkPhysics(
        mass=float(x["mass"]), com_body=np.asarray(x["com_body"]),
        inertia_body=np.asarray(x["inertia_body"]), parent=int(x["parent"]),
        joint_type=x["joint_type"], joint_axis_parent=np.asarray(x["joint_axis_parent"]),
        joint_origin_parent=np.asarray(x["joint_origin_parent"]),
    ) for x in json.loads(str(source["physics_json"]))]
    points = np.asarray(source["points_world"]); normals = np.asarray(source["normals_world"])
    priority = np.asarray(source["contact_priority"]); candidate = np.asarray(source["contact_candidate"])
    part = np.asarray(source["contact_part"]); hand = np.asarray(source["hand_contact_active"], bool)
    geometry_valid = (np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1)
                      & np.isfinite(priority) & (part >= 0))
    candidate &= geometry_valid
    force = np.zeros((T, 2, 778, 3)); environment_force = np.zeros((T, 3))
    environment_generalized = np.zeros((T, D)); fitted = np.zeros((T, D))
    internal = np.zeros((T, J)); valid = ~hand.copy(); failures = []
    # Hand-free frames are static/environment-balanced by contract.  Assign
    # their complete solved generalized wrench directly to the environment.
    environment_generalized[~hand] = target[~hand]; fitted[~hand] = target[~hand]
    edges = np.flatnonzero(np.diff(np.pad(hand.astype(np.int8), (1, 1))))

    def solve(begin: int, end: int, commit_begin: int, commit_end: int) -> None:
        local = slice(begin, end)
        try:
            result = solve_multibody_contact_qp(
                target=target[local], system_com=source["system_com_world"][local],
                joint_parts=source["joint_part_indices"], physics=physics,
                joint_axis=source["joint_axis_world"][local],
                joint_pivot=source["joint_pivot_world"][local], points=points[local],
                normals=normals[local], priority=priority[local], candidate=candidate[local],
                contact_part=part[local],
                environment_active=source["environment_contact_active"][local],
                support_direction=source["support_direction_world"][local],
                temporal_regularizer=args.force_temporal_regularizer,
            )
        except RuntimeError as error:
            if end - begin > 1:
                middle = begin + (end - begin) // 2
                solve(begin, middle, commit_begin, min(commit_end, middle))
                solve(middle, end, max(commit_begin, middle), commit_end)
            else:
                failures.append({"frame": begin, "error": str(error)})
            return
        if commit_end <= commit_begin:
            return
        result_slice = slice(commit_begin - begin, commit_end - begin)
        target_slice = slice(commit_begin, commit_end)
        force[target_slice] = result["vertex_force_world"][result_slice]
        environment_force[target_slice] = result["environment_force_world"][result_slice]
        internal[target_slice] = result["joint_internal_torque"][result_slice]
        fitted[target_slice] = result["fitted_generalized_wrench"][result_slice]
        environment_generalized[target_slice, :3] = result["environment_force_world"][result_slice]
        environment_generalized[target_slice, 6:] = result["joint_internal_torque"][result_slice]
        valid[target_slice] = True

    for begin, end in edges.reshape(-1, 2):
        step = args.window_size - args.overlap
        left_context = args.overlap // 2
        right_context = args.overlap - left_context
        for commit_begin in range(int(begin), int(end), step):
            commit_end = min(int(end), commit_begin + step)
            window_begin = max(int(begin), commit_begin - left_context)
            window_end = min(int(end), commit_end + right_context)
            solve(window_begin, window_end, commit_begin, commit_end)
    residual = fitted - target
    atomic_save(
        args.output, format_version=np.int64(1), status=np.array("ok" if valid.all() else "partial"),
        dataset=np.array("hoi4d_lerobot"), topology=np.array(f"articulated_{D}d"),
        start=np.int64(0), stop=np.int64(T), source_result_npz=np.array(str(args.input)),
        episode_index=source["episode_index"], object_instance_id=source["object_instance_id"],
        frame_index=source["frame_index"], original_generalized_contact_wrench=target,
        fitted_generalized_wrench=fitted, generalized_residual=residual,
        vertex_force_world=force.reshape(T, 1556, 3),
        environment_force_world=environment_force,
        environment_generalized_wrench=environment_generalized,
        joint_internal_torque=internal, force_label_valid=valid,
        hand_contact_active=hand,
        environment_contact_active=source["environment_contact_active"],
        environment_contact_policy=source["environment_contact_policy"],
        system_com_world=source["system_com_world"],
        part_to_world_smooth=source["part_to_world_smooth"],
        joint_part_indices=source["joint_part_indices"],
        joint_axis_world=source["joint_axis_world"], joint_pivot_world=source["joint_pivot_world"],
        contact_part=part, failures_json=np.array(json.dumps(failures, sort_keys=True)),
        source_npz=np.array(str(args.input)), window_size=np.int64(args.window_size),
        window_overlap=np.int64(args.overlap),
        gaussian_sigma_s=source["gaussian_sigma_s"],
        gaussian_sigma_pos_s=source["gaussian_sigma_pos_s"],
        gaussian_sigma_rot_s=source["gaussian_sigma_rot_s"],
    )
    maximum = float(np.max(np.abs(residual[valid]))) if valid.any() else float("nan")
    print(json.dumps({"status": "ok" if valid.all() else "partial",
                      "episode_index": int(source["episode_index"]),
                      "valid_frames": int(valid.sum()), "frames": T,
                      "residual_max": maximum, "failures": len(failures),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
