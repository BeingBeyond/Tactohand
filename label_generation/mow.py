#!/usr/bin/env python3
"""Allocate MOW single-frame gravity-balance wrenches without trajectory Stage 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data.mano_geometry import load_mano_world_geometry  # noqa: E402
from label_generation.adapters.part3_rigid_lerobot import load_rigid_episode  # noqa: E402
from core.physics.proximity import (  # noqa: E402
    contact_gated_force_candidates, signed_distance_candidate_mask,
    resolve_standard_contact_mask, signed_distance_contact_priority,
)
from core.physics.sequence_contact_qp import solve_sequence_contact_qp  # noqa: E402
from label_generation.io import atomic_save  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--view-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d-off-m", type=float, default=.020)
    args = parser.parse_args()

    data = load_rigid_episode(
        args.root, args.episode_index, view_id=args.view_id,
        minimum_aligned_frames=1,
    )
    if len(data["frame_index"]) != 1:
        raise ValueError("MOW static-gravity backend requires exactly one frame per episode")
    geometry = [load_mano_world_geometry(
        args.root, args.episode_index, data["view_id"], side=side,
        device=args.device,
    ) for side in ("left", "right")]
    points = np.concatenate([value[0] for value in geometry], axis=1)
    normals = np.concatenate([value[1] for value in geometry], axis=1)
    priorities, candidates, contacts = [], [], []
    for side in ("left", "right"):
        tactile = data["tactile"][side]
        signed = tactile["signed_distance_m"]
        valid = tactile["valid"] & np.isfinite(signed).all(axis=1) & (signed > -.5).all(axis=1)
        priorities.append(signed_distance_contact_priority(signed, valid=valid))
        candidates.append(signed_distance_candidate_mask(
            signed, d_off_m=args.d_off_m, valid=valid,
        ))
        contacts.append(resolve_standard_contact_mask(
            signed, valid=valid, stored_contact_mask=tactile["contact_mask"])[0])
    priority = np.concatenate(priorities, axis=1)
    candidate = np.concatenate(candidates, axis=1)
    contact = np.concatenate(contacts, axis=1)
    candidate = contact_gated_force_candidates(candidate, contact)

    gravity = np.asarray(data["physics"].gravity_world, np.float64)
    wrench = np.zeros((1, 6), np.float64)
    wrench[0, :3] = -gravity  # reference object mass is exactly 1 kg
    try:
        result = solve_sequence_contact_qp(
            wrench, data["x_com"], points, normals, priority, candidate,
            mu=1.2, friction_sides=6, temporal_regularizer=0.0,
            table_normal=-gravity / np.linalg.norm(gravity),
            table_contact_active=np.zeros(1, bool), allow_wrench_slack=False,
        )
        force = result["vertex_force_world"]
        valid = np.ones(1, bool)
        residual_f = result["force_residual_norm"]
        residual_tau = result["torque_residual_norm"]
        status, failure = "optimal", []
    except RuntimeError as error:
        force = np.zeros((1, 1556, 3), np.float64)
        valid = np.zeros(1, bool)
        residual_f = np.full(1, np.nan)
        residual_tau = np.full(1, np.nan)
        status = "partial"
        failure = [{"frame_interval": [0, 1], "error": str(error)}]

    atomic_save(
        args.output,
        format_version=np.int64(4), status=np.array(status),
        dataset=np.array(data["dataset"]), episode_index=np.int64(args.episode_index),
        object_name=np.array(data["object_name"]), object_index=np.int64(data["object_index"]),
        view_id=np.array(data["view_id"]), hand_side=np.array("both"),
        hand_sides=np.array(["left", "right"]), hand_vertex_count=np.int64(778),
        start=np.int64(0), stop=np.int64(1),
        normalization_json=np.array(json.dumps(data["mass_normalization"], sort_keys=True)),
        stage1_method=np.array("skipped_single_frame_static_gravity_v1"),
        wrench_source=np.array("static_gravity_balance_v1"),
        proximity_d_off_m=np.float64(args.d_off_m), force_temporal_regularizer=np.float64(0),
        contact_wrench_world=wrench, vertex_force_world=force,
        table_force_world=np.zeros((1, 3)), table_support_n=np.zeros(1),
        table_contact_active=np.zeros(1, bool), environment_clearance_m=np.full(1, np.nan),
        x_com_world=data["x_com"], force_residual_norm=residual_f,
        torque_residual_norm=residual_tau, contact_candidate_mask=candidate,
        signed_distance_vertices_m=np.concatenate([
            data["tactile"][side]["signed_distance_m"] for side in ("left", "right")
        ], axis=1),
        proximity_vertices=priority, hand_contact_mask=contact,
        hand_contact_object=np.where(contact, 0, -1).astype(np.int16),
        contact_candidate_object=np.where(candidate, 0, -1).astype(np.int16),
        contact_geometry_method=np.array("stored_source_signed_distance_v1"),
        force_label_valid=valid,
        force_label_source=np.array(["static_gravity_independent_qp"]),
        direct_table_support_mask=np.zeros(1, bool),
        invalid_reason_json=np.array(json.dumps({"strict_infeasible": failure}, sort_keys=True)),
        solver_backend=np.array("single_frame_direct_sequence_strict"),
        qp_timing_json=np.array("[]"),
    )
    print({"status": status, "episode": args.episode_index,
           "candidates": int(candidate.sum()), "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
