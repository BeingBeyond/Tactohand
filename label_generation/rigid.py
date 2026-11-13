#!/usr/bin/env python3
"""Solve a complete rigid episode while preserving every interacting object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data.mano_geometry import load_mano_world_geometry  # noqa: E402
from core.physics.environment_balance import direct_environment_balance  # noqa: E402
from core.physics.force_activation import coherent_force_activation  # noqa: E402
from label_generation.adapters.part3_rigid_lerobot import (  # noqa: E402
    HAND_INTERVAL_HEIGHT_POLICY, _episode_row, object_bottom_support_geometry,
    gaussian_observed_transition_mask, hocap_tabletop_lifecycle,
    hand_interval_height_lifecycle, hograspnet_completion_contract,
    load_rigid_episode_objects, should_use_stored_contact_gate,
)
from core.physics.proximity import (  # noqa: E402
    ABSENT_HAND_DISTANCE_M, contact_gated_force_candidates, signed_distance_contact_priority,
)
from core.physics.robust_sequence_contact import solve_per_frame_then_refine  # noqa: E402
from core.physics.rigid_multi_contact import (  # noqa: E402
    cached_object_trimesh, nearest_object_surface, normal_consistent_surface_mask,
)
from core.physics.sequence_contact_qp import solve_sequence_contact_qp  # noqa: E402
from core.physics.solver_timeline import (  # noqa: E402
    SOLVER_FPS, interpolate_linear, interpolate_poses, make_solver_timeline, nearest_source,
    transition_timeline_payload,
)
from core.physics.math3d import matrix_to_quat  # noqa: E402
from core.physics.trajectory_wrench import GAUSSIAN_HALF_WINDOW_S, estimate_rigid_discrete_gaussian_wrench, gaussian_radius_for_sigma
from label_generation.adapters.trajectory_policy import resolve_rigid_gaussian_sigmas
from label_generation.stage_pipeline import (  # noqa: E402
    acquire_cpu_stage2_lease, acquire_gpu_stage1_lease,
)
from label_generation.io import atomic_save  # noqa: E402


def solve_strict_interval_isolating_infeasible_frames(
    *, begin: int, end: int, wrench: np.ndarray, com: np.ndarray,
    points: np.ndarray, normals: np.ndarray, priority: np.ndarray,
    candidate: np.ndarray, table_normal: np.ndarray,
    table_contact_active: np.ndarray, temporal_regularizer: float,
    table_support_point_world: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, dict]], list[dict]]:
    """Bisect failed sequence QPs so one bad frame cannot invalidate a long run."""
    solved: list[tuple[int, int, dict]] = []
    failures: list[dict] = []

    def visit(start: int, stop: int) -> None:
        sl = slice(start, stop)
        try:
            result = solve_sequence_contact_qp(
                wrench[sl], com[sl], points[sl], normals[sl], priority[sl],
                candidate[sl], mu=1.2, friction_sides=6,
                temporal_regularizer=temporal_regularizer,
                table_normal=table_normal[sl] if table_normal.ndim == 2 else table_normal,
                table_contact_active=table_contact_active[sl],
                table_support_point_world=(None if table_support_point_world is None else
                                           table_support_point_world[sl]),
                table_temporal_regularizer=1., allow_wrench_slack=False,
            )
            solved.append((start, stop, result))
        except RuntimeError as error:
            if stop - start == 1:
                failures.append({"frame_interval": [int(start), int(stop)],
                                 "error": str(error)})
                return
            middle = start + (stop - start) // 2
            visit(start, middle)
            visit(middle, stop)

    visit(begin, end)
    return solved, failures


def assign_candidates_to_objects(
    points: np.ndarray, candidate: np.ndarray, objects: list[dict],
) -> np.ndarray:
    """Assign each active MANO vertex to its nearest object surface.

    Distance is evaluated in each object's local frame.  A nearest mesh-vertex
    query is used only for attribution; the canonical signed-distance value
    still controls whether the hand vertex is a valid contact candidate.
    """
    _, nearest = nearest_object_surface(points, objects)
    return np.where(candidate, nearest, -1).astype(np.int16)


def normal_consistent_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    distance: np.ndarray,
    owner: np.ndarray,
    objects: list[dict],
    *,
    closest_surface_world: np.ndarray | None = None,
    cutoff_m: float,
    safe_dist_m: float = 0.003,
    angle_deg: float = 90.0,
) -> np.ndarray:
    """Apply contact_generation's hand-normal filter to a relaxed candidate set.

    The fused geometry query supplies the nearest surface point so the normal
    test does not repeat a CPU trimesh closest-point pass.  The optional
    fallback preserves compatibility for callers that only have distances.
    Inside/near-surface vertices (<= safe_dist_m) are protected exactly like
    contact_generation.
    """
    if closest_surface_world is not None:
        return normal_consistent_surface_mask(
            points, normals, distance, closest_surface_world,
            threshold_m=cutoff_m, safe_distance_m=safe_dist_m,
            angle_deg=angle_deg,
        )
    points = np.asarray(points, np.float64)
    normals = np.asarray(normals, np.float64)
    distance = np.asarray(distance, np.float64)
    owner = np.asarray(owner, np.int16)
    closest_all = (None if closest_surface_world is None else
                   np.asarray(closest_surface_world, np.float64))
    if closest_all is not None and closest_all.shape != points.shape:
        raise ValueError("closest_surface_world must match points")
    keep = np.zeros(distance.shape, dtype=bool)
    cosine = float(np.cos(np.deg2rad(angle_deg)))
    for object_id, item in enumerate(objects):
        active = (owner == object_id) & (distance < float(cutoff_m))
        if not np.any(active):
            continue
        frame_ids, vertex_ids = np.where(active)
        if closest_all is not None:
            closest_world = closest_all[frame_ids, vertex_ids]
        else:
            mesh = cached_object_trimesh(item)
            poses = np.asarray(item["object_to_world"], np.float64)
            local = np.einsum(
                "ni,nij->nj", points[frame_ids, vertex_ids] - poses[frame_ids, :3, 3],
                poses[frame_ids, :3, :3],
            )
            closest, _, _ = trimesh.proximity.closest_point(mesh, local)
            closest_world = np.einsum(
                "nij,nj->ni", poses[frame_ids, :3, :3], closest,
            ) + poses[frame_ids, :3, 3]
        direction = closest_world - points[frame_ids, vertex_ids]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        dot = np.sum(normals[frame_ids, vertex_ids] * direction, axis=1)
        protected = distance[frame_ids, vertex_ids] < float(safe_dist_m)
        valid = protected | (dot >= cosine)
        keep[frame_ids[valid], vertex_ids[valid]] = True
    return keep


def main(argv: list[str] | None = None) -> None:
    wall_begin = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--view-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d-off-m", type=float, default=.020)
    parser.add_argument(
        "--contact-threshold-m", "--multi-object-contact-m",
        dest="contact_threshold_m", type=float, default=.010,
        help="Per hand-object contact onset; activates the relaxed force neighborhood",
    )
    parser.add_argument("--force-temporal-regularizer", type=float, default=1.)
    parser.add_argument(
        "--sequence-parts", type=int, default=1,
        help=("split the complete episode timeline into this many independent "
              "Stage-2 intervals; 1 preserves the original behavior"),
    )
    parser.add_argument("--robust-stage2", action="store_true",
                        help="use strict per-frame baseline plus bounded window refinement")
    parser.add_argument("--refine-window-size", type=int, default=64)
    parser.add_argument("--refine-overlap", type=int, default=16)
    parser.add_argument("--per-frame-time-limit-s", type=float, default=30.0)
    parser.add_argument("--window-time-limit-s", type=float, default=60.0)
    parser.add_argument(
        "--refinement-warm-start", action=argparse.BooleanOptionalAction, default=True,
        help="initialize refinement windows from their strict per-frame baseline",
    )
    parser.add_argument(
        "--normal-filter-candidates", action=argparse.BooleanOptionalAction, default=True,
        help="opt-in contact_generation-compatible normal filtering for 10/20 mm candidates",
    )
    parser.add_argument("--normal-filter-safe-dist-m", type=float, default=0.003)
    parser.add_argument("--normal-filter-angle-deg", type=float, default=90.0)
    parser.add_argument(
        "--coherent-force-activation", action=argparse.BooleanOptionalAction, default=None,
        help=("stabilize multi-object owner and require coherent MANO patches; "
              "enabled by default for OakInk-v2"),
    )
    parser.add_argument(
        "--stored-contact-gated-distance", action="store_true",
        help=("use the standard stored 10 mm contact mask to select frames for "
              "20 mm geometry queries; currently intended for DexYCB"),
    )
    parser.add_argument(
        "--gaussian-sigma-pos-s", type=float, default=None,
        help=("translation Gaussian bandwidth in seconds; default resolves from "
              "GAUSSIAN_SIGMA_POS_S then the per-dataset constant"),
    )
    parser.add_argument(
        "--gaussian-sigma-rot-s", type=float, default=None,
        help=("rotation Gaussian bandwidth in seconds; default resolves from "
              "GAUSSIAN_SIGMA_ROT_S then the per-dataset constant "
              "(0.15 s for dexycb_lerobot)"),
    )
    args = parser.parse_args(argv)
    if args.sequence_parts < 1:
        parser.error("--sequence-parts must be at least 1")
    stage1_begin = time.perf_counter()
    stage1_lease = acquire_gpu_stage1_lease()
    print(json.dumps({
        "event": "stage1_gpu_start", "episode": args.episode_index,
        "slot": None if stage1_lease.path is None else str(stage1_lease.path),
    }), flush=True)

    completion_contract = {
        "hand_completion_version": "",
        "object_completion_version": "",
    }
    if args.root.name == "hograspnet_lerobot":
        completion_contract = hograspnet_completion_contract(args.root)
    load_stored_tactile = should_use_stored_contact_gate(
        args.root.name, args.stored_contact_gated_distance)
    objects = load_rigid_episode_objects(
        args.root, args.episode_index, view_id=args.view_id,
        load_tactile=load_stored_tactile,
    )
    use_coherent_force_activation = (
        args.coherent_force_activation
        if args.coherent_force_activation is not None
        else args.root.name == "oakink-v2_lerobot"
    )
    data = objects[0]
    episode = _episode_row(args.root, args.episode_index)
    expected = np.arange(int(episode["length"]), dtype=np.int64)
    for item in objects:
        if not np.array_equal(item["frame_index"], expected):
            raise ValueError(f"incomplete track for {item['object_instance_id']}")
    timeline = make_solver_timeline(len(expected), data["fps"], solver_fps=SOLVER_FPS)
    solver_state_count = len(timeline.solver_timestamp_s)
    if solver_state_count < 2:
        raise ValueError("episode is too short for one 30 Hz solver transition")
    T = solver_state_count - 1
    source_objects = objects
    objects = []
    for source_item in source_objects:
        item = dict(source_item)
        pose = interpolate_poses(source_item["object_to_world"], timeline)
        gravity = interpolate_linear(source_item["gravity_world"], timeline)
        gravity_direction = interpolate_linear(
            source_item["gravity_direction_world"], timeline)
        gravity_direction /= np.linalg.norm(gravity_direction, axis=1, keepdims=True)
        rotation = pose[:, :3, :3]
        item["object_to_world"] = pose
        item["x_com"] = pose[:, :3, 3] + np.einsum(
            "tij,j->ti", rotation, np.asarray(source_item["physics"].com_body))
        item["quat_xyzw"] = matrix_to_quat(
            torch.as_tensor(rotation, dtype=torch.float64)).numpy()
        item["gravity_world"] = gravity
        item["gravity_direction_world"] = gravity_direction
        item["object_pose_observed"] = nearest_source(
            np.asarray(source_item.get(
                "object_pose_observed", np.ones(len(source_item["object_to_world"]), bool)
            ), bool),
            timeline,
        )
        item["fps"] = SOLVER_FPS
        objects.append(item)
    # Partition the canonical timeline, not individual contact runs.  Each
    # object's active intervals are intersected with these boundaries below.
    # This makes the split deterministic and identical across all objects.
    partition_boundaries = np.linspace(
        0, T, num=args.sequence_parts + 1, dtype=np.int64,
    )
    partition_boundaries = np.unique(partition_boundaries)
    gaussian_sigma_pos_s, gaussian_sigma_rot_s = resolve_rigid_gaussian_sigmas(
        data["dataset"],
        overrides=(args.gaussian_sigma_pos_s, args.gaussian_sigma_rot_s),
    )
    radius_pos = gaussian_radius_for_sigma(SOLVER_FPS, gaussian_sigma_pos_s)
    radius_rot = gaussian_radius_for_sigma(SOLVER_FPS, gaussian_sigma_rot_s)
    # Observation support must cover the wider of the two kernels.
    radius = max(radius_pos, radius_rot)
    print(json.dumps({
        "event": "smoothing_config", "episode": args.episode_index,
        "gaussian_sigma_pos_s": gaussian_sigma_pos_s,
        "gaussian_sigma_rot_s": gaussian_sigma_rot_s,
        "gaussian_radius_pos_frames": radius_pos,
        "gaussian_radius_rot_frames": radius_rot,
    }), flush=True)

    hand_geometry = [load_mano_world_geometry(
        args.root, args.episode_index, data["view_id"], side=side, device=args.device)
        for side in ("left", "right")]
    point_states = np.concatenate([
        interpolate_linear(item[0][expected], timeline) for item in hand_geometry
    ], axis=1)
    normal_states = np.concatenate([
        interpolate_linear(item[1][expected], timeline) for item in hand_geometry
    ], axis=1)
    normal_norm = np.linalg.norm(normal_states, axis=-1, keepdims=True)
    normal_states = np.divide(normal_states, normal_norm, out=np.zeros_like(normal_states),
                              where=normal_norm > 1e-12)
    points = point_states[:-1]
    normals = normal_states[:-1]
    source_hand_present = np.stack([
        np.linalg.norm(item[0], axis=-1).max(axis=1) > 0 for item in hand_geometry
    ], axis=1)
    hand_present = nearest_source(source_hand_present, timeline)[:-1]
    vertex_present = np.repeat(hand_present, 778, axis=1)
    transition_objects = []
    for item in objects:
        poses = item["object_to_world"][:-1].copy()
        pose_available = np.asarray(
            item.get("object_pose_observed", np.ones(len(item["object_to_world"]), bool)),
            bool,
        )[:-1]
        # A partial Show3D object track is interpolated only to preserve the
        # complete 30 Hz state axis. Never let its held endpoint participate
        # in a contact query outside the source-annotated frames.
        poses[~pose_available, :3, 3] = 1.0e6
        transition_objects.append({**item, "object_to_world": poses})
    stored_contact = None
    hybrid_geometry_contact = False
    # HOGraspNet's installed tactile stream predates its independent 30 Hz
    # MANO/object completions.  Reusing that sparse mask would silently gate
    # out contacts created by the completed trajectories.  Always recompute
    # its complete hand/object geometry; the CLI flag remains valid for the
    # datasets whose stored contact was authored on their current timeline.
    use_stored_contact_gated_distance = should_use_stored_contact_gate(
        data["dataset"], args.stored_contact_gated_distance)
    if use_stored_contact_gated_distance and data.get("tactile") is not None:
        stored_contact = np.concatenate([
            nearest_source(np.asarray(data["tactile"][side]["contact_mask"], bool), timeline)
            for side in ("left", "right")
        ], axis=1)[:-1]
        stored_signed = np.concatenate([
            interpolate_linear(
                np.asarray(data["tactile"][side]["signed_distance_m"], np.float64), timeline)
            for side in ("left", "right")
        ], axis=1)[:-1]
        stored_contact &= vertex_present
        hybrid_geometry_contact = data["dataset"] == "hocap_multiview_lerobot"
        # Stored contact masks already include the standard normal filtering.
        # HOCAP's legacy mask is known to omit contacts while switching
        # between scene objects. Query its complete timeline and take the union;
        # DexYCB's verified mask can safely gate the expensive geometry pass.
        active_frames = (np.ones(T, dtype=bool) if hybrid_geometry_contact else
                         stored_contact.reshape(T, 2, 778).any(axis=(1, 2)))
        surface_distance = stored_signed.copy()
        geometric_owner = np.full((T, points.shape[1]), -1, dtype=np.int16)
        if active_frames.any():
            query_objects = [
                {**item, "object_to_world": item["object_to_world"][active_frames]}
                for item in transition_objects
            ]
            queried_distance, queried_owner, queried_closest = nearest_object_surface(
                points[active_frames], query_objects, device=args.device,
                return_closest=True)
            surface_distance[active_frames] = queried_distance
            geometric_owner[active_frames] = queried_owner
            closest_surface_world = np.full_like(points, np.nan)
            closest_surface_world[active_frames] = queried_closest
        else:
            closest_surface_world = np.full_like(points, np.nan)
    else:
        surface_distance, geometric_owner, closest_surface_world = nearest_object_surface(
            points, transition_objects, device=args.device, return_closest=True)
    # Zero-filled missing MANO vertices can lie inside a scene object. Their
    # distance is undefined, so use the absent-hand fill before deriving labels.
    surface_distance[~vertex_present] = ABSENT_HAND_DISTANCE_M
    priority = signed_distance_contact_priority(surface_distance)
    candidate = surface_distance < args.d_off_m
    hand_contact = (
        stored_contact | (surface_distance <= args.contact_threshold_m)
        if hybrid_geometry_contact else
        stored_contact.copy() if stored_contact is not None else
        surface_distance <= args.contact_threshold_m
    )
    candidate &= vertex_present
    hand_contact &= vertex_present
    owner = np.where(candidate, geometric_owner, -1).astype(np.int16)
    contact_owner = np.where(hand_contact, geometric_owner, -1).astype(np.int16)
    if args.normal_filter_candidates:
        normal_keep = normal_consistent_candidates(
            points, normals, surface_distance, owner, objects,
            closest_surface_world=closest_surface_world,
            cutoff_m=args.d_off_m, safe_dist_m=args.normal_filter_safe_dist_m,
            angle_deg=args.normal_filter_angle_deg,
        )
        candidate &= normal_keep
        hand_contact &= normal_keep
        owner = np.where(candidate, geometric_owner, -1).astype(np.int16)
        contact_owner = np.where(hand_contact, geometric_owner, -1).astype(np.int16)
    # Scene-object tables also contain untouched distractors.  They must not
    # contribute their weight/support wrench to a hand-interaction label.
    touched = np.bincount(contact_owner[contact_owner >= 0], minlength=len(objects)) > 0
    if touched.any() and not touched.all():
        objects = [item for item, keep in zip(objects, touched) if keep]
        transition_objects = []
        for item in objects:
            poses = item["object_to_world"][:-1].copy()
            pose_available = np.asarray(
                item.get("object_pose_observed", np.ones(len(item["object_to_world"]), bool)),
                bool,
            )[:-1]
            poses[~pose_available, :3, 3] = 1.0e6
            transition_objects.append({**item, "object_to_world": poses})
        # The nearest distance must be recomputed over the retained object
        # set as well.  Reassigning only the owner would leave a candidate's
        # distance/priority inherited from a discarded distractor.
        if stored_contact is not None:
            active_frames = (np.ones(T, dtype=bool) if hybrid_geometry_contact else
                             stored_contact.reshape(T, 2, 778).any(axis=(1, 2)))
            geometric_owner.fill(-1)
            if active_frames.any():
                query_objects = [
                    {**item, "object_to_world": item["object_to_world"][active_frames]}
                    for item in transition_objects
                ]
                queried_distance, queried_owner, queried_closest = nearest_object_surface(
                    points[active_frames], query_objects, device=args.device,
                    return_closest=True)
                surface_distance[active_frames] = queried_distance
                geometric_owner[active_frames] = queried_owner
                closest_surface_world[active_frames] = queried_closest
        else:
            surface_distance, geometric_owner, closest_surface_world = nearest_object_surface(
                points, transition_objects, device=args.device, return_closest=True)
        surface_distance[~vertex_present] = ABSENT_HAND_DISTANCE_M
        priority = signed_distance_contact_priority(surface_distance)
        candidate = surface_distance < args.d_off_m
        hand_contact = (
            stored_contact | (surface_distance <= args.contact_threshold_m)
            if hybrid_geometry_contact else
            stored_contact.copy() if stored_contact is not None else
            surface_distance <= args.contact_threshold_m
        )
        candidate &= vertex_present
        hand_contact &= vertex_present
        owner = np.where(candidate, geometric_owner, -1).astype(np.int16)
        contact_owner = np.where(hand_contact, geometric_owner, -1).astype(np.int16)
        if args.normal_filter_candidates:
            normal_keep = normal_consistent_candidates(
                points, normals, surface_distance, owner, objects,
                closest_surface_world=closest_surface_world,
                cutoff_m=args.d_off_m, safe_dist_m=args.normal_filter_safe_dist_m,
                angle_deg=args.normal_filter_angle_deg,
            )
            candidate &= normal_keep
            hand_contact &= normal_keep
            owner = np.where(candidate, geometric_owner, -1).astype(np.int16)
            contact_owner = np.where(hand_contact, geometric_owner, -1).astype(np.int16)

    geometric_contact_owner = contact_owner.copy()
    activation = None
    if use_coherent_force_activation:
        activation = coherent_force_activation(
            hand_contact, contact_owner, surface_distance, hand_geometry[0][2], len(objects))
        contact_owner = np.where(
            hand_contact, activation["stabilized_owner"], -1).astype(np.int16)
        # Every current-contact vertex must use the same stabilized owner in
        # the wider force neighborhood. Non-contact proximity vertices retain
        # their nearest geometric owner.
        owner = np.where(hand_contact, contact_owner, owner).astype(np.int16)

    # Inside/safe-distance contact semantics do not require a reliable hand
    # normal, but a friction-cone force variable does.  Keep such vertices in
    # the contact annotation while excluding them from force allocation.
    force_normal_valid = (
        np.isfinite(normals).all(axis=-1)
        & (np.linalg.norm(normals, axis=-1) >= 1.0e-8)
    )
    candidate &= force_normal_valid
    owner = np.where(candidate, owner, -1).astype(np.int16)

    # Geometry, signed proximity, contact, normal filtering, and force
    # candidate ownership are complete. Release the scarce GPU Stage-1 slot
    # before entering inverse dynamics and CPU QP allocation.
    stage1_lease.release()
    print(json.dumps({
        "event": "stage1_gpu_end", "episode": args.episode_index,
        "wall_s": time.perf_counter() - stage1_begin,
    }), flush=True)
    stage2_lease = acquire_cpu_stage2_lease()
    print(json.dumps({
        "event": "stage2_cpu_start", "episode": args.episode_index,
        "slot": None if stage2_lease.path is None else str(stage2_lease.path),
    }), flush=True)

    object_wrench = np.full((T, len(objects), 6), np.nan)
    object_environment_wrench = np.zeros((T, len(objects), 6))
    object_fitted_wrench = np.zeros((T, len(objects), 6))
    object_hand_active = np.zeros((T, len(objects)), bool)
    object_environment_active = np.zeros((T, len(objects)), bool)
    object_hocap_release_event = np.zeros((T, len(objects)), bool)
    object_placement_clearance_offset = np.zeros((T, len(objects)), np.float64)
    object_missing_contact_source = np.zeros((T, len(objects)), bool)
    object_direct_environment_balance = np.zeros((T, len(objects)), bool)
    object_com = np.full((T, len(objects), 3), np.nan)
    object_pose = np.full((T, len(objects), 4, 4), np.nan)
    object_position_states = np.full((T + 1, len(objects), 3), np.nan)
    object_rotation_states = np.full((T + 1, len(objects), 3, 3), np.nan)
    object_velocity_before = np.full((T, len(objects), 6), np.nan)
    object_velocity_after = np.full((T, len(objects), 6), np.nan)
    object_gravity_world = np.full((T, len(objects), 3), np.nan)
    force = np.zeros((T, 1556, 3))
    # Persist the actual force allocation neighborhood, not the raw 20 mm
    # proximity mask.  A relaxed candidate is enabled only for a hand/object
    # block that has at least one canonical 10 mm contact vertex.
    force_candidate_mask = np.zeros_like(candidate, dtype=bool)
    force_candidate_owner = np.full_like(owner, -1, dtype=np.int16)
    support = np.zeros((T, 3))
    support_n = np.zeros(T)
    residual_f = np.zeros(T)
    residual_tau = np.zeros(T)
    all_valid = np.ones(T, bool)
    refined_any = np.zeros(T, bool)
    environment_active = np.zeros(T, bool)
    clearance = np.full((T, len(objects)), np.nan)
    failures = []
    direct_table_support = np.zeros(T, bool)

    for object_id, item in enumerate(objects):
        stage1 = estimate_rigid_discrete_gaussian_wrench(
            item["x_com"], item["quat_xyzw"], item["physics"], np.arange(T + 1),
            fps=SOLVER_FPS, sigma_s=gaussian_sigma_pos_s, radius=radius_pos,
            sigma_rot_s=gaussian_sigma_rot_s, radius_rot=radius_rot,
            gravity_world=item["gravity_world"][:-1])
        wrench = stage1["contact_wrench_world"]
        com = stage1["smoothed_position_world"][:-1]
        pose = item["object_to_world"][:-1].copy()
        pose[:, :3, :3] = stage1["smoothed_rotation_world"][:-1]
        pose[:, :3, 3] = com - np.einsum(
            "tij,j->ti", pose[:, :3, :3], np.asarray(item["physics"].com_body))
        object_wrench[:, object_id] = wrench
        object_com[:, object_id] = com
        object_pose[:, object_id] = pose
        object_position_states[:, object_id] = stage1["smoothed_position_world"]
        object_rotation_states[:, object_id] = stage1["smoothed_rotation_world"]
        object_velocity_before[:, object_id] = stage1["generalized_velocity"]
        object_velocity_after[:, object_id] = stage1["next_generalized_velocity"]
        object_gravity_world[:, object_id] = item["gravity_world"][:-1]
        pose_observed = np.asarray(
            item.get("object_pose_observed", np.ones(T + 1, bool)), bool)
        if pose_observed.shape != (T + 1,):
            raise ValueError("object_pose_observed must follow the solver state timeline")
        # Gaussian inverse dynamics at transition t uses a neighborhood around
        # both boundary states. Mark a transition usable only when the entire
        # derivative support is source-observed, avoiding interpolation-edge
        # impulses in partially annotated Show3D tracks.
        dynamics_valid = gaussian_observed_transition_mask(pose_observed, radius)
        object_candidate = candidate & (owner == object_id) & dynamics_valid[:, None]
        object_contact = hand_contact & (contact_owner == object_id)
        object_contact &= dynamics_valid[:, None]
        force_contact = object_contact.copy()
        if activation is not None:
            force_contact[:, :778] &= activation["force_active"][:, object_id, 0, None]
            force_contact[:, 778:] &= activation["force_active"][:, object_id, 1, None]
        object_candidate = contact_gated_force_candidates(object_candidate, force_contact)
        force_candidate_mask |= object_candidate
        force_candidate_owner = np.where(object_candidate, object_id, force_candidate_owner)
        support_geometry = object_bottom_support_geometry(
            item["object_to_world"][:-1], item["mesh"][0],
            item["gravity_direction_world"][:-1])
        gravity = np.asarray(item["gravity_world"][:-1])
        table_normal = -gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
        # Contact/table semantics use raw object geometry and raw canonical
        # hand contact. Gaussian dynamics validity is applied only afterwards.
        lifecycle = hand_interval_height_lifecycle(
            support_geometry["bottom_height_along_up_m"], object_contact.any(axis=1))
        hand_active = lifecycle["stabilized_hand_active"]
        clearance[:, object_id] = lifecycle["corrected_clearance_m"]
        object_hocap_release_event[:, object_id] = lifecycle["placement_event"]
        object_hand_active[:, object_id] = hand_active
        # A hand-free object is supported by the environment by contract.  In
        # those frames the environment owns the complete balancing wrench;
        # there is no hand-force allocation problem to solve.  Keep the
        # detected table lifecycle for hand-contact boundary frames, where the
        # QP must decide the hand/environment split.
        # Lifecycle contact is deliberately tolerant to short annotation gaps,
        # but an empty force neighborhood has no QP variables.  Preserve the
        # lifecycle/table state while assigning those frames directly to the
        # environment, exactly like any other frame without usable hand force.
        allocation_hand_active = hand_active & object_candidate.any(axis=1)
        direct_balance = direct_environment_balance(
            wrench, dynamics_valid, allocation_hand_active)
        direct_environment = direct_balance["active"]
        object_direct_environment_balance[:, object_id] = direct_environment
        table_active = ((np.asarray(lifecycle["active"], bool) | ~hand_active)
                        & dynamics_valid)
        object_environment_active[:, object_id] = table_active
        # Dataset contract: an interacting object cannot be simultaneously
        # off-table and out of hand contact.  Such a frame indicates missing
        # or misaligned contact evidence, not valid free flight.
        missing_source = ~hand_active & ~table_active & dynamics_valid
        object_missing_contact_source[:, object_id] = missing_source
        direct_table_support |= direct_environment
        # Transitions outside this object's observed dynamics support are
        # neutral in the multi-object validity intersection.  Hand-free
        # transitions are exactly balanced by the environment below.
        local_valid = direct_balance["valid"].copy()
        local_support = direct_balance["environment_force"].copy()
        object_environment_wrench[:, object_id] = direct_balance["environment_wrench"]
        object_fitted_wrench[:, object_id] = direct_balance["environment_wrench"]

        # QP allocation is needed only while hand contact is active, including
        # the lifecycle-stabilized contact boundaries.  The environment may
        # remain active there so the QP can split support between hand/table.
        environment_active |= table_active
        optimize = direct_balance["qp_active"]
        if args.robust_stage2:
            def log_event(event: dict) -> None:
                print(json.dumps({
                    "episode": args.episode_index,
                    "object_instance_id": item["object_instance_id"],
                    **event,
                }), flush=True)

            robust = solve_per_frame_then_refine(
                wrench=wrench, com=com, points=points, normals=normals,
                priority=priority, candidate=object_candidate,
                table_normal=table_normal,
                table_contact_active=lifecycle["active"], active=optimize,
                table_support_point_world=support_geometry["support_point_world"],
                temporal_regularizer=args.force_temporal_regularizer,
                window_size=args.refine_window_size, overlap=args.refine_overlap,
                per_frame_time_limit_s=args.per_frame_time_limit_s,
                window_time_limit_s=args.window_time_limit_s,
                refinement_warm_start=args.refinement_warm_start,
                log=log_event,
            )
            force += robust["vertex_force_world"]
            local_support[optimize] = robust["table_force_world"][optimize]
            object_environment_wrench[optimize, object_id, :3] = robust["table_force_world"][optimize]
            object_environment_wrench[optimize, object_id, 3:] = np.cross(
                support_geometry["support_point_world"][optimize] - com[optimize],
                robust["table_force_world"][optimize])
            object_fitted_wrench[optimize, object_id] = robust["fitted_contact_wrench"][optimize]
            residual_f[optimize] += robust["force_residual_norm"][optimize]
            residual_tau[optimize] += robust["torque_residual_norm"][optimize]
            local_valid[optimize] = robust["force_label_valid"][optimize]
            refined_any |= robust["refined_frame_mask"]
            failures.extend({"object_instance_id": item["object_instance_id"], **failure}
                            for failure in robust["failures"])
        else:
            edges = np.flatnonzero(np.diff(np.pad(optimize.astype(np.int8), (1, 1))))
            for begin, end in edges.reshape(-1, 2):
                cuts = np.concatenate((
                    np.array([begin], dtype=np.int64),
                    partition_boundaries[(partition_boundaries > begin) &
                                         (partition_boundaries < end)],
                    np.array([end], dtype=np.int64),
                ))
                for part_begin, part_end in zip(cuts[:-1], cuts[1:]):
                    print(json.dumps({
                    "event": "stage2_interval_start",
                    "episode": args.episode_index,
                    "object_instance_id": item["object_instance_id"],
                    "begin": int(part_begin),
                    "end": int(part_end),
                    "frames": int(part_end - part_begin),
                    "candidates": int(object_candidate[part_begin:part_end].sum()),
                    "sequence_parts": int(args.sequence_parts),
                    }), flush=True)
                    interval_begin = time.perf_counter()
                    solved, local_failures = solve_strict_interval_isolating_infeasible_frames(
                    begin=int(part_begin), end=int(part_end), wrench=wrench, com=com,
                    points=points, normals=normals, priority=priority,
                    candidate=object_candidate, table_normal=table_normal,
                    table_contact_active=lifecycle["active"],
                    table_support_point_world=support_geometry["support_point_world"],
                    temporal_regularizer=args.force_temporal_regularizer,
                    )
                    print(json.dumps({
                    "event": "stage2_interval_end",
                    "episode": args.episode_index,
                    "object_instance_id": item["object_instance_id"],
                    "begin": int(part_begin),
                    "end": int(part_end),
                    "solved_subintervals": len(solved),
                    "failed_frames": len(local_failures),
                    "wall_s": time.perf_counter() - interval_begin,
                    }), flush=True)
                    for local_begin, local_end, result in solved:
                        sl = slice(local_begin, local_end)
                        force[sl] += result["vertex_force_world"]
                        local_support[sl] = result["table_force_world"]
                        object_environment_wrench[sl, object_id, :3] = result["table_force_world"]
                        object_environment_wrench[sl, object_id, 3:] = np.cross(
                            support_geometry["support_point_world"][sl] - com[sl],
                            result["table_force_world"])
                        object_fitted_wrench[sl, object_id] = result["fitted_contact_wrench"]
                        residual_f[sl] += result["force_residual_norm"]
                        residual_tau[sl] += result["torque_residual_norm"]
                        local_valid[sl] = True
                    failures.extend({"object_instance_id": item["object_instance_id"], **failure}
                                    for failure in local_failures)
        support += local_support
        support_n += np.einsum("ti,ti->t", local_support, table_normal)
        all_valid &= local_valid

    system_com = object_com.mean(axis=1)
    total_wrench = np.zeros((T, 6))
    total_wrench[:, :3] = object_wrench[:, :, :3].sum(axis=1)
    total_wrench[:, 3:] = (object_wrench[:, :, 3:] + np.cross(
        object_com - system_com[:, None], object_wrench[:, :, :3])).sum(axis=1)
    total_environment_wrench = np.zeros((T, 6))
    total_environment_wrench[:, :3] = object_environment_wrench[:, :, :3].sum(axis=1)
    total_environment_wrench[:, 3:] = (object_environment_wrench[:, :, 3:] + np.cross(
        object_com - system_com[:, None], object_environment_wrench[:, :, :3])).sum(axis=1)
    total_fitted_wrench = np.zeros((T, 6))
    total_fitted_wrench[:, :3] = object_fitted_wrench[:, :, :3].sum(axis=1)
    total_fitted_wrench[:, 3:] = (object_fitted_wrench[:, :, 3:] + np.cross(
        object_com - system_com[:, None], object_fitted_wrench[:, :, :3])).sum(axis=1)
    atomic_save(
        args.output, format_version=np.int64(3),
        force_allocation_policy=np.array(
            "hand_contact_qp_direct_environment_balance_v5"),
        environment_contact_policy=np.array(
            HAND_INTERVAL_HEIGHT_POLICY),
        hand_completion_version=np.array(
            completion_contract["hand_completion_version"]),
        object_completion_version=np.array(
            completion_contract["object_completion_version"]),
        status=np.array("partial" if failures or not all_valid.all() else "optimal"),
        dataset=np.array(data["dataset"]), episode_index=np.int64(args.episode_index),
        object_name=np.array("multi-object"), object_index=np.int64(-1),
        object_names=np.asarray([item["object_name"] for item in objects]),
        object_instance_ids=np.asarray([item["object_instance_id"] for item in objects]),
        view_id=np.array(data["view_id"]), hand_side=np.array("both"),
        hand_sides=np.array(["left", "right"]), hand_vertex_count=np.int64(778),
        hand_present=hand_present,
        start=np.int64(0), stop=np.int64(T),
        normalization_json=np.array(json.dumps(data["mass_normalization"], sort_keys=True)),
        gaussian_sigma_s=np.float64(gaussian_sigma_pos_s),
        gaussian_sigma_pos_s=np.float64(gaussian_sigma_pos_s),
        gaussian_sigma_rot_s=np.float64(gaussian_sigma_rot_s),
        gaussian_half_window_s=np.float64(GAUSSIAN_HALF_WINDOW_S),
        gaussian_radius_frames=np.int64(radius),
        gaussian_radius_pos_frames=np.int64(radius_pos),
        gaussian_radius_rot_frames=np.int64(radius_rot),
        proximity_d_off_m=np.float64(args.d_off_m),
        contact_threshold_m=np.float64(args.contact_threshold_m),
        contact_geometry_method=np.array(
            "original_surface_multiray_inside_normal_filter_v1"),
        force_candidate_gate=np.array("per_frame_hand_object_contact_required_v1"),
        contact_mask_source=np.array(
            "stored_or_signed_distance_10mm_multi_object_v1"
            if hybrid_geometry_contact else
            "stored_contact_mask_10mm_gated_geometry_v1"
            if stored_contact is not None else "signed_distance_10mm_online_geometry"),
        stored_contact_gated_distance_requested=np.array(
            bool(args.stored_contact_gated_distance)),
        stored_contact_gated_distance_applied=np.array(
            bool(use_stored_contact_gated_distance)),
        force_temporal_regularizer=np.float64(args.force_temporal_regularizer),
        sequence_parts=np.int64(args.sequence_parts),
        sequence_partition_boundaries=partition_boundaries,
        sequence_partition_policy=np.array(
            "none" if args.sequence_parts == 1 else "canonical_timeline_equal_parts_v1"),
        stage2_solve_policy=np.array(
            "strict_per_frame_baseline_bounded_window_refine_v1"
            if args.robust_stage2 else "strict_partitioned_sequence_v1"),
        refine_window_size=np.int64(args.refine_window_size if args.robust_stage2 else 0),
        refine_overlap=np.int64(args.refine_overlap if args.robust_stage2 else 0),
        per_frame_time_limit_s=np.float64(args.per_frame_time_limit_s if args.robust_stage2 else 0),
        window_time_limit_s=np.float64(args.window_time_limit_s if args.robust_stage2 else 0),
        refined_frame_mask=refined_any,
        contact_wrench_world=total_wrench, object_contact_wrench_world=object_wrench,
        fitted_contact_wrench_world=total_fitted_wrench,
        object_fitted_contact_wrench_world=object_fitted_wrench,
        environment_wrench_world=total_environment_wrench,
        object_environment_wrench_world=object_environment_wrench,
        vertex_force_world=force, table_force_world=support, table_support_n=support_n,
        table_contact_active=environment_active, environment_clearance_m=clearance.min(axis=1),
        object_environment_clearance_m=clearance, x_com_world=system_com,
        object_x_com_world=object_com, smoothed_object_to_world=object_pose,
        object_state_position_world=object_position_states,
        object_state_rotation_world=object_rotation_states,
        object_velocity_before_world=object_velocity_before,
        object_velocity_after_world=object_velocity_after,
        object_gravity_world=object_gravity_world,
        force_residual_norm=residual_f, torque_residual_norm=residual_tau,
        signed_distance_vertices_m=surface_distance,
        proximity_vertices=signed_distance_contact_priority(surface_distance),
        contact_candidate_mask=force_candidate_mask,
        contact_candidate_object=force_candidate_owner,
        hand_contact_mask=hand_contact, hand_contact_object=contact_owner,
        geometric_hand_contact_object=geometric_contact_owner,
        force_object_active=(activation["force_active"] if activation is not None else
                             np.stack([
                                 np.stack([
                                     (hand_contact[:, :778] & (contact_owner[:, :778] == oid)).any(axis=1),
                                     (hand_contact[:, 778:] & (contact_owner[:, 778:] == oid)).any(axis=1),
                                 ], axis=1) for oid in range(len(objects))
                             ], axis=1)),
        force_contact_largest_patch_vertices=(
            activation["largest_patch_vertices"] if activation is not None
            else np.zeros((T, len(objects), 2), dtype=np.int16)),
        force_activation_policy=np.array(
            "mano_connected_patch_hysteresis_v1" if activation is not None else "any_contact_v1"),
        refinement_warm_start=np.array(bool(args.refinement_warm_start)),
        object_hand_contact_active=object_hand_active,
        object_environment_contact_active=object_environment_active,
        object_hocap_release_event=object_hocap_release_event,
        object_placement_clearance_offset_m=object_placement_clearance_offset,
        object_missing_contact_source=object_missing_contact_source,
        object_direct_environment_balance=object_direct_environment_balance,
        force_label_valid=all_valid,
        force_label_source=np.where(
            object_direct_environment_balance.all(axis=1),
            "direct_environment_balance",
            np.where(object_direct_environment_balance.any(axis=1),
                     "hybrid_environment_hand_qp", "hand_contact_strict_qp"),
        ).astype("<U32"),
        direct_table_support_mask=direct_table_support,
        missing_contact_source_mask=object_missing_contact_source.any(axis=1),
        static_environment_balance_mask=direct_table_support,
        invalid_reason_json=np.array(json.dumps({"strict_infeasible": failures}, sort_keys=True)),
        solver_backend=np.array("multi_object_direct_sequence_strict"),
        qp_timing_json=np.array("[]"),
        **transition_timeline_payload(timeline))
    stage2_lease.release()
    print(json.dumps({"episode": args.episode_index, "objects": len(objects),
                      "frames": T, "valid": int(all_valid.sum()),
                      "failures": len(failures), "wall_s": time.perf_counter() - wall_begin,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
