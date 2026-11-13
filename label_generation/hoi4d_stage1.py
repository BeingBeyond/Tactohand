#!/usr/bin/env python3
"""Build generalized inverse-dynamics/contact geometry for one HOI4D episode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_generation.adapters.arctic_lerobot import NormalizedPartPhysics  # noqa: E402
from core.physics.articulated_gaussian_wrench import estimate_articulated_discrete_gaussian_wrench  # noqa: E402
from core.physics.articulated_torch import ArticulatedState, ArticulatedTorchBackend  # noqa: E402
from label_generation.adapters.hoi4d_articulated_lerobot import (  # noqa: E402
    load_hoi4d_articulated_episode,
)
from core.data.mano_geometry import load_mano_world_geometry  # noqa: E402
from label_generation.adapters.part3_rigid_lerobot import (  # noqa: E402
    HAND_INTERVAL_HEIGHT_POLICY, hand_interval_height_lifecycle,
)
from core.physics.proximity import (  # noqa: E402
    contact_gated_force_candidates, signed_distance_candidate_mask,
    resolve_standard_contact_mask, signed_distance_contact_priority,
)
from core.physics.rigid_multi_contact import (  # noqa: E402
    nearest_object_surface, normal_consistent_surface_mask,
)
from core.physics.trajectory_wrench import gaussian_radius_for_sigma
from label_generation.adapters.trajectory_policy import resolve_rigid_gaussian_sigmas


def atomic_save(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def environment_contact(data: dict, hand_active: np.ndarray) -> dict:
    poses = data["part_to_world"]; up = -data["gravity_direction_world"]
    T = len(poses); bottom = np.empty(T); point = np.empty((T, 3))
    for frame in range(T):
        vertices = np.concatenate([
            mesh[0] @ poses[frame, part, :3, :3].T + poses[frame, part, :3, 3]
            for part, mesh in enumerate(data["meshes"])
        ])
        height = vertices @ up[frame]
        bottom[frame] = height.min(); mask = height <= bottom[frame] + .002
        point[frame] = vertices[mask].mean(axis=0)
    lifecycle = hand_interval_height_lifecycle(bottom, hand_active)
    return {"active": lifecycle["active"], "point": point,
            "clearance": lifecycle["corrected_clearance_m"],
            "pickup_event": lifecycle["pickup_event"],
            "placement_event": lifecycle["placement_event"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-id", default="ego_view")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d-off-m", type=float, default=.02)
    args = parser.parse_args()
    data = load_hoi4d_articulated_episode(
        args.root, args.episode_index, view_id=args.view_id)
    bodies = [NormalizedPartPhysics(
        part_index=index, name=("base" if index == 0 else "child"), mass=item.mass,
        volume_m3=1.0, com_part=item.com_body, inertia_com_part=item.inertia_body,
        joint_type=item.joint_type, joint_axis_parent=item.joint_axis_parent,
        joint_origin_parent=item.joint_origin_parent,
    ) for index, item in enumerate(data["physics"])]
    if len(bodies) != 2 or bodies[1].joint_type != "revolute":
        raise ValueError("HOI4D v2 currently requires a two-link revolute asset")
    child_zero = np.asarray(data["part_to_root"], np.float64)[0, 1]
    sigma_pos_s, sigma_rot_s = resolve_rigid_gaussian_sigmas(args.root.name)
    radius_pos = gaussian_radius_for_sigma(data["fps"], sigma_pos_s)
    radius_rot = gaussian_radius_for_sigma(data["fps"], sigma_rot_s)
    dynamics = estimate_articulated_discrete_gaussian_wrench(
        data["part_to_world"][:, 0], data["joint_values"][:, 0], bodies, child_zero,
        np.arange(len(data["frame_index"]) + 1), fps=data["fps"],
        sigma_s=sigma_pos_s, sigma_rot_s=sigma_rot_s,
        radius=radius_pos, radius_rot=radius_rot,
        gravity_world=tuple(data["gravity_world"][0]), device=args.device)
    backend = ArticulatedTorchBackend(bodies, child_zero, fps=data["fps"],
                                     gravity_world=tuple(data["gravity_world"][0]), device=args.device)
    system_com = []; axes = []; pivots = []
    for position, rotation, angle, velocity in zip(
            dynamics["smoothed_base_position_world"][:-1], dynamics["smoothed_base_rotation_world"][:-1],
            dynamics["smoothed_joint_angle"][:-1], dynamics["generalized_velocity"]):
        geometry = backend.geometry(ArticulatedState(*[torch.as_tensor(x, dtype=torch.float64, device=args.device)
            for x in (position, rotation, angle, velocity)]))
        system_com.append(geometry["system_com_world"].detach().cpu().numpy())
        axes.append(geometry["axis_world"].detach().cpu().numpy())
        pivots.append(geometry["pivot_world"].detach().cpu().numpy())
    stage1 = {"target_generalized_wrench": dynamics["generalized_contact_wrench"],
              "system_com_world": np.asarray(system_com), "joint_part_indices": np.array([1]),
              "joint_axis_world": np.asarray(axes)[:, None], "joint_pivot_world": np.asarray(pivots)[:, None],
              "part_to_world_smooth": data["part_to_world"]}
    hands = [load_mano_world_geometry(
        args.root, args.episode_index, args.view_id, side=side, device=args.device)
        for side in ("left", "right")]
    points = np.stack([item[0] for item in hands], axis=1)
    normals = np.stack([item[1] for item in hands], axis=1)
    flat_points = points.reshape(len(points), -1, 3)
    flat_normals = normals.reshape(len(normals), -1, 3)
    query_parts = [{
        "dataset": args.root.name,
        "object_geometry_id": f"{data['object_instance_id']}:part:{part_id}",
        "mesh": mesh,
        "object_to_world": data["part_to_world"][:, part_id],
    } for part_id, mesh in enumerate(data["meshes"])]
    surface_distance, surface_owner, closest_surface = nearest_object_surface(
        flat_points, query_parts, device=args.device, return_closest=True)
    contact = normal_consistent_surface_mask(
        flat_points, flat_normals, surface_distance, closest_surface,
        threshold_m=.010, safe_distance_m=.001, angle_deg=90.).reshape(points.shape[:-1])
    candidate = normal_consistent_surface_mask(
        flat_points, flat_normals, surface_distance, closest_surface,
        threshold_m=args.d_off_m, safe_distance_m=.001,
        angle_deg=90.).reshape(points.shape[:-1])
    priority = signed_distance_contact_priority(surface_distance).reshape(points.shape[:-1])
    candidate = contact_gated_force_candidates(
        candidate.reshape(len(candidate), -1), contact.reshape(len(contact), -1),
    ).reshape(candidate.shape)
    distance = surface_distance.reshape(points.shape[:-1]).astype(np.float32)
    owner = surface_owner.reshape(points.shape[:-1])
    part = np.where(candidate, owner, -1).astype(np.int8)
    contact_part = np.where(contact, owner, -1).astype(np.int8)
    hand_active = contact.any(axis=(1, 2))
    environment = environment_contact(data, hand_active)
    atomic_save(
        args.output, format_version=np.int64(3),
        contact_geometry_method=np.array(
            "original_surface_multiray_inside_normal_filter_v1"),
        episode_index=np.int64(args.episode_index),
        fps=np.float64(data["fps"]),
        gaussian_sigma_s=np.float64(sigma_pos_s),
        gaussian_sigma_pos_s=np.float64(sigma_pos_s),
        gaussian_sigma_rot_s=np.float64(sigma_rot_s),
        gaussian_radius_pos_frames=np.int64(radius_pos),
        gaussian_radius_rot_frames=np.int64(radius_rot),
        object_instance_id=np.array(data["object_instance_id"]), frame_index=data["frame_index"],
        original_generalized_contact_wrench=stage1["target_generalized_wrench"],
        system_com_world=stage1["system_com_world"],
        joint_part_indices=stage1["joint_part_indices"],
        joint_axis_world=stage1["joint_axis_world"], joint_pivot_world=stage1["joint_pivot_world"],
        part_to_world_smooth=stage1["part_to_world_smooth"],
        points_world=points, normals_world=normals, contact_priority=priority,
        surface_distance_m=surface_distance.reshape(points.shape[:-1]).astype(np.float32),
        surface_owner=owner.astype(np.int8), contact_mask=contact,
        contact_candidate=candidate, contact_part=part, tactile_contact_part=contact_part,
        hand_contact_active=hand_active, environment_contact_active=environment["active"],
        environment_contact_policy=np.array(HAND_INTERVAL_HEIGHT_POLICY),
        environment_contact_world=environment["point"],
        environment_clearance_m=environment["clearance"],
        support_direction_world=-data["gravity_direction_world"],
        physics_json=np.array(json.dumps([
            {"mass": x.mass, "com_body": x.com_body.tolist(),
             "inertia_body": x.inertia_body.tolist(), "parent": x.parent,
             "joint_type": x.joint_type,
             "joint_axis_parent": x.joint_axis_parent.tolist(),
             "joint_origin_parent": x.joint_origin_parent.tolist()}
            for x in data["physics"]], sort_keys=True)),
    )
    print(json.dumps({"status": "ok", "episode_index": args.episode_index,
                      "parts": len(data["parts"]), "joints": len(stage1["joint_part_indices"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
