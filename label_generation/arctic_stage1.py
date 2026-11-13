"""Build geometry-normalized ARCTIC articulated sidecars for small smokes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_generation.adapters.arctic_lerobot import (  # noqa: E402
    detect_environment_contact, load_arctic_episode,
)
from core.physics.articulated_contact_geometry import articulated_contact_geometry  # noqa: E402
from core.physics.articulated_gaussian_wrench import (  # noqa: E402
    estimate_articulated_discrete_gaussian_wrench,
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
from core.physics.trajectory_wrench import GAUSSIAN_HALF_WINDOW_S, gaussian_radius_for_sigma
from label_generation.adapters.trajectory_policy import resolve_rigid_gaussian_sigmas

TACTILE_CONTACT_MIN_VERTICES = 5


def atomic_savez(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def valid_stage1(path: Path) -> bool:
    try:
        with np.load(path) as z:
            expected_pos, expected_rot = resolve_rigid_gaussian_sigmas("arctic_lerobot")
            return (str(z["stage1_status"])=="ok" and int(z["format_version"])==14
                    and str(z["environment_detector"])==HAND_INTERVAL_HEIGHT_POLICY
                    and str(z["stage1_method"])=="articulated_newton_compatible_inverse_dynamics_v2"
                    and "gaussian_sigma_pos_s" in z.files
                    and "gaussian_sigma_rot_s" in z.files
                    and np.isclose(float(z["gaussian_sigma_pos_s"]), expected_pos,
                                   rtol=0.0, atol=1e-12)
                    and np.isclose(float(z["gaussian_sigma_rot_s"]), expected_rot,
                                   rtol=0.0, atol=1e-12))
    except (OSError,KeyError,ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", type=Path, default=Path(
        "data/beingh_pretrain/egocentric_data/_processed/arctic_lerobot"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/label_solver/arctic_normalized_smoke"))
    parser.add_argument("--num-shards",type=int,default=1)
    parser.add_argument("--shard-id",type=int,default=0)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    if args.num_shards<1 or not 0<=args.shard_id<args.num_shards:
        parser.error("expected num_shards>=1 and 0<=shard_id<num_shards")
    episodes=args.episodes
    if episodes is None:
        episodes=sorted(map(int,pq.read_table(args.root/"meta/episodes/chunk-000/file-000.parquet",
                                             columns=["episode_index"])["episode_index"].to_pylist()))
    if args.limit is not None:episodes=episodes[:args.limit]
    episodes=[episode for index,episode in enumerate(episodes) if index%args.num_shards==args.shard_id]
    for episode in episodes:
        path = args.output_dir / f"episode_{episode:06d}.npz"
        if not args.overwrite and valid_stage1(path):
            print({"status":"skipped","episode":episode,"output":str(path)},flush=True);continue
        data = load_arctic_episode(args.root, episode)
        hand_geometry = {
            side: load_mano_world_geometry(
                args.root, episode, data["view_id"], side=side, device=args.device)
            for side in ("left", "right")
        }
        all_points = np.concatenate([hand_geometry[side][0] for side in ("left", "right")], axis=1)
        all_normals = np.concatenate([hand_geometry[side][1] for side in ("left", "right")], axis=1)
        query_parts = [{
            "dataset": args.root.name,
            "object_geometry_id": f"{data['object_name']}:part:{part_id}",
            "mesh": mesh,
            "object_to_world": data["part_to_world"][:, part_id],
        } for part_id, mesh in enumerate(data["part_meshes"])]
        full_distance, full_owner, full_closest = nearest_object_surface(
            all_points, query_parts, device=args.device, return_closest=True)
        full_contact = normal_consistent_surface_mask(
            all_points, all_normals, full_distance, full_closest,
            threshold_m=.010, safe_distance_m=.001, angle_deg=90.)
        speed = np.abs(np.gradient(np.unwrap(data["joint_angle"])))
        canonical_contact = {
            "left": full_contact[:, :778], "right": full_contact[:, 778:],
        }
        simultaneous = ((canonical_contact["left"].sum(1) >= 5)
                        & (canonical_contact["right"].sum(1) >= 5))
        score = speed / max(float(speed.max()), 1e-12) + simultaneous.astype(float)
        interval_count=len(score)-1
        if args.window <= 0 or args.window >= interval_count:
            start, stop = 0, interval_count
        else:
            center = int(np.argmax(score)); start = max(0, min(center - args.window // 2,
                                                               interval_count - args.window))
            stop = start + args.window
        sl = slice(start, stop)
        geometry = articulated_contact_geometry(data["part_to_world"], data["part_physics"])
        axis=np.asarray(data["part_physics"][1].joint_axis_parent,np.float64)
        pivot=np.asarray(data["part_physics"][1].joint_origin_parent,np.float64)
        q0=float(data["joint_angle"][start])
        rotation=Rotation.from_rotvec(axis*q0).as_matrix()
        joint_transform=np.eye(4);joint_transform[:3,:3]=rotation
        joint_transform[:3,3]=pivot-rotation@pivot
        relative=np.linalg.inv(data["part_to_world"][start,0])@data["part_to_world"][start,1]
        child_zero=np.linalg.inv(joint_transform)@relative
        sigma_pos_s, sigma_rot_s = resolve_rigid_gaussian_sigmas(args.root.name)
        radius_pos = gaussian_radius_for_sigma(data["fps"], sigma_pos_s)
        radius_rot = gaussian_radius_for_sigma(data["fps"], sigma_rot_s)
        radius=max(radius_pos, radius_rot)
        start=max(start,radius);stop=min(stop,len(data["joint_angle"])-radius-1)
        stage1=estimate_articulated_discrete_gaussian_wrench(
            data["part_to_world"][:,0],data["joint_angle"],data["part_physics"],child_zero,
            np.arange(start,stop+1), fps=data["fps"], sigma_s=sigma_pos_s,
            sigma_rot_s=sigma_rot_s, radius=radius_pos, radius_rot=radius_rot,
            device=args.device)
        sl=slice(start,stop)
        dynamics={key:value[sl] for key,value in geometry.items()}
        dynamics["generalized_contact_wrench"]=stage1["generalized_contact_wrench"]
        dynamics["generalized_net_wrench"]=stage1["generalized_net_wrench"]
        dynamics["generalized_gravity_wrench"]=stage1["generalized_gravity_wrench"]
        left_tactile_count=canonical_contact["left"][sl].sum(axis=1).astype(np.int32)
        right_tactile_count=canonical_contact["right"][sl].sum(axis=1).astype(np.int32)
        hand_contact_active=(left_tactile_count+right_tactile_count)>=TACTILE_CONTACT_MIN_VERTICES
        output = {"stage1_status":np.array("ok"),"format_version":np.int64(14),
                  "contact_geometry_method":np.array(
                      "original_surface_multiray_inside_normal_filter_v1"),
                  "stage1_method":np.array("articulated_newton_compatible_inverse_dynamics_v2"),
                  "environment_detector":np.array(HAND_INTERVAL_HEIGHT_POLICY),
                  "episode_index": np.int64(episode), "start": np.int64(start),
                  "stop": np.int64(stop), "object_name": np.array(data["object_name"]),
                  "normalization_json": np.array(json.dumps({**data["mass_normalization"],
                      "fps": data["fps"]}, sort_keys=True)),
                  "left_tactile_contact_count":left_tactile_count,
                  "right_tactile_contact_count":right_tactile_count,
                  "hand_contact_active":hand_contact_active,
                  "tactile_contact_min_vertices":np.int64(TACTILE_CONTACT_MIN_VERTICES),
                  "part_to_world": data["part_to_world"][sl], "joint_angle": data["joint_angle"][sl],
                  "smoothed_base_position_world":stage1["smoothed_base_position_world"],
                  "smoothed_base_rotation_world":stage1["smoothed_base_rotation_world"],
                  "smoothed_joint_angle":stage1["smoothed_joint_angle"],
                  "initial_generalized_velocity":stage1["initial_generalized_velocity"],
                  "generalized_velocity":stage1["generalized_velocity"],
                  "next_generalized_velocity":stage1["next_generalized_velocity"],
                  "generalized_acceleration":stage1["generalized_acceleration"],
                  "gaussian_child_zero_transform":child_zero,
                  "joint_axis_parent":data["part_physics"][1].joint_axis_parent,
                  "joint_origin_parent":data["part_physics"][1].joint_origin_parent,
                  "gaussian_sigma_s":np.float64(sigma_pos_s),
                  "gaussian_sigma_pos_s":np.float64(sigma_pos_s),
                  "gaussian_sigma_rot_s":np.float64(sigma_rot_s),
                  "gaussian_half_window_s":np.float64(GAUSSIAN_HALF_WINDOW_S),
                  "gaussian_radius":np.int64(radius),
                  "gaussian_radius_pos_frames":np.int64(radius_pos),
                  "gaussian_radius_rot_frames":np.int64(radius_rot),
                  **dynamics}
        environment = detect_environment_contact(
            data["part_to_world"], data["part_meshes"],
            canonical_contact["left"], canonical_contact["right"])
        full_hand_active = (
            canonical_contact["left"].sum(axis=1)
            + canonical_contact["right"].sum(axis=1)
        ) >= TACTILE_CONTACT_MIN_VERTICES
        lifecycle = hand_interval_height_lifecycle(
            environment["bottom_height_world"], full_hand_active)
        output["environment_contact_world"] = environment["support_point_world"][sl]
        output["environment_contact_active"] = lifecycle["active"][sl]
        output["environment_bottom_height_world"] = environment["bottom_height_world"][sl]
        output["environment_clearance_m"] = lifecycle["corrected_clearance_m"][sl]
        output["environment_reference_height_world"] = lifecycle["reference_height_m"][sl]
        output["environment_pickup_event"] = lifecycle["pickup_event"][sl]
        output["environment_placement_event"] = lifecycle["placement_event"][sl]
        for part, body in enumerate(data["part_physics"]):
            output[f"part_{part}_mass"] = np.float64(body.mass)
            output[f"part_{part}_com"] = body.com_part
            output[f"part_{part}_inertia"] = body.inertia_com_part
        for side in ("left", "right"):
            vertices, normals, faces = hand_geometry[side]
            offset = 0 if side == "left" else 778
            side_distance = full_distance[:, offset:offset + 778]
            side_closest = full_closest[:, offset:offset + 778]
            candidate = normal_consistent_surface_mask(
                vertices, normals, side_distance, side_closest,
                threshold_m=.020, safe_distance_m=.001, angle_deg=90.)[sl]
            candidate = contact_gated_force_candidates(
                candidate, canonical_contact[side][sl])
            priority = signed_distance_contact_priority(
                side_distance)[sl]
            owner = full_owner[sl, offset:offset + 778]
            assignment = np.where(candidate, owner, -1).astype(np.int8)
            distance = np.where(candidate, side_distance[sl], np.inf).astype(np.float32)
            output[f"{side}_vertices_world"] = vertices[sl].astype(np.float32)
            output[f"{side}_normals_world"] = normals[sl].astype(np.float32)
            output[f"{side}_faces"] = faces.astype(np.int32)
            output[f"{side}_contact_candidate"] = candidate
            output[f"{side}_contact_mask"] = canonical_contact[side][sl]
            output[f"{side}_surface_distance_m"] = side_distance[sl].astype(np.float32)
            output[f"{side}_surface_owner"] = owner.astype(np.int8)
            output[f"{side}_contact_priority"] = priority.astype(np.float32)
            output[f"{side}_contact_part"] = assignment
            output[f"{side}_part_classifier_vertex_distance_m"] = distance
        atomic_savez(path, **output)
        print({"episode": episode, "object": data["object_name"], "window": [start, stop],
               "left_candidates": int(output["left_contact_candidate"].sum()),
               "right_candidates": int(output["right_contact_candidate"].sum()),
               "output": str(path)}, flush=True)


if __name__ == "__main__":
    main()
