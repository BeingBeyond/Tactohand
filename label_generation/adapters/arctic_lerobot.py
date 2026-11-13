"""ARCTIC adapter with geometry-normalized articulated-object physics."""
from __future__ import annotations
from core.physics.parameters import NormalizedPartPhysics
from core.data.solver_inputs import _read_slice, _read_geometry

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.geometry.inertia import _volume_com_inertia
import numpy as np
import pyarrow.parquet as pq

from .dexycb_lerobot import load_hand_tactile


REFERENCE_TOTAL_MASS_KG = 1.0
MASS_NORMALIZATION = {
    "reference_total_mass_kg": REFERENCE_TOTAL_MASS_KG,
    "force_normalization": "per_unit_total_object_mass",
    "specific_force_unit": "N/kg",
    "specific_torque_unit": "N*m/kg",
    "mass_scaling_contract": "force_and_torque_scale_linearly_with_total_mass",
    "link_mass_distribution": "closed_mesh_volume_ratio",
}










def load_arctic_episode(
    root: Path, episode_index: int, *, view_id: str = "ego_view",
    include_legacy_tactile: bool = False,
) -> dict:
    """Load aligned base/part kinematics and unit-total-mass link physics.

    Production contact/proximity is rebuilt from MANO and object geometry, so
    the installed canonical tactile stream is not an input. The opt-in legacy
    fields remain available for old analysis callers only.
    """
    root = Path(root)
    observations = pq.read_table(
        root / "meta/observations.parquet", filters=[("episode_index", "=", int(episode_index))]
    ).to_pandas()
    match = observations[observations.view_id.astype(str) == view_id]
    if len(match) != 1:
        raise KeyError(f"episode={episode_index} view={view_id!r} missing or non-unique")
    observation = match.iloc[0]
    hmr = _read_slice(root, observation, "hmr").sort_values("frame_index").reset_index(drop=True)
    objects = _read_slice(root, observation, "objects").sort_values("frame_index").reset_index(drop=True)
    if len(hmr) != len(objects) or not np.array_equal(hmr.frame_index, objects.frame_index):
        raise ValueError("HMR and object frames are not aligned")
    instance_ids = objects.object_instance_id.astype(str).unique()
    if len(instance_ids) != 1:
        raise ValueError(f"expected one articulated instance in view, got {len(instance_ids)}")
    instance = pq.read_table(
        root / "objects/instances.parquet",
        filters=[("object_instance_id", "=", str(instance_ids[0]))],
    ).to_pandas().iloc[0]
    asset_id = str(instance.object_asset_id)
    asset = pq.read_table(root / "objects/assets.parquet").to_pandas()
    asset = asset[asset.object_asset_id.astype(str) == asset_id].iloc[0]
    if str(asset.topology_type) != "articulated":
        raise ValueError(f"ARCTIC asset {asset_id} is not declared articulated")
    parts_table = pq.read_table(root / "objects/parts.parquet").to_pandas()
    parts_table = parts_table[parts_table.object_asset_id.astype(str) == asset_id].sort_values("part_index")
    if len(parts_table) != 2:
        raise ValueError(f"expected two ARCTIC parts, got {len(parts_table)}")

    meshes = [_read_geometry(str(root.resolve()), str(row.object_geometry_id))
              for _, row in parts_table.iterrows()]
    raw_volumes = np.array([_volume_com_inertia(v, f, 1.0)[0] for v, f in meshes])
    masses = REFERENCE_TOTAL_MASS_KG * raw_volumes / raw_volumes.sum()
    physics = []
    for local, (_, row) in enumerate(parts_table.iterrows()):
        volume, com, inertia = _volume_com_inertia(*meshes[local], float(masses[local]))
        physics.append(NormalizedPartPhysics(
            part_index=int(row.part_index), name=str(row.object_part_id).rsplit(":", 1)[-1],
            mass=float(masses[local]), volume_m3=volume, com_part=com, inertia_com_part=inertia,
            joint_type=str(row.joint_type), joint_axis_parent=np.asarray(row.joint_axis_parent, np.float64),
            joint_origin_parent=np.asarray(row.joint_origin_parent, np.float64)))

    camera_to_world = np.stack([np.asarray(x).reshape(4, 4) for x in hmr["camera.c2w"]])
    object_to_camera = np.stack([np.asarray(x).reshape(4, 4) for x in objects.pose_cam])
    object_to_world = camera_to_world @ object_to_camera
    part_to_object = np.stack([
        np.stack([np.asarray(p).reshape(4, 4) for p in values])
        for values in objects.part_poses_object
    ])
    if part_to_object.shape[1:] != (2, 4, 4):
        raise ValueError(f"unexpected part pose shape {part_to_object.shape}")
    part_to_world = object_to_world[:, None] @ part_to_object
    joint_values = np.stack([np.asarray(x, np.float64) for x in objects.joint_values])
    if joint_values.shape != (len(objects), 1):
        raise ValueError(f"expected one ARCTIC movable DoF, got {joint_values.shape}")
    result = {
        "episode_index": int(episode_index), "view_id": view_id, "asset_id": asset_id,
        "object_instance_id": str(instance_ids[0]),
        "object_name": str(asset["name"]), "frame_index": hmr.frame_index.to_numpy(np.int64),
        "object_to_world": object_to_world, "part_to_object": part_to_object,
        "part_to_world": part_to_world, "joint_values": joint_values,
        "joint_angle": joint_values[:, 0], "part_meshes": meshes, "part_physics": physics,
        "gravity_world": np.array((0.0, 0.0, -9.80665), np.float64), "fps": 25.0,
        "mass_normalization": dict(MASS_NORMALIZATION),
    }
    if include_legacy_tactile:
        result["left_tactile"] = load_hand_tactile(root, episode_index, side="left")
        result["right_tactile"] = load_hand_tactile(root, episode_index, side="right")
    return result


def detect_environment_contact(
    part_to_world: np.ndarray,
    part_meshes: list[tuple[np.ndarray, np.ndarray]],
    left_contact_mask: np.ndarray,
    right_contact_mask: np.ndarray,
    *,
    calibration_frames: int = 25,
    support_band_m: float = 0.002,
    enter_clearance_m: float = 0.003,
    leave_clearance_m: float = 0.007,
) -> dict:
    """Detect repeated table contact from a fixed, sequence-initial table plane.

    ARCTIC physical sequences start with the object resting on the table before
    either hand contacts it. Those leading no-hand-contact frames calibrate one
    fixed table height. Contact is then evaluated over the *full* sequence with
    hysteresis, so later put-down and pick-up events remain detectable.
    """
    poses = np.asarray(part_to_world, np.float64)
    T, P = poses.shape[:2]
    if poses.shape[2:] != (4, 4) or P != len(part_meshes):
        raise ValueError("part poses and meshes are inconsistent")
    left = np.asarray(left_contact_mask, bool)
    right = np.asarray(right_contact_mask, bool)
    if left.shape[0] != T or right.shape[0] != T:
        raise ValueError("hand contact masks must match the sequence length")
    if not 0 < enter_clearance_m < leave_clearance_m:
        raise ValueError("expected 0 < enter clearance < leave clearance")

    bottom = np.empty(T, np.float64)
    support_point = np.empty((T, 3), np.float64)
    for t in range(T):
        world_parts = [vertices @ poses[t, part, :3, :3].T + poses[t, part, :3, 3]
                       for part, (vertices, _) in enumerate(part_meshes)]
        minimum = min(float(vertices[:, 2].min()) for vertices in world_parts)
        support_vertices = np.concatenate([
            vertices[vertices[:, 2] <= minimum + support_band_m]
            for vertices in world_parts
        ], axis=0)
        bottom[t] = minimum
        support_point[t] = support_vertices.mean(axis=0)

    prefix = min(int(calibration_frames), T)
    no_hand_contact = ~(left[:prefix].any(axis=1) | right[:prefix].any(axis=1))
    candidates = np.flatnonzero(no_hand_contact)
    if not len(candidates):
        raise ValueError("no initial hand-free frames available to calibrate the table")
    provisional = float(np.median(bottom[candidates]))
    static = candidates[np.abs(bottom[candidates] - provisional) <= support_band_m]
    if not len(static):
        static = candidates
    table_z = float(np.median(bottom[static]))
    clearance = bottom - table_z

    active = np.empty(T, bool)
    state = bool(clearance[0] <= leave_clearance_m)
    for t, distance in enumerate(clearance):
        if state and distance > leave_clearance_m:
            state = False
        elif not state and distance <= enter_clearance_m:
            state = True
        active[t] = state
    return {
        "support_point_world": support_point,
        "bottom_height_world": bottom,
        "table_height_world": table_z,
        "clearance_m": clearance,
        "active": active,
        "calibration_frame_indices": static.astype(np.int64),
    }


def assign_contact_vertices_to_parts(
    hand_vertices_world: np.ndarray, part_to_world: np.ndarray,
    part_meshes: list[tuple[np.ndarray, np.ndarray]], contact_mask: np.ndarray,
    *, device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Assign active MANO vertices to the nearest articulated triangle surface."""
    T, V, _ = hand_vertices_world.shape
    if part_to_world.shape != (T, len(part_meshes), 4, 4) or contact_mask.shape != (T, V):
        raise ValueError("inconsistent hand/part/contact shapes")
    from core.physics.rigid_multi_contact import nearest_object_surface

    parts = [
        {
            "dataset": "articulated",
            "object_geometry_id": "part:" + hashlib.sha1(
                np.asarray(mesh[0], np.float32).tobytes()
                + np.asarray(mesh[1], np.int32).tobytes()).hexdigest(),
            "mesh": mesh,
            "object_to_world": part_to_world[:, part],
        }
        for part, mesh in enumerate(part_meshes)
    ]
    distance, owner = nearest_object_surface(
        hand_vertices_world, parts, device=device)
    active = np.asarray(contact_mask, bool)
    active &= np.isfinite(hand_vertices_world).all(axis=2)
    return (np.where(active, owner, -1).astype(np.int8),
            np.where(active, distance, np.inf).astype(np.float32))
