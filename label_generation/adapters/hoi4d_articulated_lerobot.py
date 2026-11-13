"""Canonical HOI4D multibody loader and link-space inverse dynamics."""
from __future__ import annotations
from core.physics.parameters import LinkPhysics
from core.physics.multibody import descendants, estimate_multibody_generalized_wrench, contact_generalized_map

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .arctic_lerobot import _read_geometry, _volume_com_inertia
from core.physics.backends import Physics
from core.physics.math3d import matrix_to_quat
from .part3_rigid_lerobot import _episode_gravity_directions, _episode_row, _slice
from core.physics.trajectory_wrench import estimate_rigid_discrete_gaussian_wrench




def load_hoi4d_articulated_episode(
    root: Path, episode_index: int, *, view_id: str = "ego_view",
) -> dict:
    root = Path(root)
    observations = pq.read_table(
        root / "meta/observations.parquet",
        filters=[("episode_index", "=", int(episode_index))],
    ).to_pandas()
    selected = observations[observations.view_id.astype(str) == view_id]
    if len(selected) != 1:
        raise KeyError(f"episode={episode_index} view={view_id!r} missing or non-unique")
    observation = selected.iloc[0]
    hmr = _slice(root, observation, "hmr").sort_values("frame_index").reset_index(drop=True)
    objects = _slice(root, observation, "objects").sort_values("frame_index").reset_index(drop=True)
    if len(hmr) != len(objects) or not np.array_equal(hmr.frame_index, objects.frame_index):
        raise ValueError("HMR and articulated object frames are not aligned")
    ids = objects.object_instance_id.astype(str).unique()
    if len(ids) != 1 or "multibody" not in ids[0]:
        raise ValueError(f"episode={episode_index} is not canonical articulated HOI4D")
    instance = pq.read_table(
        root / "objects/instances.parquet",
        filters=[("object_instance_id", "=", ids[0])],
    ).to_pylist()[0]
    asset_id = str(instance["object_asset_id"])
    asset = pq.read_table(
        root / "objects/assets.parquet",
        filters=[("object_asset_id", "=", asset_id)],
    ).to_pylist()[0]
    if str(asset["topology_type"]) != "articulated":
        raise ValueError(f"HOI4D asset {asset_id} is not declared articulated")
    parts = pq.read_table(
        root / "objects/parts.parquet",
        filters=[("object_asset_id", "=", asset_id)],
    ).to_pylist()
    parts.sort(key=lambda value: int(value["part_index"]))
    if len(parts) < 2:
        raise ValueError("articulated HOI4D asset must contain at least two parts")
    meshes = [_read_geometry(str(root.resolve()), str(part["object_geometry_id"]))
              for part in parts]
    volumes = np.asarray([_volume_com_inertia(v, f, 1.0)[0] for v, f in meshes])
    masses = volumes / volumes.sum()
    physics: list[LinkPhysics] = []
    for part, mesh, mass in zip(parts, meshes, masses, strict=True):
        _, com, inertia = _volume_com_inertia(*mesh, float(mass))
        physics.append(LinkPhysics(
            mass=float(mass), com_body=com, inertia_body=inertia,
            parent=(-1 if part["parent_part_index"] is None else int(part["parent_part_index"])),
            joint_type=str(part["joint_type"]),
            joint_axis_parent=np.asarray(part["joint_axis_parent"], np.float64),
            joint_origin_parent=np.asarray(part["joint_origin_parent"], np.float64),
        ))
    c2w = np.stack([np.asarray(value).reshape(4, 4) for value in hmr["camera.c2w"]])
    root_cam = np.stack([np.asarray(value).reshape(4, 4) for value in objects.pose_cam])
    root_world = c2w @ root_cam
    part_root = np.stack([
        np.stack([np.asarray(value).reshape(4, 4) for value in frame])
        for frame in objects.part_poses_object
    ])
    part_world = root_world[:, None] @ part_root
    frame_index = hmr.frame_index.to_numpy(np.int64)
    episode = _episode_row(root, episode_index)
    gravity_direction = _episode_gravity_directions(root, episode, frame_index)
    info = __import__("json").loads((root / "meta/info.json").read_text())
    fps = float(info.get("fps", 15.0))
    return {
        "episode_index": int(episode_index), "view_id": view_id,
        "object_instance_id": ids[0], "asset_id": asset_id,
        "frame_index": frame_index, "root_to_world": root_world,
        "part_to_root": part_root, "part_to_world": part_world,
        "joint_values": np.stack([np.asarray(x, np.float64) for x in objects.joint_values]),
        "parts": parts, "meshes": meshes, "physics": physics,
        "gravity_direction_world": gravity_direction,
        "gravity_world": gravity_direction * 9.80665, "fps": fps,
    }
