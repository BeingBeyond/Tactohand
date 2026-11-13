"""Adapter from BeingH part3 `dexycb_lerobot` to object-level wrench inputs."""
from __future__ import annotations
from core.data.solver_inputs import _read_mesh_cached, _read_mesh, _assets_table, _dexycb_episode_row as _episode_row

import json
from functools import lru_cache
from pathlib import Path

from core.geometry.inertia import _inertia_from_mesh
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.spatial import ConvexHull, QhullError

from core.physics.backends import Physics
from core.physics.math3d import matrix_to_quat


REFERENCE_TOTAL_MASS_KG = 1.0
MASS_NORMALIZATION = {
    "reference_total_mass_kg": REFERENCE_TOTAL_MASS_KG,
    "force_normalization": "per_unit_total_object_mass",
    "specific_force_unit": "N/kg",
    "specific_torque_unit": "N*m/kg",
    "mass_scaling_contract": "force_and_torque_scale_linearly_with_total_mass",
    "link_mass_distribution": "single_rigid_part",
}












def load_episode(root: Path, episode_index: int, *, view_id: str | None = None) -> dict:
    """Load canonical unit-total-mass world-COM physics for one episode."""
    root = Path(root)
    episode = _episode_row(root, episode_index)
    gravity = np.asarray(episode["gravity_dir_world"], dtype=np.float64) * 9.80665
    if not np.isfinite(gravity).all() or not np.isclose(np.linalg.norm(gravity), 9.80665, atol=1e-4):
        raise ValueError(f"invalid gravity for episode={episode_index}")
    obs = pq.read_table(root / "meta/observations.parquet", filters=[("episode_index", "=", int(episode_index))]).to_pandas()
    if view_id is None:
        eligible = obs[(obs.hmr_available == True) & (obs.objects_available == True)]  # noqa: E712
        if eligible.empty:
            raise ValueError(f"episode={episode_index} has no HMR+object view")
        view_id = str(eligible.sort_values("view_id").iloc[0].view_id)
    row = obs[obs.view_id.astype(str) == str(view_id)]
    if len(row) != 1:
        raise KeyError(f"episode={episode_index} view={view_id} not found")
    row = row.iloc[0]
    instance = pq.read_table(
        root / "objects/instances.parquet",
        filters=[("episode_index", "=", int(episode_index))],
    ).to_pandas()
    target = instance[instance.role == "target"]
    if len(target) != 1:
        raise ValueError(f"episode={episode_index}: expected one target instance")
    target_instance_id = str(target.iloc[0].object_instance_id)
    hmr = pq.ParquetFile(root / str(row.hmr_path)).read_row_group(int(row.hmr_row_group_index)).slice(int(row.hmr_row_offset), int(row.hmr_num_rows)).to_pandas()
    obj = pq.ParquetFile(root / str(row.objects_path)).read_row_group(int(row.objects_row_group_index)).slice(int(row.objects_row_offset), int(row.objects_num_rows)).to_pandas()
    hmr = hmr.sort_values("frame_index").reset_index(drop=True)
    obj = obj[(obj.object_instance_id.astype(str) == target_instance_id) &
              (obj.visible == True)].sort_values("frame_index").reset_index(drop=True)  # noqa: E712
    if len(obj) != len(hmr) or not np.array_equal(obj.frame_index.to_numpy(), hmr.frame_index.to_numpy()):
        raise ValueError(f"episode={episode_index}: object frames do not fully align to canonical timeline")
    asset_id = str(target.iloc[0].object_asset_id)
    assets = _assets_table(str(root.resolve())); asset = assets[assets.object_asset_id == asset_id].iloc[0]
    name = str(asset["name"]); mesh_id = int(json.loads(asset.attributes_json)["mesh_id"])
    mass = REFERENCE_TOTAL_MASS_KG
    com_body, inertia = _inertia_from_mesh(*_read_mesh(root, mesh_id), mass)
    c2w = np.stack([np.asarray(v, dtype=np.float64).reshape(4, 4) for v in hmr["camera.c2w"]])
    pose = np.stack([np.asarray(v, dtype=np.float64).reshape(4, 4) for v in obj.pose_cam])
    world = c2w @ pose
    R = world[:, :3, :3]; x_com = world[:, :3, 3] + np.einsum("tij,j->ti", R, com_body)
    q = matrix_to_quat(torch.as_tensor(R, dtype=torch.float64)).numpy()
    return {
        "episode_index": int(episode_index), "sequence_id": str(episode["sequence_id"]), "view_id": str(view_id),
        "frame_index": obj.frame_index.to_numpy(np.int64), "x_com": x_com, "quat_xyzw": q,
        "physics": Physics(mass, tuple(com_body), inertia, tuple(gravity)), "object_name": name,
        "mesh_id": mesh_id, "object_instance_id": target_instance_id,
        "object_index": int(obj.object_index.iloc[0]),
        "mass_normalization": dict(MASS_NORMALIZATION),
        "gravity_world_frame": str(episode["gravity_world_frame"]),
    }


def load_hand_tactile(root: Path, episode_index: int, *, side: str = "right") -> dict:
    """Load aligned MANO-vertex tactile geometry fields without creating a sidecar."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    root = Path(root)
    obs = pq.read_table(
        root / "meta/observations.parquet",
        filters=[("episode_index", "=", int(episode_index))],
    ).to_pandas()
    tactile_rows = obs[obs.tactile_available == True]  # noqa: E712
    if tactile_rows.empty:
        raise ValueError(f"episode={episode_index} has no tactile stream")
    row = tactile_rows.sort_values("view_id").iloc[0]
    if bool(row.get("hand_instances_available", False)):
        table = pq.ParquetFile(root / str(row.hand_instances_path)).read_row_group(
            int(row.hand_instances_row_group_index)
        ).slice(int(row.hand_instances_row_offset), int(row.hand_instances_num_rows))
        frame = table.to_pandas()
        from core.data.mano_geometry import _select_hand_instances

        frame = _select_hand_instances(frame, side).reset_index(drop=True)
        length = int(row.hmr_num_rows)
        frame_index = np.arange(length, dtype=np.int64)
        signed = np.zeros((length, 778), dtype=np.float32)
        proximity = np.zeros((length, 778), dtype=np.float32)
        contact = np.zeros((length, 778), dtype=bool)
        valid = np.zeros(length, dtype=bool)
        if not frame.empty:
            indices = frame.frame_index.to_numpy(np.int64)
            signed[indices] = np.stack([np.frombuffer(value, dtype=np.float32, count=778)
                                        for value in frame.signed_distance_vertices_m]).copy()
            proximity[indices] = np.stack([np.frombuffer(value, dtype=np.float32, count=778)
                                           for value in frame.proximity_vertices]).copy()
            contact[indices] = np.stack([np.frombuffer(value, dtype=np.uint8, count=778)
                                         for value in frame.contact_vertices]).astype(bool)
            valid[indices] = frame.proximity_valid.to_numpy(bool)
        return {"frame_index": frame_index, "signed_distance_m": signed,
                "proximity": proximity, "valid": valid,
                "contact_mask": contact & valid[:, None]}
    table = pq.ParquetFile(root / str(row.tactile_path)).read_row_group(
        int(row.tactile_row_group_index)
    ).slice(int(row.tactile_row_offset), int(row.tactile_num_rows))
    frame = table.to_pandas()
    columns = set(frame.columns)
    if "contact_vertices" in columns and "solver_sample_index" in columns:
        # Unified flat two-hand labels stream (installed 20260902+): one row
        # per solver sample with packed 1556-vertex binary payloads; map onto
        # the canonical source-frame grid via source_frame_left.
        offset = 0 if side == "left" else 778
        order = np.argsort(frame.source_frame_left.to_numpy(np.int64),
                           kind="stable")
        frame = frame.iloc[order].reset_index(drop=True)
        length = max(int(row.hmr_num_rows),
                     int(frame.source_frame_right.max()) + 1)
        frame_index = np.arange(length, dtype=np.int64)
        signed = np.zeros((length, 778), dtype=np.float32)
        proximity = np.zeros((length, 778), dtype=np.float32)
        contact = np.zeros((length, 778), dtype=bool)
        valid = np.zeros(length, dtype=bool)
        indices = frame.source_frame_left.to_numpy(np.int64)
        keep = indices < length
        rows = np.flatnonzero(keep)
        signed[indices[rows]] = np.stack([
            np.frombuffer(value, dtype=np.float32, count=1556)
            for value in frame.signed_distance_vertices_m.iloc[rows]
        ])[:, offset:offset + 778]
        proximity_col = frame.columns[
            frame.columns.str.contains("proximity_vertices")][0]
        proximity[indices[rows]] = np.stack([
            np.frombuffer(value, dtype=np.float32, count=1556)
            for value in frame[proximity_col].iloc[rows]
        ])[:, offset:offset + 778]
        contact[indices[rows]] = np.stack([
            np.frombuffer(value, dtype=np.uint8, count=1556)
            for value in frame.contact_vertices.iloc[rows]
        ])[:, offset:offset + 778].astype(bool)
        valid[indices[rows]] = np.stack([
            np.frombuffer(value, dtype=np.uint8, count=2)
            for value in frame.proximity_valid.iloc[rows]
        ])[:, 0 if side == "left" else 1].astype(bool)
        return {"frame_index": frame_index, "signed_distance_m": signed,
                "proximity": proximity, "valid": valid,
                "contact_mask": contact & valid[:, None]}
    frame = frame.sort_values("frame_index").reset_index(drop=True)
    signed = np.stack([
        np.frombuffer(value, dtype=np.float32, count=778)
        for value in frame[f"{side}.signed_distance_vertices_m"]
    ]).copy()
    proximity = np.stack([
        np.frombuffer(value, dtype=np.float32, count=778)
        for value in frame[f"{side}.proximity_vertices"]
    ]).copy()
    contact = np.stack([
        np.frombuffer(value, dtype=np.uint8, count=778)
        for value in frame[f"{side}.contact_vertices"]
    ]).astype(bool)
    valid = frame[f"{side}.proximity_valid"].to_numpy(bool)
    if signed.shape != (len(frame), 778) or proximity.shape != signed.shape:
        raise ValueError("unexpected MANO tactile vertex layout")
    return {
        "frame_index": frame.frame_index.to_numpy(np.int64),
        "signed_distance_m": signed,
        "proximity": proximity,
        "valid": valid,
        "contact_mask": contact & valid[:, None],
    }
