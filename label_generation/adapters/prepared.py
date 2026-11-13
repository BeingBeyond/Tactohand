"""Solver normalization of prepared target geometry."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from core.physics.backends import Physics
from .part3_rigid_lerobot import REFERENCE_TOTAL_MASS_KG, _inertia_from_mesh, _mesh
from core.data.targets import target_instance_id

def target_physics(root: Path, episode_index: int, instance_id: str, gravity: np.ndarray) -> Physics:
    instances = pq.read_table(
        root / "objects/instances.parquet",
        filters=[("episode_index", "=", int(episode_index))],
    ).to_pylist()
    matched = [row for row in instances if str(row["object_instance_id"]) == instance_id]
    if len(matched) != 1:
        raise ValueError(f"episode {episode_index}: target instance metadata is non-unique")
    asset_id = str(matched[0]["object_asset_id"])
    assets = pq.read_table(root / "objects/assets.parquet").to_pylist()
    asset = [row for row in assets if str(row["object_asset_id"]) == asset_id]
    if len(asset) != 1:
        raise ValueError(f"asset {asset_id}: metadata is missing or non-unique")
    geometry_id = str(asset[0]["object_geometry_id"])
    vertices, faces = _mesh(str(root.resolve()), geometry_id)
    com, inertia = _inertia_from_mesh(vertices, faces, REFERENCE_TOTAL_MASS_KG)
    return Physics(
        mass=REFERENCE_TOTAL_MASS_KG,
        com_body=tuple(com),
        inertia_body=inertia,
        gravity_world=tuple(np.asarray(gravity, dtype=np.float64)),
        dt=1.0 / 120.0,
        substeps=12,
    )



from typing import Any
import torch
from core.physics.backends import RigidState
from core.physics.math3d import matrix_to_quat
from core.data.inspection import decode_f64_rows, read_locator_rows

def extract_target_track(
    root: Path,
    locator: dict[str, Any],
    target_fps: float,
    *,
    require_all_valid: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rows = read_locator_rows(root, locator, "force")
    episode_index = int(locator["episode_index"])
    wanted = target_instance_id(root, episode_index)
    solver_fps = float(rows[0]["solver_fps"])
    stride_float = solver_fps / target_fps
    stride = int(round(stride_float))
    if stride < 1 or not np.isclose(stride_float, stride, atol=1e-9):
        raise ValueError("target fps must divide solver fps")

    ids = [str(value) for value in rows[0]["object_instance_ids"]]
    if wanted not in ids:
        raise ValueError(f"episode {episode_index}: target object is absent from force payload")
    object_index = ids.index(wanted)
    for row in rows:
        if [str(value) for value in row["object_instance_ids"]] != ids:
            raise ValueError(f"episode {episode_index}: object ordering changes within stream")
        if require_all_valid and not bool(row["force_label_valid"]):
            raise ValueError(f"episode {episode_index}: replay sample crosses invalid force row")

    position = np.stack(
        [decode_f64_rows(row["object_state_position_world"], 3)[object_index] for row in rows]
    )
    rotation = np.stack(
        [decode_f64_rows(row["object_state_rotation_world"], 9)[object_index].reshape(3, 3) for row in rows]
    )
    velocity_before = np.stack(
        [decode_f64_rows(row["object_velocity_before_world"], 6)[object_index] for row in rows]
    )
    contact = np.stack(
        [decode_f64_rows(row["object_contact_wrench"], 6)[object_index] for row in rows]
    )
    gravity = np.stack(
        [decode_f64_rows(row["object_gravity_world"], 3)[object_index] for row in rows]
    )
    dt = np.asarray([float(row["solver_dt_s"]) for row in rows], dtype=np.float64)

    # A final target boundary is required, so stop before the last incomplete
    # 10 Hz interval. Installed row t stores the state at transition start t.
    starts = np.arange(0, len(rows) - stride, stride, dtype=np.int64)
    if not len(starts):
        raise ValueError(f"episode {episode_index}: no complete 10 Hz replay interval")
    boundaries = np.concatenate((starts, [starts[-1] + stride]))
    impulse = []
    point = []
    for start in starts:
        sl = slice(int(start), int(start + stride))
        duration = float(dt[sl].sum())
        net = contact[sl] + np.pad(gravity[sl], ((0, 0), (0, 3)))
        impulse.append(np.sum(net * dt[sl, None], axis=0) / duration)
        point.append(net[0])
    target_position = torch.as_tensor(position[boundaries], dtype=torch.float64)
    target_rotation = torch.as_tensor(rotation[boundaries], dtype=torch.float64)
    initial_velocity = velocity_before[starts[0]]
    initial = RigidState(
        target_position[0],
        matrix_to_quat(target_rotation[0]),
        torch.as_tensor(initial_velocity[:3], dtype=torch.float64),
        torch.as_tensor(initial_velocity[3:], dtype=torch.float64),
    )

    physics = target_physics(root, episode_index, wanted, gravity[0])
    meta = {
        "episode_index": episode_index,
        "split": None,
        "target_object_instance_id": wanted,
        "object_count": len(ids),
        "solver_fps": solver_fps,
        "target_fps": target_fps,
        "stride": stride,
        "interval_count": len(starts),
        "physics": physics,
        "initial": initial,
    }
    targets = {
        "position": target_position,
        "rotation": target_rotation,
        "velocity": torch.as_tensor(velocity_before[boundaries], dtype=torch.float64),
    }
    wrenches = {
        "impulse_preserving": torch.as_tensor(np.stack(impulse), dtype=torch.float64),
        "point_sample": torch.as_tensor(np.stack(point), dtype=torch.float64),
    }
    return meta, targets, wrenches
