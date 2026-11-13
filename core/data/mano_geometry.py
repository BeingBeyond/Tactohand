"""Load MANO vertices and normals in the dataset world frame."""
from __future__ import annotations
import core.data.processed_compat as processed_contract
from typing import Any

import json
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import torch
from core.geometry.mano_layer import REPO, ManoTorchLayer as ManoLayer, basis_for_dataset, validate_migrated_hmr_file
from scipy.spatial.transform import Rotation, Slerp


def _interpolate_mano(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate translation/shape linearly and all 16 MANO rotations on SO(3)."""
    result = (1.0 - alpha) * left + alpha * right
    left_rotation = left[3:51].reshape(16, 3)
    right_rotation = right[3:51].reshape(16, 3)
    result[3:51] = np.stack([
        Slerp(
            [0.0, 1.0],
            Rotation.from_rotvec(np.stack([left_rotation[joint], right_rotation[joint]])),
        )([alpha]).as_rotvec()[0]
        for joint in range(16)
    ]).reshape(48)
    return result


def complete_view_local_mano(
    raw: np.ndarray, present: np.ndarray, *, max_gap: int = 30,
    endpoint_hold: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete one camera's MANO track without cross-view coordinate jumps."""
    values = np.asarray(raw, np.float32).copy()
    available = np.asarray(present, bool).copy()
    source = np.full(len(values), "missing", dtype="<U16")
    anchors = np.flatnonzero(available & np.isfinite(values).all(axis=1))
    available[:] = False
    available[anchors] = True
    source[anchors] = "observed"
    for left, right in zip(anchors, anchors[1:], strict=False):
        gap = int(right - left - 1)
        if gap < 1 or gap > max_gap:
            continue
        for frame in range(left + 1, right):
            values[frame] = _interpolate_mano(
                values[left], values[right], (frame - left) / (right - left))
            available[frame] = True
            source[frame] = "interpolated"
    if len(anchors):
        first, last = int(anchors[0]), int(anchors[-1])
        begin = max(0, first - endpoint_hold)
        values[begin:first] = values[first]
        available[begin:first] = True
        source[begin:first] = "endpoint_hold"
        end = min(len(values), last + endpoint_hold + 1)
        values[last + 1:end] = values[last]
        available[last + 1:end] = True
        source[last + 1:end] = "endpoint_hold"
    values[~available] = 0.0
    return values, available, source


def _select_hand_instances(instances, side: str):
    """Pick one hand's rows from a hand_instances sidecar.

    ``hand_side`` is authoritative when present.  ``subject_flag`` is a person
    index in some datasets (e.g. OakInk-v1 writes subject_0/subject_1, both
    right hands) and must never be interpreted as left/right on its own; it
    only disambiguates two same-sided hands, preferring the actor.
    """
    if "hand_side" in instances.columns:
        selected = instances[instances.hand_side.astype(str) == side]
        if len(selected) and "role" in instances.columns:
            actor = selected[selected.role.astype(str) == "actor"]
            if len(actor):
                selected = actor
        if len(selected):
            return selected.sort_values("frame_index")
        return selected
    subject_flag = 0 if side == "left" else 1
    return instances[instances.subject_flag.astype(int) == subject_flag].sort_values("frame_index")


@lru_cache(maxsize=16)
def _mano_layer(side: str, device_string: str, basis: str = "corrected") -> Any:
    """Reuse each immutable left/right MANO module in one batch worker."""
    repository_root = REPO
    assets_root = Path(os.environ.get(
        "MANO_ASSETS_ROOT", str(repository_root / "assets" / "mano" / "corrected")
    )).expanduser().resolve()
    for model_side in ("LEFT", "RIGHT"):
        model = assets_root / "models" / f"MANO_{model_side}.pkl"
        if not model.is_file():
            raise FileNotFoundError(
                f"MANO model is missing: {model}; set MANO_ASSETS_ROOT explicitly"
            )
    layer = ManoLayer(
        basis=basis, side=side, mano_assets_root=str(assets_root),
        flat_hand_mean=True, use_pca=False, ncomps=45, center_idx=None,
    ).to(torch.device(device_string)).eval()
    return layer


def load_mano_world_geometry(
    root: Path,
    episode_index: int,
    view_id: str,
    *,
    side: str = "right",
    device: str | torch.device = "cuda",
    frame_start: int | None = None,
    frame_stop: int | None = None,
    canonical_hand_path: Path | None = None,
    observation: dict[str, Any] | None = None,
    compute_normals: bool = True,
    hmr_frame: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vertices/normals/faces; compute_normals=False returns empty normals."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if observation is None:
        import pandas as pd
        observations = pd.DataFrame(processed_contract.observations(root, episode_index))
        row = observations[observations.view_id.astype(str) == view_id].iloc[0]
    else:
        import pandas as pd
        if int(observation['episode_index']) != episode_index or str(observation['view_id']) != view_id:
            raise ValueError('MANO observation locator does not match episode/view')
        row = pd.Series(observation)
    if processed_contract.is_processed(root):
        reader = processed_contract.reader_for(root)
        for path in {r['path'] for r in reader.ranges('hmr', episode_index, view_id)}:
            validate_migrated_hmr_file(root, path)
    else:
        validate_migrated_hmr_file(root, str(row.hmr_path))
    hmr = (processed_contract.observation_table(root, row).to_pandas().sort_values("frame_index")
           if hmr_frame is None else hmr_frame)
    completion_path = canonical_hand_path
    if completion_path is None:
        info = processed_contract.read_info(root)
        relative = info.get("beingh_human", {}).get(
            "hand_temporal_completion", {}).get("path")
        if relative:
            completion_path = root / str(relative)
    use_view_local_completion = root.name == "hograspnet_lerobot"
    world_coordinates = completion_path is not None and not use_view_local_completion
    multi = bool(row.get("hand_instances_available", False))
    direct_override_raw = None
    direct_override_present = None
    direct_override_c2w = None
    if use_view_local_completion:
        source_raw = np.stack(hmr[f"{side}.mano_axis_angle"].to_numpy()).astype(
            np.float32)
        source_present = hmr[f"{side}.present"].to_numpy(bool)
        raw_all, present_all, _ = complete_view_local_mano(
            source_raw, source_present, max_gap=30, endpoint_hold=5)
        raw = raw_all[slice(frame_start, frame_stop)]
        present = present_all[slice(frame_start, frame_stop)]
        hmr = hmr.iloc[slice(frame_start, frame_stop)]
        model_side = side
    elif world_coordinates:
        if processed_contract.is_processed(root) and 'canonical_hands' in processed_contract.reader_for(root).manifest['modalities']:
            completed = processed_contract.reader_for(root).read_episode('canonical_hands',int(episode_index),side).to_pandas().sort_values('frame_index')
        else:
            completed = pq.read_table(
                completion_path,
                filters=[("episode_index", "=", int(episode_index)), ("side", "=", side)],
            ).to_pandas().sort_values("frame_index")
        timeline = np.arange(int(row.hmr_num_rows), dtype=np.int64)
        raw_all = np.zeros((len(timeline), 61), dtype=np.float32)
        present_all = np.zeros(len(timeline), dtype=bool)
        if not completed.empty:
            frames = completed.frame_index.to_numpy(np.int64)
            inside = (frames >= 0) & (frames < len(timeline))
            raw_all[frames[inside]] = np.stack(
                completed.mano_axis_angle_world.to_numpy()[inside]).astype(np.float32)
            present_all[frames[inside]] = completed.present.to_numpy(bool)[inside]
        # Preserve the current view's direct fit wherever it exists.  The
        # canonical multiview track is a gap filler, not a replacement for a
        # better image-aligned observation; small cross-view calibration
        # biases otherwise erase real hand/object contacts.
        source_present = hmr[f"{side}.present"].to_numpy(bool)
        source_raw = np.stack(hmr[f"{side}.mano_axis_angle"].to_numpy()).astype(
            np.float32)
        source_c2w = np.stack([
            np.asarray(value).reshape(4, 4) for value in hmr["camera.c2w"]])
        present_all[source_present] = True
        direct_override_raw = source_raw[slice(frame_start, frame_stop)]
        direct_override_present = source_present[slice(frame_start, frame_stop)]
        direct_override_c2w = source_c2w[slice(frame_start, frame_stop)]
        raw = raw_all[slice(frame_start, frame_stop)]
        present = present_all[slice(frame_start, frame_stop)]
        hmr = hmr.iloc[slice(frame_start, frame_stop)]
        model_side = side
    elif multi:
        instances = processed_contract.observation_table(root, row, 'hand_instances').to_pandas()
        selected = _select_hand_instances(instances, side)
        timeline = np.arange(int(row.hmr_num_rows), dtype=np.int64)
        raw_all = np.zeros((len(timeline), 61), dtype=np.float32)
        present = np.zeros(len(timeline), dtype=bool)
        if not selected.empty:
            frames = selected.frame_index.to_numpy(np.int64)
            raw_all[frames] = np.stack(selected.mano_axis_angle.to_numpy()).astype(np.float32)
            present[frames] = selected.present.to_numpy(bool)
        raw = raw_all[slice(frame_start, frame_stop)]
        present = present[slice(frame_start, frame_stop)]
        hmr = hmr.iloc[slice(frame_start, frame_stop)]
        model_side = "right"
    else:
        hmr = hmr.iloc[slice(frame_start, frame_stop)]
        raw = np.stack(hmr[f"{side}.mano_axis_angle"].to_numpy()).astype(np.float32)
        present = hmr[f"{side}.present"].to_numpy(bool)
        model_side = side
    torch_device = torch.device(device)
    layer = _mano_layer(model_side, str(torch_device), basis_for_dataset(root, side=model_side))
    if not present.any():
        # Absence is applied after the world transform below. If the whole
        # slice is absent, both geometry outputs are known before skinning.
        vertices = np.zeros((len(present), 778, 3), np.float32)
        normals = np.zeros_like(vertices) if compute_normals else vertices[:0]
        faces = layer.th_faces.cpu().numpy().astype(np.int64)
        return vertices, normals, faces
    with torch.no_grad():
        vertices = (
            layer(
                torch.from_numpy(raw[:, 3:51]).to(torch_device),
                torch.from_numpy(raw[:, 51:61]).to(torch_device),
            ).verts
            + torch.from_numpy(raw[:, :3]).to(torch_device).unsqueeze(1)
        )
        vertices[~torch.tensor(
            np.asarray(present, dtype=bool).copy(), device=torch_device)] = 0
        faces_device=layer.th_faces.to(device=torch_device,dtype=torch.long)
        normals = None
        if compute_normals:
            face_normals=torch.linalg.cross(
                vertices[:,faces_device[:,1]]-vertices[:,faces_device[:,0]],
                vertices[:,faces_device[:,2]]-vertices[:,faces_device[:,0]])
            normals=torch.zeros_like(vertices)
            for corner in range(3):
                normals.index_add_(1,faces_device[:,corner],face_normals)
            normals=torch.nn.functional.normalize(normals,dim=-1,eps=1e-12)
        if not world_coordinates:
            camera_to_world=torch.as_tensor(
                np.stack([np.asarray(value).reshape(4,4) for value in hmr["camera.c2w"]]),
                dtype=vertices.dtype,device=torch_device)
            rotation=camera_to_world[:,:3,:3]
            vertices=(torch.einsum("tij,tvj->tvi",rotation,vertices)
                      +camera_to_world[:,None,:3,3])
            if normals is not None:
                normals=torch.nn.functional.normalize(
                    torch.einsum("tij,tvj->tvi",rotation,normals),dim=-1,eps=1e-12)
        elif direct_override_raw is not None and bool(direct_override_present.any()):
            # Reproduce direct current-view fits through the exact legacy
            # geometry path.  MANO global orientation is joint-centered, so
            # converting only its root axis-angle/translation is not exactly
            # equivalent to transforming the resulting mesh.
            override_raw = torch.from_numpy(direct_override_raw).to(torch_device)
            override_vertices = (
                layer(override_raw[:, 3:51], override_raw[:, 51:61]).verts
                + override_raw[:, :3].unsqueeze(1)
            )
            override_normals = None
            if normals is not None:
                override_face_normals = torch.linalg.cross(
                    override_vertices[:, faces_device[:, 1]]
                    - override_vertices[:, faces_device[:, 0]],
                    override_vertices[:, faces_device[:, 2]]
                    - override_vertices[:, faces_device[:, 0]],
                )
                override_normals = torch.zeros_like(override_vertices)
                for corner in range(3):
                    override_normals.index_add_(
                        1, faces_device[:, corner], override_face_normals)
            override_c2w = torch.as_tensor(
                direct_override_c2w, dtype=vertices.dtype, device=torch_device)
            override_rotation = override_c2w[:, :3, :3]
            override_vertices = (
                torch.einsum("tij,tvj->tvi", override_rotation, override_vertices)
                + override_c2w[:, None, :3, 3]
            )
            if override_normals is not None:
                override_normals = torch.nn.functional.normalize(
                    torch.einsum("tij,tvj->tvi", override_rotation, override_normals),
                    dim=-1, eps=1e-12)
            direct = torch.as_tensor(
                direct_override_present, device=torch_device, dtype=torch.bool)
            vertices[direct] = override_vertices[direct]
            if normals is not None and override_normals is not None:
                normals[direct] = override_normals[direct]
        # Absent frames must stay exactly zero even after the camera
        # transform: ``0 @ R + t = t`` would otherwise fake presence for
        # every consumer that tests geometry-nonzero (the force solver's
        # source_hand_present does exactly that).
        absent = ~torch.tensor(
            np.asarray(present, dtype=bool).copy(), device=torch_device)
        vertices[absent] = 0
        if normals is not None:
            normals[absent] = 0
        vertices_numpy=vertices.cpu().numpy()
        normals_numpy=(normals.cpu().numpy() if normals is not None
                       else np.empty((0, 778, 3), np.float32))
    faces=faces_device.cpu().numpy().astype(np.int64)
    return vertices_numpy,normals_numpy,faces
