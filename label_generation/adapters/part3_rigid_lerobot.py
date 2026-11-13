"""Generic unit-mass adapter for rigid-object Part3 Human-v1 datasets."""
from __future__ import annotations
from core.data.solver_inputs import _slice, _mesh, _episode_row, _episode_gravity_directions

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp

from core.physics.backends import Physics
from .dexycb_lerobot import _inertia_from_mesh, load_hand_tactile
from core.physics.math3d import matrix_to_quat
from core.physics.solver_timeline import gaussian_observed_transition_mask


REFERENCE_TOTAL_MASS_KG = 1.0
HO3D_SEQUENCE_TABLE_POLICY = "ho3d_sequence_no_hand_table_v1"
HAND_INTERVAL_HEIGHT_POLICY = "hand_interval_height_lifecycle_v1"
HOGRASPNET_HAND_COMPLETION_VERSION = "view_local_camera_mano_slerp_v2"
HOGRASPNET_OBJECT_COMPLETION_VERSION = "v1"


def hograspnet_completion_contract(root: Path) -> dict[str, str]:
    """Validate the two independent 30 Hz completion streams used by force labels."""
    info = json.loads((Path(root) / "meta/info.json").read_text())
    human = info.get("beingh_human", {})
    objects = human.get("objects", {}).get("temporal_completion", {})
    object_version = str(objects.get("version", ""))
    if object_version != HOGRASPNET_OBJECT_COMPLETION_VERSION:
        raise ValueError(
            "HOGraspNet requires the installed independent 30 Hz object completion "
            f"{HOGRASPNET_OBJECT_COMPLETION_VERSION!r}"
        )
    return {
        "hand_completion_version": HOGRASPNET_HAND_COMPLETION_VERSION,
        "object_completion_version": object_version,
    }


def should_use_stored_contact_gate(dataset: str, requested: bool) -> bool:
    """Use stored masks only for datasets verified against current geometry."""
    geometry_recompute = {
        "hocap_multiview_lerobot", "hograspnet_lerobot", "hot3d_lerobot",
        "oakink-v1_lerobot", "show3d_lerobot",
    }
    return bool(requested and dataset not in geometry_recompute)


def sequence_no_hand_table_contact(
    clearance_m: np.ndarray,
    hand_active: np.ndarray,
    *,
    enter_clearance_m: float = 0.003,
    leave_clearance_m: float = 0.007,
) -> dict:
    """Infer a sequence table plane only when no-hand evidence exists.

    This is the HO3D contract: a sequence that is hand-held throughout has
    no observed table contact.  Otherwise, no-hand frames are table placements
    and calibrate the fixed plane used for hand-contact frames.
    """
    clearance = np.asarray(clearance_m, np.float64)
    hand = np.asarray(hand_active, bool)
    if clearance.ndim != 1 or hand.shape != clearance.shape:
        raise ValueError("clearance_m and hand_active must be matching vectors")
    anchors = (~hand) & np.isfinite(clearance)
    if not anchors.any():
        return {
            "active": np.zeros_like(hand),
            "corrected_clearance_m": clearance.copy(),
            "plane_offset_m": np.float64(np.nan),
            "has_no_hand_anchor": np.bool_(False),
        }
    offset = float(np.median(clearance[anchors]))
    corrected = clearance - offset
    active = np.empty_like(hand)
    state = bool(corrected[0] <= leave_clearance_m)
    for frame, distance in enumerate(corrected):
        if state and distance > leave_clearance_m:
            state = False
        elif not state and distance <= enter_clearance_m:
            state = True
        active[frame] = state
    active[~hand] = True
    return {
        "active": active,
        "corrected_clearance_m": corrected,
        "plane_offset_m": np.float64(offset),
        "has_no_hand_anchor": np.bool_(True),
    }


def hand_interval_height_lifecycle(
    bottom_height_m: np.ndarray,
    hand_active: np.ndarray,
    *,
    height_change_m: float = 0.007,
    anchor_window_frames: int = 15,
    persistence_frames: int = 3,
    max_hand_dropout_frames: int = 5,
) -> dict[str, np.ndarray]:
    """Infer table/hand lifecycle from each object's hand-contact intervals.

    Every no-hand frame is a support anchor. Within a hand-contact run, a
    sustained rise from the left anchor confirms pickup, while the final
    sustained suffix near the right anchor confirms placement. Runs touching
    a sequence boundary use whichever anchor exists; a fully hand-held run
    with no anchor is conservatively classified as off-table.
    """
    height = np.asarray(bottom_height_m, np.float64)
    hand = np.asarray(hand_active, bool).copy()
    if height.ndim != 1 or hand.shape != height.shape:
        raise ValueError("bottom_height_m and hand_active must be matching vectors")
    if not np.isfinite(height).all():
        raise ValueError("bottom_height_m must be finite")
    if height_change_m <= 0 or anchor_window_frames < 1 or persistence_frames < 1:
        raise ValueError("lifecycle thresholds and windows must be positive")
    if max_hand_dropout_frames < 0:
        raise ValueError("max_hand_dropout_frames must be nonnegative")

    # Close only short false gaps bounded by contact on both sides.
    false_edges = np.flatnonzero(np.diff(np.pad((~hand).astype(np.int8), (1, 1))))
    for begin, end in false_edges.reshape(-1, 2):
        if (begin > 0 and end < len(hand)
                and end - begin <= max_hand_dropout_frames):
            hand[begin:end] = True

    table = ~hand
    reference = height.copy()  # no-hand frames have zero semantic clearance.
    pickup = np.zeros(len(hand), bool)
    placement = np.zeros(len(hand), bool)
    contact_edges = np.flatnonzero(np.diff(np.pad(hand.astype(np.int8), (1, 1))))
    for start, stop in contact_edges.reshape(-1, 2):
        left_values = height[max(0, start - anchor_window_frames):start]
        right_values = height[stop:min(len(height), stop + anchor_window_frames)]
        left_anchor = float(np.median(left_values)) if len(left_values) else None
        right_anchor = float(np.median(right_values)) if len(right_values) else None
        run_length = stop - start

        lift = None
        if left_anchor is not None:
            elevated = height[start:stop] > left_anchor + height_change_m
            for local in range(max(0, run_length - persistence_frames + 1)):
                if elevated[local:local + persistence_frames].all():
                    lift = start + local
                    break
            table[start:stop] = True
        else:
            # A sequence that starts already held has no evidence of support.
            lift = start
            table[start:stop] = False
        if lift is not None:
            pickup[lift] = True
            table[lift:stop] = False

        put = None
        if right_anchor is not None:
            near_right = height[start:stop] <= right_anchor + height_change_m
            suffix = run_length
            while suffix > 0 and near_right[suffix - 1]:
                suffix -= 1
            if run_length - suffix >= persistence_frames:
                put = start + suffix
                if lift is None or put < lift:
                    # No confirmed lift: the complete contact run stayed on support.
                    put = start
                table[put:stop] = True
                placement[put] = True

        if left_anchor is not None and right_anchor is not None:
            switch = put if put is not None else stop
            reference[start:switch] = left_anchor
            reference[switch:stop] = right_anchor
        elif left_anchor is not None:
            reference[start:stop] = left_anchor
        elif right_anchor is not None:
            reference[start:stop] = right_anchor
        else:
            reference[start:stop] = np.nan

    return {
        "active": table,
        "stabilized_hand_active": hand,
        "pickup_event": pickup,
        "placement_event": placement,
        "reference_height_m": reference,
        "corrected_clearance_m": height - reference,
    }


def object_bottom_support_geometry(
    object_to_world: np.ndarray,
    mesh_vertices: np.ndarray,
    gravity_direction_world: np.ndarray,
    *,
    support_band_m: float = 0.002,
) -> dict:
    """Return per-frame bottom height and support point along gravity."""
    poses = np.asarray(object_to_world, np.float64)
    vertices = np.asarray(mesh_vertices, np.float64)
    gravity = np.asarray(gravity_direction_world, np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("object_to_world must have shape (T,4,4)")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh_vertices must have shape (V,3)")
    if gravity.shape == (3,):
        gravity = np.broadcast_to(gravity, (len(poses), 3))
    if (gravity.shape != (len(poses), 3) or not np.isfinite(gravity).all()
            or np.any(np.linalg.norm(gravity, axis=1) < 1e-12)):
        raise ValueError("gravity_direction_world must have shape (3,) or (T,3)")
    if support_band_m <= 0:
        raise ValueError("support_band_m must be positive")
    up = -gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
    T = len(poses)

    world = np.einsum("vj,tij->tvi", vertices, poses[:, :3, :3]) + poses[:, None, :3, 3]
    height = np.einsum("tvi,ti->tv", world, up)
    bottom = height.min(axis=1)
    support_point = np.empty((T, 3), np.float64)
    for t in range(T):
        band = height[t] <= bottom[t] + support_band_m
        support_point[t] = world[t, band].mean(axis=0)
    return {
        "support_point_world": support_point,
        "bottom_height_along_up_m": bottom,
        "up_direction_world": up,
    }


def hocap_tabletop_lifecycle(
    geometric_active: np.ndarray,
    clearance_m: np.ndarray,
    hand_active: np.ndarray,
    *,
    leave_clearance_m: float = 0.007,
    max_hand_dropout_frames: int = 5,
) -> dict[str, np.ndarray]:
    """Apply the HOCAP pick-up/put-down tabletop contract per object.

    HOCAP objects start on the table and every object that is picked up is
    subsequently put back on the table.  Geometry decides when a contacted
    object has been lifted, but a confirmed hand release is authoritative for
    putting it back; CAD/pose height bias must not prevent re-entry.
    """
    geometric = np.asarray(geometric_active, bool)
    clearance = np.asarray(clearance_m, np.float64)
    hand = np.asarray(hand_active, bool).copy()
    if geometric.shape != clearance.shape or hand.shape != clearance.shape:
        raise ValueError("HOCAP lifecycle inputs must have matching shape (T,)")
    if max_hand_dropout_frames < 0:
        raise ValueError("max_hand_dropout_frames must be nonnegative")

    # Close only bounded false gaps between two observed hand-contact runs.
    false_edges = np.flatnonzero(np.diff(np.pad((~hand).astype(np.int8), (1, 1))))
    for begin, end in false_edges.reshape(-1, 2):
        if (begin > 0 and end < len(hand)
                and end - begin <= max_hand_dropout_frames):
            hand[begin:end] = True

    table = np.ones(len(hand), bool)
    lifted = False
    release = np.zeros(len(hand), bool)
    placement_offset = np.zeros(len(hand), np.float64)
    current_offset = 0.0
    for frame in range(len(hand)):
        if not lifted:
            if hand[frame] and clearance[frame] > leave_clearance_m:
                lifted = True
                table[frame] = False
            else:
                table[frame] = True
        elif hand[frame]:
            table[frame] = False
        else:
            # Dataset prior: a confirmed release after lift means put-down.
            lifted = False
            table[frame] = True
            release[frame] = True
            current_offset = float(clearance[frame])
        placement_offset[frame] = current_offset if table[frame] else 0.0
    return {
        "active": table,
        "stabilized_hand_active": hand,
        "release_event": release,
        "placement_clearance_offset_m": placement_offset,
        "geometric_active": geometric,
    }










def eligible_observations(
    root: Path, episode_index: int, *, require_tactile: bool = True,
):
    rows = pq.read_table(
        root / "meta/observations.parquet", filters=[("episode_index", "=", int(episode_index))]
    ).to_pandas()
    mask = rows.hmr_available.fillna(False) & rows.objects_available.fillna(False)
    # Show3D predates the stored contact stream. Its contact mask is rebuilt
    # from HMR/object geometry by the force solver, so tactile is not an
    # eligibility requirement for this dataset.
    if (require_tactile and "tactile_available" in rows
            and Path(root).name != "show3d_lerobot"):
        mask &= rows.tactile_available.fillna(False)
    return rows[mask].sort_values("view_id")


def _track_score(frame, camera_to_world: dict[int, np.ndarray]) -> tuple[float, int, int]:
    frame = frame.sort_values("frame_index")
    visible = frame.visible.fillna(False).to_numpy(bool)
    rows = frame.loc[visible]
    poses = [camera_to_world[int(index)] @ np.asarray(value).reshape(4, 4)
             for index, value in zip(rows.frame_index, rows.pose_cam)
             if int(index) in camera_to_world]
    if len(poses) < 2:
        return (0.0, len(poses), -int(frame.object_index.iloc[0]))
    translations = np.stack([pose[:3, 3] for pose in poses])
    motion = float(np.linalg.norm(np.diff(translations, axis=0), axis=1).sum())
    return (motion, len(poses), -int(frame.object_index.iloc[0]))


def _interpolate_world_poses(frame_index: np.ndarray, poses: np.ndarray,
                             target_index: np.ndarray) -> np.ndarray:
    """Interpolate SE(3) poses and hold the nearest endpoint outside support."""
    from core.physics.solver_timeline import interpolate_pose_samples
    return interpolate_pose_samples(frame_index, poses, target_index)


def load_rigid_episode(
    root: Path,
    episode_index: int,
    *,
    view_id: str | None = None,
    object_instance_id: str | None = None,
    minimum_aligned_frames: int = 2,
    load_tactile: bool = True,
) -> dict:
    """Load one complete rigid-object track in a physical episode.

    ``object_instance_id`` is deliberately explicit.  The old fallback (the
    object with the greatest accumulated translation) is retained for
    single-object callers, but must not be used to represent a multi-object
    episode.  Two aligned states are sufficient for one solver transition;
    callers which only inspect per-frame payloads may explicitly request one.
    """
    root = Path(root)
    observations = eligible_observations(
        root, episode_index, require_tactile=load_tactile)
    if observations.empty:
        raise ValueError(f"episode={episode_index} has no eligible HMR+object observation")
    observation = observations.iloc[0] if view_id is None else observations[
        observations.view_id.astype(str) == str(view_id)
    ].iloc[0]
    hmr = _slice(root, observation, "hmr").sort_values("frame_index").reset_index(drop=True)
    objects = _slice(root, observation, "objects")
    info = json.loads((root / "meta/info.json").read_text())
    temporal_completion = bool(
        info.get("beingh_human", {}).get("objects", {}).get("temporal_completion")
    )
    if "view_id" in hmr.columns:
        hmr = hmr[hmr.view_id.astype(str) == str(observation.view_id)].reset_index(drop=True)
    if "view_id" in objects.columns:
        objects = objects[objects.view_id.astype(str) == str(observation.view_id)].reset_index(drop=True)
    camera_to_world = {
        int(index): np.asarray(value).reshape(4, 4)
        for index, value in zip(hmr.frame_index, hmr["camera.c2w"])
    }
    # object_index is only guaranteed to be meaningful inside one source
    # annotation convention. HOT3D legitimately stores every scene object as
    # object_index=0, while object_instance_id remains globally unambiguous.
    tracks = [group for _, group in objects.groupby("object_instance_id")]
    tracks = [track for track in tracks if bool(track.visible.fillna(False).any())]
    if not tracks:
        raise ValueError("observation contains no visible object track")
    if object_instance_id is None:
        track = max(tracks, key=lambda value: _track_score(value, camera_to_world))
    else:
        requested = [value for value in tracks
                     if str(value.object_instance_id.iloc[0]) == str(object_instance_id)]
        if len(requested) != 1:
            raise KeyError(f"object instance {object_instance_id!r} has no unique visible track")
        track = requested[0]
    track = track.sort_values("frame_index").drop_duplicates("frame_index").reset_index(drop=True)
    interpolated_pose_dataset = root.name in {"hot3d_lerobot", "show3d_lerobot"}
    if not interpolated_pose_dataset and not temporal_completion:
        track = track[track.visible.fillna(False)].reset_index(drop=True)
    common = np.intersect1d(hmr.frame_index.to_numpy(np.int64), track.frame_index.to_numpy(np.int64))
    if len(common) < minimum_aligned_frames:
        raise ValueError(f"selected object track has only {len(common)} aligned frames")
    if interpolated_pose_dataset:
        if root.name == "hot3d_lerobot":
            coverage = len(common) / max(len(hmr), 1)
            if coverage < .5:
                raise ValueError(f"HOT3D object track covers only {coverage:.1%} of the episode")
        aligned_hmr = hmr[hmr.frame_index.isin(common)].sort_values("frame_index").reset_index(drop=True)
        aligned_track = track[track.frame_index.isin(common)].sort_values("frame_index").reset_index(drop=True)
    else:
        hmr = hmr[hmr.frame_index.isin(common)].sort_values("frame_index").reset_index(drop=True)
        track = track[track.frame_index.isin(common)].sort_values("frame_index").reset_index(drop=True)
    instance_id = str(track.object_instance_id.iloc[0])
    instances = pq.read_table(root / "objects/instances.parquet").to_pandas()
    instance = instances[instances.object_instance_id.astype(str) == instance_id]
    if len(instance) != 1:
        raise KeyError(f"instance={instance_id!r} missing or non-unique")
    assets = pq.read_table(root / "objects/assets.parquet").to_pandas()
    asset = assets[assets.object_asset_id.astype(str) == str(instance.iloc[0].object_asset_id)].iloc[0]
    vertices, faces = _mesh(str(root.resolve()), str(asset.object_geometry_id))
    com, inertia = _inertia_from_mesh(vertices, faces, REFERENCE_TOTAL_MASS_KG)
    if interpolated_pose_dataset:
        observed_c2w = np.stack([np.asarray(value).reshape(4, 4) for value in aligned_hmr["camera.c2w"]])
        observed_o2c = np.stack([np.asarray(value).reshape(4, 4) for value in aligned_track.pose_cam])
        observed_o2w = observed_c2w @ observed_o2c
        full_index = hmr.frame_index.to_numpy(np.int64)
        o2w = _interpolate_world_poses(common, observed_o2w, full_index)
        pose_observed = np.isin(full_index, common)
        common = full_index
    else:
        c2w = np.stack([np.asarray(value).reshape(4, 4) for value in hmr["camera.c2w"]])
        o2c = np.stack([np.asarray(value).reshape(4, 4) for value in track.pose_cam])
        o2w = c2w @ o2c
        pose_observed = np.ones(len(common), bool)
    rotation = o2w[:, :3, :3]
    x_com = o2w[:, :3, 3] + np.einsum("tij,j->ti", rotation, com)
    episode = _episode_row(root, episode_index)
    gravity_direction = _episode_gravity_directions(root, episode, common)
    fps = float(info.get("fps", 30.0))
    physics = Physics(1.0, tuple(com), inertia, tuple(gravity_direction[0] * 9.80665), dt=1 / fps, substeps=1)
    tactile = None
    if load_tactile and bool(observation.get("tactile_available", False)):
        tactile = {side: load_hand_tactile(root, episode_index, side=side)
                   for side in ("left", "right")}
        for side in tactile:
            index = {int(value): i for i, value in enumerate(tactile[side]["frame_index"])}
            selected = np.array([index[int(value)] for value in common], np.int64)
            tactile[side] = {
                key: value[selected] if isinstance(value, np.ndarray)
                and len(value) == len(index) else value
                for key, value in tactile[side].items()
            }
    return {
        "dataset": root.name, "episode_index": int(episode_index), "view_id": str(observation.view_id),
        "frame_index": common, "object_index": int(track.object_index.iloc[0]),
        "object_instance_id": instance_id, "object_name": str(asset["name"]),
        "object_geometry_id": str(asset.object_geometry_id),
        "object_to_world": o2w, "x_com": x_com,
        "object_pose_observed": pose_observed,
        "quat_xyzw": matrix_to_quat(__import__("torch").as_tensor(rotation, dtype=__import__("torch").float64)).numpy(),
        "mesh": (vertices, faces), "physics": physics, "fps": fps, "tactile": tactile,
        "gravity_direction_world": gravity_direction,
        "gravity_world": gravity_direction * 9.80665,
        "mass_normalization": {
            "reference_total_mass_kg": 1.0, "force_normalization": "per_unit_total_object_mass",
            "specific_force_unit": "N/kg", "specific_torque_unit": "N*m/kg",
            "mass_scaling_contract": "force_and_torque_scale_linearly_with_total_mass",
            "link_mass_distribution": "single_rigid_part",
        },
    }


def load_rigid_episode_objects(
    root: Path, episode_index: int, *, view_id: str | None = None,
    load_tactile: bool = True,
) -> list[dict]:
    """Load every usable rigid object in a canonical observation.

    Results are stable-sorted by instance id.  Objects which do not cover the
    complete episode are rejected, except HOT3D tracks: its schema supplies
    valid poses even when ``visible=False`` and sparse missing rows are
    interpolated in world-frame SE(3).
    """
    root = Path(root)
    observations = eligible_observations(
        root, episode_index, require_tactile=load_tactile)
    if observations.empty:
        raise ValueError(f"episode={episode_index} has no eligible HMR+object observation")
    observation = observations.iloc[0] if view_id is None else observations[
        observations.view_id.astype(str) == str(view_id)
    ].iloc[0]
    objects = _slice(root, observation, "objects")
    if "view_id" in objects.columns:
        objects = objects[objects.view_id.astype(str) == str(observation.view_id)]
    instance_ids = sorted(str(value) for value in objects.loc[
        objects.visible.fillna(False), "object_instance_id"].unique())
    episode_length = int(_episode_row(root, episode_index)["length"])
    loaded = []
    for value in instance_ids:
        try:
            loaded.append(load_rigid_episode(root, episode_index,
                                             view_id=str(observation.view_id),
                                             object_instance_id=value,
                                             load_tactile=load_tactile))
        except ValueError as error:
            if root.name != "hot3d_lerobot" or "covers only" not in str(error):
                raise
    complete = [value for value in loaded if len(value["frame_index"]) == episode_length]
    if not complete:
        raise ValueError(f"episode={episode_index} has no complete visible rigid-object track")
    return complete
