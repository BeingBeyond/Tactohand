"""HMR camera pose (camera.c2w) parquet reader shared by preflight and training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import core.data.processed_compat as processed_contract

import numpy as np


def hmr_c2w(root: Path, locator: dict[str, Any]) -> np.ndarray:
    table = processed_contract.observation_table(root, locator, columns=["camera.c2w"])
    matrices = []
    for value in table.column("camera.c2w").to_pylist():
        flat = (
            np.frombuffer(value, dtype=np.float32)
            if isinstance(value, (bytes, bytearray, memoryview))
            else np.asarray(value, dtype=np.float64).reshape(-1)
        )
        matrices.append(flat.reshape(4, 4).astype(np.float64))
    return np.stack(matrices)


def canonical_to_video_map(root: Path,
                           locator: dict[str, Any]) -> tuple[np.ndarray,
                                                             np.ndarray] | None:
    """Sparse-video mapping: per canonical row, the video-file index AND the
    canonical time of the displayed frame; None when the video carries the
    full timeline (dense, identity case).

    Interval-sampled videos (hograspnet: ~29% median density) hold a subset
    of the canonical frames; the hmr ``video_frame_index`` column maps
    canonical rows to the sampled video frames.  The fill rule is CAUSAL
    hold-last (rows before the first sample hold it) so the displayed frame
    never leads the labels, and the returned canonical sample time lets the
    CAMERA pose match the displayed frame (ego cameras move — projecting
    labels with a time-mismatched c2w offsets the mesh even at sampled
    frames).
    """
    if int(locator.get("num_frames", 0)) == int(
            locator.get("video_num_frames", -1)):
        return None
    # Sparse-frame bookkeeping needs one integer column, not MANO/camera
    # payloads. Project it before row-group reads to keep metadata audits bounded.
    try:
        table = processed_contract.observation_table(root, locator, columns=["video_frame_index"])
    except KeyError as error:
        if "video_frame_index" not in str(error):
            raise
        return None
    if "video_frame_index" not in table.column_names:
        return None
    values = table.column("video_frame_index").to_pylist()
    known = np.asarray([(index, int(value)) for index, value
                        in enumerate(values) if value is not None],
                       np.int64).reshape(-1, 2)
    if not len(known):
        return None
    if (len(known) == len(values)
            and np.array_equal(known[:, 1], np.arange(len(values)))):
        return None  # dense despite the metadata mismatch: identity
    rows = np.arange(len(values), dtype=np.int64)
    # hold-last-sample (causal): head rows before the first sample hold it
    sample = np.clip(np.searchsorted(known[:, 0], rows, side="right") - 1,
                     0, len(known) - 1)
    return known[sample, 1], known[sample, 0]


from core.geometry.camera import CameraIntrinsics

def load_cameras(root: Path) -> dict[str, CameraIntrinsics]:
    """Read ``meta/cameras.parquet`` into ``{camera_id: intrinsics}``."""
    table = processed_contract.cameras(root).to_pydict()
    return {str(camera_id): CameraIntrinsics(float(fx), float(fy),
                                             float(cx), float(cy))
            for camera_id, fx, fy, cx, cy in zip(
                table["camera_id"], table["fx"], table["fy"],
                table["cx"], table["cy"])}


def view_camera(root: Path, observation: dict[str, Any],
                cameras: dict[str, CameraIntrinsics] | None = None
                ) -> tuple[np.ndarray, CameraIntrinsics]:
    """Return ``(c2w[T, 4, 4], intrinsics)`` for one observations row.

    ``c2w`` is indexed by the view's source-frame order (HMR stream rows).
    """

    cameras = cameras if cameras is not None else load_cameras(root)
    camera_id = str(observation.get("camera_id"))
    if camera_id not in cameras:
        raise KeyError(f"camera_id {camera_id!r} missing from "
                       f"{Path(root).name}/meta/cameras.parquet")
    return hmr_c2w(Path(root), observation), cameras[camera_id]
