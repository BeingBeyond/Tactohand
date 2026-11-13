"""World→pixel pinhole projection (design §2 L1, §4 contracts)."""
from __future__ import annotations

import numpy as np


def world_to_camera(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Apply a world→camera 4×4 (rotation+translation only) to (N, 3)."""
    points = np.asarray(points_world, np.float64).reshape(-1, 3)
    return points @ w2c[:3, :3].T + w2c[:3, 3]


def project_camera(points_camera: np.ndarray, camera
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space pinhole projection; returns ``(uv, visible)``.

    Geometry closer than 5 cm is numerically explosive under perspective
    projection and normally indicates a missing/bad hand track at the
    camera, so it is flagged invisible.  ``visible`` does NOT include the
    in-frame test — use :func:`in_frame` for that.
    """
    points = np.asarray(points_camera, np.float64).reshape(-1, 3)
    visible = np.isfinite(points).all(axis=1) & (points[:, 2] > .05)
    depth = np.maximum(points[:, 2], 1e-8)
    uv = np.stack((float(camera.fx) * points[:, 0] / depth + float(camera.cx),
                   float(camera.fy) * points[:, 1] / depth
                   + float(camera.cy)), axis=1)
    return uv, visible


def project_world(points_world: np.ndarray, w2c: np.ndarray, camera
                  ) -> tuple[np.ndarray, np.ndarray]:
    return project_camera(world_to_camera(points_world, w2c), camera)


def in_frame(uv: np.ndarray, visible: np.ndarray, width: int, height: int
             ) -> np.ndarray:
    """Combine projection validity with an in-image bounds test."""
    uv = np.asarray(uv)
    return (visible & (uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height))
