"""Per-vertex force coordinates: z is surface normal, y points wristward."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import torch


CAMERA_FRAME = "camera"
VERTEX_LOCAL_FRAME = "vertex_local"


def clamp_force_magnitude(force: torch.Tensor, clamp_n: float) -> torch.Tensor:
    """Rescale [..., 3] vectors so no magnitude exceeds ``clamp_n``.

    Direction-preserving (f * c/|f|), differentiable, and an exact no-op
    below the bound — unlike a per-component clip it never rotates the force
    vector, so the aggregation identity (sum f, sum r x f) keeps its
    semantics.

    The factor is computed branch-free as c / max(|f|, c): the naive
    torch.where(|f| > c, f * c/|f|, f) evaluates c/|f| on the ~85%
    exactly-zero vertices, and its backward turns the masked zero gradient
    into NaN (0 * inf) — one such step NaNs every weight.
    """
    magnitude = force.norm(dim=-1, keepdim=True)
    return force * (clamp_n / magnitude.clamp_min(clamp_n))



def validate_vertex_force_frame(value: str) -> str:
    value = str(value).lower()
    if value not in {CAMERA_FRAME, VERTEX_LOCAL_FRAME}:
        raise ValueError("vertex_force_frame must be camera or vertex_local")
    return value


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


@lru_cache(maxsize=2)
def _mano_topology(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from core.geometry.mano_layer import ManoTorchLayer as ManoLayer

    root = Path(os.environ.get(
        "MANO_ASSETS_ROOT",
        str(Path(__file__).resolve().parents[2] / "assets" / "mano" / "stock"),
    ))
    layer = ManoLayer(
        side=side, mano_assets_root=str(root), use_pca=False,
        flat_hand_mean=True, ncomps=45, center_idx=None,
    ).eval()
    return (
        layer.th_faces.cpu().numpy().astype(np.int64),
        layer.th_J_regressor.cpu().numpy().astype(np.float32),
        layer.th_weights.argmax(dim=1).cpu().numpy().astype(np.int64),
    )


def vertex_local_axes(vertices: np.ndarray, side: str, *,
                      topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame y/z axes for posed MANO vertices ``[...,778,3]``."""
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.shape[-2:] != (778, 3):
        raise ValueError(f"expected MANO vertices [...,778,3], got {vertices.shape}")
    faces, joint_regressor, vertex_joint = _mano_topology(side) if topology is None else topology
    flat = vertices.reshape(-1, 778, 3)

    triangles = flat[:, faces]
    face_normals = np.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
    )
    normals = np.zeros_like(flat)
    batch_indices = np.arange(len(flat))[:, None]
    for corner in range(3):
        np.add.at(normals, (batch_indices, faces[None, :, corner]), face_normals)
    normal_valid = np.linalg.norm(normals, axis=-1) >= 1e-8
    z_axis = _normalize(normals)
    z_axis[~normal_valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    joints = np.einsum("jv,bvc->bjc", joint_regressor, flat)
    parents = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
    joint_y = joints[:, parents] - joints
    # Root/palm vertices share the middle-MCP-to-wrist direction.
    joint_y[:, 0] = joints[:, 0] - joints[:, 9]
    desired_y = joint_y[:, vertex_joint]
    y_axis = desired_y - np.sum(
        desired_y * z_axis, axis=-1, keepdims=True) * z_axis
    bad = np.linalg.norm(y_axis, axis=-1) < 1e-6
    if np.any(bad):
        coordinate_axes = np.eye(3, dtype=np.float32)
        fallback = coordinate_axes[np.argmin(np.abs(z_axis[bad]), axis=1)]
        y_axis[bad] = fallback - np.sum(
            fallback * z_axis[bad], axis=-1, keepdims=True) * z_axis[bad]
    y_axis = _normalize(y_axis)
    return y_axis.reshape(vertices.shape), z_axis.reshape(vertices.shape)


@lru_cache(maxsize=2)
def canonical_vertex_local_axes(side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return stable side-specific axes from the neutral MANO template."""
    from core.geometry.mano_layer import ManoTorchLayer as ManoLayer

    root = Path(os.environ.get(
        "MANO_ASSETS_ROOT",
        str(Path(__file__).resolve().parents[2] / "assets" / "mano" / "stock"),
    ))
    layer = ManoLayer(
        side=side, mano_assets_root=str(root), use_pca=False,
        flat_hand_mean=True, ncomps=45, center_idx=None,
    ).eval()
    with torch.no_grad():
        vertices = layer(torch.zeros(1, 48), torch.zeros(1, 10)).verts
    return tuple(axis[0] for axis in vertex_local_axes(
        vertices.cpu().numpy(), side))


def force_world_to_local(force: np.ndarray, y_world: np.ndarray,
                         z_world: np.ndarray) -> np.ndarray:
    x_world = np.cross(y_world, z_world)
    return np.stack((
        np.sum(force * x_world, axis=-1),
        np.sum(force * y_world, axis=-1),
        np.sum(force * z_world, axis=-1),
    ), axis=-1).astype(np.float32)


def axes_world_to_camera(y_world: np.ndarray, z_world: np.ndarray,
                         camera_c2w_rotation: np.ndarray) -> np.ndarray:
    """Return ``[...,2,3]`` camera-frame rows ``[y,z]``."""
    rotate = np.asarray(camera_c2w_rotation, dtype=np.float32)
    return np.stack((y_world @ rotate, z_world @ rotate), axis=-2).astype(np.float32)


def force_local_to_camera(force, yz_camera):
    """Differentiably expand local components along camera-frame vertex axes."""
    y_axis, z_axis = yz_camera.unbind(dim=-2)
    x_axis = torch.linalg.cross(y_axis, z_axis)
    return (force[..., 0:1] * x_axis + force[..., 1:2] * y_axis
            + force[..., 2:3] * z_axis)
