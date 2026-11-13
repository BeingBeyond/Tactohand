"""Shaded 3D mesh rendering via pyrender (design §2 L2 offscreen capability).

Complements the cv2 wireframe/dot primitives in :mod:`core.visualization.drawing`
with true shaded mesh rendering: per-vertex colored MANO/object meshes
composited over RGB frames with alpha blending.  Axis conventions are
centralized here — callers hand over world-space vertices plus ``w2c`` and
RGB vertex colors, this module performs the OpenCV→OpenGL flip.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "0")

from typing import Any, Iterable, Sequence  # noqa: E402

import cv2
import numpy as np

from core.geometry.projection import world_to_camera  # noqa: E402


def scale_intrinsics(intrinsics, factor: float):
    """Scale intrinsics for a uniformly downscaled image copy."""
    return type(intrinsics)(intrinsics.fx * factor, intrinsics.fy * factor,
                            intrinsics.cx * factor, intrinsics.cy * factor)


def letterbox_frame(image: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize with gray padding to (size, size)."""
    height, width = image.shape[:2]
    factor = size / max(height, width)
    resized = cv2.resize(image, (max(1, int(round(width * factor))),
                                 max(1, int(round(height * factor)))))
    out = np.full((size, size, 3), 114, np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    out[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return out


def render_mesh_overlay(
    image_bgr: np.ndarray,
    w2c: np.ndarray | None,
    intrinsics,
    meshes: Sequence[dict],
    *,
    alpha: float = 0.85,
    renderer: Any = None,
    background: np.ndarray | None = None,
    smooth: bool = True,
):
    """Render shaded meshes over one BGR frame; returns the composited image.

    ``w2c`` transforms world-space vertices into the camera frame; pass
    ``None`` when the vertices are already camera-space (legacy converted
    data planes store camera-space meshes).  ``meshes`` entries:
    ``{"vertices": (N, 3), "faces": (M, 3) int, "colors": (N, 3) uint8 RGB
    or None, "rgba": (4,) fallback material}``.  Per-vertex colors win;
    otherwise a flat RGBA material is used.  Pass a
    persistent ``renderer`` (``pyrender.OffscreenRenderer``) to amortize the
    GL context across frames; a mismatched viewport falls back to a
    temporary one.  ``background`` replaces the base image (e.g. a dimmed
    copy) without changing the blend.
    """
    import pyrender
    import trimesh

    height, width = image_bgr.shape[:2]
    base = background if background is not None else image_bgr
    usable = [mesh for mesh in meshes
              if len(mesh.get("faces", [])) and len(mesh.get("vertices", []))]
    if not usable:
        return base.copy()

    camera_vertices = []
    max_z = 0.0
    for mesh in usable:
        points = (world_to_camera(mesh["vertices"], w2c) if w2c is not None
                  else np.asarray(mesh["vertices"], np.float64))
        camera_vertices.append(points)
        if points.size:
            max_z = max(max_z, float(np.nanmax(points[:, 2])))
    zfar = max(10.0, max_z + 1.0)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0],
                           ambient_light=[0.35, 0.35, 0.35])
    scene.add(pyrender.IntrinsicsCamera(
        float(intrinsics.fx), float(intrinsics.fy),
        float(intrinsics.cx), float(intrinsics.cy),
        znear=0.01, zfar=zfar), pose=np.eye(4))
    # Camera-facing key light separates front faces from back faces; two
    # dimmer side/back fills keep shaded regions legible without flattening.
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.8),
              pose=np.eye(4))
    side = np.eye(4)
    side[:3, :3] = np.array([[0.7, 0.7, -0.2],
                             [-0.7, 0.7, -0.2],
                             [0.2, 0.2, 0.7]])
    scene.add(pyrender.DirectionalLight(color=[0.9, 0.9, 0.95], intensity=1.0),
              pose=side)
    back = np.eye(4)
    back[:3, 2] = -1.0
    scene.add(pyrender.DirectionalLight(color=[0.75, 0.78, 0.85], intensity=0.6),
              pose=back)

    for mesh, points in zip(usable, camera_vertices):
        vertices_gl = np.asarray(points, np.float64).copy()
        vertices_gl[:, 1] *= -1.0        # OpenCV (y down, z forward)
        vertices_gl[:, 2] *= -1.0        # -> OpenGL (y up, z back)
        colors = mesh.get("colors")
        if colors is not None:
            mesh_gl = trimesh.Trimesh(
                vertices=vertices_gl, faces=np.asarray(mesh["faces"]),
                vertex_colors=np.asarray(colors, np.uint8), process=False)
            scene.add(pyrender.Mesh.from_trimesh(mesh_gl, smooth=smooth))
        else:
            mesh_gl = trimesh.Trimesh(
                vertices=vertices_gl, faces=np.asarray(mesh["faces"]),
                process=False)
            scene.add(pyrender.Mesh.from_trimesh(
                mesh_gl, material=pyrender.MetallicRoughnessMaterial(
                    alphaMode="BLEND", doubleSided=True,
                    baseColorFactor=tuple(
                        float(x) for x in mesh.get("rgba", (1, 1, 1, 1))),
                metallicFactor=0.0, roughnessFactor=0.8), smooth=smooth))

    own: bool = (renderer is None
           or getattr(renderer, "viewport_width", -1) != width
           or getattr(renderer, "viewport_height", -1) != height)
    active: Any = (pyrender.OffscreenRenderer(width, height)
                    if own else renderer)
    try:
        rgba_render, depth = active.render(scene,
                                           flags=pyrender.RenderFlags.RGBA)
    finally:
        if own:
            active.delete()
    if rgba_render.ndim != 3 or rgba_render.shape[2] < 4:
        return base.copy()
    alpha_u8 = rgba_render[:, :, 3]
    mask = alpha_u8 > 0
    if not np.any(mask):
        return base.copy()
    alpha_mult = float(np.clip(alpha, 0.0, 1.0))
    render_bgr = rgba_render[:, :, :3][:, :, ::-1]
    out = base.copy()
    if np.all(alpha_u8[mask] == 255):
        blended = cv2.addWeighted(render_bgr, alpha_mult, base,
                                  1.0 - alpha_mult, 0.0)
        out[mask] = blended[mask]
        return out
    source = render_bgr[mask].astype(np.float32)
    target = base[mask].astype(np.float32)
    weights = (alpha_u8[mask].astype(np.float32)
               * (alpha_mult / 255.0))[:, None]
    out[mask] = np.clip(source * weights + target * (1.0 - weights),
                        0.0, 255.0).astype(np.uint8)
    return out


class OverlayRenderer:
    """Frame-loop wrapper that reuses one offscreen GL context."""

    def __init__(self):
        self._renderer = None

    def render(self, image_bgr, w2c, intrinsics, meshes, *,
               alpha: float = 0.85, background=None, smooth: bool = True):
        height, width = image_bgr.shape[:2]
        if self._renderer is not None and (
                getattr(self._renderer, "viewport_width", -1) != width
                or getattr(self._renderer, "viewport_height", -1) != height):
            self.close()  # viewport changed: EGL contexts cannot resize
        if self._renderer is None:
            # Cache the context across frames: without this, every call
            # built and deleted a temporary OffscreenRenderer, and pyrender's
            # delete() forces a full gc.collect() each time (~0.1-0.3s x
            # 100+ overlays per review window).
            import pyrender
            self._renderer = pyrender.OffscreenRenderer(width, height)
        return render_mesh_overlay(image_bgr, w2c, intrinsics, meshes,
                                   alpha=alpha, renderer=self._renderer,
                                   background=background, smooth=smooth)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None
