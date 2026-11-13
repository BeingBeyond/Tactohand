"""Drawing primitives over BGR images (design §2 L2, §4 contracts)."""
from __future__ import annotations

import cv2
import numpy as np

from core.geometry.projection import project_world


def put_text(image: np.ndarray, text: str, origin: tuple[int, int],
             scale: float = 0.42, color: tuple[int, int, int] = (255, 255, 255)
             ) -> None:
    """Outlined text: black halo + colored face, readable on any backdrop."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_mesh_projection(image: np.ndarray, uv: np.ndarray,
                         visible: np.ndarray, faces: np.ndarray,
                         color: tuple[int, int, int], alpha: float,
                         max_faces: int | None = 2500) -> None:
    safe_uv = np.clip(np.nan_to_num(uv, nan=-1e6, posinf=1e6, neginf=-1e6),
                      -1e6, 1e6)
    pixels = np.rint(safe_uv).astype(np.int32)
    valid_faces = np.asarray(faces)[visible[np.asarray(faces)].all(axis=1)]
    # A missing or badly conditioned hand can project behind the near plane to
    # enormous screen coordinates. Do not connect those vertices across the
    # complete image. Real mesh triangles remain local in image space.
    if len(valid_faces):
        triangles = pixels[valid_faces]
        height, width = image.shape[:2]
        inside = ((triangles[..., 0] >= 0) & (triangles[..., 0] < width)
                  & (triangles[..., 1] >= 0)
                  & (triangles[..., 1] < height)).any(axis=1)
        edges = np.stack((triangles[:, 1] - triangles[:, 0],
                          triangles[:, 2] - triangles[:, 1],
                          triangles[:, 0] - triangles[:, 2]), axis=1)
        local = (np.linalg.norm(edges, axis=2).max(axis=1)
                 <= .75 * max(width, height))
        valid_faces = valid_faces[inside & local]
    if max_faces is not None and len(valid_faces) > max_faces:
        valid_faces = valid_faces[::max(1, len(valid_faces) // max_faces)]
    layer = image.copy()
    for face in valid_faces:
        cv2.polylines(layer, [pixels[face]], True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, image)


def draw_mesh_world(image: np.ndarray, points_world: np.ndarray,
                    faces: np.ndarray, w2c: np.ndarray, camera,
                    color: tuple[int, int, int], alpha: float,
                    max_faces: int | None = 2500) -> None:
    uv, visible = project_world(points_world, w2c, camera)
    draw_mesh_projection(image, uv, visible, faces, color, alpha, max_faces)


def draw_vertex_cloud(image: np.ndarray, uv: np.ndarray,
                      visible: np.ndarray, colors: np.ndarray,
                      radii: np.ndarray, stride: int = 3) -> int:
    """Draw projected vertices as colored dots; returns the count drawn.

    ``colors`` is (N, 3) uint8 BGR and ``radii`` (N,) int; both indexed like
    ``uv``.  ``stride`` subsamples dense clouds (778-vertex MANO hands stay
    readable without saturating the frame).
    """
    height, width = image.shape[:2]
    uv = np.asarray(uv)
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    pick = np.flatnonzero(visible & inside)[::max(1, stride)]
    for index in pick:
        cv2.circle(image, (int(uv[index, 0]), int(uv[index, 1])),
                   int(radii[index]),
                   tuple(int(value) for value in colors[index]), -1,
                   cv2.LINE_AA)
    return len(pick)


def draw_contact_markers(image: np.ndarray, vertices_world: np.ndarray,
                         w2c: np.ndarray, camera, mask: np.ndarray, *,
                         color: tuple[int, int, int] = (0, 0, 255),
                         radius: int = 2, stride: int = 1) -> int:
    """Occlusion-immune contact markers; returns the count drawn.

    Projects the masked world-space vertices and stamps haloed dots ON TOP of
    the already-composited frame — call after mesh rendering — so contact
    stays readable where a nearer surface (other hand, curled finger, object)
    would hide it.  The dark halo keeps the dots legible over same-coloured
    mesh regions.
    """
    vertices_world = np.asarray(vertices_world, np.float64).reshape(-1, 3)
    mask = np.asarray(mask, bool).reshape(-1)
    if not mask.any():
        return 0
    uv, visible = project_world(vertices_world[mask], w2c, camera)
    height, width = image.shape[:2]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    pick = np.flatnonzero(visible & inside)[::max(1, int(stride))]
    for index in pick:
        center = (int(uv[index, 0]), int(uv[index, 1]))
        cv2.circle(image, center, int(radius) + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, center, int(radius), tuple(color), -1, cv2.LINE_AA)
    return len(pick)


def draw_arrow_world(image: np.ndarray, origin: np.ndarray,
                     vector: np.ndarray, w2c: np.ndarray, camera,
                     pixels: float, color: tuple[int, int, int],
                     thickness: int, label: str = "") -> bool:
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-10:
        return False
    pair = np.stack((origin, origin + .08 * vector / magnitude))
    uv, visible = project_world(pair, w2c, camera)
    delta = uv[1] - uv[0]
    norm = float(np.linalg.norm(delta))
    if not visible.all() or norm < .25:
        return False
    begin = np.rint(uv[0]).astype(int)
    end = np.rint(uv[0] + pixels * delta / norm).astype(int)
    cv2.circle(image, tuple(begin), max(3, thickness), color, -1, cv2.LINE_AA)
    cv2.arrowedLine(image, tuple(begin), tuple(end), color, thickness,
                    cv2.LINE_AA, tipLength=.22)
    if label:
        cv2.putText(image, label, tuple(end + np.array((5, -5))),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, color, 2, cv2.LINE_AA)
    return True


def draw_force_legend(image: np.ndarray, scale_n: float) -> None:
    height = image.shape[0]
    x0, x1, y0, y1 = 18, 178, height - 30, height - 18
    gradient = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None],
                                 cv2.COLORMAP_TURBO)
    image[y0:y1, x0:x1] = cv2.resize(gradient, (x1 - x0, y1 - y0))
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 1)
    cv2.putText(image, "0", (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, .36,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f">={scale_n:.1f} N", (x1 - 50, y0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1,
                cv2.LINE_AA)


def draw_legend(image: np.ndarray, entries: list[tuple[str,
                                                       tuple[int, int, int]]],
                origin: tuple[int, int] = (10, 18)) -> None:
    """Color-swatch legend for label maps (contact/proximity/force)."""
    x, y = origin
    for text, color in entries:
        cv2.rectangle(image, (x, y - 9), (x + 14, y + 5), color, -1)
        put_text(image, text, (x + 20, y + 2), 0.40)
        y += 18


def draw_header(image: np.ndarray,
                lines: list[tuple[str, tuple[int, int, int]]],
                *, line_height: int = 20) -> None:
    height = max(1, len(lines)) * line_height + 7
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    for index, (content, color) in enumerate(lines):
        cv2.putText(image, content, (8, 19 + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, color, 1, cv2.LINE_AA)
