"""Object mesh store access + solid-silhouette world overlay (design §2 L2).

The normalized ``_processed`` object contract stores every triangle mesh in
one binary mesh blob addressed by the static object registry (``assets.json``
for object_assets_v1, or legacy Parquet tables)
(byte offsets into the blob); ``instances.parquet`` + ``assets.parquet`` map
an ``object_instance_id`` to its geometry.  Overlay style follows the
unihandv3_tools checker: one stable palette colour per object key, a single
batched ``cv2.fillPoly`` per frame, and a 0.72/0.28 alpha blend — the
silhouette is meant to be drawn beneath the shaded hand meshes so the hands
stay readable.
"""
from __future__ import annotations
import core.data.processed_compat as processed_contract
from core.geometry.object_assets import AssetIndex, read_asset_table

from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import pyarrow.parquet as pq

from core.visualization.drawing import put_text
from core.geometry.projection import project_world

_MESH_STORES: dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
_INSTANCE_GEOMETRY: dict[tuple[str, int], dict[str, str]] = {}
_PARTS_BY_ASSET: dict[tuple[str, int], dict[str, list[dict]]] = {}

# BGR, kept clear of the hand colours (skin / contact red / rest blue).
_PALETTE_BGR = (
    (255, 220, 0),    # cyan
    (0, 140, 255),    # orange
    (60, 230, 60),    # green
    (255, 60, 220),   # magenta
    (255, 100, 30),   # teal
    (60, 60, 230),    # red-brown
    (180, 255, 80),   # mint
    (230, 160, 60),   # sky blue
)


def _cache_key(root: Path) -> tuple[str, int]:
    info = root / 'meta/info.json'
    declaration = json.loads(info.read_text()) if info.exists() else {}
    manifest = root / declaration.get('manifest', 'meta/index/manifest.json')
    identity = f"{root}::{declaration.get('revision', 'legacy')}::{manifest}"
    return identity, manifest.stat().st_mtime_ns if manifest.exists() else 0


def load_object_meshes(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the dataset mesh blob once; returns ``geometry_id -> (verts, faces)``."""
    root = Path(root).resolve()
    cached = _MESH_STORES.get(_cache_key(root))
    if cached is not None:
        return cached
    geometries = read_asset_table(
        processed_contract.object_path(root, 'geometries_path', 'objects/geometries.parquet'), 'geometries',
        columns=["object_geometry_id", "verts_offset", "verts_count",
                 "faces_offset", "faces_count"]).to_pylist()
    store = np.memmap(processed_contract.object_path(root, 'mesh_store_path', 'objects/mesh_store.bin'), dtype=np.uint8,
                      mode="r")
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in geometries:
        vertices = np.ndarray((int(row["verts_count"]), 3), dtype="<f4",
                              buffer=store,
                              offset=int(row["verts_offset"])).copy()
        faces = np.ndarray((int(row["faces_count"]), 3), dtype="<i4",
                           buffer=store,
                           offset=int(row["faces_offset"])).copy()
        meshes[str(row["object_geometry_id"])] = (vertices, faces)
    _MESH_STORES[_cache_key(root)] = meshes
    return meshes


def instance_geometry_ids(root: Path) -> dict[str, str]:
    """``object_instance_id -> object_geometry_id`` via instances × assets."""
    root = Path(root).resolve()
    cached = _INSTANCE_GEOMETRY.get(_cache_key(root))
    if cached is not None:
        return cached
    from core.geometry.object_assets import instance_geometry_ids as shared_join
    mapping = shared_join(root)
    _INSTANCE_GEOMETRY[_cache_key(root)] = mapping
    return mapping


def load_object_parts(root: Path) -> dict[str, list[dict]]:
    """``object_asset_id -> part rows`` in ``part_index`` order (empty dict
    when the dataset carries no articulated parts table)."""
    root = Path(root).resolve()
    cached = _PARTS_BY_ASSET.get(_cache_key(root))
    if cached is not None:
        return cached
    parts: dict[str, list[dict]] = {}
    try:
        parts_path = processed_contract.object_path(root, 'parts_path', 'objects/parts.parquet')
    except FileNotFoundError:
        return {}
    if parts_path.is_file():
        for row in read_asset_table(parts_path, 'parts').to_pylist():
            parts.setdefault(str(row["object_asset_id"]), []).append(row)
        for rows in parts.values():
            rows.sort(key=lambda row: int(row["part_index"]))
    _PARTS_BY_ASSET[_cache_key(root)] = parts
    return parts


def episode_articulated_asset(root: Path, episode_index: int,
                              object_name: str) -> str | None:
    """Asset id of the episode's articulated object (name-delimiter match on
    the asset id), or None when no instance/asset pair matches."""
    root = Path(root).resolve()
    instances = processed_contract.object_instances(root, int(episode_index)).to_pylist()
    marker = f":{object_name}:"
    for row in instances:
        asset_id = str(row["object_asset_id"])
        if marker in f"{asset_id}:":
            return asset_id
    return None


def object_color(key: str) -> tuple[int, int, int]:
    """Stable BGR colour for an object key (sha256 into a fixed palette)."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return _PALETTE_BGR[digest[0] % len(_PALETTE_BGR)]


def pose_object_vertices(vertices: np.ndarray, com_body: np.ndarray,
                         rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Body-frame mesh vertices → world, COM placed at ``position``.

    The solver state stores the object COM world position plus a body→world
    rotation, while the mesh blob keeps raw body-frame geometry (whose origin
    is generally not the COM, hence ``com_body``).
    """
    offset = np.asarray(vertices, np.float64) \
        - np.asarray(com_body, np.float64).reshape(3)
    return offset @ np.asarray(rotation, np.float64).reshape(3, 3).T \
        + np.asarray(position, np.float64).reshape(3)


def draw_object_silhouette(canvas: np.ndarray, vertices_world: np.ndarray,
                           faces: np.ndarray, w2c: np.ndarray, camera,
                           color: tuple[int, int, int], *, alpha: float = 0.72,
                           label: str | None = None) -> np.ndarray:
    """Solid fillPoly silhouette of a posed world-space mesh on ``canvas``.

    ``camera`` carries already-scaled intrinsics so projected uv land directly
    in canvas pixel space.  Faces are clipped and batched into one
    ``cv2.fillPoly`` call (mesh silhouettes only — no z-buffer), then blended
    at ``alpha``.  ``label`` is drawn at the projected vertex centroid.
    """
    height, width = canvas.shape[:2]
    faces = np.asarray(faces)
    uv, visible = project_world(vertices_world, w2c, camera)
    face_uv = np.rint(uv[faces]).astype(np.int32)
    face_uv = face_uv[visible[faces].all(axis=1)]
    if len(face_uv):
        on_screen = ((face_uv[:, :, 0].max(axis=1) >= 0)
                     & (face_uv[:, :, 1].max(axis=1) >= 0)
                     & (face_uv[:, :, 0].min(axis=1) < width)
                     & (face_uv[:, :, 1].min(axis=1) < height))
        if on_screen.any():
            overlay = canvas.copy()
            cv2.fillPoly(overlay, face_uv[on_screen], color, cv2.LINE_AA)
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)
    if label:
        points = np.flatnonzero(visible)
        if points.size:
            x, y = uv[points].mean(axis=0)
            put_text(canvas, label[:34], (max(2, int(round(x))),
                                          max(16, int(round(y)) - 6)), 0.42)
    return canvas
