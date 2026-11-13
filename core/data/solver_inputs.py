"""Prepared mesh and episode table access for solver inputs."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import numpy as np
import pyarrow.parquet as pq

@lru_cache(maxsize=None)
def _read_mesh_cached(root_string: str, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root_string)
    geoms = pq.read_table(root / "objects/geometries.parquet").to_pandas()
    row = geoms[geoms.mesh_id == int(mesh_id)]
    if len(row) != 1:
        raise KeyError(f"mesh_id={mesh_id} missing or non-unique")
    row = row.iloc[0]
    with (root / "objects/mesh_store.bin").open("rb") as f:
        f.seek(int(row.verts_offset)); verts = np.frombuffer(f.read(int(row.verts_count) * 12), np.float32).reshape(-1, 3).copy()
        f.seek(int(row.faces_offset)); faces = np.frombuffer(f.read(int(row.faces_count) * 12), np.int32).reshape(-1, 3).copy()
    return verts, faces


def _read_mesh(root: Path, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one immutable mesh, cached across episodes in batch generation."""
    return _read_mesh_cached(str(Path(root).resolve()), int(mesh_id))


@lru_cache(maxsize=None)
def _assets_table(root_string: str):
    return pq.read_table(Path(root_string) / "objects/assets.parquet").to_pandas()


def _dexycb_episode_row(root: Path, episode_index: int) -> dict:
    table = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet", filters=[("episode_index", "=", int(episode_index))])
    if table.num_rows != 1:
        raise KeyError(f"episode_index={episode_index} not found")
    return table.to_pylist()[0]


def _read_slice(root: Path, row, prefix: str):
    return pq.ParquetFile(root / str(row[f"{prefix}_path"])).read_row_group(
        int(row[f"{prefix}_row_group_index"])
    ).slice(int(row[f"{prefix}_row_offset"]), int(row[f"{prefix}_num_rows"])).to_pandas()


@lru_cache(maxsize=64)
def _read_geometry(root_string: str, geometry_id: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root_string)
    table = pq.read_table(root / "objects/geometries.parquet").to_pandas()
    match = table[table.object_geometry_id.astype(str) == geometry_id]
    if len(match) != 1:
        raise KeyError(f"geometry_id={geometry_id!r} missing or non-unique")
    row = match.iloc[0]
    with (root / "objects/mesh_store.bin").open("rb") as stream:
        stream.seek(int(row.verts_offset))
        vertices = np.frombuffer(stream.read(int(row.verts_count) * 12), np.float32).reshape(-1, 3).copy()
        stream.seek(int(row.faces_offset))
        faces = np.frombuffer(stream.read(int(row.faces_count) * 12), np.int32).reshape(-1, 3).copy()
    return vertices, faces


def _slice(root: Path, row, prefix: str):
    return pq.ParquetFile(root / str(row[f"{prefix}_path"])).read_row_group(
        int(row[f"{prefix}_row_group_index"])
    ).slice(int(row[f"{prefix}_row_offset"]), int(row[f"{prefix}_num_rows"])).to_pandas()


@lru_cache(maxsize=256)
def _mesh(root_string: str, geometry_id: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root_string)
    rows = pq.read_table(root / "objects/geometries.parquet").to_pandas()
    match = rows[rows.object_geometry_id.astype(str) == str(geometry_id)]
    if len(match) != 1:
        raise KeyError(f"geometry={geometry_id!r} missing or non-unique")
    row = match.iloc[0]
    with (root / "objects/mesh_store.bin").open("rb") as stream:
        stream.seek(int(row.verts_offset))
        vertices = np.frombuffer(stream.read(int(row.verts_count) * 12), np.float32).reshape(-1, 3).copy()
        stream.seek(int(row.faces_offset))
        faces = np.frombuffer(stream.read(int(row.faces_count) * 12), np.int32).reshape(-1, 3).copy()
    return vertices, faces


def _episode_row(root: Path, episode_index: int) -> dict:
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "=", int(episode_index))])
        if table.num_rows:
            return table.to_pylist()[0]
    raise KeyError(f"episode={episode_index} missing")


def _episode_gravity_directions(root: Path, episode: dict,
                                frame_index: np.ndarray) -> np.ndarray:
    """Load either constant episode gravity or a frame-sidecar gravity track."""
    locator = episode.get("gravity_data/path")
    if isinstance(locator, str) and locator:
        group = pq.ParquetFile(root / locator).read_row_group(
            int(episode.get("gravity_data/row_group_index", 0)),
            columns=["frame_index", "episode_index", "gravity_dir_world"],
        ).slice(int(episode.get("gravity_data/row_offset", 0)),
                int(episode["gravity_data/num_rows"]))
        rows = group.to_pandas().sort_values("frame_index")
        rows = rows[rows.episode_index.astype(int) == int(episode["episode_index"])]
        by_frame = {int(index): np.asarray(value, np.float64)
                    for index, value in zip(rows.frame_index, rows.gravity_dir_world)}
        missing = [int(index) for index in frame_index if int(index) not in by_frame]
        if missing:
            raise ValueError(f"gravity sidecar misses {len(missing)} aligned frames")
        gravity = np.stack([by_frame[int(index)] for index in frame_index])
    else:
        value = np.asarray(episode.get("gravity_dir_world"), np.float64)
        gravity = np.broadcast_to(value, (len(frame_index), 3)).copy()
    norm = np.linalg.norm(gravity, axis=1, keepdims=True)
    if gravity.shape != (len(frame_index), 3) or not np.isfinite(gravity).all() or np.any(norm < 1e-12):
        raise ValueError("episode has no valid gravity direction track")
    return gravity / norm
