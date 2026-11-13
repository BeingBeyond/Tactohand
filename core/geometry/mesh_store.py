from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class MeshIndexRow:
    mesh_id: int
    mesh_key: str
    obj_name: str
    obj_label: int
    mesh_path: str
    verts_offset: int
    verts_count: int
    faces_offset: int
    faces_count: int
    bbox_min: List[float]
    bbox_max: List[float]


class MeshStoreBuilder:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = store_path.open("wb")
        self._offset = 0
        self._rows: List[MeshIndexRow] = []
        self._key_to_id: Dict[str, int] = {}

    def add_mesh(
        self,
        mesh_key: str,
        mesh_path: Path,
        obj_name: str,
        obj_label: int,
        mesh_scale: float = 1.0,
        mesh_translation: List[float] | Tuple[float, float, float] | None = None,
    ) -> int:
        if mesh_key in self._key_to_id:
            return self._key_to_id[mesh_key]

        verts, faces = load_mesh(mesh_path)
        if verts.size == 0 or faces.size == 0:
            return -1

        verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        if mesh_scale != 1.0:
            verts = verts * float(mesh_scale)
        if mesh_translation is not None:
            trans = np.asarray(mesh_translation, dtype=np.float32).reshape(-1)
            if trans.size >= 3:
                verts = verts + trans[:3].reshape(1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3).astype(np.uint32)

        bbox_min = verts.min(axis=0).tolist()
        bbox_max = verts.max(axis=0).tolist()

        verts_offset = self._offset
        verts_bytes = verts.tobytes(order="C")
        self._fp.write(verts_bytes)
        self._offset += len(verts_bytes)

        faces_offset = self._offset
        faces_bytes = faces.tobytes(order="C")
        self._fp.write(faces_bytes)
        self._offset += len(faces_bytes)

        mesh_id = len(self._rows)
        row = MeshIndexRow(
            mesh_id=mesh_id,
            mesh_key=str(mesh_key),
            obj_name=str(obj_name),
            obj_label=int(obj_label),
            mesh_path=str(mesh_path),
            verts_offset=int(verts_offset),
            verts_count=int(verts.shape[0]),
            faces_offset=int(faces_offset),
            faces_count=int(faces.shape[0]),
            bbox_min=[float(x) for x in bbox_min],
            bbox_max=[float(x) for x in bbox_max],
        )
        self._rows.append(row)
        self._key_to_id[str(mesh_key)] = mesh_id
        return mesh_id

    def finalize(self, index_path: Path) -> List[MeshIndexRow]:
        self._fp.close()
        index_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [row.__dict__ for row in self._rows]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, index_path)
        return self._rows


def load_mesh_index(index_path: Path) -> List[dict]:
    if not index_path.exists():
        return []
    table = pq.read_table(index_path)
    return table.to_pylist()


def build_mesh_key_map(rows: Iterable[dict]) -> Dict[str, int]:
    mesh_map: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("mesh_key", ""))
        if key:
            mesh_map[key] = int(row.get("mesh_id", -1))
    return mesh_map


def ensure_mesh_store(
    mesh_items: Iterable[dict],
    store_path: Path,
    index_path: Path,
) -> Dict[str, int]:
    if store_path.exists() and index_path.exists():
        rows = load_mesh_index(index_path)
        return build_mesh_key_map(rows)

    mesh_items = list(mesh_items)
    if not mesh_items:
        return {}

    builder = MeshStoreBuilder(store_path)
    for item in mesh_items:
        mesh_path = Path(item["mesh_path"])
        if not mesh_path.exists():
            continue
        builder.add_mesh(
            mesh_key=str(item.get("mesh_key", "")),
            mesh_path=mesh_path,
            obj_name=item.get("obj_name", ""),
            obj_label=int(item.get("obj_label", -1)),
            mesh_scale=float(item.get("mesh_scale", 1.0) or 1.0),
            mesh_translation=item.get("mesh_translation", None),
        )

    rows = builder.finalize(index_path)
    return build_mesh_key_map([row.__dict__ for row in rows])


def load_mesh(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return load_obj(mesh_path)

    try:
        import trimesh

        mesh = trimesh.load(str(mesh_path), force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return verts, faces
    except Exception:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)


def load_obj(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    with mesh_path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except Exception:
                        continue
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    continue
                idxs: List[int] = []
                for part in parts:
                    try:
                        token = part.split("/")[0]
                        vi = int(token)
                        if vi < 0:
                            vi = len(verts) + vi + 1
                        idxs.append(vi - 1)
                    except Exception:
                        continue
                if len(idxs) < 3:
                    continue
                v0 = idxs[0]
                for i in range(1, len(idxs) - 1):
                    faces.append([v0, idxs[i], idxs[i + 1]])

    if not verts or not faces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)
