"""Vectorized labels-v5 episode reading and a byte-bounded episode cache."""
from __future__ import annotations
import core.data.processed_compat as processed_contract

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.tactile_labels.schema import (
    FORCE_TIMELINE_COLUMNS,
    LABELS_REVISION,
    LABELS_SCHEMA_NAME,
    SAMPLE_TIMELINE_COLUMNS,
)

# A9 content hash: every payload column in schema order (matching the
# staging writer); articulated columns join only for articulated episodes.
NATIVE_TACTILE_HASH_COLUMNS = ("solver_sample", "source_position", "flags",
                               "contact", "signed_distance")
FORCE_HASH_COLUMNS = ("solver_transition", "flags", "vertex_force_world",
                      "contact_wrench", "environment_wrench",
                      "object_position", "object_rotation", "object_velocity")


def episode_sha1(labels_root: Path, episode: dict) -> tuple[str, str]:
    """Content hash of one episode's label rows (and force rows if any).

    Single source of truth for the ``tactile_sha1``/``force_sha1`` values
    in ``meta/label_index.parquet`` — writers (materialize, converters,
    repair tools) and the validator MUST hash through this function so a
    legitimate rewrite never reads as a stale index.
    """
    columns = NATIVE_TACTILE_HASH_COLUMNS
    tactile = read_episode_columns(labels_root, episode["tactile_groups"], columns)
    tactile_hash = hashlib.sha1()
    for name in columns:
        tactile_hash.update(np.ascontiguousarray(tactile[name]))
    if not episode["force_groups"]:
        return tactile_hash.hexdigest(), ""
    force_columns = FORCE_HASH_COLUMNS
    if int(episode.get("generalized_wrench_dim", 0)) > 0:
        force_columns += ("generalized_wrench", "joint_state")
    force = read_episode_columns(
        labels_root, episode["force_groups"], force_columns)
    if episode.get("topology") == "static":
        force.update(read_optional_episode_columns(
            labels_root, episode["force_groups"],
            ("static_gravity_world", "static_object_index")))
    force_hash = hashlib.sha1()
    for value in force.values():
        force_hash.update(np.ascontiguousarray(value))
    return tactile_hash.hexdigest(), force_hash.hexdigest()


def validate_labels_info(labels_root: Path) -> dict:
    """Read and check ``meta/info.json``; raises on a foreign contract."""
    info = processed_contract.read_info(Path(labels_root))
    if info.get("mano_dependent_labels", {}).get("state") == "stale":
        raise ValueError(f"{labels_root}: MANO geometry changed; recompute contact/force labels before training")
    declared = info.get("beingh", {}).get("labels", {})
    if str(declared.get("schema")) != LABELS_SCHEMA_NAME or int(
            declared.get("revision", -1)) != LABELS_REVISION:
        raise ValueError(
            f"{labels_root}: labels schema {declared!r} is not "
            f"{LABELS_SCHEMA_NAME} rev {LABELS_REVISION}")
    return declared


def load_label_index(labels_root: Path, episode_indices: list[int] | None = None,
                     *, lazy_groups: bool = False) -> list[dict]:
    if processed_contract.is_processed(labels_root):
        return processed_contract.label_index(
            labels_root, episode_indices, lazy_groups=lazy_groups)
    table = pq.read_table(Path(labels_root) / "meta/label_index.parquet")
    rows = table.to_pylist()
    if episode_indices is None:
        return rows
    wanted = set(episode_indices)
    return [r for r in rows if int(r['episode_index']) in wanted]


def load_label_stats(labels_root: Path) -> dict:
    if processed_contract.is_processed(labels_root):
        return processed_contract.label_stats(labels_root)
    return json.loads((Path(labels_root) / "meta/label_stats.json").read_text())


def _decode_column(column: pa.ChunkedArray, rows: int) -> np.ndarray:
    combined = column.combine_chunks()
    if isinstance(combined, pa.ChunkedArray):
        if combined.num_chunks == 1:
            combined = combined.chunk(0)
        else:  # multi-chunk fixed-size lists decode through the values array
            widths = combined.chunk(0).type.list_size
            values = np.concatenate(
                [chunk.flatten().to_numpy(zero_copy_only=False)
                 for chunk in combined.chunks])
            if values.size != rows * widths:
                raise ValueError("inconsistent fixed_size_list payload")
            return values.reshape(rows, widths)
    if pa.types.is_fixed_size_list(combined.type):
        width = combined.type.list_size
        values = combined.flatten().to_numpy(zero_copy_only=False)
        if values.size != rows * width:
            raise ValueError(
                f"fixed_size_list column holds {values.size} values, "
                f"expected {rows}x{width}")
        return values.reshape(rows, width)
    if pa.types.is_integer(combined.type) or pa.types.is_floating(combined.type):
        return combined.to_numpy(zero_copy_only=False)
    if pa.types.is_boolean(combined.type):
        return np.asarray(combined.to_pylist(), dtype=bool)
    if pa.types.is_list(combined.type):
        values=combined.to_pylist()
        widths={len(value) for value in values if value is not None}
        if len(widths)!=1:
            raise ValueError('Numeric label lists require one declared non-null width')
        width=widths.pop()
        result=np.full((rows,width),np.nan,combined.type.value_type.to_pandas_dtype())
        for i,value in enumerate(values):
            if value is not None:result[i]=value
        return result
    raise TypeError(f"unsupported label column type {combined.type}")


def native_tactile_sha1(table: pa.Table) -> str:
    """Hash native tactile values in solver-row order, independent of packing."""
    digest = hashlib.sha1()
    for name in NATIVE_TACTILE_HASH_COLUMNS:
        values = _decode_column(table[name], table.num_rows)
        digest.update(np.ascontiguousarray(values))
    return digest.hexdigest()


class LabelFileHandle:
    """One parquet label file with cached row-group presence."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._parquet: pq.ParquetFile | None = None

    def read(self, row_group: int, columns: Sequence[str]) -> dict[str, np.ndarray]:
        if self._parquet is None:
            self._parquet = pq.ParquetFile(self.path)
        table = self._parquet.read_row_group(int(row_group), columns=list(columns))
        rows = table.num_rows
        missing = [name for name in columns if name not in table.column_names]
        if missing:
            raise ValueError(f"{self.path}: row group lacks columns {missing}")
        return {name: _decode_column(table.column(name), rows)
                for name in columns}


def read_episode_columns(labels_root: Path, groups: Iterable[dict],
                          columns: Sequence[str]) -> dict[str, np.ndarray]:
    """Read one episode's row groups (usually one) and concatenate by row order."""
    source = groups.timeline_source if isinstance(groups, processed_contract.TactileGroups) else None
    groups = list(groups)
    virtual = {}
    if source is not None:
        reader, episode, modality = source
        mapping = SAMPLE_TIMELINE_COLUMNS if modality == "tactile" else FORCE_TIMELINE_COLUMNS
        index_column = reader.spec(modality).get("solver_index_column", "solver_row")
        if index_column != "solver_row":
            mapping = {name: index_column if field == "solver_row" else field
                       for name, field in mapping.items()}
        requested = {name: mapping[name] for name in columns if name in mapping}
        if requested:
            timeline = reader.read_episode("tactile.timeline", episode, "shared",
                                           list(dict.fromkeys(requested.values())))
            if len(timeline) != sum(group['num_rows'] for group in groups):
                raise ValueError(f"episode {episode}: timeline and {modality} row counts differ")
            for name, field in requested.items():
                values = _decode_column(timeline[field], len(timeline))
                if field == "force_valid":
                    values = values.astype(np.uint8)
                elif name in ("solver_sample", "solver_transition"):
                    values = values.astype(np.int32, copy=False)
                virtual[name] = values
    physical = [name for name in columns if name not in virtual]
    if virtual and not physical:
        return virtual
    arrays: list[dict[str, np.ndarray]] = []
    total = 0
    for group in groups:
        handle = LabelFileHandle(Path(labels_root) / str(group["path"]))
        values = handle.read(int(group["row_group"]), physical)
        if 'row_offset' in group:
            start = int(group['row_offset'])
            count = int(group['num_rows'])
            values = {name: value[start:start+count] for name,value in values.items()}
        arrays.append(values)
        total += values[physical[0]].shape[0]
    if not arrays:
        raise ValueError("episode has no label row groups")
    result: dict[str, np.ndarray] = {}
    for name in columns:
        if name in virtual:
            result[name] = virtual[name]
            continue
        pieces = [piece[name] for piece in arrays]
        result[name] = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        if result[name].shape[0] != total:
            raise ValueError(f"column {name} row count mismatch")
    return result


def read_optional_episode_columns(labels_root: Path, groups: Iterable[dict],
                                  columns: Sequence[str]) -> dict[str, np.ndarray]:
    """Read additive metadata without requiring a rewrite of older labels.

    A column must exist in every fragment or none; partial metadata must not
    silently shift the source-row alignment.
    """
    groups = list(groups)
    if not groups:
        return {}
    schemas = [set(pq.ParquetFile(Path(labels_root) / str(g["path"])).schema_arrow.names)
               for g in groups]
    available = []
    for name in columns:
        present = [name in schema for schema in schemas]
        if any(present) and not all(present):
            raise ValueError(f"Inconsistent optional label column: {name}")
        if all(present):
            available.append(name)
    return read_episode_columns(labels_root, groups, available) if available else {}


class ByteBoundedEpisodeCache:
    """LRU over decoded episode label dicts, evicted by payload bytes."""

    def __init__(self, loader, max_bytes: int):
        self._loader = loader
        self._max_bytes = int(max_bytes)
        self._entries: OrderedDict[tuple, tuple[dict, int]] = OrderedDict()
        self.stats = {"loads": 0, "hits": 0, "evictions": 0}

    def __getitem__(self, key: tuple) -> dict:
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.stats["hits"] += 1
            return cached[0]
        value = self._loader(key)
        size = sum(int(array.nbytes) for array in value.values())
        self._entries[key] = (value, size)
        self._entries.move_to_end(key)
        self.stats["loads"] += 1
        while len(self._entries) > 1 and self._used_bytes() > self._max_bytes:
            self._entries.popitem(last=False)
            self.stats["evictions"] += 1
        return value

    def _used_bytes(self) -> int:
        return sum(size for _, size in self._entries.values())
