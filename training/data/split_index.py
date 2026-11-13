"""Memory-efficient access to restored frame-level dataset splits."""
from __future__ import annotations
import core.data.processed_compat as processed_contract
import pyarrow.compute as pc

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow.parquet as pq

SplitKey = tuple[int, Optional[str]]
FrameInterval = tuple[int, int]


def split_sidecar_path(root: Path) -> Optional[Path]:
    for name in ("official_splits.parquet", "local_splits.parquet"):
        path = Path(root) / "meta" / name
        if path.is_file():
            return path
    return None


def available_splits(root: Path) -> list[str]:
    if processed_contract.is_processed(root):
        table = processed_contract.split_table(root, ['split'])
        if table is None:
            return []
        return sorted(v for v in table['split'].unique().to_pylist() if v is not None)
    path = split_sidecar_path(root)
    if path is None:
        return []
    values = pq.read_table(path, columns=["split"])[0]
    return sorted(set(map(str, values.unique().to_pylist())))


def read_split_intervals(
    root: Path, splits: Iterable[str], protocol: str | None = None
) -> Optional[dict[SplitKey, tuple[FrameInterval, ...]]]:
    """Return selected frame ranges without materializing rows as Python dicts."""
    wanted = sorted(set(map(str, splits)))
    if "all" in wanted:
        return None
    if processed_contract.is_processed(root):
        table = processed_contract.split_table(root)
        if table is None:
            return None
        protocols=[v for v in table['protocol'].unique().to_pylist() if v is not None]
        if protocol is None and len(protocols)>1:
            raise ValueError('Multiple split protocols: select protocol explicitly')
        if protocol is not None:table=table.filter(pc.equal(table['protocol'],protocol))
        table=table.filter(pc.is_in(table['split'],value_set=__import__('pyarrow').array(wanted)))
        if not table.num_rows:return {}
        result=defaultdict(list)
        for row in table.select([c for c in ['episode_index','view_id','start_frame','end_frame','mapping_status'] if c in table.column_names]).to_pylist():
            if row['episode_index'] is None or row['start_frame'] is None:continue
            key=(int(row['episode_index']),row.get('view_id'))
            a,b=int(row['start_frame']),int(row['end_frame'])
            result[key].append((a,b))
        merged = {}
        for key, values in result.items():
            intervals = []
            for a, b in sorted(values):
                if intervals and a <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]))
                else:
                    intervals.append((a, b))
            merged[key] = tuple(intervals)
        return merged
    path = split_sidecar_path(root)
    if path is None:
        return None
    schema = pq.read_schema(path)
    required = {"episode_index", "frame_index", "split"}
    if not required.issubset(schema.names):
        raise ValueError(f"invalid split sidecar schema: {path}")
    columns = ["episode_index", "frame_index"]
    has_view = "view_id" in schema.names
    if has_view:
        columns.append("view_id")
    filters = [("split", "in", wanted)]
    table = pq.read_table(path, columns=columns, filters=filters)
    if table.num_rows == 0:
        return {}
    episode = table["episode_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    frame = table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    view = (np.asarray(table["view_id"].to_pylist(), dtype=object)
            if has_view else np.full(table.num_rows, None, dtype=object))
    order = np.lexsort((frame, view.astype(str), episode))
    episode, frame, view = episode[order], frame[order], view[order]
    boundary = np.ones(len(frame), dtype=bool)
    boundary[1:] = ((episode[1:] != episode[:-1])
                    | (view[1:] != view[:-1])
                    | (frame[1:] != frame[:-1] + 1))
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:] - 1, len(frame) - 1]
    result: dict[SplitKey, list[FrameInterval]] = defaultdict(list)
    for start, end in zip(starts.tolist(), ends.tolist()):
        key = (int(episode[start]), str(view[start]) if has_view else None)
        result[key].append((int(frame[start]), int(frame[end]) + 1))
    return {key: tuple(intervals) for key, intervals in result.items()}


def frames_in_intervals(frames: Iterable[int], intervals: tuple[FrameInterval, ...]) -> bool:
    """Whether every requested frame belongs to one selected half-open interval."""
    values = np.asarray(list(frames), dtype=np.int64)
    if values.size == 0:
        return False
    for start, end in intervals:
        mask = (values >= start) & (values < end)
        values = values[~mask]
        if values.size == 0:
            return True
    return False
