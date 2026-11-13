from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def build_seq_row_ranges(rows: List[dict]) -> List[Tuple[int, int, int]]:
    if not rows:
        return []

    ranges: List[Tuple[int, int, int]] = []
    current_seq = int(rows[0]["seq_id"])
    start = 0
    for i, row in enumerate(rows):
        seq_id = int(row["seq_id"])
        if seq_id != current_seq:
            ranges.append((current_seq, start, i))
            current_seq = seq_id
            start = i
    ranges.append((current_seq, start, len(rows)))
    return ranges


def write_seq_row_index(output_root: Path, entries: List[dict]) -> None:
    if not entries:
        return
    out_path = output_root / "meta" / "seq_row_index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(entries)
    pq.write_table(table, out_path)


def _get_frame_range(video_file) -> Tuple[int, int]:
    if isinstance(video_file, dict):
        frame_range = video_file.get("frame_range")
    else:
        frame_range = getattr(video_file, "frame_range", None)
    if frame_range is None:
        raise ValueError("video_file missing frame_range")
    start, end = frame_range
    return int(start), int(end)


def build_seq_video_ranges(sequences: Iterable[dict], video_files: Iterable) -> List[dict]:
    ranges: List[dict] = []
    indexed_video_files = [
        (vid, *_get_frame_range(vf))
        for vid, vf in enumerate(video_files)
    ]
    indexed_video_files.sort(key=lambda item: item[1])
    video_cursor = 0
    for seq in sequences:
        seq_id = int(seq["seq_id"])
        seq_start = int(seq.get("video_start_frame", 0))
        num_frames = int(seq.get("num_frames", 0))
        seq_end = seq_start + num_frames
        split = seq.get("split", "")
        if num_frames <= 0:
            continue

        while video_cursor < len(indexed_video_files) and indexed_video_files[video_cursor][2] <= seq_start:
            video_cursor += 1

        scan_cursor = video_cursor
        while scan_cursor < len(indexed_video_files):
            vid, vf_start, vf_end = indexed_video_files[scan_cursor]
            if vf_start >= seq_end:
                break
            if vf_end <= seq_start or vf_start >= seq_end:
                scan_cursor += 1
                continue
            inter_start = max(seq_start, vf_start)
            inter_end = min(seq_end, vf_end)
            if inter_end <= inter_start:
                scan_cursor += 1
                continue
            ranges.append({
                "seq_id": seq_id,
                "split": split,
                "global_video_id": int(vid),
                "seq_frame_start": inter_start - seq_start,
                "seq_frame_end": inter_end - seq_start,
                "global_frame_start": inter_start,
                "global_frame_end": inter_end,
                "global_video_frame_start": inter_start - vf_start,
                "global_video_frame_end": inter_end - vf_start,
            })
            scan_cursor += 1
    return ranges


def write_seq_video_ranges(output_root: Path, entries: List[dict]) -> None:
    if not entries:
        return
    out_path = output_root / "meta" / "seq_video_ranges.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(entries)
    pq.write_table(table, out_path)
