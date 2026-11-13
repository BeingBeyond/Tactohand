from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Tuple, List


def compute_global_frame_idx(seq_info: dict, frame_index: int) -> int:
    return int(seq_info["video_start_frame"]) + int(frame_index)


def _get_frame_range(video_file) -> Tuple[int, int]:
    if isinstance(video_file, dict):
        frame_range = video_file.get("frame_range")
    else:
        frame_range = getattr(video_file, "frame_range", None)
    if frame_range is None:
        raise ValueError("video_file missing frame_range")
    start, end = frame_range
    return int(start), int(end)


def build_video_index(video_files: Iterable) -> Tuple[List[int], List[int], List[int]]:
    starts: List[int] = []
    ends: List[int] = []
    ids: List[int] = []

    for idx, vf in enumerate(video_files):
        start, end = _get_frame_range(vf)
        starts.append(start)
        ends.append(end)
        ids.append(idx)

    order = sorted(range(len(starts)), key=lambda i: starts[i])
    starts = [starts[i] for i in order]
    ends = [ends[i] for i in order]
    ids = [ids[i] for i in order]

    for i in range(1, len(starts)):
        if starts[i] < ends[i - 1]:
            raise ValueError("Overlapping video frame ranges detected")

    return starts, ends, ids


def locate_video_frame(
    global_frame_idx: int,
    starts: List[int],
    ends: List[int],
    ids: List[int],
) -> Tuple[int, int]:
    if not starts:
        raise ValueError("No video files available to locate frame")

    idx = bisect_right(starts, global_frame_idx) - 1
    if idx < 0 or global_frame_idx >= ends[idx]:
        raise ValueError(f"global_frame_idx {global_frame_idx} out of video ranges")

    video_id = ids[idx]
    video_frame_idx = global_frame_idx - starts[idx]
    return int(video_id), int(video_frame_idx)
