"""Index-driven reader for installed handcrop streams of a ``_processed`` root.

Layout (rich / prox / mow): per (episode, view, ``crop_type='hand'``,
``target_id`` in {left, right}) one segment covering a canonical frame
range, packed sequentially into shared mp4 files.  The segment table
``meta/cropped_video_observations.parquet`` lists, per segment, the
canonical ``frame_offset``/``num_frames`` and the crop geometry rows
(``meta/crop_data/hand/v1/...`` for legacy data, ``crop_data/hand/...``
for v2: per-frame boxes in explicitly declared coordinate units, plus
independent ``frame_valid``/``crop_valid``). Legacy units reference a raw
or virtual canvas; v2 uses unbounded normalized observation coordinates.

In-file offsets default to cumulative ``num_frames`` in table order. Roots
declaring ``file_frame_offset_source=from_timestamp`` instead preserve gaps
between retained segments by converting each stored start timestamp to frames.

The training stream and the labels-v5 review visualizer both read hand
crops through this index so the canonical-frame -> crop-frame mapping
and the crop-aware intrinsics live in one place.
"""
from __future__ import annotations
import core.data.processed_compat as processed_contract

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import re

import pyarrow as pa
import pyarrow.compute as pc

import numpy as np
import pyarrow.parquet as pq

from .handcrop_storage import decode_cam_intr


HANDCROP_STREAM_LAYOUT = "stream_chunk_v1"


def handcrop_file_offset(row: dict, cumulative: int, fps: float, addressing: str) -> int:
    """Resolve packed offsets without losing gaps between retained segments."""
    if addressing == 'from_timestamp':
        offset = int(round(float(row['from_timestamp']) * fps))
        if offset < 0:
            raise ValueError('Negative crop file offset')
        return offset
    if addressing != 'cumulative':
        raise ValueError(f'Unknown crop frame addressing: {addressing}')
    return cumulative


def declared_crop_canvas(root: Path, video_key: str | None = None) -> tuple[int, int] | None:
    """Resolve stored box units independently of the RGB storage resolution.

    A shared processed/raw stream may have a different size from the virtual
    canvas used when crops were generated. Mixed canvases require a view key.
    """
    info = processed_contract.read_info(Path(root))
    declaration = info.get('beingh_human', {}).get('crop_coordinate_space', {})
    if declaration.get('kind') == 'normalized_observation':
        # Unit canvas describes geometry, not a physical one-pixel RGB stream.
        return 1, 1
    by_key = declaration.get('by_video_key', {})
    if by_key:
        if video_key is None:
            sizes = {(int(d['raw_width']), int(d['raw_height'])) for d in by_key.values()}
            if len(sizes) != 1:
                return None
            width, height = next(iter(sizes))
            if min(width, height) <= 0:
                raise ValueError('Invalid declared crop coordinate canvas')
            return width, height
        key = str(video_key)
        if not key.startswith('observation.images.'):
            key = f'observation.images.{key}'
        if key not in by_key:
            raise ValueError(f'No declared crop coordinate canvas for {key}')
        declaration = by_key[key]
    width, height = declaration.get('raw_width'), declaration.get('raw_height')
    if width is None or height is None:
        return None
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError('Invalid declared crop coordinate canvas')
    return int(width), int(height)


def normalized_handcrop_video_path(episode: int, view: str, hand: str) -> str:
    """One-file-per-episode crop locator for normalized installed streams."""
    if episode < 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", view):
        raise ValueError(f"invalid handcrop episode/view: {episode}/{view}")
    if hand not in {"left", "right"}:
        raise ValueError(f"invalid handcrop hand: {hand}")
    return (f"handcropped_videos/observation.images.{view}.{hand}/"
            f"chunk-{episode // 1000:03d}/file-{episode % 1000:03d}.mp4")


@dataclass(frozen=True)
class HandcropSegment:
    episode_index: int
    view_id: str
    target_id: str
    frame_offset: int
    num_frames: int
    video_path: Path
    file_frame_offset: int
    crop_data_path: Path | None
    crop_data_row_group: int
    crop_data_row_offset: int
    crop_data_num_rows: int


class HandcropStreamIndex:
    """Canonical-frame lookups into one dataset's hand-crop video stream."""

    def __init__(self, root: Path, episode_indices: list[int] | None = None,
                 *, inventory_only: bool = False):
        self.root = Path(root)
        self._episode_indices = episode_indices
        self._inventory_only = inventory_only
        self._available_keys = set()
        self._processed = None
        if processed_contract.is_processed(self.root):
            self._initialize_processed()
            return
        table_path = self.root / "meta/cropped_video_observations.parquet"
        if not table_path.is_file():
            raise FileNotFoundError(
                f"{self.root} has no meta/cropped_video_observations.parquet")
        rows = pq.read_table(table_path).to_pylist()
        info = json.loads((self.root / 'meta/info.json').read_text())
        addressing = info.get('beingh_human', {}).get('capabilities', {}).get(
            'handcropped_videos', {}).get('file_frame_offset_source', 'cumulative')
        running: dict[str, int] = {}
        self.segments: dict[tuple[int, str, str], list[HandcropSegment]] = {}
        for row in rows:
            if str(row.get("crop_type")) != "hand":
                continue
            video_rel = str(row["video_path"])
            cumulative = running.get(video_rel, 0)
            file_offset = handcrop_file_offset(row, cumulative, float(info['fps']), addressing)
            running[video_rel] = cumulative + int(row["num_frames"])
            key = (int(row["episode_index"]), str(row["view_id"]),
                   str(row["target_id"]))
            self.segments.setdefault(key, []).append(HandcropSegment(
                episode_index=int(row["episode_index"]),
                view_id=str(row["view_id"]),
                target_id=str(row["target_id"]),
                frame_offset=int(row["frame_offset"]),
                num_frames=int(row["num_frames"]),
                video_path=self.root / video_rel,
                file_frame_offset=file_offset,
                crop_data_path=self.root / str(row["crop_data_path"]),
                crop_data_row_group=int(row["crop_data_row_group_index"]),
                crop_data_row_offset=int(row["crop_data_row_offset"]),
                crop_data_num_rows=int(row["crop_data_num_rows"]),
            ))
        self._file_frames: dict[str, int] = {}

    def _initialize_processed(self) -> None:
        self._processed = processed_contract.reader_for(self.root)
        self.segments = {}
        self._file_frames = {}
        spec = self._processed.manifest['modalities'].get('handcrop_video')
        if spec is None:
            raise FileNotFoundError(f'{self.root} has no handcrop_video modality')
        for shard in spec['ranges']:
            table = self._processed._index_table(shard['path'])
            if self._episode_indices is not None:
                table = table.filter(pc.is_in(table['episode'],
                    value_set=pa.array(self._episode_indices, type=pa.int64())))
            if self._inventory_only:
                pairs = table.select(['episode', 'stream']).group_by(
                    ['episode', 'stream']).aggregate([])
                for episode, stream in zip(pairs['episode'].to_pylist(), pairs['stream'].to_pylist()):
                    binding = spec['streams'][stream]
                    self._available_keys.add((episode, binding['view_id'], binding['target_id']))
                continue
            for row in table.to_pylist():
                binding = spec['streams'][row['stream']]
                key = (row['episode'], binding['view_id'], binding['target_id'])
                geometry = []
                if self._processed.has_modality('crop_geometry'):
                    for part in self._processed.ranges('crop_geometry', row['episode'], row['stream']):
                        start = max(row['start'], part['start'])
                        stop = min(row['start'] + row['count'], part['start'] + part['count'])
                        if start >= stop:
                            continue
                        _, offsets = self._processed._file(part['path'])
                        cursor = part['offset'] + start - part['start']
                        end = cursor + stop - start
                        while cursor < end:
                            group = int(np.searchsorted(offsets, cursor, side='right') - 1)
                            count = int(min(end, offsets[group + 1]) - cursor)
                            geometry.append((start, count, self.root / part['path'],
                                             group, int(cursor - offsets[group])))
                            start += count
                            cursor += count
                if not geometry:
                    geometry = [(row['start'], row['count'], None, 0, 0)]
                for start, count, path, group, offset in geometry:
                    self.segments.setdefault(key, []).append(HandcropSegment(
                        episode_index=row['episode'],view_id=key[1],target_id=key[2],
                        frame_offset=start,num_frames=count,
                        video_path=self.root/row['path'],
                        file_frame_offset=row['offset']+start-row['start'],
                        crop_data_path=path,crop_data_row_group=group,
                        crop_data_row_offset=offset,crop_data_num_rows=count if path else 0))
        for segments in self.segments.values():
            segments.sort(key=lambda segment: segment.frame_offset)

    def available(self, episode_index: int, view_id: str,
                  target_id: str) -> bool:
        key = (int(episode_index), str(view_id), str(target_id))
        return key in self._available_keys or bool(self.segments.get(key))

    def input_side(self, episode_index: int, view_id: str,
                   present: "np.ndarray") -> int | None:
        """Which hand's crop backs a window (0=left, 1=right, None=full).

        Dominant-present hand over the window's rows, ties to the RIGHT
        hand — a plain argmax would pin every tie to the left and
        both-hands datasets would silently read left crops only.  When
        the dominant hand has no crop stream (one-hand datasets whose
        labels force-present BOTH hands, e.g. hocap), serve the other
        hand's crop rather than dropping to the full view.  Single
        source of truth: the training reader, the eval overlay, and the
        batch/review visualizers must all go through this instead of
        re-deriving the rule.
        """
        if not len(present):
            return None
        sums = np.asarray(present, bool).sum(axis=0)
        side = int(np.argmax([sums[0], sums[1] + 0.5]))
        if self.available(episode_index, view_id, ("left", "right")[side]):
            return side
        other = 1 - side
        if self.available(episode_index, view_id, ("left", "right")[other]):
            return other
        return None

    def video_paths(self) -> list[Path]:
        seen: dict[str, Path] = {}
        for segs in self.segments.values():
            for seg in segs:
                seen[str(seg.video_path)] = seg.video_path
        return list(seen.values())

    def file_frame_count(self, video_path: Path) -> int:
        """Total frames of one packed crop file (probe once, then cache)."""
        key = str(video_path)
        if key not in self._file_frames:
            import cv2
            capture = cv2.VideoCapture(str(video_path))
            self._file_frames[key] = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
        return self._file_frames[key]

    def verify_packing(self) -> dict[str, int]:
        """Compare per-file cumulative frame sums against probed counts."""
        per_file: dict[str, int] = {}
        for segs in self.segments.values():
            for seg in segs:
                total = per_file.get(str(seg.video_path), 0)
                per_file[str(seg.video_path)] = max(
                    total, seg.file_frame_offset + seg.num_frames)
        return {path: self.file_frame_count(Path(path)) - declared
                for path, declared in per_file.items()}

    def positions(self, episode_index: int, view_id: str, target_id: str,
                  frames: np.ndarray) -> np.ndarray:
        """In-file crop-video positions for canonical ``frames``.

        Uncovered frames edge-hold rather than error: frames before the
        first segment clamp to its frame 0; any other uncovered frame
        (middle gap or past the end) holds the final frame of the LAST
        segment — not the nearest edge.  Mirrors the full-video reader's
        clamp behaviour on short streams; the padding windows rely on it.
        """
        if self._processed is not None:
            locations = self._processed.locate('handcrop_video',int(episode_index),f'{view_id}/{target_id}',frames)
            if any(value is None for value in locations):
                raise KeyError('Requested handcrop frame is missing; choose temporal padding explicitly')
            return np.asarray([value[1] for value in locations],np.int64)
        segs = self.segments.get((int(episode_index), str(view_id),
                                  str(target_id)))
        if not segs:
            raise KeyError(f"no hand-crop segment for ep={episode_index} "
                           f"view={view_id} target={target_id}")
        frames = np.asarray(frames, np.int64)
        out = np.empty(len(frames), np.int64)
        for i, frame in enumerate(frames.tolist()):
            seg = next((s for s in segs
                        if s.frame_offset <= frame
                        < s.frame_offset + s.num_frames), None)
            if seg is None:
                seg = (segs[0] if frame < segs[0].frame_offset
                       else max(segs, key=lambda s: s.frame_offset))
                local = int(np.clip(frame - seg.frame_offset, 0,
                                    seg.num_frames - 1))
            else:
                local = frame - seg.frame_offset
            out[i] = seg.file_frame_offset + local
        return out

    def read_frames(self, episode_index: int, view_id: str, target_id: str,
                    frames: np.ndarray, reader, video_id=None) -> list:
        """Decode crop frames via a ``read_frames(video_id, positions)``
        reader; returns frames in caller order.  ``video_id`` resolves a
        Path to the reader's id (defaults to ``reader.video_id`` — the
        review wrapper provides it; training passes its id map lookup)."""
        if self._processed is not None:
            locations = self._processed.locate('handcrop_video',int(episode_index),f'{view_id}/{target_id}',frames)
            if any(value is None for value in locations):
                raise KeyError('Requested handcrop frame is missing; choose temporal padding explicitly')
            picks = {}
            for i,(path,position) in enumerate(locations):picks.setdefault(self.root/path,[]).append((i,position))
            resolve = video_id if video_id is not None else reader.video_id
            out = [None]*len(locations)
            for path,entries in picks.items():
                decoded = reader.read_frames(resolve(path),[value for _,value in entries])
                for (i,_),frame in zip(entries,decoded):out[i]=frame
            return out
        segs = self.segments.get((int(episode_index), str(view_id),
                                  str(target_id)))
        if not segs:
            raise KeyError(f"no hand-crop segment for ep={episode_index} "
                           f"view={view_id} target={target_id}")
        frames = np.asarray(frames, np.int64)
        picks: dict[Path, list[tuple[int, int]]] = {}
        for i, frame in enumerate(frames.tolist()):
            seg = next((s for s in segs
                        if s.frame_offset <= frame
                        < s.frame_offset + s.num_frames), None)
            if seg is None:
                seg = (segs[0] if frame < segs[0].frame_offset
                       else max(segs, key=lambda s: s.frame_offset))
            local = int(np.clip(frame - seg.frame_offset, 0,
                                seg.num_frames - 1))
            picks.setdefault(seg.video_path, []).append(
                (i, seg.file_frame_offset + local))
        resolve = video_id if video_id is not None else reader.video_id
        out: list = [None] * len(frames)
        for path, entries in picks.items():
            decoded = reader.read_frames(
                resolve(path),
                [position for _i, position in entries])
            for (i, _position), frame_rgb in zip(entries, decoded):
                out[i] = frame_rgb
        return out

    def crop_boxes(self, episode_index: int, view_id: str, target_id: str,
                   frames: np.ndarray) -> np.ndarray:
        """``(len(frames), 4)`` xyxy crop windows in declared full-frame units.

        v1 uses a declared/raw-mirror pixel canvas; v2 uses unbounded normalized
        observation coordinates. ``raw_mirror_size_for`` returns the encoding
        canvas (1x1 for v2), so existing per-axis conversions remain valid.

        Rows whose crop is invalid that frame carry NaN.  Callers projecting
        camera-frame geometry onto the crop must first scale these boxes
        into the coordinate space of their intrinsics — ``raw_mirror_size``
        gives the mirror resolution the boxes reference, the installed
        ``videos/`` are that mirror downscaled (see ``crop_adjusted_intrinsics``
        callers).  Prefer :meth:`crop_intrinsics` when the table carries
        ``cam_intr``: those are the crop frame's own per-frame intrinsics,
        immune to mirror-resolution ambiguity.
        """
        if self._processed is not None:
            if 'crop_geometry' not in self._processed.manifest['modalities']:
                return np.full((len(frames),4),np.nan)
            table,present = self._processed.read('crop_geometry',int(episode_index),f'{view_id}/{target_id}',frames,
                ['crop_box_xyxy_orig','crop_valid'])
            result = np.full((len(frames),4),np.nan)
            for position,row in zip(np.flatnonzero(present),table.to_pylist()):
                if row['crop_valid']:result[position]=row['crop_box_xyxy_orig']
            return result
        segs = self.segments.get((int(episode_index), str(view_id),
                                  str(target_id)))
        if not segs:
            raise KeyError(f"no hand-crop segment for ep={episode_index} "
                           f"view={view_id} target={target_id}")
        frames_list = np.asarray(frames, np.int64).tolist()
        wanted = set(frames_list)
        rows_by_frame: dict[int, np.ndarray] = {}
        for seg in segs:
            needed = [f for f in wanted
                      if seg.frame_offset <= f < seg.frame_offset + seg.num_frames]
            if not needed:
                continue
            data = pq.ParquetFile(seg.crop_data_path).read_row_group(
                seg.crop_data_row_group).slice(
                seg.crop_data_row_offset, seg.crop_data_num_rows)
            frame_col = np.asarray(data.column("frame_index").to_pylist(),
                                   np.int64)
            box_col = data.column("crop_box_xyxy_orig").to_pylist()
            valid_col = np.asarray(data.column("crop_valid").to_pylist(), bool)
            index_of = {int(f): int(i) for i, f in enumerate(frame_col)}
            for f in needed:
                hit = index_of.get(f)
                if hit is None:
                    rows_by_frame[f] = np.full(4, np.nan)
                elif valid_col[hit]:
                    rows_by_frame[f] = np.asarray(box_col[hit], np.float64)
                else:
                    rows_by_frame[f] = np.full(4, np.nan)
        return np.stack([rows_by_frame.get(f, np.full(4, np.nan))
                         for f in frames_list])

    def crop_intrinsics(self, episode_index: int, view_id: str,
                        target_id: str, frames: np.ndarray) -> np.ndarray | None:
        """Per-frame ``(len(frames), 3, 3)`` crop-frame intrinsics, or None.

        Reads the ``cam_intr`` column (v1 msgpack or v2 typed fx/fy/cx/cy) written
        by the handcrop generator alongside each crop box.  These describe
        the CROP FRAME's own projection — the exact K the generator used —
        so callers should prefer them over re-deriving K from the box and
        a mirror resolution (ambiguous on mixed-resolution mirrors, and
        wrong wherever the box space is not the installed/raw pixel space).
        None when the installed crop_data table predates the column.
        """
        if self._processed is not None:
            spec = self._processed.manifest['modalities'].get('crop_geometry',{})
            if not spec:return None
            schema = pq.read_schema(self._processed.path(spec['schema_example_path']))
            if 'cam_intr' not in schema.names:return None
        segs = self.segments.get((int(episode_index), str(view_id),
                                  str(target_id)))
        if not segs:
            return None
        frames_list = np.asarray(frames, np.int64).tolist()
        wanted = set(frames_list)
        out: dict[int, np.ndarray | None] = {}
        for seg in segs:
            # Padding windows run past the packed crop video's end; clamp
            # requested frames into the segment's frame range (edge-hold
            # semantics, same as read_frames) so their K edge-holds too.
            lo, hi = seg.frame_offset, seg.frame_offset + seg.num_frames
            # clamped -> [original frames that clamped to it]
            clamp_map: dict[int, list[int]] = {}
            for f in wanted:
                if 0 <= f:
                    selected = next((candidate for candidate in segs
                        if candidate.frame_offset <= f < candidate.frame_offset + candidate.num_frames), None)
                    if selected is None:
                        selected = segs[0] if f < segs[0].frame_offset else max(segs,key=lambda candidate:candidate.frame_offset)
                    if selected is not seg:
                        continue
                    clamp_map.setdefault(min(max(f, lo), hi - 1),
                                         []).append(f)
            needed = sorted(clamp_map)
            if not needed:
                continue
            schema = (schema if self._processed is not None
                      else pq.read_schema(seg.crop_data_path))
            if "cam_intr" not in schema.names:
                return None
            data = (self._processed.read('crop_geometry', int(episode_index),
                    f'{view_id}/{target_id}', range(lo, hi))[0]
                    if self._processed is not None else
                    pq.ParquetFile(seg.crop_data_path).read_row_group(
                        seg.crop_data_row_group).slice(
                        seg.crop_data_row_offset, seg.crop_data_num_rows))
            frame_col = np.asarray(data.column("frame_index").to_pylist(),
                                   np.int64)
            intr_col = data.column("cam_intr").to_pylist()
            if all(value is None for value in intr_col):
                # v2 makes the column explicit for legacy geometry which had
                # no stored intrinsics. Preserve its camera-derived fallback.
                return None
            valid_col = np.asarray(data.column("crop_valid").to_pylist(), bool)
            index_of = {int(f): int(i) for i, f in enumerate(frame_col)}
            usable = np.sort(frame_col[np.asarray(valid_col) & np.asarray(
                [v is not None for v in intr_col])])
            # Edge/nearest fallback searches only frames with a usable K.
            def _resolve(clamped):
                hit = index_of.get(int(clamped))
                if (hit is not None and valid_col[hit]
                        and intr_col[hit] is not None):
                    return hit
                if not len(usable):
                    return None
                near = int(usable[np.argmin(np.abs(usable - clamped))])
                return index_of.get(near)
            for clamped, originals in clamp_map.items():
                hit = _resolve(clamped)
                if hit is None:
                    for f in originals:
                        out[f] = None
                    continue
                try:
                    d = decode_cam_intr(intr_col[hit])
                    fx, fy = d["focal"]
                    cx, cy = d["princpt"]
                    kmat = np.array([[fx, 0.0, cx],
                                     [0.0, fy, cy],
                                     [0.0, 0.0, 1.0]], np.float64)
                except Exception:
                    for f in originals:
                        out[f] = None
                    continue
                for f in originals:
                    out[f] = kmat
        if all(out.get(f) is None for f in frames_list):
            return None
        # Absent frames (no upstream row) become NaN K — callers treat a
        # NaN/zero K as "nothing projects for this token".
        stack = []
        for f in frames_list:
            k = out.get(f)
            stack.append(k if k is not None
                         else np.full((3, 3), np.nan))
        return np.stack(stack)


@lru_cache(maxsize=8)
def raw_mirror_size(root: Path) -> tuple[int, int] | None:
    """Stored-coordinate canvas; physical raw dimensions are a legacy fallback.

    A v2 unit canvas does not imply a physical one-pixel RGB stream.

    ``crop_box_xyxy_orig`` lives in this coordinate space; ``None`` when the
    dataset installs no mirror.  Cached per root — callers derive the
    per-axis raw->installed scale from it (e.g. ``sx = installed_w /
    raw_w``) before mapping crop boxes onto installed intrinsics.

    .. deprecated:: Mixed-resolution mirrors (arctic: ego 2800x2000 vs exo
       2800x2800) make the "one resolution per root" assumption wrong; use
       :func:`raw_mirror_size_for` with the window's ``video_key`` instead.
       This root-level variant keeps the first-sorted-video behavior for
       callers that have no view context.
    """
    declared = declared_crop_canvas(root)
    if declared is not None:
        return declared
    mirrors = sorted((Path(root) / "raw_videos").rglob("*.mp4"))
    if not mirrors:
        return declared_crop_canvas(root)
    return _mirror_resolution(mirrors[0])


@lru_cache(maxsize=64)
def raw_mirror_size_for(root: Path, video_key: str) -> tuple[int, int] | None:
    """Stored-coordinate canvas for one view; raw dimensions are a fallback.

    The raw mirror keeps one directory per ``observation.images.<key>``; a
    dataset may mix resolutions across views (arctic's ego view is
    2800x2000 while its exo views are 2800x2800), so the mirror that
    ``crop_box_xyxy_orig`` coordinates belong to must be resolved per view.
    Directory matching tolerates the crop-stream's resolution suffixes
    (``exo_view_5.landscape`` vs mirror dir ``exo_view_5``); a key with no
    directory falls back to the first mirror (single-resolution roots keep
    working unchanged).
    """
    declared = declared_crop_canvas(root, video_key)
    if declared is not None:
        return declared
    if not (Path(root) / 'raw_videos').is_dir():
        return None
    key_name = (video_key if str(video_key).startswith("observation.images.")
                else f"observation.images.{video_key}")
    key_dir = Path(root) / "raw_videos" / key_name
    mirrors: list[Path] = []
    if key_dir.is_dir():
        mirrors = sorted(key_dir.rglob("*.mp4"))
    else:
        stem = key_dir.name
        for candidate in sorted((Path(root) / "raw_videos").iterdir()):
            if candidate.is_dir() and (candidate.name == stem
                                       or stem.startswith(candidate.name)):
                mirrors = sorted(candidate.rglob("*.mp4"))
                break
        if not mirrors:
            mirrors = sorted((Path(root) / "raw_videos").rglob("*.mp4"))
    if not mirrors:
        return declared_crop_canvas(root, video_key)
    return _mirror_resolution(mirrors[0])


def _mirror_resolution(video: Path) -> tuple[int, int]:
    """Read stream dimensions without decoding native high-resolution RGB."""
    import json
    import subprocess

    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'json', str(video)],
        capture_output=True, text=True, check=True, timeout=30)
    streams = json.loads(result.stdout).get('streams', [])
    if not streams:
        raise ValueError(f'raw mirror has no video stream: {video}')
    width, height = int(streams[0]['width']), int(streams[0]['height'])
    if width <= 0 or height <= 0:
        raise ValueError(f'invalid raw mirror dimensions {width}x{height}: {video}')
    return width, height


def frame_intrinsics(index: HandcropStreamIndex, episode_index: int,
                     view_id: str, target_id: str, frames: np.ndarray,
                     k_full: np.ndarray, obs_w: int, obs_h: int,
                     out_size: tuple[int, int],
                     video_key: str | None = None) -> list:
    """Per-frame crop intrinsics for projecting onto hand-crop frames.

    Takes the full-frame intrinsics ``k_full`` (3x3, or ``(T,3,3)`` matching
    ``frames`` for per-frame full-frame K), scales the segment
    boxes from raw-mirror pixels into the installed observation space
    (per axis), maps them through each crop window, and returns one
    :class:`CameraIntrinsics` per frame (NaN-box frames get a zero K —
    nothing projects). Single source of truth for declared units, including
    v2 normalized coordinates; overlays in the review stack and training-batch
    visualizer must go through this instead of re-deriving it.

    ``video_key`` (the obs row's ``video_key``, e.g.
    ``observation.images.exo_view_5``) selects the raw mirror whose
    resolution the boxes reference; ``None`` falls back to the root-level
    first-mirror size, which is wrong for mixed-resolution roots.
    """
    from core.geometry.camera import CameraIntrinsics

    if video_key is not None:
        raw = raw_mirror_size_for(index.root, video_key)
    else:
        raw = raw_mirror_size(index.root)
    if raw is None:
        raise RuntimeError(
            f"{index.root.name} has no raw_videos mirror to resolve "
            "crop-box coordinates")
    box_scale = np.asarray([obs_w / raw[0], obs_h / raw[1],
                            obs_w / raw[0], obs_h / raw[1]])
    per_frame = []
    for i, box in enumerate(index.crop_boxes(
            episode_index, view_id, target_id, frames)):
        if not np.isfinite(box).all():
            per_frame.append(CameraIntrinsics(0.0, 0.0, 0.0, 0.0))
            continue
        k_frame = k_full[i] if np.ndim(k_full) == 3 else k_full
        k = crop_adjusted_intrinsics(k_frame, box * box_scale, out_size)
        per_frame.append(CameraIntrinsics(float(k[0, 0]), float(k[1, 1]),
                                          float(k[0, 2]), float(k[1, 2])))
    return per_frame


def crop_adjusted_intrinsics(k: np.ndarray, crop_xyxy: np.ndarray,
                             out_size: tuple[int, int]) -> np.ndarray:
    """Map full-frame intrinsics onto a square hand-crop frame.

    ``crop_xyxy`` is the crop window in the ORIGINAL video's pixels; the
    crop frame is that window resized to ``out_size``.
    """
    x1, y1, x2, y2 = (float(v) for v in crop_xyxy)
    out_w, out_h = out_size
    sx = out_w / max(x2 - x1, 1e-6)
    sy = out_h / max(y2 - y1, 1e-6)
    k = np.asarray(k, np.float64)
    out = k.copy()
    out[0, 0] = k[0, 0] * sx
    out[1, 1] = k[1, 1] * sy
    out[0, 2] = (k[0, 2] - x1) * sx
    out[1, 2] = (k[1, 2] - y1) * sy
    return out
