"""Shared writer for handcrop v1 segments — the single place that owns the
storage contract, so producers cannot re-derive (and mis-derive) it.

Contract owned here (see AGENTS.md "Handcrop v1 契约与消费者路径验证"):

* the crop video and crop_data rows are indexed by VIDEO FRAME
  (``frame_index`` == 0..n_video-1); geometry for frame k is the solver
  row implied by the labels' A1 mapping (``source_position``), fitted via
  :func:`fit_source_mapping` — never a positional row==frame assumption
  (arctic's mapping is row+12; most datasets are 0);
* ``crop_box_xyxy_orig`` / ``hand_bbox_xywh_orig`` are stored in
  RAW-MIRROR pixels (readers rescale by obs/raw before mapping K) — for
  fisheye datasets this is the canonical box pre-multiplied by raw/obs;
* ``cam_intr`` msgpack carries the crop frame's own K;
* the index table append is an exclusive-lock read-modify-write;
* every install records the A1 mapping it used in
  ``meta/handcrop_generation.json`` so the labels publish gate can detect
  stale crops without a GPU.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import cv2
import msgpack
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .handcrop import (HANDCROP_STREAM_LAYOUT, crop_adjusted_intrinsics,
                       normalized_handcrop_video_path)

CROP_SIZE = 256
A1_RESID_TOL = 0.01


def fit_source_mapping(sp: np.ndarray,
                       resid_tol: float = A1_RESID_TOL
                       ) -> tuple[float, float, float]:
    """Fit ``source_video_frame = rate * solver_row + offset`` from the
    labels' per-row ``source_position`` (the A1 mapping).  Raises when the
    mapping is not affine — a generator must never guess the pairing."""
    sp = np.asarray(sp, np.float64)
    rows = np.arange(len(sp), dtype=np.float64)
    design = np.vstack([rows, np.ones_like(rows)]).T
    (rate, offset), *_ = np.linalg.lstsq(design, sp, rcond=None)
    resid = float(np.abs(sp - (rate * rows + offset)).max())
    if resid > resid_tol:
        raise ValueError(
            f"source mapping not affine (resid {resid:.3f} frames); "
            "refusing row/frame pairing")
    return float(rate), float(offset), resid


def video_frame_rows(n_video: int, n_rows: int, rate: float, offset: float
                     ) -> np.ndarray:
    """Solver row backing each video frame (-1 when the labels do not
    cover it — those frames are crop_valid=False edge-holds)."""
    rows = np.floor((np.arange(n_video) - offset) / rate + 0.5).astype(np.int64)
    rows[(rows < 0) | (rows >= n_rows)] = -1
    return rows


def project_boxes(verts: np.ndarray, present: np.ndarray, c2w: np.ndarray,
                  k_obs: np.ndarray, context: float = 1.25,
                  min_side: float = 96, max_side: float = 512
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame square boxes (obs pixels) from camera/world-frame MANO
    vertices.  Returns (crop_box_xyxy, hand_bbox_xywh, valid); frames
    without a tracked hand hold the previous box and mark valid=False."""
    n = len(verts)
    boxes = np.zeros((n, 4), np.float64)
    hands = np.zeros((n, 4), np.float64)
    valid = np.zeros(n, bool)
    last = np.array([0.0, 0.0, 200.0, 200.0])
    ones = np.ones((1, 1), np.float64)
    for i in range(n):
        if not present[i] or not np.isfinite(verts[i]).all():
            boxes[i], hands[i] = last, last
            continue
        cam = (np.linalg.inv(c2w[i]) @ np.hstack(
            [verts[i], np.repeat(ones, len(verts[i]), 0)]).T).T[:, :3]
        # A mesh crossing the camera plane has no finite pinhole bounding
        # box. Do not turn negative depth into a huge positive projection.
        if not np.isfinite(cam).all() or (cam[:, 2] <= 1e-6).any():
            boxes[i] = last
            continue
        z = cam[:, 2]
        u = k_obs[0, 0] * cam[:, 0] / z + k_obs[0, 2]
        v = k_obs[1, 1] * cam[:, 1] / z + k_obs[1, 2]
        x0, x1 = float(u.min()), float(u.max())
        y0, y1 = float(v.min()), float(v.max())
        hands[i] = (x0, y0, x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hand_span = max(x1 - x0, y1 - y0, 1.0)
        side = float(np.clip(context * hand_span,
                             min_side, max_side))
        # Cap the context margin, never the hand itself. Close-up ego hands
        # can exceed max_side; a smaller valid crop would truncate the mesh.
        side = max(side, hand_span)
        boxes[i] = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        valid[i] = True
        last = boxes[i]
    return boxes, hands, valid


def reference_crop(frame: np.ndarray, box: np.ndarray,
                   size: int = CROP_SIZE) -> np.ndarray:
    """Canonical-pixel centerpad crop (verify reference)."""
    h, w = frame.shape[:2]
    if not np.isfinite(box).all():
        raise ValueError('nonfinite crop box')
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    crop_w, crop_h = max(x1 - x0, 1), max(y1 - y0, 1)
    if crop_w * crop_h > 4096 * 4096:
        # Rasterize directly into the output grid rather than allocating a
        # potentially unbounded padded source canvas. Match resize's pixel
        # centers; source samples outside the image retain black padding.
        xs = x0 + (np.arange(size, dtype=np.float64) + 0.5) * (crop_w / size) - 0.5
        ys = y0 + (np.arange(size, dtype=np.float64) + 0.5) * (crop_h / size) - 0.5
        map_x, map_y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    patch = np.zeros((crop_h, crop_w, 3), np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


class CropEncoder:
    """libx264 crf-17 short-GOP pipe writer (seek-friendly, near-lossless
    relative to the fresh remap)."""

    def __init__(self, path: Path, fps: float, size: int = CROP_SIZE):
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{size}x{size}", "-r", f"{fps:.6f}", "-i", "pipe:0",
             "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
             "-threads", os.environ.get('HANDCROP_FFMPEG_THREADS', '2'),
             "-g", "4", "-keyint_min", "4", "-sc_threshold", "0", "-bf", "0",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(
                f"crop encoder failed rc={code}: {self.path}")


def store_scale_for(obs_wh: tuple[int, int],
                    raw_wh: tuple[int, int]) -> np.ndarray:
    """Per-axis raw/obs ratio applied to *_orig columns at write time (the
    reader's obs/raw multiply recovers the obs-pixel window exactly)."""
    return np.asarray([raw_wh[0] / obs_wh[0], raw_wh[1] / obs_wh[1],
                       raw_wh[0] / obs_wh[0], raw_wh[1] / obs_wh[1]])


def update_generation_record(root: Path, view: str, *,
                             a1_rate: float, a1_offset: float,
                             source: str) -> None:
    """Record the mapping/frame-indexing a handcrop install used, so the
    labels publish gate can detect stale crops without a GPU."""
    path = root / "meta/handcrop_generation.json"
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = json.loads(path.read_text()) if path.is_file() else {}
        record[view] = {"a1_rate": a1_rate, "a1_offset": a1_offset,
                        "source": source, "indexing": "video_frame"}
        temporary = path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(record, indent=1, sort_keys=True))
        os.replace(temporary, path)


def replace_episode_rows(index_table, rows, episode, view, *, replace_existing=True):
    """Keep unrelated rows in Arrow; only materialize the new segment rows."""
    import pyarrow.compute as pc
    if replace_existing:
        match = pc.and_(pc.equal(index_table['episode_index'], episode),
                        pc.equal(index_table['view_id'], view))
        index_table = index_table.filter(pc.invert(match))
    replacement = pa.Table.from_pylist(rows, schema=index_table.schema)
    return pa.concat_tables([index_table, replacement])


def build_crop_geometry(episode: int, view: str, hand: str,
                        boxes: np.ndarray, hands: np.ndarray,
                        valid: np.ndarray, k_obs: np.ndarray,
                        crop_schema: pa.Schema, store_scale: np.ndarray,
                        size: int = CROP_SIZE) -> pa.Table:
    """Serialize video-indexed crop geometry for both packed and episode writers."""
    n = len(boxes)
    side = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
    hand_in_crop = np.stack([
        (hands[:, 0] - boxes[:, 0]) * size / side,
        (hands[:, 1] - boxes[:, 1]) * size / side,
        hands[:, 2] * size / side,
        hands[:, 3] * size / side], axis=1)
    intr = []
    for box in boxes if 'cam_intr' in crop_schema.names else []:
        k_adj = crop_adjusted_intrinsics(k_obs, box, (size, size))
        value = {"focal": [float(k_adj[0, 0]), float(k_adj[1, 1])],
                 "princpt": [float(k_adj[0, 2]), float(k_adj[1, 2])]}
        if 'cam_intr' in crop_schema.names and pa.types.is_struct(crop_schema.field('cam_intr').type):
            from .handcrop_storage import typed_cam_intr
            intr.append(typed_cam_intr(value))
        else:
            intr.append(msgpack.dumps(value))
    payload = {"episode_index": pa.array([episode] * n, pa.int64()),
         "view_id": pa.array([view] * n),
         "target_id": pa.array([hand] * n),
         "frame_index": pa.array(np.arange(n), pa.int64()),
         "hand_bbox_xywh_orig": pa.array((hands * store_scale).tolist()),
         "hand_bbox_xywh": pa.array(hand_in_crop.tolist()),
         "crop_box_xyxy_orig": pa.array((boxes * store_scale).tolist()),
         "frame_valid": pa.array([True] * n),
         "crop_valid": pa.array(valid.tolist())}
    if 'cam_intr' in crop_schema.names:
        payload['cam_intr'] = pa.array(intr, type=crop_schema.field('cam_intr').type)
    return pa.table(payload, schema=crop_schema)


def write_episode_segments(root: Path, episode: int, view: str, fps: float,
                           per_hand: dict[str, tuple[np.ndarray, ...]],
                           frames_iter: Iterator[tuple[int, np.ndarray]],
                           crop_fn: Callable[[np.ndarray, np.ndarray],
                                             np.ndarray],
                           k_obs: np.ndarray, crop_schema: pa.Schema,
                           store_scale: np.ndarray,
                           a1_rate: float = 1.0, a1_offset: float = 0.0,
                           size: int = CROP_SIZE, namespace: str | None = None,
                           replace_existing: bool = False) -> None:
    """Encode + commit one episode's hand-crop segments under the contract.

    ``per_hand[hand] = (boxes, hand_bboxes, valid)`` are per VIDEO frame
    (obs pixels); ``frames_iter`` yields ``(k, BGR frame)`` for k in
    0..n-1; ``crop_fn(frame, box)`` renders one size x size BGR crop.
    Writes the mp4s atomically, the crop_data parquets (ONE row group,
    raw-scaled *_orig, cam_intr msgpack, frame_index == video frame) and
    appends the index rows under an exclusive lock.
    """
    n = len(next(iter(per_hand.values()))[0])
    video_rels = {h: (f"handcropped_videos/v1/videos/chunk-000/"
                      f"ego_ep{episode:06d}_{h}.mp4")
                  for h in per_hand}
    crop_rels = {h: (f"meta/crop_data/hand/v1/chunk-000/"
                     f"ego_ep{episode:06d}_{h}.parquet") for h in per_hand}
    if namespace is not None:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+', namespace):
            raise ValueError('unsafe handcrop namespace')
        tag = re.sub(r'[^A-Za-z0-9_-]', '_', view)
        video_rels = {h: f'handcropped_videos/v1/videos/{namespace}/{tag}_ep{episode:06d}_{h}.mp4'
                      for h in per_hand}
        crop_rels = {h: f'meta/crop_data/hand/v1/{namespace}/{tag}_ep{episode:06d}_{h}.parquet'
                     for h in per_hand}
    info = json.loads((root / "meta/info.json").read_text())
    geometry_version = info.get('beingh_human', {}).get('crop_data', {}).get('version')
    if geometry_version in ('v2', 'v3'):
        from .handcrop_storage import GEOMETRY_SCHEMA_V2, GEOMETRY_SCHEMA_V3, reserve_geometry_path
        crop_schema = GEOMETRY_SCHEMA_V3 if geometry_version == 'v3' else GEOMETRY_SCHEMA_V2
        observations = pq.read_table(root / info['beingh_human'].get('observation_table', 'meta/observations.parquet'),
            columns=['width', 'height'], filters=[('episode_index', '=', episode), ('view_id', '=', view)]).to_pylist()
        if len(observations) != 1:
            raise ValueError('v2 crop writer needs exactly one observation raster')
        obs = observations[0]
        store_scale = np.array([1.0 / obs['width'], 1.0 / obs['height']] * 2)
        crop_rels = {h: reserve_geometry_path(root, view, h) for h in per_hand}
    layout = (info.get("beingh_human", {}).get("capabilities", {})
              .get("handcropped_videos", {}).get("layout"))
    if layout == HANDCROP_STREAM_LAYOUT:
        video_rels = {
            h: normalized_handcrop_video_path(episode, view, h)
            for h in per_hand
        }
    for relative in [*video_rels.values(), *crop_rels.values()]:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    tmp_videos = {h: (root / video_rels[h]).with_suffix(".tmp.mp4")
                  for h in per_hand}
    encoders = {h: CropEncoder(tmp_videos[h], fps, size) for h in per_hand}
    seen = 0
    for k, frame in frames_iter:
        if k != seen:
            raise ValueError(f"frame stream gap: got {k}, expected {seen}")
        seen += 1
        for hand, (boxes, _hands, _valid) in per_hand.items():
            encoders[hand].write(crop_fn(frame, boxes[k]))
    if seen != n:
        raise ValueError(f"frame stream short: {seen} != {n}")
    for encoder in encoders.values():
        encoder.release()
    for hand in per_hand:
        os.replace(tmp_videos[hand], root / video_rels[hand])

    # Prepare per-job geometry outside the shared index lock.
    replacement_rows = []
    for hand, (boxes, hands, valid) in per_hand.items():
        seg = build_crop_geometry(episode, view, hand, boxes, hands, valid,
                                  k_obs, crop_schema, store_scale, size)
        pq.write_table(seg, root / crop_rels[hand], row_group_size=n)
        replacement_rows.append({
            "episode_index": episode, "view_id": view,
            "crop_type": "hand", "target_id": hand,
            "segment_index": 0, "frame_offset": 0,
            "stream_num_frames": n, "video_path": video_rels[hand],
            "from_timestamp": 0.0, "to_timestamp": n / fps,
            "num_frames": n, "width": size, "height": size,
            "crop_data_path": crop_rels[hand],
            "crop_data_row_group_index": 0, "crop_data_row_offset": 0,
            "crop_data_num_rows": n})
    index_path = root / "meta/cropped_video_observations.parquet"
    with open(root / "meta/cropped_video_observations.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_table = pq.read_table(index_path)
        staged = replace_episode_rows(index_table, replacement_rows, episode, view,
                                      replace_existing=replace_existing)
        tmp = root / "meta/cropped_video_observations.tmp.parquet"
        pq.write_table(staged, tmp)
        os.replace(tmp, index_path)
    update_generation_record(root, view, a1_rate=a1_rate,
                             a1_offset=a1_offset,
                             source=f"ep{episode} via handcrop_writer")


def labels_handcrop_geometry_consistency(live_root: Path, staged_root: Path) -> list[str]:
    """Check new MANO geometry through the actual training crop consumer."""
    index_path = live_root / "meta/cropped_video_observations.parquet"
    if not index_path.is_file():
        return []
    from core.tactile_labels.handcrop_contract import run_contract
    from .decode import load_label_index
    available = {int(r['episode_index']) for r in load_label_index(staged_root)}
    by_view: dict[str, set[int]] = {}
    for row in pq.read_table(index_path).to_pylist():
        episode = int(row['episode_index'])
        if episode in available:
            by_view.setdefault(str(row['view_id']), set()).add(episode)
    failures = []
    for view, episode_set in by_view.items():
        episodes = sorted(episode_set)
        selected = sorted({episodes[0], episodes[len(episodes)//2], episodes[-1]})
        report = run_contract(live_root, staged_root, selected, device='cpu',
                              frames_per_ep=4, view=view)
        failures.extend(f'{view}: {item}' for item in report['failures'])
    return failures


def labels_handcrop_consistency(live_root: Path, staged_root: Path,
                                max_episodes: int = 4) -> list[str]:
    """No-GPU publish gate: are the installed hand-crop streams still
    consistent with the STAGED labels about to go live?

    Checks (per handcrop-bearing view, sampled episodes): the A1 mapping
    fitted from staged labels equals the one recorded at handcrop install
    time (meta/handcrop_generation.json); crop rows == video_num_frames;
    frame_index == 0..n-1; stored hand_bbox centers at (size/2, size/2) on
    valid frames.  The reader-K == stored cam_intr identity is NOT
    re-checked here (no obs camera K in scope) — it is enforced at install
    time by ``scripts/data_checks/check_handcrop_consumer_contract.py``.
    Returns the list of violations (empty = consistent)."""
    from .decode import load_label_index, read_episode_columns
    from .handcrop import HandcropStreamIndex, frame_intrinsics

    index_path = live_root / "meta/cropped_video_observations.parquet"
    if not index_path.is_file():
        return []
    index_rows = pq.read_table(index_path).to_pylist()
    views = sorted({str(r["view_id"]) for r in index_rows})
    # CROP_SCHEMA has no width column; the crop size rides the index rows.
    crop_widths = {(int(r["episode_index"]), str(r["view_id"]),
                    str(r["target_id"])): int(r["width"])
                   for r in index_rows}
    record_path = live_root / "meta/handcrop_generation.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() \
        else {}
    direct_source = json.loads((live_root / 'meta/info.json').read_text()).get(
        'beingh_human', {}).get('capabilities', {}).get('handcropped_videos', {}).get(
        'geometry_sampling') == 'source_frame_positions'
    label_index = {int(r["episode_index"]): r
                   for r in load_label_index(staged_root)}
    failures: list[str] = []
    index = HandcropStreamIndex(live_root)
    obs_all = {(int(r["episode_index"]), str(r['view_id'])): r for r in pq.read_table(
        live_root / "meta/observations.parquet").to_pylist()}
    eps = sorted(set(label_index) & {key[0] for key in obs_all})
    eps = eps[::max(1, len(eps) // max_episodes)][:max_episodes]
    for ep in eps:
        payload = read_episode_columns(
            staged_root, label_index[ep]["tactile_groups"],
            ("source_position",))
        sp = np.asarray(payload["source_position"], np.float64)
        try:
            rate, offset, _ = fit_source_mapping(sp)
        except ValueError as error:
            if not direct_source or not np.isfinite(sp).all() or np.any(np.diff(sp) < 0):
                failures.append(f"ep{ep}: staged labels {error}")
                continue
            rate = offset = None
        for view in views:
            obs = obs_all.get((ep, view))
            if obs is None:
                continue
            rec = record.get(view, {})
            if not direct_source and (abs(rec.get("a1_offset", offset) - offset) > 0.51
                    or abs(rec.get("a1_rate", rate) - rate) > 1e-3):
                failures.append(
                    f"ep{ep} {view}: staged A1 (rate {rate:.5f}, offset "
                    f"{offset:.2f}) != handcrop generation record "
                    f"({rec.get('a1_rate')}, {rec.get('a1_offset')}) — "
                    "re-run the handcrop installer before/after this "
                    "labels install")
            for hand in ("left", "right"):
                if not index.available(ep, view, hand):
                    continue
                segments = sorted(index.segments[(ep, view, hand)], key=lambda s: s.frame_offset)
                n_video = int(obs.get("video_num_frames") or obs["num_frames"])
                total_rows = sum(s.crop_data_num_rows for s in segments)
                if total_rows != n_video:
                    failures.append(
                        f"ep{ep} {view}/{hand}: crop rows "
                        f"{total_rows} != video frames "
                        f"{n_video}")
                    continue
                rows = pa.concat_tables([
                    pq.ParquetFile(seg.crop_data_path).read_row_group(
                        seg.crop_data_row_group).slice(seg.crop_data_row_offset, seg.crop_data_num_rows)
                    for seg in segments]).to_pandas()
                fidx = np.asarray(rows["frame_index"], np.int64)
                if not np.array_equal(fidx, np.arange(len(fidx))):
                    failures.append(f"ep{ep} {view}/{hand}: frame_index "
                                    "not 0..n-1")
                valid = np.asarray(rows["crop_valid"], bool)
                if valid.any():
                    hb = np.asarray(
                        rows.loc[valid.tolist(),
                                 "hand_bbox_xywh"].tolist(), np.float64)
                    size = int(crop_widths.get((ep, view, hand), 256))
                    center = size / 2.0
                    off = float(np.abs(
                        hb[:, 0] + hb[:, 2] / 2 - center).max())
                    if off > 2.0:
                        # 2px absorbs the legacy converter's integer box
                        # rounding (~1px); real drift (A1 mismatch,
                        # convention errors) lands at tens of pixels.
                        failures.append(
                            f"ep{ep} {view}/{hand}: stored bbox center off "
                            f"{off:.2f}px")
    return failures
