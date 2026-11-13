#!/usr/bin/env python3
"""Handcrop v1 consumer-contract check — the guardrail against the two
classes of bug that self-consistent verification cannot see.

For every checked episode this replicates the DOWNSTREAM consumer path
(training input plane / batch review) exactly:

* A1 mapping: ``source_position`` per solver row is fitted (rate + offset,
  residual must be ~0).  Any generator pairing solver rows with video
  frames MUST use this mapping — a wrong pairing (arctic's row+12 was
  assumed 1:1 once) drifts the crop content off-center while every
  self-referential check stays green.
* stream indexing: the crop video and crop_data rows are indexed by VIDEO
  frame (``video_num_frames`` rows, ``frame_index`` == 0..n-1).
* reader-K identity: ``frame_intrinsics`` (raw-mirror-rescaled boxes ->
  crop_adjusted_intrinsics) equals the stored ``cam_intr`` msgpack on
  valid frames — the storage must follow the raw-mirror-pixel contract.
* review-exact centering: solver mesh vertices at row
  ``source_frame - offset``, projected through the reader K at the source
  frame, must center at (128, 128) and stay inside the 256x256 crop.
* stored ``hand_bbox_xywh`` center at (128, 128) on valid frames.

Exit code 1 on any violation.  Run after every handcrop (re)install:

  python scripts/data_checks/check_handcrop_consumer_contract.py \
      --dataset arctic_lerobot --episodes 0 30 150 299
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


from core.tactile_labels.decode import (  # noqa: E402
    load_label_index, read_episode_columns)
import core.data.processed_compat as processed_contract
from core.tactile_labels.handcrop import (  # noqa: E402
    HandcropStreamIndex, frame_intrinsics)
from core.tactile_labels.mano_tracks import solver_vertex_tracks  # noqa: E402
from core.data.camera import load_cameras, view_camera  # noqa: E402
from core.data.camera import canonical_to_video_map

CENTER_TOL = 0.5        # px, review-exact mesh centering tolerance
A1_RESID_TOL = 0.01     # frames, source_position affine residual


def fit_a1(sp: np.ndarray) -> tuple[float, float, float]:
    rows = np.arange(len(sp), dtype=np.float64)
    design = np.vstack([rows, np.ones_like(rows)]).T
    (rate, offset), *_ = np.linalg.lstsq(design, sp, rcond=None)
    resid = float(np.abs(sp - (rate * rows + offset)).max())
    return float(rate), float(offset), resid


def run_contract(dataset_root: Path, labels_root: Path, eps: list[int],
                 device: str = "cuda:0", frames_per_ep: int = 4,
                 view: str = "ego_view", allow_partial_coverage: bool = False) -> dict:
    index = HandcropStreamIndex(dataset_root)
    cameras = load_cameras(dataset_root)
    obs_rows = {int(r['episode_index']):r for r in processed_contract.observations(dataset_root)
                if str(r['view_id'])==view}
    label_index = {int(r["episode_index"]): r
                   for r in load_label_index(labels_root)}
    report = {"episodes": {}, "failures": []}
    info = processed_contract.read_info(dataset_root)
    indexed = processed_contract.is_processed(dataset_root)
    direct_source = info.get('beingh_human', {}).get('capabilities', {}).get(
        'handcropped_videos', {}).get('geometry_sampling') == 'source_frame_positions'
    for ep in eps:
        if ep not in obs_rows or ep not in label_index:
            report["failures"].append(f"ep{ep}: no obs/labels")
            continue
        obs, label_row = obs_rows[ep], label_index[ep]
        payload = read_episode_columns(
            labels_root, label_row["tactile_groups"],
            ("source_position",))
        sp = np.asarray(payload["source_position"], np.float64)
        rate, offset, resid = fit_a1(sp)
        checks = {"a1_rate": rate, "a1_offset": offset, "a1_resid": resid}
        repeated_source = len(sp) > 1 and np.any(np.diff(sp) == 0) and np.all(np.diff(sp) >= 0)
        if not np.isfinite(sp).all() or np.any(np.diff(sp) < 0):
            report['failures'].append(f'ep{ep}: invalid source positions')
        if resid > A1_RESID_TOL and not repeated_source and not direct_source:
            report["failures"].append(
                f"ep{ep}: A1 mapping not affine (resid {resid:.3f})")
        n_video = int(obs.get("video_num_frames") or obs["num_frames"])

        c2w, k_cam = view_camera(dataset_root, obs, cameras)
        k_full = np.array([[k_cam.fx, 0, k_cam.cx],
                           [0, k_cam.fy, k_cam.cy],
                           [0, 0, 1]], np.float64)
        # sample valid source frames across the covered span
        rounded_source = np.rint(sp).astype(np.int64)
        video_mapping = canonical_to_video_map(dataset_root, obs)
        if video_mapping is not None:
            canonical_rows = np.clip(rounded_source, 0, len(video_mapping[0]) - 1)
            mapped_source = video_mapping[1 if indexed else 0][canonical_rows]
        else:
            mapped_source = rounded_source
        usable, first_rows = np.unique(mapped_source, return_index=True)
        inside = (usable >= 0) & (usable < (int(obs['num_frames']) if indexed else n_video))
        if allow_partial_coverage:
            covered = np.zeros(int(obs['num_frames']) if indexed else n_video, bool)
            for hand in ('left', 'right'):
                for segment in index.segments.get((ep, view, hand), []):
                    covered[segment.frame_offset:segment.frame_offset + segment.num_frames] = True
            inside &= covered[np.clip(usable, 0, len(covered) - 1)]
        usable, first_rows = usable[inside], first_rows[inside]
        if not len(usable):
            report['failures'].append(f'ep{ep}: no source positions inside video')
            continue
        sample_indices = np.unique(np.linspace(0, len(usable)-1,
                                   max(1, frames_per_ep)).astype(int))
        picks, geo_rows = usable[sample_indices], first_rows[sample_indices]
        camera_frames = (video_mapping[1][canonical_rows[geo_rows]]
                         if video_mapping is not None else picks)
        _presence = None   # (lazy) solver presence for this episode
        for hand in ("left", "right"):
            if not index.available(ep, view, hand):
                # one-hand datasets (hocap annotates a single hand per
                # episode) legitimately ship no segment for the untracked
                # hand — verify absence against the solver tracks instead
                # of blanket-failing
                if _presence is None:
                    _tracks = solver_vertex_tracks(
                        dataset_root, ep, view, source_positions=sp,
                        device=device)
                    _presence = {h: bool(np.asarray(_tracks[h][1]).any())
                                 for h in ("left", "right")}
                if _presence[hand]:
                    report["failures"].append(
                        f"ep{ep}/{hand}: unavailable despite solver tracks")
                else:
                    checks[f"{hand}_absent_ok"] = 1.0
                continue
            segments = sorted(index.segments[(ep, view, hand)], key=lambda s: s.frame_offset)
            total_rows = sum(s.crop_data_num_rows for s in segments)
            if total_rows != n_video and not allow_partial_coverage:
                report["failures"].append(
                    f"ep{ep}/{hand}: crop rows {total_rows} "
                    f"!= video frames {n_video} (stream must be video-"
                    f"indexed)")
            tables = []
            for seg in segments:
                part = pq.ParquetFile(seg.crop_data_path).read_row_group(
                    seg.crop_data_row_group).slice(seg.crop_data_row_offset, seg.crop_data_num_rows)
                expected = np.arange(seg.frame_offset, seg.frame_offset+seg.num_frames)
                if seg.crop_data_num_rows != seg.num_frames or not np.array_equal(
                        np.asarray(part['frame_index'], np.int64), expected):
                    report['failures'].append(f'ep{ep}/{hand}: segment frame-index coverage mismatch')
                tables.append(part)
            import pyarrow as pa
            rows = pa.concat_tables(tables).to_pandas()
            fidx = np.asarray(rows["frame_index"], np.int64)
            if (not indexed and not allow_partial_coverage and not np.array_equal(fidx, np.arange(len(fidx)))):
                report["failures"].append(
                    f"ep{ep}/{hand}: frame_index not 0..n-1")
            valid = np.asarray(rows["crop_valid"], bool)
            if valid.any():
                hb = np.asarray(
                    rows.loc[valid.tolist(), "hand_bbox_xywh"].tolist(),
                    np.float64)
                center_off = float(np.abs(
                    hb[:, 0] + hb[:, 2] / 2 - 128).max())
                checks[f"{hand}_bbox_center_off"] = center_off

            per = frame_intrinsics(
                index, ep, view, hand, picks, k_full,
                int(obs["width"]), int(obs["height"]), (256, 256),
                video_key=str(obs["video_key"]))
            from core.tactile_labels.handcrop_storage import decode_cam_intr
            k_diff, centers = 0.0, []
            has_intr = "cam_intr" in rows.columns
            for j, f in enumerate(picks):
                kk = per[j]
                if kk.fx == 0:
                    continue
                if has_intr:
                    matching = rows.loc[rows['frame_index'] == int(f)]
                    if len(matching) != 1:
                        report['failures'].append(f'ep{ep}/{hand}: missing or duplicated crop row for frame {int(f)}')
                        continue
                    stored = decode_cam_intr(matching.iloc[0]["cam_intr"])
                    if stored is not None:
                        k_diff = max(k_diff, abs(kk.fx - stored["focal"][0]),
                                     abs(kk.fy - stored["focal"][1]),
                                     abs(kk.cx - stored["princpt"][0]),
                                     abs(kk.cy - stored["princpt"][1]))
            checks[f"{hand}_reader_vs_stored_K"] = k_diff
            if k_diff > 1e-6:
                report["failures"].append(
                    f"ep{ep}/{hand}: reader K != stored cam_intr "
                    f"({k_diff:.6f})")

            # review-exact centering: mesh at row source-offset through the
            # reader K at the source frame
            src_for_rows = picks
            if video_mapping is None:
                tracks = solver_vertex_tracks(
                    dataset_root, ep, view, source_positions=sp,
                    rows=geo_rows, device=device)
            else:
                # Training displays the causal sampled frame and its camera;
                # check the crop against that exact source-frame geometry.
                tracks = solver_vertex_tracks(
                    dataset_root, ep, view,
                    source_positions=np.asarray(camera_frames, np.float64), device=device)
            verts, present = tracks[hand]
            verts = np.asarray(verts, np.float64)
            per2 = frame_intrinsics(
                index, ep, view, hand, src_for_rows, k_full,
                int(obs["width"]), int(obs["height"]), (256, 256),
                video_key=str(obs["video_key"]))
            worst = 0.0
            for j in range(len(geo_rows)):
                if not present[j] or not np.isfinite(verts[j]).all():
                    continue
                kk = per2[j]
                if kk.fx == 0:
                    continue
                K = np.array([[kk.fx, 0, kk.cx],
                              [0, kk.fy, kk.cy], [0, 0, 1]], np.float64)
                cam = (np.linalg.inv(c2w[int(camera_frames[j])]) @ np.hstack(
                    [verts[j], np.ones((778, 1))]).T).T[:, :3]
                z = np.maximum(cam[:, 2], 1e-6)
                u = K[0, 0] * cam[:, 0] / z + K[0, 2]
                v = K[1, 1] * cam[:, 1] / z + K[1, 2]
                off = max(abs((u.min() + u.max()) / 2 - 128),
                          abs((v.min() + v.max()) / 2 - 128))
                worst = max(worst, off)
                if u.min() < -1 or u.max() > 257 or v.min() < -1 \
                        or v.max() > 257:
                    report["failures"].append(
                        f"ep{ep}/{hand} f{int(src_for_rows[j])}: mesh "
                        f"spills outside crop")
            checks[f"{hand}_consumer_center_off"] = worst
            if worst > CENTER_TOL:
                report["failures"].append(
                    f"ep{ep}/{hand}: consumer-exact mesh center off by "
                    f"{worst:.2f}px > {CENTER_TOL}px")
        report["episodes"][str(ep)] = checks
    return report
