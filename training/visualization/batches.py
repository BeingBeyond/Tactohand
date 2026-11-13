"""Adapt learning batches to the shared label review renderer."""
from __future__ import annotations
from pathlib import Path
import html
import re
from types import SimpleNamespace
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from core.tactile_labels.assets import (  # noqa: E402
    load_asset_names, object_display_name,
)
from core.tactile_labels.decode import (  # noqa: E402
    load_label_index, read_episode_columns,
)
from core.tactile_labels.mano_tracks import solver_vertex_tracks  # noqa: E402
from core.tactile_labels.handcrop import (  # noqa: E402
    HandcropStreamIndex, frame_intrinsics, raw_mirror_size)
from core.tactile_labels.schema import (  # noqa: E402
    FLAG_LEFT_PRESENT, FLAG_RIGHT_PRESENT,
)
from core.geometry.camera import CameraIntrinsics
from core.data.camera import load_cameras  # noqa: E402
from core.visualization.prepared import render_window
from core.visualization.prepared import hand_payloads

from core.visualization.prepared import ReviewContext
from training.data.preprocessing import edge_hold

PROXIMITY_CLIP_M = 0.25

def write_gallery(root: Path, records: list[dict]) -> None:
    """Incremental browser index; existing review artifacts remain untouched."""
    parts = ['<!doctype html><meta charset="utf-8"><title>Training input label review</title>',
             '<style>body{font:16px sans-serif;background:#eee;margin:24px}video,img{max-width:100%}section{background:white;padding:16px;margin:16px 0}</style>',
             '<h1>Training input + saved labels</h1>',
             '<p>Inspect hand/object alignment visually. Tensor consistency is not proof of pixel alignment. '
             'Source and selected windows: <a href="selection.json">selection.json</a>.</p>']
    for record in records:
        caption = f"ep {record['episode']} / {record.get('input_view', record['view'])} / source {record.get('video_source', 'see config')}"
        parts.append('<section><h2>' + html.escape(caption) + '</h2>')
        for key, tag in (('video', 'video controls preload="metadata"'), ('sheet', 'img loading="lazy"')):
            path = Path(record[key]).resolve().relative_to(root.resolve()).as_posix()
            src = html.escape(path, quote=True)
            parts.append(f'<{tag} src="{src}">' + ('</video>' if key == 'video' else ''))
        parts.append('</section>')
    temporary = root / 'index.tmp.html'
    temporary.write_text('\n'.join(parts) + '\n')
    temporary.replace(root / 'index.html')


def render_window_sample(context: ReviewContext, batch: dict, index: int,
                         dataset: str, tau: float, batch_index: int,
                         sample: int, split: str,
                         out_dir: Path) -> dict:
    """One training window through the review renderer + cross-checks."""
    episode = int(batch["episode_index"][index])
    view_id = str(batch["view_id"][index])
    mode = str(batch["dataset_mode"][index])
    labels = context.index[episode]
    tactile = read_episode_columns(
        context.labels_root, labels["tactile_groups"],
        ("flags", "contact", "signed_distance", "source_position"))

    tokens_total = int(batch["real_mask"].shape[1])
    left = int(batch["left_padding"][index])
    right = int(batch["right_padding"][index])
    real = slice(left, tokens_total - right)
    source_frames = batch["source_frames"][index].numpy()[real]

    rows = batch["geometry_solver_rows"][index].numpy()[real]
    force_rows = batch["force_solver_rows"][index].numpy()[real]
    target_idx = int(batch["target_object_index"][index])
    if mode == "static_frame":
        rows = rows[:1]
        force_rows = force_rows[:1]

    contact = tactile["contact"].reshape(-1, 2, 778)[rows]
    distance = tactile["signed_distance"].astype(np.float32).reshape(
        -1, 2, 778)[rows]
    present = np.stack((
        (tactile["flags"] & FLAG_LEFT_PRESENT) > 0,
        (tactile["flags"] & FLAG_RIGHT_PRESENT) > 0), axis=1)[rows]
    if mode == "static_frame":
        # Static windows tile ONE row across the whole sequence; tile the
        # re-read row the same way so the cross-checks compare equal shapes
        # (np.array_equal refuses to broadcast).
        n_real = tokens_total - left - right
        contact = np.repeat(contact, n_real, axis=0)
        distance = np.repeat(distance, n_real, axis=0)
        present = np.repeat(present, n_real, axis=0)

    # Hand-crop windows: re-derive the reader's per-window crop side and
    # replay its supervision transforms on the re-read rows, so the
    # cross-checks compare against exactly what _row_targets produced
    # (HandcropStreamIndex.input_side: dominant-present hand, ties RIGHT,
    # availability check + other-hand fallback, full-frame fallback).
    crop_side: int | None = None
    if context.handcrop is not None:
        crop_side = context.handcrop.input_side(episode, view_id, present)
    native_present = present.copy()
    if crop_side is not None:
        present = present.copy()
        present[:, 1 - crop_side] = True
        contact = contact.copy()
        distance = distance.copy()
        contact[:, 1 - crop_side, :] = 0.0
        distance[:, 1 - crop_side, :] = np.inf

    canonical = str(labels.get("canonical_view") or view_id)
    # Degenerate-world datasets (HMR camera.c2w == identity, e.g. decaf)
    # have no cross-view calibration: each view's HMR mesh lives in its OWN
    # camera frame, so the canonical track lands offset on every other
    # view (parallax).  Per part3_standard §"standard visualization",
    # project the DISPLAYED view's own solve there — contact/proximity are
    # per-vertex hand properties, valid on either track.  Calibrated
    # datasets (real c2w) keep the world-frame canonical track.
    track_view = canonical
    if view_id != canonical and np.allclose(
            batch["camera_c2w"][index].numpy(), np.eye(4), atol=1e-6):
        track_view = view_id
    # rows= scopes the MANO forward to the window's bracket frames and CUDA
    # batches it — the episode-wide CPU default made a single random window
    # cost minutes on long-episode datasets (obman ~5k rows x 2 hands).
    tracks = solver_vertex_tracks(
        context.root, episode, track_view,
        source_positions=tactile["source_position"], rows=rows,
        with_faces=True, device="cuda")
    vertices = np.stack(
        [tracks["left"][0], tracks["right"][0]], axis=1)
    force_vectors, wrench_contact, force_scale, object_position, hand_force, \
        overlay = hand_payloads(
            context.labels_root, context.root, labels, force_rows, target_idx)
    if crop_side is not None and force_vectors is not None:
        force_vectors = force_vectors.copy()
        force_vectors[:, 1 - crop_side] = 0.0
    # The supervised GT is the HAND wrench (contact - environment), not the
    # raw contact wrench hand_payloads returns: render and cross-check the
    # quantity _wrench_losses actually matches (environment carries the
    # gravity reaction, ~mass*g N on supported objects).
    wrench = None
    if wrench_contact is not None:
        force_columns = read_episode_columns(
            context.labels_root, labels["force_groups"],
            ("contact_wrench", "environment_wrench"))
        wrench = (force_columns["contact_wrench"][force_rows]
                  - force_columns["environment_wrench"][force_rows]).astype(np.float32)
        # Reproduce the current reader's mode-specific behavior: trajectory
        # crops mask a three-component block; static crops retain the full
        # aggregate force/torque pair. A static wrench has no hand-slot axis.
        if crop_side is not None and mode != 'static_frame':
            wrench[:, 3 * (1 - crop_side):3 * (1 - crop_side) + 3] = 0.0

    # --- cross-checks: batch tensors vs the re-read label rows ----------
    checks: dict[str, object] = {}
    batch_contact = (batch["contact_target_10hz"][index].numpy()[real] > 0)
    checks["contact_equal"] = bool(
        np.array_equal(batch_contact, contact > 0))
    batch_present = batch["hand_present_10hz"][index].numpy()[real]
    expected_present = present
    if context.reader is not None:
        # Presence in the batch also applies native identity/stream gates
        # (e.g. DexYCB right-only); reuse the reader's shared rules.
        record = SimpleNamespace(episode_index=episode, view_id=view_id)
        native = context.reader._native_evaluable_present(record, native_present)
        expected_present, _ = context.reader._crop_presence_override(native, crop_side)
    checks["present_equal"] = bool(np.array_equal(batch_present, expected_present))
    batch_proximity = batch["proximity_target_10hz"][index].numpy()[real]
    expect = np.zeros_like(distance, np.float32)
    finite = np.isfinite(distance)
    expect[finite] = np.exp(-tau * np.clip(distance[finite], 0.0,
                                           PROXIMITY_CLIP_M))
    checks["proximity_max_err"] = float(
        np.abs(batch_proximity - expect).max())
    if wrench is not None:
        batch_wrench = batch["hand_wrench_camera_30hz"][
            index].numpy()[real][:, 0, :]
        checks["wrench_force_norm_err"] = float(np.abs(
            np.linalg.norm(batch_wrench[:, :3], axis=-1)
            - np.linalg.norm(wrench[:, :3], axis=-1)).max())
        checks["wrench_torque_norm_err"] = float(np.abs(
            np.linalg.norm(batch_wrench[:, 3:], axis=-1)
            - np.linalg.norm(wrench[:, 3:], axis=-1)).max())
    # Proximity threshold sits above signed_distance's float16 storage
    # wobble (1-2 ulp at the 0.25 m clip ~ 0.007 proximity) and far below
    # any real transform bug (wrong rows / wrong tau / wrong gating land
    # at 0.1+).  Contact/present stay bit-exact requirements.
    mismatch = (not checks["contact_equal"] or not checks["present_equal"]
                or checks["proximity_max_err"] > 2e-2
                or checks.get("wrench_force_norm_err", 0.0) > 1e-2
                or checks.get("wrench_torque_norm_err", 0.0) > 1e-2)

    # --- extend the re-read label rows to the full 32-token window -------
    # The network receives sequence_length tokens; short runs are edge-held
    # (trajectory) or tiled (static), so the video shows every token,
    # padding included, exactly as the model does.
    if mode == "static_frame":
        def full_window(array):
            if array is None:
                return None
            if array.shape[0] == tokens_total:  # already tiled above
                return array
            return np.repeat(array, tokens_total, axis=0)
    else:
        def full_window(array):
            return None if array is None else edge_hold(
                array, tokens_total, left, right)
    contact = full_window(contact)
    distance = full_window(distance)
    present = full_window(present)
    vertices = full_window(vertices)
    force_vectors = full_window(force_vectors)
    wrench = full_window(wrench)
    object_position = full_window(object_position)
    overlay = None if overlay is None else [
        {**item, "vertices": full_window(np.asarray(item["vertices"]))}
        for item in overlay]

    padding = f"  pad:L{left}/R{right}" if (left or right) else ""
    crop_tag = (f"  crop={'L' if crop_side == 0 else 'R'}"
                if crop_side is not None else "")
    header = (f"TRAIN {split} b{batch_index:05d} s{sample:02d} {dataset} "
              f"ep={episode} view={view_id} {mode} "
              + f"solver row {int(rows[0])}, frames={len(rows)}"
              + padding + crop_tag
              + ("  MISMATCH(batch vs labels)!" if mismatch else ""))
    force_valid = bool(batch["force_supervision_valid"][index])
    real_mask = batch["real_mask"][index].numpy().astype(bool)
    batch_force_mag = np.linalg.norm(
        batch["vertex_force_camera_30hz"][index].numpy()[real][:, 0],
        axis=-1)
    extras = []
    for t in range(tokens_total):
        if not real_mask[t]:
            extras.append(f"t{t}: PAD (edge-held duplicate)")
            continue
        ri = t - left
        note = (f"batch t{t}: contact L{int((batch_contact[ri, 0]).sum())}"
                f"/R{int((batch_contact[ri, 1]).sum())}"
                f" |F|max={batch_force_mag[ri].max():.2f}N")
        if wrench is not None:
            note += (f" |wF|={np.linalg.norm(batch_wrench[ri, :3]):.2f}N"
                     f" |wT|={np.linalg.norm(batch_wrench[ri, 3:]):.3f}Nm")
        extras.append(note + ("" if force_valid else "  force-masked"))

    # --- the view IS the network input: the batch's own 256x256 letter-
    # boxed (train: photometric-jittered) frames, dedup-expanded, all 32
    # tokens; overlays project through the same intrinsics after the exact
    # letterbox_square affine.  No raw-video re-reading can drift from the
    # training pixels (the reinterhand gray panels came from that path).
    images = batch["images"]
    if "image_inverse" in batch:
        frames_chw = images[batch["image_inverse"][index]]
    else:
        frames_chw = images[index]
    frames = frames_chw.numpy()
    if frames.dtype != np.uint8:  # uint8_images=false fallback
        mean = np.array((.485, .456, .406)).reshape(1, 3, 1, 1)
        std = np.array((.229, .224, .225)).reshape(1, 3, 1, 1)
        frames = frames * std + mean
        frames = (frames.clip(0, 1) * 255).astype(np.uint8)
    size = int(frames.shape[-1])
    frames_bgr = np.ascontiguousarray(
        frames.transpose(0, 2, 3, 1)[..., ::-1])

    # Intrinsics/letterbox MUST come from the window's OWN view: the frames
    # and c2w above are that view's.  Using the canonical view's camera
    # (arctic: ego 512x384 fx457 vs exo 274x384 fx639) scales/centers the
    # overlay wrong — single-view datasets never showed it (canonical ==
    # view), and gigahands' odroid cams share intrinsics.  ``canonical``
    # stays the MANO-track view only.
    obs = context.observation(episode, view_id)
    camera = context.cameras.get(str(obs.get("camera_id")))
    if camera is None or not obs.get("width") or not obs.get("height"):
        raise RuntimeError(
            f"no usable intrinsics for {dataset} ep{episode} view "
            f"{view_id} — cannot project overlays onto the input")
    c2w = batch["camera_c2w"][index].numpy().astype(np.float64)
    obs_w, obs_h = int(obs["width"]), int(obs["height"])
    if crop_side is None:
        scale = min(float(size) / obs_w, float(size) / obs_h)
        offset_x = (size - max(1, int(round(obs_w * scale)))) // 2
        offset_y = (size - max(1, int(round(obs_h * scale)))) // 2
        k: object = CameraIntrinsics(
            camera.fx * scale, camera.fy * scale,
            camera.cx * scale + offset_x, camera.cy * scale + offset_y)
    else:
        crop_index = context.handcrop
        if crop_index is None or context.raw_size is None:
            raise RuntimeError(
                f"{dataset} crop-side window without a crop index or a "
                "raw_videos mirror to resolve crop-box coordinates")
        target = ("left", "right")[crop_side]
        # Prefer the generator's own per-frame crop intrinsics (cam_intr
        # column): they are the exact K the crop pipeline used, immune to
        # mirror-resolution ambiguity.  Fall back to box-derived K only
        # when the installed table predates that column.
        ks: list[CameraIntrinsics] = []
        crop_ks = crop_index.crop_intrinsics(
            episode, view_id, target, source_frames)
        if crop_ks is not None:
            for kmat in crop_ks:
                if kmat is None:
                    ks.append(CameraIntrinsics(0.0, 0.0, 0.0, 0.0))
                else:
                    ks.append(CameraIntrinsics(float(kmat[0, 0]),
                                               float(kmat[1, 1]),
                                               float(kmat[0, 2]),
                                               float(kmat[1, 2])))
        else:
            k_full = np.array([[camera.fx, 0.0, camera.cx],
                               [0.0, camera.fy, camera.cy],
                               [0.0, 0.0, 1.0]], np.float64)
            ks = frame_intrinsics(
                crop_index, episode, view_id, target, source_frames,
                k_full, obs_w, obs_h, (size, size),
                video_key=str(obs.get("video_key")))
        # Padding tokens repeat the edge frame's pixels, so they repeat the
        # edge frame's K too (same edge-hold the label tensors get).
        if left:
            ks = [ks[0]] * left + ks
        if right:
            ks = ks + [ks[-1]] * right
        k = ks
    view = SimpleNamespace(frames=frames_bgr, c2w=c2w, k=k, ok=True, note="")
    sheet, info = render_window(
        out_dir, header, view, contact, distance, force_vectors, vertices,
        tracks["faces"], present, force_scale, object_position, wrench,
        extras, object_overlays=overlay)
    return {"video": info["video"], "sheet": str(sheet),
            "input_view": view_id,
            "video_source": "handcrop" if crop_side is not None else "videos",
            "episode": episode, "view": canonical, "mode": mode,
            "crop_side": (None if crop_side is None
                          else ("left" if crop_side == 0 else "right")),
            "tokens": int(len(rows)),
            "geometry_solver_rows": rows.tolist(),
            "force_solver_rows": force_rows.tolist(),
            "source_frames": source_frames.tolist(), "checks": checks,
            "mismatch": bool(mismatch), **{
                key: value for key, value in info.items()
                if key not in ("video", "sheet")}}
