"""GT-vs-prediction comparison renders for eval windows.

Optional eval-side visualization driven by ``--visualize N``: for the first
``N`` validation windows per dataset, render one frame per token comparing
the GT supervision against the network's prediction, as videos + contact
sheets + per-frame PNGs + per-frame hand-mesh GLBs.  The rendered modality
set follows ``--metrics`` automatically (proximity / proximity_bands ->
proximity panels, contact -> contact panels, force -> force panels) and can
be narrowed with ``--viz-modalities``.

Every modality window renders up to three background variants:

* ``input/`` — the network's own letterboxed input frames (what the model
  actually saw);
* ``full/``  — the native full-view video of the window's view (decoded
  from the dataset video, downscaled to ``FULL_MAX_SIDE``);
* ``crop/``  — the dominant hand's hand-crop stream (only when the crop
  stream covers the episode/view; skipped otherwise, e.g. dexycb).

Each variant directory holds ``gt_vs_pred.mp4``, ``sheet.png`` and
``frames/tNNN.png`` (the composed GT|PRED row of every frame).  The
modality window dir additionally holds ``meshes/tNNN_{gt,pred}.glb``:
camera-frame hand meshes whose vertex colors mirror the 2D panels exactly
(same colormap, same thresholds, same validity masking), left hand tinted
blue and right hand pink as the non-hot base.

Faithfulness contract (mirrors
``scripts/visualization/visualize_training_batches.py``):

* the ``input/`` backgrounds ARE the network input — the batch's own
  letterboxed uint8 frames, dedup-expanded via ``image_inverse``;
* overlays project world-space MANO tracks (``solver_vertex_tracks``) of
  the label's canonical view through the displayed view's ``camera_c2w``
  and the same letterbox-adjusted intrinsics, with the shared
  ``ReviewContext`` (cameras / observations / handcrop index) and the
  reference script's token-chain resolution and edge-hold padding —
  single source of truth, no drift;
* GT/predicted quantities come from the harness ``BatchView`` decoders
  (the exact tensors the metric families accumulate), NOT from a parallel
  re-derivation: rendered colors and reported numbers cannot disagree.

Outputs are namespaced to never collide across runs: ``<viz_root>/
<checkpoint>_<split>/<dataset>/<modality>/epXXXXXX_<view>_wNNN/`` with a
per-modality ``windows.json`` index; a second eval with different
modalities, splits, or checkpoints cannot overwrite earlier renders.

Panels per modality (GT always left of PRED, same colormap + scale):

* proximity: GT distance | PRED distance (near red -> far blue) |
  |err| turbo 0-50 mm;
* contact: GT contact vertices (red) | PRED contact (prob > 0.5, red);
* force: GT |F| | PRED |F| per-vertex magnitude, shared per-window scale
  (frame-independent magnitude coloring; vertex force frames may differ
  from the displayed mesh frame, so no direction arrows).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from core.tactile_labels.decode import read_episode_columns
from core.tactile_labels.mano_tracks import solver_vertex_tracks
from core.tactile_labels.handcrop import frame_intrinsics
from core.data.camera import canonical_to_video_map
from core.visualization import (
    OverlayRenderer, compose_row, contact_sheet, encode_frames,
    proximity_color, put_text, force_color,
)
from core.geometry.camera import CameraIntrinsics
from training.evaluation.families_contact import THRESHOLD as CONTACT_THRESHOLD
from core.visualization.prepared import ReviewContext
from training.data.preprocessing import edge_hold

PROXIMITY_CLIP_M = 0.25
ERROR_SCALE_MM = 50.0     # turbo error-color clamp, documented in the footer
SHEET_FRAMES = (0, 10, 21, 31)
VIZ_MODALITIES = ("proximity", "contact", "force")
FULL_MAX_SIDE = 640       # full-view variant frames are downscaled to this
CONTACT_RED = np.array([226, 58, 58], np.uint8)      # RGB, mirrors panels
SKIN_LEFT = np.array([158, 197, 255], np.uint8)      # pastel blue
SKIN_RIGHT = np.array([255, 179, 193], np.uint8)     # pastel pink
INVALID_GRAY = np.array([185, 185, 185], np.uint8)


def modalities_from_metrics(metric_selectors: Sequence[str]) -> List[str]:
    """Map metric-family selectors to renderable modalities (stable order)."""
    selected = set()
    for selector in metric_selectors:
        if selector in ("proximity", "proximity_bands"):
            selected.add("proximity")
        elif selector == "contact":
            selected.add("contact")
        elif selector == "force":
            selected.add("force")
    return [m for m in VIZ_MODALITIES if m in selected]


class ComparisonRenderer:
    """Per-dataset contexts + window renders for the eval CLI."""

    def __init__(self, processed_root: Path, labels_root: Path,
                 handcrop_datasets: List[str], device: str = "cuda:0",
                 tau: float = 10.0,
                 modalities: Sequence[str] = ("proximity",),
                 video_backend: Optional[str] = None) -> None:
        self.processed_root = Path(processed_root)
        self.labels_root = Path(labels_root)
        self.handcrop_datasets = set(handcrop_datasets)
        self.device = torch.device(device)
        self.tau = float(tau)
        self.video_backend = video_backend  # None = VideoReader auto
        unknown = [m for m in modalities if m not in VIZ_MODALITIES]
        if unknown:
            raise ValueError(f"unknown viz modalities {unknown}; "
                             f"available: {VIZ_MODALITIES}")
        self.modalities = list(modalities)
        self.contexts: Dict[str, ReviewContext] = {}
        self.records: List[Dict[str, Any]] = []
        self._counters: Dict[tuple, int] = defaultdict(int)

    def _context(self, dataset: str) -> ReviewContext:
        if dataset not in self.contexts:
            # Crop availability is a property of the DATA (does the dataset
            # ship a handcrop stream?), not of the run's training input
            # choice — so every dataset gets a HandcropStreamIndex attempt
            # and the crop variant appears whenever the stream covers the
            # window, regardless of handcrop_datasets.
            try:
                self.contexts[dataset] = ReviewContext(
                    dataset,
                    self.processed_root, self.labels_root,
                    handcrop=True)
            except FileNotFoundError:
                self.contexts[dataset] = ReviewContext(
                    dataset,
                    self.processed_root, self.labels_root,
                    handcrop=False)
        return self.contexts[dataset]

    # -- window bookkeeping ------------------------------------------------

    def render(self, view, batch: Dict[str, Any], index: int,
               dataset: str, out_dir: Path) -> List[Dict[str, Any]]:
        """Render one batch sample's window in every selected modality.

        Each modality window gets an ``input/`` variant (the network's own
        background) plus ``full/`` and, when the crop stream covers the
        window, ``crop/`` variants decoded from the dataset videos — and a
        ``meshes/`` folder of per-frame GT/PRED GLBs colored by the
        modality's colormap.
        """
        prepared = self._prepare_window(view, batch, index, dataset)
        records: List[Dict[str, Any]] = []
        for modality in self.modalities:
            counter = self._counters[(dataset, modality)]
            self._counters[(dataset, modality)] = counter + 1
            leaf = (f"ep{prepared['episode']:06d}_"
                    f"{re.sub(r'[^A-Za-z0-9_.-]', '_', prepared['view_id'])}"
                    f"_w{counter:03d}")
            window_dir = out_dir / modality / leaf
            variants: List[tuple[str, np.ndarray, Any]] = [
                ("input", prepared["frames_bgr"], prepared["k"])]
            full = self._full_variant_frames(prepared)
            if full is not None:
                variants.append(("full", *full))
            crop = self._crop_variant_frames(prepared)
            if crop is not None:
                variants.append(("crop", *crop))
            record = {
                "dataset": dataset, "modality": modality,
                "episode": prepared["episode"], "view": prepared["view_id"],
                "mode": prepared["mode"], "tokens": prepared["tokens_total"],
                # Exact cross-predictor join key: the window's first
                # canonical frame (render order alone is fragile if two
                # evals filter windows differently).
                "frame_start": int(prepared["real_frames"][0]),
                "variants": {},
            }
            for name, frames_bgr, k in variants:
                variant_dir = window_dir / name
                variant_dir.mkdir(parents=True, exist_ok=True)
                panels = self._render_frames(prepared, modality, frames_bgr, k)
                video_path = variant_dir / "gt_vs_pred.mp4"
                encode_frames(video_path, np.stack(panels), fps=10.0, crf=20)
                sheet = contact_sheet(
                    [panels[i] for i in SHEET_FRAMES if i < len(panels)],
                    self._legend(modality, prepared["header"]))
                sheet_path = variant_dir / "sheet.png"
                import cv2
                cv2.imwrite(str(sheet_path), sheet)
                frames_dir = variant_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                frames_clean_dir = variant_dir / "frames_clean"
                frames_clean_dir.mkdir(exist_ok=True)
                for t, panel in enumerate(panels):
                    cv2.imwrite(str(frames_dir / f"t{t:03d}.png"), panel)
                    cv2.imwrite(str(frames_clean_dir / f"t{t:03d}.png"),
                                frames_bgr[t])
                record["variants"][name] = {
                    "video": str(video_path), "sheet": str(sheet_path),
                    "frames": str(frames_dir), "n_frames": len(panels),
                }
            record["meshes"] = self._export_window_glbs(
                window_dir, prepared, modality)
            record["zero_pose"] = self._export_zero_pose_glbs(
                window_dir, prepared, modality)
            # Flat input-variant pointers for backward compatibility with
            # the harness print and any windows.json consumers.
            record["video"] = record["variants"]["input"]["video"]
            record["sheet"] = record["variants"]["input"]["sheet"]
            record.update(self._window_scores(view, index, modality))
            self.records.append(record)
            records.append(record)
        return records

    def _legend(self, modality: str, header: str) -> str:
        if modality == "proximity":
            return (f"{header}  |  GT near-red->far-blue  |  PRED same "
                    f"scale  |  |err| turbo 0-{ERROR_SCALE_MM:.0f} mm")
        if modality == "contact":
            return (f"{header}  |  GT contact vertices red  |  PRED "
                    f"prob>{CONTACT_THRESHOLD:.2f} red")
        return (f"{header}  |  |F| turbo, shared per-window scale "
                f"(GT max)")

    @staticmethod
    def _window_scores(view, index: int, modality: str) -> Dict[str, Any]:
        """One headline number per modality for the windows.json index."""
        if modality == "proximity":
            valid = np.isfinite(view.distance_mm_gt[index].cpu().numpy())
            err = (view.distance_mm_hat[index].cpu().numpy()
                   - view.distance_mm_gt[index].cpu().numpy())
            return {"ade_mm": float(np.abs(err[valid]).mean())
                    if valid.any() else None}
        if modality == "contact":
            # Same binarisation threshold the contact family scores with
            # (per-predictor operating point — e.g. EgoPHI 0.3, not 0.5):
            # the panel must show what the metrics count.
            from training.evaluation.families_contact import THRESHOLD as contact_threshold
            mask = view.contact_mask[index].bool().expand_as(
                view.contact_prob[index]).cpu().numpy()
            pred = view.contact_prob[index].cpu().numpy() > contact_threshold
            gt = view.contact_target[index].bool().cpu().numpy()
            tp = float((pred & gt & mask).sum())
            fp = float((pred & ~gt & mask).sum())
            fn = float((~pred & gt & mask).sum())
            return {"f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
                    "precision": tp / max(tp + fp, 1.0),
                    "recall": tp / max(tp + fn, 1.0)}
        mask = view.force_mask.bool()[index]          # [T,S]
        gt = view.force_target[index].cpu().numpy()   # [T,S,H,V,3]
        pred = view.force_pred[index].cpu().numpy()
        if bool(mask.all()) or not bool(mask.any()):
            kept = slice(None)
        else:
            kept = mask.cpu().numpy()                 # gather per substep
        sel = kept if isinstance(kept, slice) else kept
        err = np.linalg.norm(pred - gt, axis=-1)[sel]
        return {"epe_n": float(err.mean()) if err.size else None}

    # -- per-window preparation (modality-independent) ---------------------

    def _prepare_window(self, view, batch: Dict[str, Any], index: int,
                        dataset: str) -> Dict[str, Any]:
        context = self._context(dataset)
        episode = int(batch["episode_index"][index])
        view_id = str(batch["view_id"][index])
        mode = str(batch["dataset_mode"][index])
        labels = context.index[episode]

        tokens_total = int(batch["real_mask"].shape[1])
        left = int(batch["left_padding"][index])
        right = int(batch["right_padding"][index])
        source_frames = batch["source_frames"][index].cpu().numpy()
        real_frames = (source_frames[left:tokens_total - right]
                       if tokens_total - right > left else source_frames)

        tactile = read_episode_columns(
            context.labels_root, labels["tactile_groups"],
            ("source_position",))
        rows = batch["geometry_solver_rows"][index].cpu().numpy()
        rows = rows[left:tokens_total - right] if tokens_total - right > left else rows
        if mode == "static_frame":
            rows = rows[:1]

        # MANO vertex tracks (canonical view, world space) scoped to the
        # window's bracket frames — the same geometry the GT labels were
        # rendered from, so overlay vs pixels drift is impossible.
        canonical = str(labels.get("canonical_view") or view_id)
        track_view = canonical
        if view_id != canonical and np.allclose(
                batch["camera_c2w"][index].cpu().numpy(), np.eye(4),
                atol=1e-6):
            track_view = view_id   # degenerate-world dataset: per-view solve
        tracks = solver_vertex_tracks(
            context.root, episode, track_view,
            source_positions=tactile["source_position"], rows=rows,
            with_faces=True,
            device="cuda" if self.device.type == "cuda" else "cpu")
        vertices = np.stack([tracks["left"][0], tracks["right"][0]], axis=1)
        present = batch["hand_present_10hz"][index].cpu().numpy().astype(bool)

        if mode == "static_frame":
            def full_window(array: Optional[np.ndarray]):
                return None if array is None else np.repeat(
                    array, tokens_total, axis=0)
        else:
            def full_window(array: Optional[np.ndarray]):
                return None if array is None else edge_hold(
                    array, tokens_total, left, right)
        vertices = full_window(vertices)

        # --- network input as background (dedup-expanded uint8 frames) ----
        images = batch["images"]
        if "image_inverse" in batch:
            frames_chw = images[batch["image_inverse"][index]]
        else:
            frames_chw = images[index]
        frames = frames_chw.cpu().numpy()
        if frames.dtype != np.uint8:  # uint8_images=false fallback
            mean = np.array((.485, .456, .406)).reshape(1, 3, 1, 1)
            std = np.array((.229, .224, .225)).reshape(1, 3, 1, 1)
            frames = frames * std + mean
            frames = (frames.clip(0, 1) * 255).astype(np.uint8)
        size = int(frames.shape[-1])
        frames_bgr = np.ascontiguousarray(
            frames.transpose(0, 2, 3, 1)[..., ::-1])

        # Letterbox-adjusted intrinsics of the window's OWN view (crop
        # datasets: per-frame crop-adjusted K via the shared index).
        obs = context.observation(episode, view_id)
        camera = context.cameras.get(str(obs.get("camera_id")))
        if camera is None or not obs.get("width") or not obs.get("height"):
            raise RuntimeError(
                f"no usable intrinsics for {dataset} ep{episode} view "
                f"{view_id} — cannot project overlays")
        c2w = batch["camera_c2w"][index].cpu().numpy().astype(np.float64)
        obs_w, obs_h = int(obs["width"]), int(obs["height"])
        crop_side = self._crop_side(context, episode, view_id, mode, present)
        # Honor the loader's own crop-side decision when the window was
        # read through the hand-crop stream.  Re-deriving the side from
        # ``hand_present_10hz`` is wrong on hand-crop batches: the off-crop
        # hand is force-present there (present-with-zeros supervision), so
        # bimanual windows would flip to the wrong hand.
        loader_side = batch.get("crop_side")
        if loader_side is not None:
            if torch.is_tensor(loader_side):
                loader_side = int(loader_side.detach().cpu().reshape(-1)[0])
            else:
                loader_side = int(np.asarray(loader_side).reshape(-1)[0])
            if loader_side >= 0:
                crop_side = loader_side
        if crop_side is None:
            scale = min(float(size) / obs_w, float(size) / obs_h)
            offset_x = (size - max(1, int(round(obs_w * scale)))) // 2
            offset_y = (size - max(1, int(round(obs_h * scale)))) // 2
            k: Any = CameraIntrinsics(
                camera.fx * scale, camera.fy * scale,
                camera.cx * scale + offset_x, camera.cy * scale + offset_y)
        else:
            if context.raw_size is None:
                raise RuntimeError(
                    f"{dataset} crop window without a raw_videos mirror")
            k_full = np.array([[camera.fx, 0.0, camera.cx],
                               [0.0, camera.fy, camera.cy],
                               [0.0, 0.0, 1.0]], np.float64)
            ks = frame_intrinsics(
                context.handcrop, episode, view_id,
                ("left", "right")[crop_side],
                real_frames, k_full, obs_w, obs_h, (size, size),
                video_key=view_id)
            if left:
                ks = [ks[0]] * left + ks
            if right:
                ks = ks + [ks[-1]] * right
            k = ks

        header = (f"EVAL {dataset} ep={episode} view={view_id} {mode} "
                  + (f"row {int(rows[0])}" if mode == "static_frame"
                     else f"tokens {int(real_frames[0])}+{len(rows)}"))
        return {
            "header": header, "episode": episode, "view_id": view_id,
            "mode": mode, "tokens_total": tokens_total,
            "dataset": dataset, "left": left, "right": right,
            "real_frames": real_frames, "crop_side": crop_side,
            "frames_bgr": frames_bgr, "c2w": c2w, "k": k,
            "vertices": vertices, "faces": tracks["faces"],
            "present": present, "view": view, "index": index,
        }

    @staticmethod
    def _crop_side(context: ReviewContext, episode: int, view_id: str,
                   mode: str, present: np.ndarray) -> Optional[int]:
        """Reader's per-window crop side (HandcropStreamIndex.input_side).

        Static-frame datasets (mow, obman) carry crop streams too — crop
        availability is decided by the stream index, not the topology.
        """
        del mode  # topology does not gate crop availability
        if context.handcrop is None:
            return None
        return context.handcrop.input_side(episode, view_id, present)

    # -- full / crop background variants ------------------------------------

    def _video_reader(self, context: ReviewContext):
        """Lazy per-dataset VideoReader; returns ``(reader, get_id)`` where
        ``get_id(path)`` registers a video path on first use (the reader's
        path dict is shared, so late registrations are visible).  The
        backend is the run's configured ``video_backend`` (torchcodec for
        the formal runs) — cv2 random seeks over NFS are pathologically
        slow on the long chunk files."""
        cached = getattr(self, "_video_readers", None)
        if cached is None:
            cached = self._video_readers = {}
        if context.name not in cached:
            from core.io.video_reader import VideoReader
            ids: dict[Path, int] = {}
            by_id: dict[int, Path] = {}

            def get_id(path) -> int:
                path = Path(path)
                if path not in ids:
                    ids[path] = len(ids)
                    by_id[ids[path]] = path
                return ids[path]

            cached[context.name] = (
                VideoReader(by_id, cache_size=4, image_mode="rgb",
                            backend=self.video_backend), get_id)
        return cached[context.name]

    def _episode_row(self, context: ReviewContext, episode: int) -> dict:
        """The dataset's ``meta/episodes`` row (video chunk/file/timestamps
        live there, not in the label index)."""
        cached = getattr(self, "_episode_rows", None)
        if cached is None:
            cached = self._episode_rows = {}
        if context.name not in cached:
            from core.data.inspection import read_episode_rows
            cached[context.name] = {int(r["episode_index"]): r
                                    for r in read_episode_rows(context.root)}
        return cached[context.name][int(episode)]

    def _full_variant_frames(self, prep):
        """Native full-view background + intrinsics, or None on any gap.

        Frames decode from the dataset video at the window's canonical
        positions (interval-map aware, edge-held over the padding) and are
        downscaled so the longest side is FULL_MAX_SIDE; K scales with
        them.  Overlay geometry (c2w, tracks) is unchanged — same view.
        """
        context = self._context(prep["dataset"])
        episode, view_id = prep["episode"], prep["view_id"]
        obs = context.observation(episode, view_id)
        video_key = str(obs.get("video_key") or "")
        if not video_key:
            return None
        row = self._episode_row(context, episode)
        prefix = f"videos/{video_key}"
        relative = row.get(f"{prefix}/path")
        if relative is None and f"{prefix}/chunk_index" not in row:
            return None
        try:
            if relative is None:
                relative = (f"{prefix}/chunk-{int(row[f'{prefix}/chunk_index']):03d}"
                            f"/file-{int(row[f'{prefix}/file_index']):03d}.mp4")
            video_path = context.root / relative
            reader, get_id = self._video_reader(context)
            positions = self._video_positions(context, episode, obs,
                                              prep["real_frames"])
            frames = reader.read_frames(get_id(video_path), positions.tolist())
            if frames is None or len(frames) == 0:
                return None
            frames = np.stack([np.asarray(f, np.uint8) for f in frames])
        except Exception as error:  # noqa: BLE001 — full variant is best-effort
            print(f"[viz] full variant unavailable for ep{episode}: {error!r}",
                  flush=True)
            return None
        frames = edge_hold(frames, prep["tokens_total"],
                           prep["left"], prep["right"])
        import cv2
        h, w = frames.shape[1:3]
        scale = min(1.0, FULL_MAX_SIDE / max(h, w))
        if scale < 1.0:
            frames = np.stack([
                cv2.resize(f, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
                for f in frames])
        camera = context.cameras.get(str(obs.get("camera_id")))
        if camera is None:
            return None
        k = CameraIntrinsics(camera.fx * scale, camera.fy * scale,
                             camera.cx * scale, camera.cy * scale)
        return np.ascontiguousarray(frames[..., ::-1]), k

    def _video_positions(self, context: ReviewContext, episode: int,
                         obs: dict, real_frames: np.ndarray) -> np.ndarray:
        """Video-file positions for the window's canonical frames."""
        try:
            mapping = canonical_to_video_map(context.root, obs)
        except Exception:  # noqa: BLE001 — dense fallback
            mapping = None
        video_key = str(obs["video_key"])
        row = self._episode_row(context, episode)
        fps = float(json.loads(
            (context.root / "meta/info.json").read_text()).get("fps", 30.0))
        start = int(round(float(row[f"videos/{video_key}/from_timestamp"]) * fps))
        positions = real_frames if mapping is None else mapping[0][real_frames]
        return start + positions

    def _crop_variant_frames(self, prep):
        """Dominant-hand crop background + per-frame crop K, or None.

        Works for both rigid_trajectory and static_frame windows (static
        real_frames are a single canonical row; edge_hold fills the rest).
        """
        context = self._context(prep["dataset"])
        if context.handcrop is None or prep["crop_side"] is None:
            return None
        size = int(prep["frames_bgr"].shape[1])   # match the input variant
        side = ("left", "right")[prep["crop_side"]]
        try:
            reader, get_id = self._video_reader(context)
            frames = context.handcrop.read_frames(
                prep["episode"], prep["view_id"], side,
                np.asarray(prep["real_frames"], np.int64), reader,
                video_id=get_id)
            if frames is None or len(frames) == 0:
                return None
            frames = np.stack([np.asarray(f, np.uint8) for f in frames])
            # NO visibility fill: the stream's hand-absent black frames are
            # exactly what the network sees at training time (the absent
            # hand is supervised present-with-zeros there), so the crop
            # variant shows them as-is to stay faithful to training.
        except Exception as error:  # noqa: BLE001 — crop variant is best-effort
            print(f"[viz] crop variant unavailable for ep{prep['episode']}: "
                  f"{error!r}", flush=True)
            return None
        frames = edge_hold(frames, prep["tokens_total"],
                           prep["left"], prep["right"])
        camera = context.cameras.get(
            str(context.observation(prep["episode"], prep["view_id"])
                .get("camera_id")))
        if camera is None:
            return None
        if context.raw_size is None:
            return None
        k_full = np.array([[camera.fx, 0.0, camera.cx],
                           [0.0, camera.fy, camera.cy],
                           [0.0, 0.0, 1.0]], np.float64)
        obs = context.observation(prep["episode"], prep["view_id"])
        ks = frame_intrinsics(
            context.handcrop, prep["episode"], prep["view_id"], side,
            prep["real_frames"], k_full, int(obs["width"]),
            int(obs["height"]), (size, size))
        ks = edge_hold(np.asarray(ks), prep["tokens_total"],
                       prep["left"], prep["right"])
        return np.ascontiguousarray(frames[..., ::-1]), list(ks)

    # -- frame rendering ----------------------------------------------------

    def _render_frames(self, prep: Dict[str, Any], modality: str,
                       frames_bgr: np.ndarray,
                       k) -> List[np.ndarray]:
        view, index = prep["view"], prep["index"]
        c2w = prep["c2w"]
        vertices, faces, present = (prep["vertices"], prep["faces"],
                                    prep["present"])
        overlay = OverlayRenderer()
        rows_out: List[np.ndarray] = []
        mesh_present = present | (
            np.linalg.norm(vertices, axis=-1).max(axis=2) > 0)
        try:
            for t in range(len(frames_bgr)):
                base = frames_bgr[t].copy()
                dim = (base * 0.6).astype(np.uint8)
                w2c = np.linalg.inv(c2w[t])
                k_frame = k[int(t)] if isinstance(k, list) else k
                meshes = [dict(vertices=vertices[t, side], faces=faces[name])
                          for side, name in ((0, "left"), (1, "right"))
                          if mesh_present[t, side]]
                panels: List[np.ndarray] = []
                if modality == "proximity":
                    dist_gt = (view.distance_mm_gt[index].detach()
                               .cpu().numpy() / 1000.0).clip(
                        0.0, PROXIMITY_CLIP_M)
                    dist_pred = (view.distance_mm_hat[index].detach()
                                 .cpu().numpy() / 1000.0).clip(
                        0.0, PROXIMITY_CLIP_M)
                    err_mm = np.abs(
                        view.distance_mm_hat[index].cpu().numpy()
                        - view.distance_mm_gt[index].cpu().numpy())
                    panels += self._proximity_panels(
                        overlay, base, dim, w2c, k_frame, meshes,
                        mesh_present, dist_gt, dist_pred, err_mm, t)
                elif modality == "contact":
                    panels += self._contact_panels(
                        overlay, base, dim, w2c, k_frame, meshes,
                        mesh_present, view, index, t)
                else:
                    panels += self._force_panels(
                        overlay, base, dim, w2c, k_frame, meshes,
                        mesh_present, view, index, t)
                footer = (prep["header"] if t == 0 else
                          f"{prep['header']}  t={t}/{len(frames_bgr)}")
                rows_out.append(compose_row(panels, gap=6, header="",
                                            footer=footer))
        finally:
            overlay.close()
        return rows_out

    def _proximity_panels(self, overlay, base, dim, w2c, k_frame, meshes,
                          mesh_present, dist_gt, dist_pred, err_mm,
                          t) -> List[np.ndarray]:
        panels: List[np.ndarray] = []
        for tag, dist in (("GT PROXIMITY", dist_gt),
                          ("PRED PROXIMITY", dist_pred)):
            colors = [proximity_color(dist[t, side])[..., ::-1].copy()
                      for side in (0, 1) if mesh_present[t, side]]
            panel = overlay.render(
                base, w2c, k_frame,
                [dict(mesh, colors=c) for mesh, c in zip(meshes, colors)],
                alpha=.98, background=dim)
            valid = np.isfinite(dist[t])
            median = (float(np.median(dist[t][valid])) * 1000.0
                      if valid.any() else float("nan"))
            put_text(panel, tag, (8, 20), 0.55)
            put_text(panel, f"median {median:.0f} mm",
                     (8, panel.shape[0] - 12), 0.45, (180, 255, 180))
            panels.append(panel)
        err_colors = [
            force_color(err_mm[t, side], ERROR_SCALE_MM)[0][..., ::-1].copy()
            for side in (0, 1) if mesh_present[t, side]]
        p_err = overlay.render(
            base, w2c, k_frame,
            [dict(mesh, colors=c) for mesh, c in zip(meshes, err_colors)],
            alpha=.98, background=dim)
        e_valid = np.isfinite(err_mm[t])
        e_med = (float(np.median(err_mm[t][e_valid]))
                 if e_valid.any() else float("nan"))
        put_text(p_err, "|PRED-GT| ERROR", (8, 20), 0.55)
        put_text(p_err, f"median {e_med:.0f} mm (turbo 0-50)",
                 (8, p_err.shape[0] - 12), 0.45, (180, 255, 180))
        panels.append(p_err)
        return panels

    def _contact_panels(self, overlay, base, dim, w2c, k_frame, meshes,
                        mesh_present, view, index, t) -> List[np.ndarray]:
        contact_rgb = np.array([40, 40, 255], np.uint8)   # RGB red
        skin = np.array([150, 172, 224], np.uint8)
        prob = view.contact_prob[index].detach().cpu().numpy()
        gt = view.contact_target[index].bool().cpu().numpy()
        valid = view.contact_mask[index].bool()
        valid = valid.expand_as(view.contact_prob[index]).cpu().numpy()
        panels: List[np.ndarray] = []
        for tag, hot in (("GT CONTACT", gt),
                         ("PRED CONTACT", prob > CONTACT_THRESHOLD)):
            colors = []
            for side in (0, 1):
                if not mesh_present[t, side]:
                    continue
                vcolors = np.tile(skin, (778, 1))
                vcolors[hot[t, side] & valid[t, side]] = contact_rgb
                colors.append(vcolors)
            panel = overlay.render(
                base, w2c, k_frame,
                [dict(mesh, colors=c) for mesh, c in zip(meshes, colors)],
                alpha=.85)
            counts = []
            for side in (0, 1):
                if mesh_present[t, side]:
                    n = int((hot[t, side] & valid[t, side]).sum())
                    counts.append(f"{'L' if side == 0 else 'R'} {n}")
            put_text(panel, tag, (8, 20), 0.55)
            put_text(panel, "  ".join(counts) + " contact vtx",
                     (8, panel.shape[0] - 12), 0.45, (180, 255, 180))
            panels.append(panel)
        return panels

    def _force_panels(self, overlay, base, dim, w2c, k_frame, meshes,
                      mesh_present, view, index, t) -> List[np.ndarray]:
        """Per-vertex |F| coloring; magnitudes are frame-independent, so
        this is safe under any ``vertex_force_frame`` (no direction arrows)."""
        gt = view.force_target[index, t, 0].float().cpu().numpy()
        pred = view.force_pred[index, t, 0].float().cpu().numpy()
        valid = bool(view.force_mask[index, t, 0])
        mag_gt = np.linalg.norm(gt, axis=-1)
        mag_pred = np.linalg.norm(pred, axis=-1)
        scale = max(float(mag_gt.max()), 1e-6)
        panels: List[np.ndarray] = []
        for tag, mag in ((f"GT |F| (scale {scale:.1f}N)", mag_gt),
                         ("PRED |F|", mag_pred)):
            colors = [force_color(np.clip(mag[side], 0.0, scale),
                                  scale)[0][..., ::-1].copy()
                      for side in (0, 1) if mesh_present[t, side]]
            if not valid:
                for color in colors:
                    color[:] = INVALID_GRAY
            panel = overlay.render(
                base, w2c, k_frame,
                [dict(mesh, colors=c) for mesh, c in zip(meshes, colors)],
                alpha=.98, background=dim)
            put_text(panel, tag, (8, 20), 0.55)
            put_text(panel, f"max {mag.max():.2f} N" if valid else "force masked",
                     (8, panel.shape[0] - 12), 0.45, (180, 255, 180))
            panels.append(panel)
        return panels

    # -- per-frame hand-mesh GLBs -------------------------------------------

    @staticmethod
    def _zero_pose_templates():
        """Canonical zero-pose MANO templates + faces, cached.

        The SAME template the label frames are built from (flat-hand mean,
        zeros pose), so per-frame vertex values paint onto it without any
        correspondence ambiguity.  Returns {side: (verts [778,3], faces)}.
        """
        cached = getattr(ComparisonRenderer, "_templates", None)
        if cached is not None:
            return cached
        from core.geometry.mano_layer import ManoTorchLayer as ManoLayer
        cached = {}
        with torch.no_grad():
            for side, faces_key in (("left", "left"), ("right", "right")):
                layer = ManoLayer(
                    side=side, use_pca=False,
                    flat_hand_mean=True, ncomps=45, center_idx=None).eval()
                verts = layer(
                    torch.zeros(1, 48), torch.zeros(1, 10)).verts[0]
                from core.geometry.force_frames import _mano_topology
                faces, _, _ = _mano_topology(side)
                cached[side] = (verts.numpy().astype(np.float64), faces)
        ComparisonRenderer._templates = cached
        return cached

    def _export_zero_pose_glbs(self, window_dir: Path,
                               prep: Dict[str, Any],
                               modality: str) -> Optional[str]:
        """Per-frame GT/PRED GLBs on the zero-pose MANO template.

        Both hands are always shown (absent hands render all-gray — the
        template geometry is valid even when the track is not), offset
        along x so they do not overlap.  Vertex colors come from the same
        ``_mesh_colors`` decisions as the posed ``meshes/`` GLBs.
        """
        view, index = prep["view"], prep["index"]
        zero_dir = window_dir / "zero_pose"
        try:
            import trimesh
        except ImportError:
            return None
        templates = self._zero_pose_templates()
        zero_dir.mkdir(parents=True, exist_ok=True)
        total = prep["tokens_total"]
        offset = {"left": np.array([-0.12, 0.0, 0.0]),
                  "right": np.array([0.12, 0.0, 0.0])}
        written = 0
        try:
            for t in range(total):
                for tag, colors_by_side in self._mesh_colors(
                        modality, view, index, t, None):
                    scene = trimesh.Scene()
                    for side, name in ((0, "left"), (1, "right")):
                        verts, faces = templates[name]
                        mesh = trimesh.Trimesh(
                            vertices=verts + offset[name],
                            faces=faces, process=False)
                        mesh.visual.vertex_colors = colors_by_side[side]
                        scene.add_geometry(mesh, node_name=name)
                    scene.export(file_obj=str(
                        zero_dir / f"t{t:03d}_{tag}.glb"))
                    written += 1
        except Exception as error:  # noqa: BLE001 — zero-pose is best-effort
            print(f"[viz] zero-pose GLBs unavailable for ep{prep['episode']}: "
                  f"{error!r}", flush=True)
            return None
        return str(zero_dir) if written else None

    def _export_window_glbs(self, window_dir: Path, prep: Dict[str, Any],
                            modality: str) -> Optional[str]:
        """Per-frame GT/PRED hand meshes as GLBs, colored like the panels.

        Vertices are transformed to the camera frame of their frame (the
        space the 2D overlays project); left hand is tinted blue and right
        hand pink as the non-hot base, hot vertices follow the modality's
        colormap/thresholds exactly.  Returns the mesh folder path.
        """
        view, index = prep["view"], prep["index"]
        c2w = prep["c2w"]
        vertices, faces, present = (prep["vertices"], prep["faces"],
                                    prep["present"])
        mesh_present = present | (
            np.linalg.norm(vertices, axis=-1).max(axis=2) > 0)
        mesh_dir = window_dir / "meshes"
        try:
            import trimesh
        except ImportError:
            return None
        mesh_dir.mkdir(parents=True, exist_ok=True)
        total = prep["tokens_total"]
        written = 0
        try:
            for t in range(total):
                w2c = np.linalg.inv(c2w[t])
                # Skip zero/degenerate sides: absent-hand tracks carry
                # all-zero vertices, and a degenerate mesh in the scene can
                # blank out the whole file in some GLB viewers.
                sides = [s for s in (0, 1) if mesh_present[t, s]
                         and np.linalg.norm(vertices[t, s], axis=-1).max()
                         > 1e-6]
                if not sides:
                    continue
                for tag, colors_by_side in self._mesh_colors(
                        modality, view, index, t, mesh_present):
                    scene = trimesh.Scene()
                    for side in sides:
                        name = "left" if side == 0 else "right"
                        world = vertices[t, side]
                        cam = world @ w2c[:3, :3].T + w2c[:3, 3]
                        mesh = trimesh.Trimesh(
                            vertices=cam.astype(np.float64),
                            faces=faces[name], process=False)
                        mesh.visual.vertex_colors = colors_by_side[side]
                        scene.add_geometry(mesh, node_name=name)
                    scene.export(file_obj=str(
                        mesh_dir / f"t{t:03d}_{tag}.glb"))
                    written += 1
        except Exception:  # noqa: BLE001 — GLB export is best-effort
            return None
        return str(mesh_dir) if written else None

    def _mesh_colors(self, modality: str, view, index: int, t: int,
                     mesh_present: np.ndarray):
        """Yield ``(tag, {side: (778,4) uint8 RGBA})`` for GT then PRED.

        Colors mirror the 2D panels' colormap/threshold/validity decisions
        so the GLBs and the rendered images cannot disagree.
        """
        if modality == "contact":
            prob = view.contact_prob[index].detach().cpu().numpy()
            gt = view.contact_target[index].bool().cpu().numpy()
            valid = (view.contact_mask[index].bool()
                     .expand_as(view.contact_prob[index]).cpu().numpy())
            hot = {"gt": gt, "pred": prob > 0.5}
            for tag, mask in hot.items():
                colors = {}
                for side in (0, 1):
                    base = (SKIN_LEFT if side == 0 else SKIN_RIGHT).copy()
                    vcolors = np.tile(base, (778, 1))
                    vcolors[~valid[t, side]] = INVALID_GRAY
                    vcolors[mask[t, side] & valid[t, side]] = CONTACT_RED
                    colors[side] = np.concatenate(
                        [vcolors, np.full((778, 1), 255, np.uint8)], axis=1)
                yield tag, colors
            return
        if modality == "proximity":
            dist = {
                "gt": (view.distance_mm_gt[index].detach().cpu().numpy()
                       / 1000.0).clip(0.0, PROXIMITY_CLIP_M),
                "pred": (view.distance_mm_hat[index].detach().cpu().numpy()
                         / 1000.0).clip(0.0, PROXIMITY_CLIP_M),
            }
            for tag, d in dist.items():
                colors = {}
                for side in (0, 1):
                    rgb = proximity_color(d[t, side])[..., ::-1]
                    rgb = np.asarray(rgb, np.uint8)
                    colors[side] = np.concatenate(
                        [rgb, np.full((778, 1), 255, np.uint8)], axis=1)
                yield tag, colors
            return
        gt = view.force_target[index, t, 0].float().cpu().numpy()
        pred = view.force_pred[index, t, 0].float().cpu().numpy()
        valid = bool(view.force_mask[index, t, 0])
        mag = {"gt": np.linalg.norm(gt, axis=-1),
               "pred": np.linalg.norm(pred, axis=-1)}
        scale = max(float(mag["gt"].max()), 1e-6)
        for tag in ("gt", "pred"):
            colors = {}
            for side in (0, 1):
                rgb = force_color(np.clip(mag[tag][side], 0.0, scale),
                                  scale)[0][..., ::-1]
                rgb = np.asarray(rgb, np.uint8).copy()
                if not valid:
                    rgb[:] = INVALID_GRAY
                colors[side] = np.concatenate(
                    [rgb, np.full((778, 1), 255, np.uint8)], axis=1)
            yield tag, colors

    def write_index(self, out_root: Path) -> None:
        if self.records:
            (out_root / "windows.json").write_text(
                json.dumps(self.records, indent=1, default=str))
