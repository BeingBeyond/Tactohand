"""Shared rendering for prepared labels, training inputs and predictions.

Reads existing videos and labels and writes visualization outputs. Contains no
label materialization, episode sampling, solver loading or dataset installation.
"""
from __future__ import annotations

from pathlib import Path
import sys


import json
import cv2
import numpy as np
from core.io.video_reader import VideoReader
from core.data.camera import canonical_to_video_map as shared_canonical_to_video_map
from core.visualization import OverlayRenderer, compose_row, contact_sheet, draw_arrow_world, draw_contact_markers, draw_force_legend, draw_object_silhouette, encode_frames, force_color, proximity_color, put_text, scale_intrinsics, target_force_color, turbo_color, view_camera
from core.tactile_labels.decode import read_episode_columns
from core.tactile_labels.schema import REQUIRED_FORCE_COLUMNS
import pyarrow.parquet as pq
from core.visualization import instance_geometry_ids, load_object_meshes, object_color, pose_object_vertices


GAP = 6


SHEET_FRAMES = (0, 10, 21, 31)


def canonical_to_video_map(root: Path, obs: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Shared implementation: core.data.camera."""
    return shared_canonical_to_video_map(root, obs)


class WindowView:
    """Native-aspect downscaled frames + camera context for one window."""

    def __init__(self, root: Path, episode_row: dict, obs: dict,
                 cameras: dict, source_frames: np.ndarray, reader,
                 max_width: int = 640, handcrop=None):
        """``handcrop=(HandcropStreamIndex, target_id)`` reads the input
        frames from the v1 hand-crop stream of that hand instead of the
        full view video; intrinsics become per-frame (the crop window
        tracks the hand), so ``self.k`` is ``(T, 3, 3)``.
        """
        self.ok = True
        self.note = ""
        self.scale = 1.0
        video_key = str(obs["video_key"])
        camera_rows = np.asarray(source_frames, np.int64)
        if handcrop is not None:
            crop_index, target_id = handcrop
            try:
                from core.tactile_labels.handcrop import frame_intrinsics
                frames = crop_index.read_frames(
                    int(episode_row["episode_index"]), str(obs["view_id"]),
                    target_id, np.asarray(source_frames, np.int64), reader)
                # VideoReader yields RGB; the drawing stack is BGR.
                self.frames = np.stack(
                    [np.asarray(f, np.uint8) for f in frames])[..., ::-1]
                self.scale = min(1.0, max_width / self.frames.shape[2])
                _c2w, intrinsics = view_camera(root, obs, cameras)
                if not isinstance(intrinsics, np.ndarray):
                    # view_camera may yield a CameraIntrinsics value.
                    intrinsics = np.array(
                        [[intrinsics.fx, 0.0, intrinsics.cx],
                         [0.0, intrinsics.fy, intrinsics.cy],
                         [0.0, 0.0, 1.0]])
                self.c2w = _c2w[camera_rows]
                out_size = (int(self.frames.shape[2]),
                            int(self.frames.shape[1]))
                from core.visualization.camera import CameraIntrinsics
                # Shared raw-mirror convention: frame_intrinsics scales the
                # boxes from raw-mirror pixels into the installed obs space
                # and maps the full-frame intrinsics through each window.
                per_frame = frame_intrinsics(
                    crop_index, int(episode_row["episode_index"]),
                    str(obs["view_id"]), target_id,
                    np.asarray(source_frames, np.int64), intrinsics,
                    int(obs["width"]), int(obs["height"]), out_size,
                    video_key=video_key)
                # The drawing stack ducks on CameraIntrinsics; a crop window
                # moves with the hand, so K is per displayed frame.
                self.k = [CameraIntrinsics(value.fx * self.scale,
                                           value.fy * self.scale,
                                           value.cx * self.scale,
                                           value.cy * self.scale)
                          for value in per_frame]
                return
            except Exception as error:  # noqa: BLE001 — fall back to full
                self.note = f"handcrop unavailable ({error}); full video"
        try:
            chunk = int(episode_row[f"videos/{video_key}/chunk_index"])
            file_index = int(episode_row[f"videos/{video_key}/file_index"])
            path = root / (f"videos/{video_key}/chunk-{chunk:03d}"
                           f"/file-{file_index:03d}.mp4")
            fps = float(json.loads((root / "meta/info.json").read_text())
                        .get("fps", 30.0))
            start = int(round(float(
                episode_row[f"videos/{video_key}/from_timestamp"]) * fps))
            positions = np.asarray(source_frames, np.int64)
            video_map = canonical_to_video_map(root, obs)
            if video_map is not None:
                # Sparse video: the map holds (video index, canonical time
                # of that frame).  The mesh is projected with the CAMERA AT
                # THE DISPLAYED FRAME'S OWN TIME — a time-mismatched c2w
                # offsets the overlay even when the frame is a real sample.
                video_positions, sample_rows = video_map
                # Sparse positions are segment-relative: rebase onto the
                # episode's span inside the shared chunk video, exactly like
                # the dense branch below.
                positions = start + video_positions[positions]
                camera_rows = sample_rows[np.asarray(source_frames, np.int64)]
            else:
                positions = start + positions
            frames = reader.read_frames(
                reader.video_id(path), positions.tolist())
            first = np.asarray(frames[0], np.uint8)
            self.scale = min(1.0, max_width / first.shape[1])
            size = (max(1, int(round(first.shape[1] * self.scale))),
                    max(1, int(round(first.shape[0] * self.scale))))
            resized = [cv2.resize(np.asarray(f, np.uint8), size)
                       for f in frames]
            # VideoReader yields RGB; the drawing stack is BGR.
            self.frames = np.stack(resized)[..., ::-1].copy()
        except Exception as error:  # noqa: BLE001 — visualize with placeholders
            self.ok = False
            self.note = f"video unavailable ({error})"
            self.frames = np.full((len(source_frames), 360, 640, 3),
                                  40, np.uint8)
        c2w, intrinsics = view_camera(root, obs, cameras)
        self.c2w = c2w[camera_rows]
        self.k = scale_intrinsics(intrinsics, self.scale) if self.ok \
            else intrinsics


def render_window(out_dir: Path, header: str, view: WindowView,
                  contact: np.ndarray, distance: np.ndarray,
                  force_vectors: np.ndarray | None, vertices: np.ndarray,
                  faces: dict, present: np.ndarray, force_scale: float,
                  object_position: np.ndarray | None,
                  wrench: np.ndarray | None,
                  extras: list[str],
                  object_overlays: list[dict] | None = None,
                  extra_views: list | None = None
                  ) -> tuple[Path, dict]:
    """Four-panel pyrender video + contact sheet via core.visualization.

    Panels: INPUT(+shaded contact-colored hand meshes) | CONTACT | PROXIMITY
    | FORCE (force-colored meshes + per-vertex arrows + wrench arrow).

    ``extra_views`` is a list of ``(tag, WindowView)`` — one per extra
    INPUT background (hand-crop datasets: the other hand's crop).  Each
    frame renders one four-panel strip per view and stacks them
    vertically; window statistics are collected from the primary view
    only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_count = len(view.frames)
    video_path = out_dir / "labels_video.mp4"
    skin = np.array([150, 172, 224], np.uint8)          # RGB
    contact_rgb = np.array([40, 40, 255], np.uint8)     # RGB red
    rest_rgb = np.array([230, 130, 40], np.uint8)       # RGB blue
    gray_rgb = np.array([120, 120, 120], np.uint8)
    overlay = OverlayRenderer()
    contact_counts, d50s, fmaxs = [], [], []
    rows_out = []
    # Hand-hand datasets (interhand26m) mark label presence from contact
    # validity: hands with no contact carry present=0 while their MANO
    # geometry exists.  Draw the shaded mesh whenever vertices exist so
    # those windows still show the tracked hands (uncolored, no contact);
    # every label-driven readout stays gated on the label flags below.
    mesh_present = present | (
        np.linalg.norm(vertices, axis=-1).max(axis=2) > 0)
    def render_strip(v: WindowView, t: int, tag: str,
                     collect_stats: bool) -> list[np.ndarray]:
        base = v.frames[t].copy()
        if not v.ok:
            put_text(base, v.note[:60], (6, base.shape[0] - 12), 0.4,
                     (0, 255, 255))
        w2c = np.linalg.inv(v.c2w[t])
        # Hand-crop windows carry per-frame intrinsics: the crop box
        # moves with the hand, so K is adjusted frame by frame.
        k_frame = (v.k[int(t)] if isinstance(v.k, list) else v.k)
        dim = (base * 0.6).astype(np.uint8)
        for ov in object_overlays or []:
            alpha = float(ov.get("alpha", .72))
            base = draw_object_silhouette(
                base, ov["vertices"][t], ov["faces"], w2c, k_frame,
                ov["color"], alpha=alpha, label=ov.get("label"))
            dim = draw_object_silhouette(
                dim, ov["vertices"][t], ov["faces"], w2c, k_frame,
                ov["color"], alpha=alpha)
        meshes = [dict(vertices=vertices[t, side], faces=faces[name])
                  for side, name in ((0, "left"), (1, "right"))
                  if mesh_present[t, side]]
        contact_l = int(contact[t, 0].sum()) if present[t, 0] else 0
        contact_r = int(contact[t, 1].sum()) if present[t, 1] else 0
        if collect_stats:
            contact_counts.append(contact_l + contact_r)
        d = distance[t]
        finite = np.isfinite(d)
        d50 = float(np.median(d[finite])) if finite.any() \
            else float("nan")
        if collect_stats:
            d50s.append(d50)

        def color_hands(values_side, base_color, hot_mask=None,
                        hot_color=None):
            out = []
            for side in (0, 1):
                if not mesh_present[t, side]:
                    continue
                colors = np.tile(base_color, (778, 1))
                if hot_mask is not None:
                    colors[hot_mask[t, side]] = hot_color
                out.append(colors)
            return out

        # P1: INPUT + shaded hands, contact vertices red
        p1 = overlay.render(
            base, w2c, k_frame,
            [dict(mesh, colors=colors) for mesh, colors in zip(
                meshes, color_hands(d, skin, contact > 0, contact_rgb))],
            alpha=.80)
        for side in (0, 1):
            if present[t, side]:
                draw_contact_markers(p1, vertices[t, side], w2c, k_frame,
                                     contact[t, side] > 0)
        put_text(p1, f"INPUT + contact{tag}", (8, 20), 0.55)

        # P2: CONTACT — the hand mesh itself, lightly shaded so the photo
        # keeps showing, with the red contact patches on top.  The mesh
        # anchors WHERE on the hand the contact sits — patches alone are a
        # few sub-pixel triangles in wide exo views (arctic) and vanish in
        # rasterization.  P1 keeps the occlusion-marked input view.
        contact_patches = []
        for side, name in ((0, "left"), (1, "right")):
            if not present[t, side]:
                continue
            hot = contact[t, side] > 0
            if not hot.any():
                continue
            side_faces = np.asarray(faces[name])
            keep = hot[side_faces].any(axis=1)
            if keep.any():
                contact_patches.append(dict(
                    vertices=vertices[t, side], faces=side_faces[keep],
                    colors=np.tile(contact_rgb, (len(vertices[t, side]), 1))))
        p2 = overlay.render(
            base, w2c, k_frame,
            [dict(mesh, colors=np.tile(skin, (778, 1))) for mesh in meshes],
            alpha=.40)
        if contact_patches:
            p2 = overlay.render(p2, w2c, k_frame, contact_patches, alpha=1.0)
        put_text(p2, "CONTACT mesh + red patches", (8, 20), 0.55)
        put_text(p2, f"L {contact_l} | R {contact_r} contact vtx",
                 (8, p2.shape[0] - 12), 0.45, (180, 255, 180))

        # P3: PROXIMITY (near red -> far blue).  Colors follow mesh_present
        # (the meshes list's filter): absent-label hands carry all-clip
        # distance, i.e. a uniform far-blue — filtering on ``present`` here
        # would shift the remaining hand's colors onto the wrong mesh.
        prox_colors = [proximity_color(np.where(finite, d, 0.25)[side])
                       [..., ::-1].copy()          # BGR -> RGB
                       for side in (0, 1) if mesh_present[t, side]]
        p3 = overlay.render(
            base, w2c, k_frame,
            [dict(mesh, colors=colors) for mesh, colors in zip(
                meshes, prox_colors)],
            alpha=.98, background=dim)
        put_text(p3, "PROXIMITY signed dist", (8, 20), 0.55)
        p3_rows = []
        for side, tag in ((0, "L"), (1, "R")):
            if not present[t, side]:
                p3_rows.append(f"{tag} absent")
                continue
            side_vals = d[side]
            side_vals = side_vals[np.isfinite(side_vals)]
            if side_vals.size == 0:
                p3_rows.append(f"{tag} --/--/-- mm")
            else:
                p3_rows.append(
                    f"{tag} {np.median(side_vals) * 1000:.0f}"
                    f"/{side_vals.max() * 1000:.0f}"
                    f"/{side_vals.min() * 1000:.0f} mm")
        for i, row in enumerate(p3_rows):
            put_text(p3, row,
                     (8, p3.shape[0] - 12 - 14 * (len(p3_rows) - 1 - i)),
                     0.45, (180, 255, 180))

        # P4: FORCE (force-colored meshes + arrows)
        if force_vectors is not None:
            fv = force_vectors[t]
            magnitude = np.linalg.norm(fv, axis=-1)
            fmax = float(magnitude.max()) if magnitude.size else 0.0
            if collect_stats:
                fmaxs.append(fmax)
            force_colors = [force_color(magnitude[side],
                                        force_scale)[0][..., ::-1].copy()
                            for side in (0, 1) if mesh_present[t, side]]
            p4 = overlay.render(
                base, w2c, k_frame,
                [dict(mesh, colors=colors) for mesh, colors in zip(
                    meshes, force_colors)],
                alpha=.98, background=dim)
            for side, name in ((0, "left"), (1, "right")):
                if not present[t, side]:
                    continue
                verts = vertices[t, side]
                ids = np.flatnonzero(magnitude[side] > 1e-4)
                ids = ids[np.argsort(magnitude[side, ids])][-80:]
                for vertex in ids:
                    ratio = min(float(magnitude[side, vertex])
                               / force_scale, 1.0)
                    draw_arrow_world(
                        p4, verts[vertex], fv[side, vertex], w2c, k_frame,
                        3 + 40 * np.sqrt(ratio), turbo_color(ratio),
                        1 + int(2 * np.sqrt(ratio)))
            if wrench is not None and object_position is not None:
                net = wrench[t, :3]
                ratio = min(float(np.linalg.norm(net))
                           / max(force_scale, 1e-9), 1.0)
                draw_arrow_world(p4, object_position[t], net, w2c,
                                 k_frame, 36 + 30 * ratio,
                                 target_force_color(.25 + .5 * ratio),
                                 4, "F")
            draw_force_legend(p4, force_scale)
            put_text(p4, "FORCE |v| N/kg (turbo arrows)", (8, 20), 0.55)
            put_text(p4, f"max {fmax:.2f} (scale {force_scale:.2f})",
                     (8, p4.shape[0] - 12), 0.45, (180, 255, 180))
        else:
            p4 = dim.copy()
            put_text(p4, "FORCE n/a", (8, 20), 0.55, (120, 120, 120))
        return [p1, p2, p3, p4]

    try:
        for t in range(frames_count):
            strips = [compose_row(
                render_strip(view, t, "", True), gap=GAP, header=header,
                footer=extras[t] if t < len(extras) else "")]
            for tag, extra in extra_views or []:
                strips.append(compose_row(
                    render_strip(extra, t, tag, False), gap=GAP, header="",
                    footer=f"input: {tag.strip()}"))
            if len(strips) > 1:
                width = max(strip.shape[1] for strip in strips)
                strips = [strip if strip.shape[1] == width else
                          cv2.copyMakeBorder(
                              strip, 0, 0, 0, width - strip.shape[1],
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
                          for strip in strips]
                rows_out.append(np.vstack(strips))
            else:
                rows_out.append(strips[0])
    finally:
        overlay.close()
    encode_frames(video_path, np.stack(rows_out), fps=10.0, crf=20)
    sheet = contact_sheet([rows_out[index] for index in SHEET_FRAMES
                           if index < len(rows_out)],
                          f"{header}  |  P1 skin mesh, red=contact  "
                          "P2 light mesh + red contact patches  "
                          "P3 near-red->far-blue  P4 turbo force")
    sheet_path = out_dir / "sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    info = {"video": str(video_path), "sheet": str(sheet_path),
            "contact_vtx_sum": contact_counts,
            "distance_median_mm": [None if np.isnan(x) else round(x * 1000, 1)
                                    for x in d50s],
            "force_max": [round(x, 4) for x in fmaxs]}
    return sheet_path, info


class _Reader:
    def __init__(self, root: Path):
        self._root = root
        self._reader = None
        self._ids: dict[Path, int] = {}

    def video_id(self, path: Path) -> int:
        path = Path(path)
        if path not in self._ids:
            self._ids[path] = len(self._ids)
            self._reader = None
        return self._ids[path]

    def read_frames(self, video_id: int, frames):
        if self._reader is None:
            self._reader = VideoReader(
                dict((i, p) for p, i in self._ids.items()), cache_size=48,
                backend="opencv", image_mode="rgb")
        return self._reader.read_frames(video_id, frames)


def _v4_object_states(root: Path, index_row: dict):
    """Per-object position/rotation tracks from the installed v4 force
    stream — the multi-object pose detail the v5 tree deliberately omits
    (transitional: review-only source until per-object detail exists)."""
    path = root / "meta/observations.parquet"
    if not path.is_file():
        return None
    observations = pq.read_table(path, filters=[
        ("episode_index", "=", int(index_row["episode_index"])),
    ]).to_pylist()
    row = next((o for o in observations
                if int(o["episode_index"]) == int(index_row["episode_index"])
                and bool(o.get("force_available"))), None)
    if row is None:
        return None
    table = pq.ParquetFile(root / str(row["force_path"])).read_row_group(
        int(row["force_row_group_index"])).slice(
        int(row["force_row_offset"]), int(row["force_num_rows"]))
    ids = [str(v) for v in table.column("object_instance_ids")[0].as_py()]
    positions = np.stack([
        np.frombuffer(value, np.float64).reshape(len(ids), 3)
        for value in table.column("object_state_position_world").to_pylist()])
    rotations = np.stack([
        np.frombuffer(value, np.float64).reshape(len(ids), 3, 3)
        for value in table.column("object_state_rotation_world").to_pylist()])
    return positions, rotations


def preview_object_overlays(root: Path, index_row: dict,
                            force: dict, rows: np.ndarray,
                            target_idx: int) -> list[dict] | None:
    """Target-object overlay from the label state plus dimmed non-target
    overlays from the v4 per-object state.

    The target is drawn last at full alpha; non-targets at 0.28 so a hand
    grasping a non-target object is interpretable under union-force
    semantics.  Articulated episodes and store-missing geometry contribute
    no overlay (decoration, never fatal)."""
    object_ids = [str(v) for v in (index_row.get("object_ids") or [])]
    if not object_ids or target_idx >= len(object_ids):
        return None
    if str(index_row.get("topology")) == "articulated":
        return None
    try:
        meshes = load_object_meshes(root)
        geometry_of = instance_geometry_ids(root)

        def overlay_for(index: int, rotation, position, alpha: float,
                        suffix: str = "") -> dict | None:
            geometry = geometry_of.get(object_ids[index])
            if geometry is None or geometry not in meshes:
                return None
            verts, faces = meshes[geometry]
            com_body = np.asarray(index_row["object_com_body"][index], np.float64)
            posed = np.stack([
                pose_object_vertices(verts, com_body, rotation[t], position[t])
                for t in range(len(rows))])
            return {"vertices": posed.astype(np.float32), "faces": faces,
                    "color": object_color(object_ids[index]),
                    "alpha": alpha,
                    "label": object_ids[index] + suffix}

        label_rotation = np.asarray(
            force["object_rotation"], np.float32).reshape(-1, 3, 3)[rows]
        label_position = np.asarray(force["object_position"], np.float32)[rows]
        overlays: list[dict] = []
        if len(object_ids) > 1:
            states = _v4_object_states(root, index_row)
            if states is not None:
                positions, rotations = states
                for index in range(len(object_ids)):
                    if index == target_idx:
                        continue
                    overlay = overlay_for(
                        index, rotations[rows, index],
                        positions[rows, index], .28, " (non-target)")
                    if overlay is not None:
                        overlays.append(overlay)
        target = overlay_for(target_idx, label_rotation, label_position, .72)
        if target is not None:
            overlays.append(target)
        return overlays or None
    except Exception:  # noqa: BLE001 — overlay is decoration, never fatal
        return None


def hand_payloads(labels_root: Path, root: Path, labels: dict,
                  rows: np.ndarray,
                  target_idx: int = 0) -> tuple[np.ndarray, np.ndarray, float,
                                                np.ndarray, np.ndarray,
                                                list | None]:
    """Per-window force data (or all-None for contact-only episodes)."""
    if not labels["force_groups"]:
        return None, None, 1.0, None, None, None
    force = read_episode_columns(labels_root, labels["force_groups"],
                                 REQUIRED_FORCE_COLUMNS)
    force_vectors = force["vertex_force_world"].reshape(
        -1, 2, 778, 3)[rows]
    force_scale = float(np.percentile(
        np.linalg.norm(force_vectors, axis=-1), 99)) or 1.0
    wrench = force["contact_wrench"][rows]
    object_position = force["object_position"][rows]
    overlay = preview_object_overlays(root, labels, force, rows, target_idx)
    return (force_vectors, wrench, force_scale, object_position,
            np.linalg.norm(wrench[:, :3], axis=1), overlay)


from core.data.camera import load_cameras
from core.data.processed_compat import observations
from core.tactile_labels.decode import load_label_index
from core.tactile_labels.handcrop import HandcropStreamIndex, raw_mirror_size

class ReviewContext:
    """Per-dataset lookups the overlays need (intrinsics, view metadata)."""

    def __init__(self, name: str, processed_root: Path, labels_root: Path,
                 handcrop: bool = False, reader=None):
        self.reader = reader
        self.name = name
        self.root = processed_root / f"{name}_lerobot"
        self.labels_root = labels_root / f"{name}_lerobot"
        self.index = {int(r["episode_index"]): r
                      for r in load_label_index(self.labels_root)}
        self.cameras = load_cameras(self.root)
        # Shared crop-stream index — the SAME class the training reader
        # builds for video_source=handcrop, so canonical-frame -> crop-file
        # mapping and crop boxes can never drift between the two.
        self.handcrop = (HandcropStreamIndex(self.root) if handcrop
                         else None)
        # Crop boxes live in the raw hi-res mirror's pixel space; the
        # overlays must scale them into this obs row's (installed) space
        # before adjusting the intrinsics.  None => no mirror installed
        # and crop-K rendering is impossible (render raises).  Per-root
        # only as a fallback: frame_intrinsics resolves the mirror per
        # video_key at projection time (arctic mixes ego 2800x2000 and
        # exo 2800x2800 mirrors under one root).
        self.raw_size = (raw_mirror_size(self.root) if self.handcrop is not None
                         else None)
        # (episode, view) -> observations row (camera_id, width, height).
        self.views: dict[int, list[dict]] = {}

    def observation(self, episode: int, canonical: str) -> dict:
        if episode not in self.views:
            self.views[episode] = [
                row for row in observations(self.root, episode)
                if bool(row.get("hmr_available"))
            ]
        views = self.views[episode]
        if not views:
            raise ValueError(f"{self.root.name} episode {episode} has no HMR views")
        return next((v for v in views if str(v["view_id"]) == canonical),
                    views[0])
