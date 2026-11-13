"""Read native solver rows and sample clips from decodable source frames."""
from __future__ import annotations
import core.data.processed_compat as processed_contract

from dataclasses import dataclass, replace
from collections import OrderedDict
from itertools import groupby
import json
import math
import os
import random
import time
from pathlib import Path
import zlib
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from core.data.video import PreparedVideoReader as VideoReader
from core.data.resources import resolve_resource_path
from core.tactile_labels.assets import load_asset_names, object_display_name
from core.tactile_labels.handcrop import HandcropStreamIndex, frame_intrinsics
from core.tactile_labels.decode import (
    ByteBoundedEpisodeCache, load_label_index, read_episode_columns,
    read_optional_episode_columns,
    validate_labels_info,
)
from core.tactile_labels.static_physics import static_row_physics
from core.tactile_labels.trajectory_physics import solver_gravity_world
from core.tactile_labels.processed_training import episode_metadata, view_metadata
from core.tactile_labels.mano_tracks import solver_vertex_tracks
from core.tactile_labels.native_timeline import (
    read_native_timeline, target_spans, validate_native_timeline,
)
from core.tactile_labels.schema import (
    FLAG_CONTACT_VALID, FLAG_FORCE_VALID, FLAG_LEFT_PRESENT,
    FLAG_PROXIMITY_VALID, FLAG_RIGHT_PRESENT, MANO_VERTICES, PROXIMITY_CLIP_M,
    REQUIRED_FORCE_COLUMNS, REQUIRED_TACTILE_COLUMNS, TOPOLOGY_ARTICULATED,
    TOPOLOGY_RIGID, TOPOLOGY_STATIC,
)
from core.data.inspection import read_episode_rows
from core.data.camera import canonical_to_video_map, hmr_c2w, load_cameras
from core.geometry.projection import in_frame, project_camera
from core.physics.backends import Physics
from training.data.clip_sampling import (
    native_frame_origins, source_clip_plan,
)
from training.data.preprocessing import letterbox_square
from training.data.rigid_scene import attach_single_body, attach_multi_body
from core.geometry.force_frames import VERTEX_LOCAL_FRAME, axes_world_to_camera, canonical_vertex_local_axes, force_world_to_local, validate_vertex_force_frame, vertex_local_axes

SUBSTEPS_PER_TOKEN = 3
MIN_IN_FRAME_HAND_VERTICES = 8


def edge_pad(values: np.ndarray, left: int, right: int) -> np.ndarray:
    return np.concatenate((
        np.repeat(values[:1], left, axis=0),
        values,
        np.repeat(values[-1:], right, axis=0),
    ), axis=0)


def _tail_leftover(real_length: int, sequence_length: int, stride: int,
                   phase: int) -> int:
    """Frames past the last full grid window under this phase (0 = none)."""
    return ((real_length - sequence_length) % stride) - (int(phase) % stride)


def window_count(real_length: int, sequence_length: int, stride: int,
                 include_tail: bool, phase: int = 0) -> int:
    """Window count for one run; the tail depends on the phase.

    A tail window exists only when this phase leaves frames past the last
    full grid window uncovered.  When the shifted grid already reaches the
    run end, the tail would be a redundant clamp onto the final frame, so
    it is dropped.  The phase-0 default is the maximum count (tail always
    included when a remainder exists).
    """
    if real_length <= 0:
        return 0
    if real_length < sequence_length:
        return (sequence_length - 1) // stride + 1 if include_tail else 0
    count = (real_length - sequence_length) // stride + 1
    if include_tail and _tail_leftover(
            real_length, sequence_length, stride, phase) > 0:
        count += 1
    return count


def window_starts(real_length: int, sequence_length: int, stride: int,
                  phase: int, include_tail: bool) -> list[int]:
    count = window_count(real_length, sequence_length, stride, include_tail,
                         phase)
    if count <= 0:
        return []
    phase = int(phase) % stride
    if real_length < sequence_length:
        if stride >= sequence_length:
            # Only one window fits a short run; a phase-shifted start would
            # just drop head frames (degenerating to 1 real frame at the
            # largest phases), so anchor it to cover the whole run.
            return [real_length - sequence_length] if include_tail else []
        base = real_length - sequence_length
        return [base + phase + index * stride for index in range(count)]
    # Non-tail grid starts never pass real_length - sequence_length.  The
    # tail (emitted only when this phase leaves a remainder) covers exactly
    # those leftover frames; the old min(start, real_length - 1) clamp
    # instead materialized a redundant 1-real-frame window whenever the
    # shifted grid already reached the run end.
    return [phase + index * stride for index in range(count)]


@dataclass(frozen=True)
class LabelRunRecord:
    """One contiguous force-valid token run expressed for one camera view.

    ``topology="static"`` marks the independent-frame twin: every row is
    its own physical problem, so one record covers the whole episode and
    window ``start`` indexes ROWS (each window stacks one row's frame and
    labels ``sequence_length`` times).
    """

    episode_index: int
    view_id: str
    split: str
    video_path: Path
    video_segment_start: int
    source_frames: np.ndarray   # (real_length,) canonical frame per token
    token_start: int            # first global token id of the run
    object_idx: int             # run-wide target object (label_index object_ids)
    segment_index: int
    real_length: int
    topology: str = "rigid"
    # contact-only sources (topology none) supervise contact/proximity
    # only: force/wrench/rollout losses stay masked off for their windows.
    force_supervised: bool = True
    # Union-wrench episodes (hands also touch non-target objects, §5.5):
    # the aggregated wrench cannot drive the target body alone, so their
    # physics rollout stays off (per-episode flag; conservative per run).
    rollout_eligible: bool = True
    # Source rows are partitioned across an episode/view's admitted runs.
    # RGB coverage is independent of the force-valid token intervals.
    source_origin_start: int = 0
    source_origin_stop: int = 0
    solver_span: tuple[int, int] | None = None


def episode_training_role(labels: dict) -> str:
    """Single admission policy for one label_index row.

    ``force``       rigid/static with force streams (full supervision)
    ``force_no_rollout`` articulated: contact/proximity/force, no rollout
    ``contact_only`` topology none: contact/proximity supervision only
    ``skip``        everything else (unsupported or inconsistent)
    """
    topology = str(labels["topology"])
    force = bool(labels["has_force"])
    if topology in (TOPOLOGY_RIGID, TOPOLOGY_STATIC):
        return "force" if force else "skip"
    if topology == TOPOLOGY_ARTICULATED:
        return "force_no_rollout" if force else "skip"
    if topology == "none":
        return "contact_only" if not force else "skip"
    return "skip"


class LabelStreamDataset(Dataset):
    """Sliding-window reader over one labels-v5 dataset root."""

    # Filled by DataModule._build_child from published dataset statistics.
    label_stats: dict[str, Any]

    def __init__(self, cfg: dict[str, Any], split: str, max_episodes=None):
        started = time.perf_counter()
        self.init_timings = {}
        self.cfg = cfg
        self.training_data_fast_path = bool(cfg.get("training_data_fast_path", False))
        self.split = str(split)
        self.sequence_length = int(cfg["sequence_length"])
        self.window_stride = int(cfg.get(
            "window_stride", max(1, self.sequence_length // 2)))
        if not 0 < self.window_stride <= self.sequence_length:
            raise ValueError("window_stride must be in (0, sequence_length]")
        self.window_include_tail = bool(cfg.get("window_include_tail", True))
        self.window_mode = str(cfg.get(
            "train_window_mode" if self.split == "train" else "val_window_mode",
            "fixed")).strip().lower()
        if self.window_mode not in {"fixed", "phase_offset"}:
            raise ValueError(f"unsupported window mode {self.window_mode}")
        self.window_seed = int(cfg.get("window_sampling_seed", 42))
        native_root = cfg.get("native_timeline_root")
        self.native_timeline_root = Path(native_root).expanduser().resolve() if native_root else None
        self.source_clip_target_fps = float(cfg.get("target_fps", 10.0))
        self._source_origin_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        # Order-level randomness (train split only — val keeps the canonical
        # order everywhere).  Both are zero-cost: video groups stay contiguous
        # so the LRU handle cache sees the same open/seek pattern.
        self.shuffle_video_order = (
            self.split == "train"
            and bool(cfg.get("shuffle_video_order", True)))
        self.shuffle_window_order = (
            self.split == "train"
            and bool(cfg.get("shuffle_window_order", True)))
        self.photometric_augment = (
            self.split == "train"
            and bool(cfg.get("photometric_augment", True)))
        self.proximity_tau = float(cfg.get("proximity_tau", 10.0))
        self.vertex_force_frame = validate_vertex_force_frame(
            cfg.get("vertex_force_frame", "camera"))
        self.pad_label_mode = str(cfg.get("pad_label_mode", "repeat_edge"))
        self.supervise_hand_visibility = bool(cfg.get("supervise_hand_visibility", False))
        self.root = Path(cfg["root"]).expanduser().resolve()
        self.labels_root = Path(
            cfg.get("labels_root") or self.root).expanduser().resolve()
        self._visibility_cameras = load_cameras(self.root) if self.supervise_hand_visibility else {}
        reader = (processed_contract.reader_for(self.labels_root)
                  if processed_contract.is_processed(self.labels_root) else None)
        self._native_reader = (reader if reader is not None and
                               reader.has_modality("tactile.timeline") else None)
        label_contract = validate_labels_info(self.labels_root)
        self._standalone_native_revision = (label_contract.get("timeline_revision")
                                            if reader is None else None)
        self._sealed_native_timeline = (self._native_reader is not None
            and reader.manifest["contracts"].get("beingh", {}).get("labels", {})
            .get("timeline") == "solver_rows_v1")
        self.native_solver_timeline = (
            self.native_timeline_root is not None or self._native_reader is not None
            or self._standalone_native_revision is not None)
        self._native_revision = (reader.revision if reader is not None and self.native_solver_timeline
                                 else self._standalone_native_revision)
        if not self.native_solver_timeline:
            raise ValueError(f"{self.labels_root}: a native solver timeline is required")
        self.random_source_clip_origin = True
        if (
                self.source_clip_target_fps != 10.0 or not self.window_include_tail):
            raise ValueError("native clip origins require target_fps=10 and window_include_tail")
        # Per-dataset curation (both no-ops unless configured):
        # (a) object allowlist — keep only label runs whose target object's
        #     asset display name is listed in a JSON file keyed by dataset
        #     name ({"gigahands": ["flute_scan", ...]}); the names are the
        #     same ones the review visualizers print, via the shared join
        #     in core.tactile_labels.assets.  Unlisted datasets are
        #     unfiltered; runs with an unresolvable object id are dropped
        #     (only explicitly picked objects pass).
        # (b) contact_only_datasets — force labels exist but must not
        #     supervise: the dataset's episodes drop to the exact natively
        #     contact-only path (native source spans, zeroed
        #     force_valid, placeholder zero force streams) while the rest
        #     of a mixture keeps force supervision.
        self.dataset_name = str(cfg.get("dataset_name") or self.root.name)
        self._window_scoped_geometry = (
            self.training_data_fast_path and self.split == "train"
            and self.dataset_name.removesuffix("_lerobot") in {
                str(name).removesuffix("_lerobot")
                for name in cfg.get("window_scoped_geometry_datasets", ())
            })
        exclude_map = cfg.get("rollout_exclude_episodes") or {}
        self._rollout_excluded = {
            int(episode)
            for name in {self.dataset_name,
                         self.dataset_name.removesuffix("_lerobot")}
            for episode in (exclude_map.get(name) or [])
        }
        if self._rollout_excluded:
            print(f"[labels-v5] {self.dataset_name}: rollout-excluded episodes "
                  f"{sorted(self._rollout_excluded)} (force/contact retained)",
                  flush=True)
        self._asset_names: dict[str, str] = {}
        self._source_names: dict[str, str] = {}
        self._allowlist: set[str] | None = None
        allowlist_file = cfg.get("object_allowlist_file")
        if allowlist_file:
            mapping = json.loads(
                resolve_resource_path(allowlist_file).resolve().read_text())
            names = mapping.get(self.dataset_name)
            if names is not None:
                assets_path = processed_contract.object_path(self.root,'assets_path','objects/assets.parquet')
                if not assets_path.exists():
                    raise FileNotFoundError(
                        f"object_allowlist_file lists {self.dataset_name!r} "
                        f"but {assets_path} is missing")
                self._asset_names, self._source_names = load_asset_names(
                    assets_path)
                self._allowlist = set(map(str, names))
            elif int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))) == 0:
                # Normal for mixture children the JSON does not target —
                # but also the symptom of a mistyped JSON key, so say it.
                print(f"[labels-v5] {self.dataset_name}: no entry in "
                      f"{Path(allowlist_file).name} — runs unfiltered",
                      flush=True)
        self._force_masked = self.dataset_name in (
            cfg.get("contact_only_datasets") or [])
        # mow annotates exactly one hand: the other hand is UNLABELED, not
        # observed-contact-free.  Supervising it toward zeros would teach
        # "no contact anywhere" on a hand nobody annotated, so its rows
        # stay invalid (no loss).  Every other dataset keeps the legacy
        # zero-supervision of absent hands.  (Hand-crop windows override
        # this: the off-crop hand is explicitly zero-supervised there.)
        absent_mode = str(cfg.get(
            "absent_hand_supervision",
            "none" if self.root.name == "mow_lerobot" else "zero",
        )).strip().lower()
        if absent_mode not in {"zero", "none"}:
            raise ValueError(
                f"absent_hand_supervision must be zero|none, got "
                f"{absent_mode!r}")
        self._supervise_absent_hand = absent_mode == "zero"
        # rich/prox are contact-only sources and never enter force training;
        # hand-crop input is opt-in per dataset config (video_source:
        # handcrop) for whichever rigid dataset wants crop frames.
        processed = processed_contract.is_processed(self.root)
        fast_metadata = processed and bool(processed_contract.reader_for(self.root).manifest.get(
            'contracts', {}).get('unified_data'))
        episodes = (episode_metadata(self.root) if fast_metadata else
                    processed_contract.reader_for(self.root).metadata('episodes').to_pylist()
                    if processed else read_episode_rows(self.root))
        manifest_file = cfg.get("split_manifest_file")
        manifest_ids = None
        if manifest_file:
            from training.data.split_manifest import select_manifest_episodes
            manifest_ids = select_manifest_episodes(self.root, str(manifest_file), self.split)
        selected = [row for row in episodes
                    if (int(row["episode_index"]) in manifest_ids
                        if manifest_ids is not None else
                        self.split == "all" or str(row.get("split")) == self.split)]
        if manifest_ids is not None:
            selected = [dict(row, split=self.split) for row in selected]
        # EgoPHI official ARCTIC protocol reproduces its held-out-subject
        # split: keep only episodes of one subject.  subject_id carries the
        # full "s05|activity" sequence id, so match on the subject part
        # before the "|" (same convention as egophi_predictor.episode_subject).
        subject_filter = str(
            cfg.get("episode_subject_filter") or "").strip().lower()
        if subject_filter:
            selected = [row for row in selected
                        if str(row.get("subject_id") or "").split("|")[0]
                        .strip().lower() == subject_filter]
            if not selected:
                raise ValueError(
                    f"episode_subject_filter={subject_filter!r} matched no "
                    f"{self.split}-split episodes in {self.root}")
        skipped_topologies: dict[str, int] = {}
        include = cfg.get("include_episode_indices")
        if include is not None:
            include = set(map(int, include))
            selected = [row for row in selected
                        if int(row["episode_index"]) in include]
        if max_episodes is not None:
            selected = selected[:int(max_episodes)]

        indexed_observations = None
        if processed and not fast_metadata:
            selected = processed_contract.episode_rows(self.root,
                [int(ep['episode_index']) for ep in selected])
        self.init_timings['episode_views_s'] = time.perf_counter() - started
        index_started = time.perf_counter()
        label_rows = load_label_index(self.labels_root,
            [int(ep['episode_index']) for ep in selected], lazy_groups=fast_metadata)
        self.index_by_episode: dict[int, dict] = {
            int(row['episode_index']): row for row in label_rows}
        del label_rows
        self.init_timings['labels_s'] = time.perf_counter() - index_started
        if fast_metadata:
            view_started = time.perf_counter()
            visible = [ep for ep in selected
                       if int(ep['episode_index']) in self.index_by_episode]
            indexed_observations = view_metadata(self.root, visible, cfg.get('views'))
            self.init_timings['episode_views_s'] += time.perf_counter() - view_started

        video_source = str(cfg.get("video_source", "full")).strip().lower()
        if video_source not in {"full", "handcrop"}:
            raise ValueError(f"unsupported video_source {video_source!r}")
        self._handcrop = None
        # The installed crop index also records which hand identities are
        # actually annotated for an episode/view.  Some legacy label streams
        # (notably single-hand HOCAP episodes) force both presence bits on;
        # intersecting with this identity inventory restores the native
        # evaluable-hand set for full-view and crop-view methods alike.
        try:
            self._eval_handcrop = HandcropStreamIndex(self.root,
                episode_indices=[int(ep["episode_index"]) for ep in selected],
                inventory_only=fast_metadata and video_source == 'full')
        except FileNotFoundError:
            self._eval_handcrop = None
        # Eval-only protocol for single-hand crop baselines: expose one
        # image stream per hand while leaving the historical primary stream
        # and all training supervision untouched.
        self._handcrop_both_hands = bool(cfg.get("handcrop_both_hands", False))
        if video_source == "handcrop":
            if self._eval_handcrop is None:
                raise ValueError(
                    f"video_source=handcrop but {self.root} has no "
                    f"hand-crop stream")
            self._handcrop = self._eval_handcrop

        # Column-projected observations read: init touches only the locator
        # columns (the full 38-column pylist conversion was the biggest
        # fixed cost on the multiview datasets).
        obs_path = self.root / "meta/observations.parquet"
        if processed and indexed_observations is None:
            if max_episodes is None:
                indexed_observations = processed_contract.observations(self.root)
            else:
                indexed_observations = [obs for ep in selected
                    for obs in processed_contract.observations(self.root, int(ep['episode_index']))]
        obs_schema = (set(indexed_observations[0]) if indexed_observations else set()
                      if indexed_observations is not None else
                      set(pq.ParquetFile(obs_path).schema_arrow.names))
        obs_columns = ("episode_index", "view_id", "video_key",
                       "hmr_available", "hmr_path", "hmr_row_group_index",
                       "hmr_row_offset", "hmr_num_rows", "hmr_segment_count",
                       "num_frames", "video_num_frames", "camera_id", "width", "height")
        observations = indexed_observations if indexed_observations is not None else pq.read_table(
            obs_path,
            columns=[c for c in obs_columns if c in obs_schema]).to_pylist()
        self._view_locators: dict[int, list[dict]] = {}
        for row in observations:
            if bool(row.get("hmr_available")):
                self._view_locators.setdefault(
                    int(row["episode_index"]), []).append(row)
        requested_views = (
            set(map(str, cfg["views"])) if cfg.get("views") is not None else None)
        self._dataset_fps = float(processed_contract.read_info(self.root).get('fps',30.0))

        self.records: list[LabelRunRecord] = []
        self._video_paths_by_relative: dict[str, Path] = {}
        self._video_map_cache: OrderedDict[tuple[int, str], tuple[np.ndarray, np.ndarray] | None] = OrderedDict()
        self._video_map_cache_max = 4096
        total_runs = kept_runs = 0
        for episode in selected:
            episode_index = int(episode["episode_index"])
            labels = self.index_by_episode.get(episode_index)
            if labels is None or int(labels["n_rows"]) == 0:
                continue
            topology = str(labels["topology"])
            role = episode_training_role(labels)
            if role == "skip":
                # Mixed datasets (hoi4d) keep their eligible episodes; a
                # pure-unsupported dataset ends with no records below.
                skipped_topologies[topology] = \
                    skipped_topologies.get(topology, 0) + 1
                continue
            force_episode = (role != "contact_only") and not self._force_masked
            rollout_ok = (role == "force") and not self._force_masked
            if force_episode and topology == TOPOLOGY_RIGID and len(labels['object_ids']) > 1:
                if episode_index in self._rollout_excluded:
                    # Explicit curation: retain force/contact supervision but
                    # keep this episode out of multi-body rollout.
                    rollout_ok = False
                else:
                    from core.tactile_labels.rigid_reference_store import reference_status
                    metadata = reference_status(cfg.get('rigid_body_reference_root'), self.root, labels)
                    reference_ready = metadata is not None and metadata.get('status') == 'ready'
                    if cfg.get('require_rigid_body_references', False) and not reference_ready:
                        raise ValueError(f'Prepare offline rigid references first: '
                                         f'{self.root.name}/{episode_index}: {metadata}')
                    rollout_ok = rollout_ok and reference_ready
            views = sorted(
                (view for view in self._view_locators.get(episode_index, [])
                 if requested_views is None
                 or str(view["view_id"]) in requested_views),
                key=lambda view: str(view["view_id"]))
            if not views:
                continue
            if topology == TOPOLOGY_STATIC:
                # Independent frames: no tokens/runs — one record per view,
                # real_length counts ROWS; each window stacks a single row.
                total_runs += 1
                if not self._object_allowed(labels, 0):
                    continue
                kept_runs += 1
                source_frames = np.arange(int(labels["n_rows"]), dtype=np.int64)
                if self.native_solver_timeline:
                    positions = self._native_episode_timeline(labels)['source_position'].to_numpy()
                    if not np.isfinite(positions).all() or not np.array_equal(positions, np.rint(positions)):
                        raise ValueError(f"episode {episode_index}: invalid static source-frame mapping")
                    source_frames = positions.astype(np.int64)
                for view in views:
                    if self.native_solver_timeline and (
                            source_frames.min(initial=0) < 0
                            or source_frames.max(initial=-1) >= int(view["num_frames"])):
                        raise ValueError(f"episode {episode_index}: static source frame outside {view['view_id']}")
                    self.records.append(LabelRunRecord(
                        episode_index=episode_index,
                        view_id=str(view["view_id"]),
                        split=str(episode.get("split", "")),
                        video_path=self._view_video_path(episode, view),
                        video_segment_start=self._view_segment_start(
                            episode, view),
                        source_frames=source_frames,
                        token_start=0,
                        object_idx=0,
                        segment_index=0,
                        real_length=int(labels["n_rows"]),
                        topology=TOPOLOGY_STATIC,
                        force_supervised=True,
                        rollout_eligible=(
                            not bool(labels.get("has_non_target_force"))),
                    ))
                continue
            total, kept = self._append_native_records(episode, labels, views, force_episode, rollout_ok)
            total_runs += total
            kept_runs += kept
        if skipped_topologies:
            print(f"[labels-v5] {self.root.name}: skipped episodes by "
                  f"topology {skipped_topologies}", flush=True)
        if self._allowlist is not None or self._force_masked:
            message = (f"[labels-v5] {self.dataset_name}: object allowlist "
                       f"kept {kept_runs}/{total_runs} runs")
            if self._force_masked:
                message += "; force supervision masked (contact-only dataset)"
            print(message, flush=True)
        if not self.records:
            raise ValueError(f"no {self.split!r} windows under {self.root}")

        self._labels_cache = ByteBoundedEpisodeCache(
            self._decode_episode,
            int(cfg.get("episode_cache_bytes", 512 << 20)))
        self._c2w_cache = ByteBoundedEpisodeCache(
            self._decode_view_c2w, 64 << 20)
        self._last_scoped_mano = None
        # canonical→video frame maps for interval-sampled videos
        # (hograspnet): tiny recency LRU — sequential access keeps hits high
        # while bounding memory on the 200k-record datasets.
        self._video_reader: VideoReader | None = None
        self._video_ids: dict[Path, int] = {}
        self.init_timings['total_s'] = time.perf_counter() - started
        if int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))) == 0:
            print(f"[labels-v5 init] {self.dataset_name} split={self.split} "
                  f"episodes={len(selected)} records={len(self.records)} "
                  f"total={self.init_timings['total_s']:.2f}s "
                  f"episode_views={self.init_timings['episode_views_s']:.2f}s "
                  f"labels={self.init_timings['labels_s']:.2f}s", flush=True)

    # ------------------------------------------------------------------ setup

    def _native_episode_timeline(self, labels):
        revision = self._native_revision
        assert revision is not None
        eid = int(labels["episode_index"])
        if self._native_reader is not None:
            return validate_native_timeline(
                self._native_reader.read_episode("tactile.timeline", eid, "shared"), labels, revision)
        if self._standalone_native_revision is not None:
            return read_native_timeline(
                self.labels_root / "timeline_data" / f"episode_{eid:06d}.parquet", labels, revision)
        root = self.native_timeline_root
        assert root is not None
        path = root / self.labels_root.name / revision / f"episode_{eid:06d}.parquet"
        return read_native_timeline(path, labels, revision)

    def _append_native_records(self, episode, labels, views, force_supervised, rollout_ok):
        eid = int(labels["episode_index"])
        single_target = labels["has_force"] and len(labels["object_ids"]) == 1
        if self._allowlist is None and (not labels["has_force"]
                                        or (self._sealed_native_timeline and single_target)):
            # These episodes have one span. Load source positions on first use.
            spans = [(0, int(labels["n_rows"]), 0 if single_target else -1, 0, 0)]
        else:
            spans = list(target_spans(self._native_episode_timeline(labels)))
        kept = 0
        fps = float(labels["source_fps"])
        for segment, (begin, end, target, source_start, source_stop) in enumerate(spans):
            if not self._object_allowed(labels, target):
                continue
            kept += 1
            for view in views:
                view_id = str(view["view_id"])
                frame_count = int(view["num_frames"])
                if len(spans) == 1 and self._allowlist is None:
                    first, stop = 0, frame_count
                else:
                    first, stop = max(0, source_start), min(source_stop, frame_count)
                if first >= stop:
                    continue
                # A full-view span owns every decoded frame. Only a target
                # subspan needs the sparse map to test that it contains one.
                if (first != 0 or stop != frame_count) and (
                        int(view.get("video_num_frames") or 0) != frame_count):
                    mapping = self._video_frame_map(eid, view_id)
                    if mapping is not None and not len(native_frame_origins(first, stop, mapping[1])):
                        continue
                length = max(1, math.ceil((stop - first) * self.source_clip_target_fps / fps))
                self.records.append(LabelRunRecord(
                    episode_index=eid, view_id=view_id, split=str(episode.get("split", "")),
                    video_path=self._view_video_path(episode, view),
                    video_segment_start=self._view_segment_start(episode, view),
                    source_frames=np.asarray([first], np.int64), token_start=0,
                    object_idx=max(target, 0), segment_index=segment, real_length=length,
                    topology=str(labels["topology"]), force_supervised=force_supervised,
                    rollout_eligible=(rollout_ok and (len(labels["object_ids"]) > 1
                                      or not bool(labels.get("has_non_target_force")))),
                    source_origin_start=first, source_origin_stop=stop,
                    solver_span=(begin, end),
                ))
        return len(spans), kept

    def _object_allowed(self, labels: dict, object_idx: int) -> bool:
        """Allowlist gate for one run's target object (open when unset)."""
        if self._allowlist is None:
            return True
        ids = labels.get("object_ids") or []
        if not 0 <= int(object_idx) < len(ids):
            return False
        name = object_display_name(
            str(ids[int(object_idx)]), self._asset_names, self._source_names)
        return name in self._allowlist

    def _decode_episode(self, key: tuple) -> dict[str, np.ndarray]:
        episode_index = int(key[0])
        labels = self.index_by_episode[episode_index]
        tactile = read_episode_columns(
            self.labels_root, labels["tactile_groups"], REQUIRED_TACTILE_COLUMNS)
        rows = tactile["flags"].shape[0]
        if int(labels["n_rows"]) != rows:
            raise ValueError(
                f"episode {episode_index}: tactile row mismatch")
        skip_force_geometry = self.training_data_fast_path and (
            self._force_masked or episode_training_role(labels) == "contact_only"
        ) and str(labels["topology"]) != TOPOLOGY_STATIC
        if not skip_force_geometry and labels["force_groups"]:
            force = read_episode_columns(
                self.labels_root, labels["force_groups"],
                REQUIRED_FORCE_COLUMNS)
            if force["flags"].shape[0] != rows:
                raise ValueError(
                    f"episode {episode_index}: force row mismatch")
        else:
            # Contact-only sources (topology none): placeholder zero force
            # streams — every force-side loss is masked off per sample.
            zeros_3 = np.zeros((rows, 2, MANO_VERTICES, 3), np.float32)
            force = {
                "flags": np.zeros(rows, bool),
                "vertex_force_world": zeros_3,
                "contact_wrench": np.zeros((rows, 6), np.float32),
                "environment_wrench": np.zeros((rows, 6), np.float32),
                "object_position": np.zeros((rows, 3), np.float32),
                "object_rotation": np.repeat(
                    np.eye(3, dtype=np.float32)[None], rows, axis=0),
                "object_velocity": np.zeros((rows, 6), np.float32),
            }
        defer_levers = (self._window_scoped_geometry and not skip_force_geometry
                        and str(labels["topology"]) == TOPOLOGY_RIGID
                        and rows >= 8 * self.sequence_length * SUBSTEPS_PER_TOKEN)
        if defer_levers:
            vertices = None
        elif skip_force_geometry:
            # These values are not supervised or scored. Keep finite,
            # shape-compatible placeholders so the force head remains in the
            # DDP graph without reading force payloads or running MANO.
            vertices = np.zeros((rows, 2, MANO_VERTICES, 3), np.float32)
        else:
            tracks = solver_vertex_tracks(
                self.root, episode_index, str(labels["canonical_view"]),
                source_positions=tactile["source_position"])
            vertices = np.stack(
                [tracks["left"][0], tracks["right"][0]], axis=1)
        present = np.stack((
            (tactile["flags"] & FLAG_LEFT_PRESENT) > 0,
            (tactile["flags"] & FLAG_RIGHT_PRESENT) > 0), axis=1)
        # Solver-failed rows can carry non-finite object positions (obman
        # preview labels: 7% of rows).  Their FORCE_VALID flag is cleared, but
        # the lever feeds the PREDICTION-side wrench aggregation
        # (cross(lever, force)), which no loss gate can sanitize: NaN * 0 =
        # NaN survives the masked sum and crashes the finiteness guard.  Such
        # rows are geometry-absent; zero the pose so every derived target
        # stays finite.
        finite_position = np.isfinite(force["object_position"]).all(axis=-1)
        if not bool(finite_position.all()):
            force["object_position"] = np.where(
                finite_position[:, None], force["object_position"],
                np.zeros(3, force["object_position"].dtype))
        lever_fields = (
            {"lever_source_positions": tactile["source_position"], "lever_position_valid": finite_position}
            if vertices is None else
            {"lever": self._vertex_levers(vertices, force["object_position"], finite_position)})
        static_parameters = {}
        if str(labels["topology"]) == TOPOLOGY_STATIC and labels["has_force"]:
            optional = read_optional_episode_columns(
                self.labels_root, labels["force_groups"],
                ("static_gravity_world", "static_object_index"))
            object_index = optional.get("static_object_index")
            if self._sealed_native_timeline:
                timeline = self._native_episode_timeline(labels)
                native_slots = timeline["target_object"].fill_null(-1).to_numpy()
                if object_index is not None and not np.array_equal(object_index, native_slots):
                    raise ValueError(f"episode {episode_index}: static target slots disagree")
                object_index = native_slots
            slots, gravity = static_row_physics(
                self.root, labels, tactile["source_position"],
                (force["flags"] & FLAG_FORCE_VALID) > 0,
                force["contact_wrench"],
                object_index=object_index,
                gravity_world=optional.get("static_gravity_world"))
            static_parameters = {"static_object_index": slots,
                                 "static_gravity_world": gravity}
        gravity_world = (solver_gravity_world(
            self.root, episode_index, tactile["source_position"], labels["gravity_world"])
            if str(labels["topology"]) != TOPOLOGY_STATIC and not skip_force_geometry
            else np.broadcast_to(np.asarray(labels["gravity_world"], np.float64), (rows, 3)))
        body_parameters = {}
        if (str(labels['topology']) == TOPOLOGY_RIGID and labels['has_force']
                and not self._force_masked and len(labels['object_ids']) > 1):
            if episode_index in self._rollout_excluded:
                body_parameters = {
                    'body_ready': np.asarray(False),
                    'body_recovery_report': np.asarray(
                        f'excluded from rollout by rollout_exclude_episodes: '
                        f'{self.root.name}/{episode_index}'),
                }
            else:
                from core.tactile_labels.rigid_reference_store import read_reference
                recovered, metadata = read_reference(
                    self.cfg.get('rigid_body_reference_root'), self.root, labels)
                if recovered is None:
                    if self.cfg.get('require_rigid_body_references', False):
                        raise ValueError(f'Missing/failed offline rigid references: '
                                         f'{self.root.name}/{episode_index}: {metadata}')
                    body_parameters = {'body_ready': np.asarray(False)}
                else:
                    body_parameters = {f'body_{key}': value for key, value in recovered.items()}
                    body_parameters['body_ready'] = np.asarray(True)
                    body_parameters['body_recovery_report'] = np.asarray(json.dumps(metadata['report']))
        return {
            "force_geometry_available": np.asarray(not skip_force_geometry),
            **static_parameters,
            **body_parameters,
            **lever_fields,
            "source_positions": tactile["source_position"],
            "contact": tactile["contact"].reshape(rows, 2, MANO_VERTICES),
            "signed_distance": tactile["signed_distance"].astype(
                np.float32).reshape(rows, 2, MANO_VERTICES),
            "present": present,
            "contact_valid": (tactile["flags"] & FLAG_CONTACT_VALID) > 0,
            "proximity_valid": (tactile["flags"] & FLAG_PROXIMITY_VALID) > 0,
            "force": force["vertex_force_world"].reshape(
                rows, 2, MANO_VERTICES, 3),
            "force_valid": (force["flags"] & FLAG_FORCE_VALID) > 0,
            "contact_wrench": force["contact_wrench"],
            "environment_wrench": force["environment_wrench"],
            "hand_wrench": (force["contact_wrench"]
                            - force["environment_wrench"]),
            "position": force["object_position"],
            "rotation": force["object_rotation"].reshape(rows, 3, 3),
            "velocity": force["object_velocity"],
            "gravity_world": gravity_world,
            "episode_hand_present": present.any(axis=0),
        }

    @staticmethod
    def _vertex_levers(vertices, positions, finite_position):
        lever = vertices.astype(np.float64) - positions[:, None, None, :]
        # Geometry-absent rows carry zero vertices, so subtracting the
        # object position must not create a lever on those rows.
        geometry_absent = ((np.linalg.norm(vertices, axis=-1).max(axis=-1) == 0)
                           | ~finite_position[:, None])
        lever[geometry_absent] = 0.0
        return lever.astype(np.float32)

    def _scoped_mano_source(self, episode_index, view_id):
        key = (episode_index, view_id)
        cached = self._last_scoped_mano
        if cached is None or cached[0] != key:
            observation = next(
                row for row in processed_contract.observations(self.root, episode_index)
                if str(row["view_id"]) == view_id)
            frame = processed_contract.observation_table(self.root, observation)
            cached = (key, observation, frame.to_pandas().sort_values("frame_index"))
            self._last_scoped_mano = cached
        return cached[1], cached[2]

    def _episode_levers(self, episode_index, episode, rows=None):
        if "lever" in episode:
            return episode["lever"] if rows is None else episode["lever"][rows]
        labels = self.index_by_episode[episode_index]
        view_id = str(labels["canonical_view"])
        observation, hmr_frame = self._scoped_mano_source(episode_index, view_id)
        tracks = solver_vertex_tracks(
            self.root, episode_index, view_id,
            source_positions=episode["lever_source_positions"], rows=rows,
            observation=observation, hmr_frame=hmr_frame)
        vertices = np.stack([tracks["left"][0], tracks["right"][0]], axis=1)
        selection = slice(None) if rows is None else rows
        return self._vertex_levers(vertices, episode["position"][selection],
                                   episode["lever_position_valid"][selection])

    def _decode_view_c2w(self, key: tuple) -> dict[str, np.ndarray]:
        episode_index, view_id = int(key[0]), str(key[1])
        for view in self._view_locators.get(episode_index, []):
            if str(view["view_id"]) == view_id:
                return {"c2w": hmr_c2w(self.root, view)}
        raise KeyError(f"no HMR view {view_id} for episode {episode_index}")

    def _video_frame_map(self, episode_index: int, view_id: str):
        """canonical→video positions for interval-sampled videos (None for
        dense videos — the common case pays one dict lookup)."""
        key = (int(episode_index), str(view_id))
        cached = self._video_map_cache.get(key)
        if cached is None and key not in self._video_map_cache:
            view_row = next(
                (view for view in self._view_locators.get(key[0], [])
                 if str(view["view_id"]) == key[1]), None)
            if view_row is None:
                mapping = None
            else:
                try:
                    mapping = canonical_to_video_map(self.root, view_row)
                except Exception:  # noqa: BLE001 — dense fallback
                    mapping = None
            self._video_map_cache[key] = mapping
            while len(self._video_map_cache) > self._video_map_cache_max:
                self._video_map_cache.popitem(last=False)
        else:
            self._video_map_cache.move_to_end(key)
        return self._video_map_cache.get(key)

    def _handcrop_input_side(self, record: "LabelRunRecord",
                             present: np.ndarray) -> int | None:
        """Which hand's crop backs this window (0=left, 1=right, None=full)."""
        if self._handcrop is None or not len(present):
            return None
        return self._handcrop.input_side(
            record.episode_index, record.view_id, present)

    def _hand_visibility_targets(
            self, record: LabelRunRecord, episode, geometry_rows: np.ndarray,
            source_frames: np.ndarray, camera_rows: np.ndarray, c2w: np.ndarray,
            crop_side: int | None, content_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project the two tracked hands onto the actual decoded image plane."""
        target = np.zeros((len(geometry_rows), 2), dtype=bool)
        valid = np.zeros_like(target)
        positions = episode["source_positions"]
        aligned = np.flatnonzero(np.abs(positions[geometry_rows] - camera_rows) <= 1e-4)
        if not len(aligned):
            return target, valid
        view = next(view for view in self._view_locators[record.episode_index]
                    if str(view["view_id"]) == record.view_id)
        camera = self._visibility_cameras.get(str(view.get("camera_id")))
        full_width = int(view.get("width") or 0)
        full_height = int(view.get("height") or 0)
        if camera is None or full_width <= 0 or full_height <= 0:
            return target, valid
        width, height = full_width, full_height
        cameras = [camera] * len(aligned)
        if crop_side is not None:
            assert self._handcrop is not None
            crop_frames = camera_rows if self._handcrop._processed is not None else source_frames
            height_out, width_out = content_hw
            cameras = frame_intrinsics(
                self._handcrop, record.episode_index, record.view_id,
                ("left", "right")[crop_side], crop_frames[aligned],
                np.array([[camera.fx, 0.0, camera.cx],
                          [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]]),
                width, height, (width_out, height_out), video_key=str(view["video_key"]))
            width, height = width_out, height_out

        # Use the same world-space hand identities as the force labels.
        canonical = str(self.index_by_episode[record.episode_index]["canonical_view"])
        observation, hmr_frame = self._scoped_mano_source(record.episode_index, canonical)
        rows = geometry_rows[aligned]
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        tracks = solver_vertex_tracks(
            self.root, record.episode_index, canonical, source_positions=positions,
            rows=unique_rows, observation=observation, hmr_frame=hmr_frame)
        native_present = episode["present"][rows]
        for offset, (frame, intrinsics) in enumerate(zip(aligned, cameras)):
            pose = c2w[frame]
            if (not np.isfinite([intrinsics.fx, intrinsics.fy,
                                 intrinsics.cx, intrinsics.cy]).all()
                    or min(intrinsics.fx, intrinsics.fy) <= 0
                    or not np.isfinite(pose).all()):
                continue
            for side, name in enumerate(("left", "right")):
                vertices, present = tracks[name]
                if not present[inverse[offset]]:
                    # An unlabeled hand or a missing track on a present hand is unknown.
                    valid[frame, side] = (self._supervise_absent_hand
                                          and not native_present[offset, side]
                                          and native_present[offset].any())
                    continue
                points = (vertices[inverse[offset]] - pose[:3, 3]) @ pose[:3, :3]
                if not np.isfinite(points).all():
                    continue
                uv, projectable = project_camera(points, intrinsics)
                if not projectable.any() and points[:, 2].max() > 0:
                    continue
                valid[frame, side] = True
                inside = in_frame(uv, projectable, width, height)
                if crop_side is not None:
                    # A crop can extend beyond the source image into padding.
                    full_uv, full_projectable = project_camera(points, camera)
                    inside &= in_frame(full_uv, full_projectable, full_width, full_height)
                # A few boundary vertices alone are too sensitive to fit noise.
                target[frame, side] = inside.sum() >= MIN_IN_FRAME_HAND_VERTICES
        return target, valid

    def _view_video_path(self, episode: dict, view: dict) -> Path:
        """videos/<video_key>/chunk-XXX/file-XXX.mp4 for one observation."""
        video_key = str(view["video_key"])
        relative = episode.get(f'videos/{video_key}/path')
        if not relative:
            chunk = int(episode[f"videos/{video_key}/chunk_index"])
            file_index = int(episode[f"videos/{video_key}/file_index"])
            relative = f"videos/{video_key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        if relative not in self._video_paths_by_relative:
            self._video_paths_by_relative[relative] = self.root / relative
        return self._video_paths_by_relative[relative]

    def _view_segment_start(self, episode: dict, view: dict) -> int:
        video_key = str(view["video_key"])
        return int(round(float(
            episode[f"videos/{video_key}/from_timestamp"]) * self._dataset_fps))

    def _row_targets(self, episode, rows: np.ndarray,
                      crop_side: int | None) -> tuple:
        """Contact/proximity targets + validity for solver rows.

        Single source for both the trajectory path (``rows`` = token start
        rows) and the static path (``rows`` = one row): gathers the tactile
        arrays, applies the hand-crop presence override, zeroes absent-hand
        targets, computes the exponential proximity encoding, and combines
        the per-row packed validity flags with per-hand presence (spec §5.3;
        single-hand datasets deviate via absent_hand_supervision=none).
        Returns (contact, proximity, contact_valid, proximity_valid,
        contact_native, contact_valid_native); callers that predate the
        native-scoring pair take ``[:4]``.
        """
        contact = episode["contact"][rows].astype(np.float32).copy()
        distance = episode["signed_distance"][rows].astype(np.float32)
        present, absent_active = self._crop_presence_override(
            episode["present"][rows], crop_side)
        missing = ~present
        distance[missing] = np.inf
        contact[missing] = 0.0
        # NATIVE scoring targets, captured BEFORE the crop override: the
        # off-crop zeroing below is a supervision contract ("not visible in
        # this crop"), not a statement about the scene.  Evaluating a
        # handcrop method against the zeroed targets would hand it free
        # credit for frames where the unseen hand is actually in contact
        # (2026-09-22 arctic: L-frame GT collapsed 85% -> 9%).  The harness
        # scores contact against these native targets for every method.
        # contact_valid is a per-ROW scalar flag (any-hand), so the native
        # targets stay per-hand as the labels wrote them; the row flag
        # gates the whole frame together.
        contact_native = contact.copy()
        contact_native[~episode["contact_valid"][rows]] = 0.0
        if crop_side is not None:
            # The crop never shows the off-crop hand: present-with-zeros
            # ("no contact in view", not "hand absent") per the
            # _crop_presence_override contract its docstring states.
            contact[:, 1 - crop_side, :] = 0.0
            distance[:, 1 - crop_side, :] = np.inf
        proximity = np.zeros_like(distance, np.float32)
        finite = np.isfinite(distance)
        proximity[finite] = np.exp(-self.proximity_tau * np.clip(
            distance[finite], 0.0, PROXIMITY_CLIP_M))
        contact_valid = ((episode["contact_valid"][rows][..., None] & present)
                         | absent_active)
        proximity_valid = ((episode["proximity_valid"][rows][..., None]
                            & present) | absent_active)
        # Native validity: the label's own per-row flag broadcast to both
        # hands, independent of any crop pipeline.  Paired with
        # contact_native for scoring.
        contact_valid_native = np.broadcast_to(
            episode["contact_valid"][rows][..., None, None],
            contact_native.shape[:2] + (1,)).copy()
        return (contact, proximity, contact_valid, proximity_valid,
                contact_native, contact_valid_native)

    def _crop_presence_override(
            self, present: np.ndarray,
            crop_side: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Hand-crop supervision rule, single definition.

        The off-crop hand is supervised as PRESENT-WITH-ZEROS: the callers
        zero its contact/proximity/force targets, and its validity flag is
        forced on (the crop simply does not show that hand — "no contact
        in view", not "hand absent").  Works for row-level (2,) and
        window-level (N, 2) presence arrays alike.
        """
        present = present.copy()
        if crop_side is not None:
            present[..., 1 - crop_side] = True
        absent_active = (~present if self._supervise_absent_hand
                         else np.zeros_like(present, bool))
        if crop_side is not None:
            absent_active[..., 1 - crop_side] = True
        return present, absent_active

    def _to_model_images(self, frames: np.ndarray) -> torch.Tensor:
        """Convert sample-owned uint8 HWC frames to CHW model inputs."""
        images = torch.from_numpy(frames).permute(0, 3, 1, 2)
        if not bool(self.cfg.get("uint8_images", False)):
            images = images.float().div_(255.0)
            mean = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1)
            std = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1)
            images = (images - mean) / std
        return images

    def _maybe_photometric(self, frames: np.ndarray,
                           content_hw: tuple[int, int]) -> np.ndarray:
        """Brightness/contrast/saturation jitter on letterboxed uint8 frames.

        Runs after the letterbox downscale (a few float32 passes at model
        resolution, absorbed by the dataloader workers); the black borders
        are restored afterwards so train inputs keep exactly the val border
        geometry.  One parameter draw per window keeps the temporal signal
        flicker-free.  The draw is deliberately NOT seeded: augmentation
        randomness must not couple to the deterministic sampling order.
        """
        if not self.photometric_augment or frames.size == 0:
            return frames
        rng = np.random.default_rng()
        out = frames.astype(np.float32)
        out *= float(rng.uniform(0.8, 1.2))
        gray_mean = float(out.mean())
        contrast = float(rng.uniform(0.8, 1.2))
        saturation = float(rng.uniform(0.85, 1.15))

        def adjust(rgb):
            rgb -= gray_mean
            rgb *= contrast
            rgb += gray_mean
            gray = rgb[..., 0] + rgb[..., 1]
            gray += rgb[..., 2]
            gray /= 3.0
            gray = gray[..., None]
            rgb -= gray
            rgb *= saturation
            rgb += gray
            np.rint(rgb, out=rgb)
            np.clip(rgb, 0.0, 255.0, out=rgb)
            return rgb.astype(np.uint8)

        size = int(out.shape[-2])
        height, width = int(content_hw[0]), int(content_hw[1])
        scale = min(size / width, size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        if out.ndim == 4:
            # The mean above includes the full clip. The remaining math is
            # pixel-local, so process one content rectangle at a time and
            # leave the letterbox borders black from the start.
            result = np.zeros(out.shape, np.uint8)
            content = (slice(None), slice(top, top + resized_height),
                       slice(left, left + resized_width))
            for frame, destination in zip(out[content], result[content]):
                destination[:] = adjust(frame)
            return result
        out = adjust(out)
        out[:, :top] = 0
        out[:, top + resized_height:] = 0
        out[:, :, :left] = 0
        out[:, :, left + resized_width:] = 0
        return out

    def _read_input_frames(self, record: "LabelRunRecord",
                           frames: np.ndarray,
                           crop_side: int | None) -> np.ndarray:
        """Input frames for one window.

        Hand-crop mode (video_source=handcrop): frames come from the given
        hand's crop stream — the same side the supervision override zeroed
        the other hand for (segment grouping lives in
        ``core.tactile_labels.handcrop``, shared with the review renderer).
        Falls back to the full view video when no crop segment covers the
        episode/view.
        """
        reader = self._ensure_video_reader()
        if crop_side is not None:
            # crop_side is None unless _handcrop_input_side already verified
            # the crop stream covers this episode/view.
            target = ("left", "right")[int(crop_side)]
            if self._handcrop._processed is not None:
                frames = self._camera_rows(record,np.asarray(frames,np.int64))
            frames = self._handcrop.read_frames(
                record.episode_index, record.view_id, target,
                np.asarray(frames, np.int64), reader,
                video_id=self._video_ids.__getitem__)
            return np.stack([np.asarray(f, np.uint8) for f in frames])
        return np.asarray(reader.read_frames(
            self._video_ids[record.video_path],
            self._video_positions(record, frames).tolist()), dtype=np.uint8)

    def _read_resized_frames(self, record, frames, crop_side):
        """Resize each distinct source image once, before clip augmentation."""
        keys = (self._video_positions(record, frames)
                if crop_side is None else np.asarray(frames, np.int64))
        _unique, first, inverse = np.unique(
            keys, return_index=True, return_inverse=True)
        decoded = self._read_input_frames(record, frames[first], crop_side)
        content_hw = tuple(np.asarray(decoded[0]).shape[:2])
        resized = np.stack([
            letterbox_square(frame, int(self.cfg["image_size"]))
            for frame in decoded])
        # Restore occurrences BEFORE augmentation: keep its random draws,
        # temporal shape and per-occurrence arithmetic unchanged.
        if len(inverse) == len(resized) and np.array_equal(inverse, np.arange(len(resized))):
            return resized, content_hw
        return resized[inverse], content_hw

    def _read_both_hand_frames(self, record: "LabelRunRecord",
                               frames: np.ndarray,
                               native_present: np.ndarray) -> np.ndarray | None:
        """Return [hand,time,H,W,C] crops for eval-only dual-crop baselines."""
        if not self._handcrop_both_hands or self._handcrop is None:
            return None
        reader = self._ensure_video_reader()
        streams = []
        required = np.asarray(native_present, bool).reshape(-1, 2).any(axis=0)
        for side, target in enumerate(("left", "right")):
            if not self._handcrop.available(record.episode_index,
                                            record.view_id, target):
                if bool(required[side]):
                    raise RuntimeError(
                        f"dual-crop eval requires {target} crop for a native-"
                        f"valid hand: dataset={self.root.name} episode="
                        f"{record.episode_index} view={record.view_id}")
                streams.append(np.zeros(
                    (len(frames), int(self.cfg["image_size"]),
                     int(self.cfg["image_size"]), 3), np.uint8))
                continue
            decoded = self._handcrop.read_frames(
                record.episode_index, record.view_id, target,
                self._camera_rows(record,np.asarray(frames,np.int64)) if self._handcrop._processed is not None else np.asarray(frames, np.int64), reader,
                video_id=self._video_ids.__getitem__)
            streams.append(np.stack([np.asarray(f, np.uint8) for f in decoded]))
        return np.stack(streams)

    def _native_evaluable_present(self, record: "LabelRunRecord",
                                  present: np.ndarray) -> np.ndarray:
        """Original presence intersected with the installed hand identities."""
        out = np.asarray(present, bool).copy()
        # Keep the annotated side: DexYCB sequences may contain either a
        # left or a right hand. A right-hand-only baseline convention must
        # not mask left-hand force/contact supervision in this shared reader.
        if self._eval_handcrop is None:
            return out
        available = np.asarray([
            self._eval_handcrop.available(record.episode_index, record.view_id,
                                          target)
            for target in ("left", "right")], bool)
        return out & available

    def _video_positions(self, record: "LabelRunRecord",
                         frames: np.ndarray) -> np.ndarray:
        """Video-file positions for the window's canonical frames."""
        mapping = self._video_frame_map(record.episode_index, record.view_id)
        if mapping is None:
            return np.asarray(record.video_segment_start, np.int64) + frames
        video_positions, _sample_rows = mapping
        # Sparse positions are segment-relative (0-based inside the episode's
        # span of the shared chunk video): rebase onto the segment start, or
        # the reader fetches a different episode's frames.
        return (np.asarray(record.video_segment_start, np.int64)
                + video_positions[np.asarray(frames, np.int64)])

    def _camera_rows(self, record: "LabelRunRecord",
                     frames: np.ndarray) -> np.ndarray:
        """Canonical rows whose camera poses match the DISPLAYED frames.

        Dense videos: identity.  Sparse videos: the sampled frame's own
        canonical time — the model is shown that image, so every
        camera-frame quantity (c2w, rotation reference) must use the
        camera of the image, not of the label row.
        """
        mapping = self._video_frame_map(record.episode_index, record.view_id)
        if mapping is None:
            return np.asarray(frames, np.int64)
        _video_positions, sample_rows = mapping
        return sample_rows[np.asarray(frames, np.int64)]

    @property
    def video_paths(self) -> list[Path]:
        paths = dict.fromkeys(record.video_path for record in self.records)
        if self._handcrop is not None:
            for path in self._handcrop.video_paths():
                paths.setdefault(path)
        return list(paths)

    def attach_video_reader(self, reader: VideoReader, video_ids: dict) -> None:
        self._video_reader = reader
        self._video_ids = video_ids

    def _ensure_video_reader(self) -> VideoReader:
        if self._video_reader is None:
            paths = self.video_paths
            self._video_ids = {path: index for index, path in enumerate(paths)}
            self._video_reader = VideoReader(
                {index: path for path, index in self._video_ids.items()},
                cache_size=int(self.cfg.get("video_cache_size", 48)),
                backend=str(self.cfg["video_backend"]), image_mode="rgb",
                torchcodec_seek_mode=str(
                    self.cfg.get("torchcodec_seek_mode", "exact")),
                torchcodec_num_ffmpeg_threads=max(1, int(
                    self.cfg.get("torchcodec_num_ffmpeg_threads", 4))))
        return self._video_reader

    def close(self) -> None:
        self._last_scoped_mano = None
        if self._video_reader is not None:
            self._video_reader.close()
            self._video_reader = None

    def dataset_episode_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        """Public decode hook for validation/replay tooling."""
        episode_index = int(episode_index)
        episode = self._labels_cache[(episode_index,)]
        if "lever" in episode:
            return episode
        result = dict(episode, lever=self._episode_levers(episode_index, episode))
        result.pop("lever_source_positions")
        result.pop("lever_position_valid")
        return result

    # ---------------------------------------------------------------- windows

    def window_count(self, record: LabelRunRecord, phase: int = 0) -> int:
        if record.topology == TOPOLOGY_STATIC:
            return int(record.real_length)
        return math.ceil(record.real_length / self.window_stride)

    def window_starts(self, record: LabelRunRecord, phase: int = 0) -> list[int]:
        stride = 1 if record.topology == TOPOLOGY_STATIC else self.window_stride
        return list(range(0, record.real_length, stride))

    def __len__(self) -> int:
        return sum(self.window_count(record) for record in self.records)

    # ----------------------------------------------------------------- sample

    @staticmethod
    def _sample_address(index: int | tuple[int, ...]) -> tuple[int, int, int | None]:
        if isinstance(index, tuple):
            return int(index[0]), int(index[1]), int(index[2]) if len(index) > 2 else None
        return int(index), 0, None

    def _clip_window_plan(self, stream_index, record, start, origin_seed, episode):
        plan = self._source_window_plan(stream_index, record, start, origin_seed, episode)
        return 0, self.sequence_length - len(plan.source_frames), plan

    def _source_window_plan(self, stream_index, record, start, origin_seed, episode):
        """Resolve an epoch's native origin on the same timeline as labels."""
        if origin_seed is None:
            origin_seed = zlib.crc32(f"{self.window_seed}:{record.episode_index}:{record.view_id}:"
                                     f"{stream_index}:{start}".encode())
        labels = self.index_by_episode[record.episode_index]
        solver_fps = float(labels["solver_fps"])
        view = next(view for view in self._view_locators[record.episode_index]
                    if str(view["view_id"]) == record.view_id)
        mapping = self._video_frame_map(record.episode_index, record.view_id)
        displayed_rows = None if mapping is None else mapping[1]
        frame_count = int(view["num_frames"])
        if displayed_rows is not None:
            frame_count = min(frame_count, len(displayed_rows))
        elif int(view.get("video_num_frames") or 0) > 0:
            frame_count = min(frame_count, int(view["video_num_frames"]))
        assert record.solver_span is not None
        run_start, exclusive_stop = record.solver_span
        run_stop = exclusive_stop - 1
        frame_count = min(frame_count, record.source_origin_stop)
        first = record.source_origin_start
        origin_stop = frame_count
        origins = self._source_origin_cache.get(stream_index)
        if origins is None:
            origins = native_frame_origins(first, origin_stop, displayed_rows)
            if not len(origins):
                raise ValueError(f"episode {record.episode_index}: no decodable source frame")
            self._source_origin_cache[stream_index] = origins
            if len(self._source_origin_cache) > 64:
                self._source_origin_cache.popitem(last=False)
        else:
            self._source_origin_cache.move_to_end(stream_index)
        bucket_index = start // self.window_stride
        bucket_count = self.window_count(record)
        first = bucket_index * len(origins) // bucket_count
        stop = (bucket_index + 1) * len(origins) // bucket_count
        first = min(first, len(origins) - 1)
        bucket = origins[first:max(first + 1, stop)]
        # A seeded affine permutation gives varied starts without keeping
        # a sampler-sized shuffle, and visits every candidate on a cycle.
        seed = origin_seed & 0xffffffff
        epoch = origin_seed >> 32
        count = len(bucket)
        step = 1 + (seed >> 16) % max(1, count - 1)
        while math.gcd(step, count) != 1:
            step += 1
        return source_clip_plan(
            episode["source_positions"], run_start, run_stop,
            int(bucket[(seed + epoch * step) % count]), self.sequence_length,
            float(labels["source_fps"]), solver_fps, self.source_clip_target_fps,
            frame_count, displayed_rows)

    def __getitem__(self, index: int | tuple[int, ...]) -> dict[str, Any]:
        return self._get_sample(index)

    def __getitems__(self, indices):
        if not self._window_scoped_geometry or len(indices) < 2:
            return [self[index] for index in indices]
        results = []
        for episode_index, group in groupby(
                indices, key=lambda index: self.records[self._sample_address(index)[0]].episode_index):
            items = list(group)
            episode = self._labels_cache[(episode_index,)]
            if len(items) < 2 or "lever" in episode:
                results.extend(self[index] for index in items)
                continue
            view_id = str(self.index_by_episode[episode_index]["canonical_view"])
            _, hmr_frame = self._scoped_mano_source(episode_index, view_id)
            # The stored-vertices fallback compares the first requested row.
            # Keep its original per-window selection on those sources.
            if "left.vertices_camera" in hmr_frame.columns:
                results.extend(self[index] for index in items)
                continue
            windows = []
            for position, index in enumerate(items):
                stream, start, origin_seed = self._sample_address(index)
                record = self.records[stream]
                _, _, plan = self._clip_window_plan(stream, record, start, origin_seed, episode)
                windows.append((position, plan.substep_rows))
            # Fuse neighboring or overlapping windows. Disjoint windows
            # retain their own scope so MANO never forwards a large gap.
            clusters: list[list[tuple[int, np.ndarray]]] = []
            last_row = -1
            for position, rows in sorted(windows, key=lambda item: int(item[1][0])):
                if not clusters or int(rows[0]) > last_row + 1:
                    clusters.append([])
                clusters[-1].append((position, rows))
                last_row = max(last_row, int(rows[-1]))
            prepared = {}
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                sizes = [len(rows) for _, rows in cluster]
                unique, inverse = np.unique(np.concatenate([rows for _, rows in cluster]), return_inverse=True)
                levers = self._episode_levers(episode_index, episode, unique)
                for (position, _), selection in zip(cluster, np.split(inverse, np.cumsum(sizes)[:-1])):
                    prepared[position] = levers[selection]
            results.extend(self._get_sample(index, lever_rows=prepared.get(position))
                           for position, index in enumerate(items))
        return results

    def _get_sample(self, index: int | tuple[int, ...], *, lever_rows=None) -> dict[str, Any]:
        stream_index, start, origin_seed = self._sample_address(index)
        record = self.records[stream_index]
        if record.topology == TOPOLOGY_STATIC:
            return self._static_window(record, start)
        total = self.sequence_length

        episode = self._labels_cache[(record.episode_index,)]
        left, right, plan = self._clip_window_plan(stream_index, record, start, origin_seed, episode)
        real_frames = plan.source_frames
        available = len(real_frames)
        substep_rows, closing_row = plan.substep_rows, plan.closing_row
        geometry_rows = plan.geometry_rows
        c2w_view = self._c2w_cache[(record.episode_index, record.view_id)]["c2w"]
        camera_rows = self._camera_rows(record, real_frames)
        camera_reference = c2w_view[int(camera_rows[0])]
        rotate = camera_reference[:3, :3]
        labels = self.index_by_episode[record.episode_index]
        physics = self._physics(labels, record.object_idx, rotate)

        boundary_rows = np.concatenate(
            (substep_rows, np.asarray([closing_row], np.int64)))

        repeat_edge = self.pad_label_mode == "repeat_edge"
        sup = (np.clip(np.arange(total) - left, 0, available - 1) if repeat_edge
               else None)

        # Hand-crop side decision must be identical for the input frames and
        # the supervision override below: dominant-present hand, ties to the
        # right hand (mirrors the review visualizer's primary-crop rule).
        crop_side = self._handcrop_input_side(
            record, episode["present"][geometry_rows])

        force = episode["force"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN, 2, MANO_VERTICES, 3)
        world_lever = (self._episode_levers(record.episode_index, episode, substep_rows)
                       if lever_rows is None else lever_rows)
        lever = world_lever.reshape(
            available, SUBSTEPS_PER_TOKEN, 2, MANO_VERTICES, 3)
        if crop_side is not None:
            force[:, :, 1 - crop_side] = 0.0
            lever = lever.copy()
            lever[:, :, 1 - crop_side] = 0.0
        environment = episode["environment_wrench"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN, 2, 3)
        hand = episode["hand_wrench"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN, 2, 3)
        if crop_side is not None:
            hand[:, :, 1 - crop_side] = 0.0

        def rot6(values: np.ndarray, count: int) -> np.ndarray:
            if repeat_edge:
                return (values[sup].reshape(-1, 2, 3) @ rotate).reshape(
                    count, SUBSTEPS_PER_TOKEN, 6)
            out = np.zeros((count, SUBSTEPS_PER_TOKEN, 6), np.float32)
            out[left:left + available] = (
                values.reshape(-1, 2, 3) @ rotate).reshape(
                available, SUBSTEPS_PER_TOKEN, 6)
            return out

        if repeat_edge:
            force_out = ((force[sup] @ rotate) * physics.mass).astype(np.float32)
            lever_out = (lever[sup] @ rotate).astype(np.float32)
        else:
            force_out = np.zeros(
                (total, SUBSTEPS_PER_TOKEN, 2, MANO_VERTICES, 3), np.float32)
            lever_out = np.zeros_like(force_out)
            force_out[left:left + available] = ((force @ rotate)
                                                * physics.mass).astype(np.float32)
            lever_out[left:left + available] = (lever @ rotate).astype(np.float32)
        environment_out = rot6(environment, total)
        hand_out = rot6(hand, total)
        if not repeat_edge:
            padding = np.ones((total, SUBSTEPS_PER_TOKEN), bool)
            padding[left:left + available] = False
            environment_out[padding, :3] = -physics.mass * np.asarray(
                physics.gravity_world)
            environment_out[padding, 3:] = 0.0

        real_mask = np.zeros(total, bool)
        real_mask[left:left + available] = True
        mask30 = np.repeat(real_mask, SUBSTEPS_PER_TOKEN)

        native_present = self._native_evaluable_present(
            record, episode["present"][geometry_rows])
        present, _absent = self._crop_presence_override(
            native_present, crop_side)
        contact, proximity_real, contact_valid_real, proximity_valid_real, \
            contact_native, contact_valid_native = \
            self._row_targets(episode, geometry_rows, crop_side)
        if plan.geometry_aligned is not None:
            contact_valid_real &= plan.geometry_aligned[:, None]
            proximity_valid_real &= plan.geometry_aligned[:, None]
            contact_valid_native &= plan.geometry_aligned[:, None, None]

        def hold(values: np.ndarray, fill: np.ndarray | None = None) -> np.ndarray:
            out = np.zeros((total,) + values.shape[1:], values.dtype)
            out[left:left + available] = values
            if repeat_edge:
                out[:left] = out[left]
                out[left + available:] = out[left + available - 1]
            elif fill is not None:
                out[:left] = fill
                out[left + available:] = fill
            return out

        contact_10hz = hold(contact)
        proximity_10hz = hold(proximity_real)
        contact_valid_10hz = hold(contact_valid_real)
        # NATIVE scoring targets: undecayed by the crop pipeline (see
        # _row_targets).  The harness scores contact against these.
        contact_native_10hz = hold(contact_native)
        contact_valid_native_10hz = hold(contact_valid_native)
        proximity_valid_10hz = hold(proximity_valid_real)
        hand_present_10hz = hold(present)

        gravity_rows = episode["gravity_world"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN, 3) @ rotate
        if repeat_edge:
            gravity_30hz = gravity_rows[sup].reshape(-1, 3)
        else:
            gravity_30hz = np.broadcast_to(np.asarray(physics.gravity_world),
                (total, SUBSTEPS_PER_TOKEN, 3)).copy()
            gravity_30hz[left:left + available] = gravity_rows
            gravity_30hz = gravity_30hz.reshape(-1, 3)

        position_rows = episode["position"][boundary_rows]
        rotation_rows = episode["rotation"][boundary_rows]
        velocity_rows = episode["velocity"][boundary_rows]
        position_cam = (position_rows - camera_reference[None, :3, 3]) @ rotate
        rotation_cam = rotate.T @ rotation_rows
        velocity_cam = velocity_rows.copy()
        velocity_cam[..., :3] = velocity_rows[..., :3] @ rotate
        velocity_cam[..., 3:] = velocity_rows[..., 3:] @ rotate
        state_count = total * SUBSTEPS_PER_TOKEN + 1
        begin = left * SUBSTEPS_PER_TOKEN
        end = (left + available) * SUBSTEPS_PER_TOKEN + 1
        position_30hz = np.empty((state_count, 3), np.float32)
        rotation_30hz = np.empty((state_count, 3, 3), np.float32)
        velocity_30hz = np.zeros((state_count, 6), np.float32)
        position_30hz[:begin] = position_cam[0]
        position_30hz[begin:end] = position_cam
        position_30hz[end:] = position_cam[-1]
        rotation_30hz[:begin] = rotation_cam[0]
        rotation_30hz[begin:end] = rotation_cam
        rotation_30hz[end:] = rotation_cam[-1]
        velocity_30hz[begin:end] = velocity_cam

        dt = float(labels["dt_s"])
        # Per-row force supervision from the packed flags, laid out exactly
        # like force_out: repeat_edge holds the edge row's flag, zero-pad
        # rows stay invalid.  Rows whose stored vertex forces cannot
        # reproduce the stored wrench columns (QP-failed or dynamics-invalid
        # frames) carry a cleared FLAG_FORCE_VALID and must never supervise
        # the point/wrench losses, or the two heads' targets contradict.
        stored_force_valid = episode["force_valid"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN)
        row_force_valid = stored_force_valid.copy()
        if plan.force_intervals_valid is not None:
            row_force_valid &= plan.force_intervals_valid
        if getattr(self, "random_source_clip_origin", False):
            prediction_valid = np.zeros_like(row_force_valid)
            label_valid = stored_force_valid if record.solver_span is not None else row_force_valid
            prediction_valid[:, 0] = label_valid[:, 0] & record.force_supervised
            if plan.force_prediction_aligned is not None:
                prediction_valid[:, 0] &= plan.force_prediction_aligned
        if repeat_edge:
            valid30 = row_force_valid[sup].reshape(-1)
        else:
            valid30 = np.zeros((total, SUBSTEPS_PER_TOKEN), bool)
            valid30[left:left + available] = row_force_valid
            valid30 = valid30.reshape(-1)
        wrench_rows = episode["contact_wrench"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN, 6)
        wrench_token = wrench_rows.mean(axis=1)
        wrench_camera = np.zeros((total, 6), np.float32)
        wrench_camera[left:left + available, :3] = wrench_token[:, :3] @ rotate
        wrench_camera[left:left + available, 3:] = wrench_token[:, 3:] @ rotate
        static_force = (-physics.mass * np.asarray(physics.gravity_world)
                        ).astype(np.float32)
        wrench_camera[:left, :3] = static_force
        wrench_camera[left + available:, :3] = static_force

        if self.training_data_fast_path:
            frames, content_hw = self._read_resized_frames(
                record, real_frames, crop_side)
        else:
            frames = self._read_input_frames(record, real_frames, crop_side)
            content_hw = tuple(np.asarray(frames[0]).shape[:2])
            frames = np.stack([
                letterbox_square(frame, int(self.cfg["image_size"]))
                for frame in frames])
        both_hand_frames = self._read_both_hand_frames(
            record, real_frames, native_present)
        frames = self._maybe_photometric(frames, content_hw)
        if left or right:
            frames = edge_pad(frames, left, right)
        images = self._to_model_images(frames)
        images_by_hand = None
        if both_hand_frames is not None:
            dual = np.stack([[letterbox_square(frame, int(self.cfg["image_size"]))
                              for frame in stream]
                             for stream in both_hand_frames])
            dual = np.stack([edge_pad(stream, left, right) for stream in dual])
            images_by_hand = torch.stack(
                [self._to_model_images(stream) for stream in dual])
        c2w_out = edge_pad(c2w_view[camera_rows], left, right).astype(np.float32)

        result: dict[str, Any] = {
            "images": images,
            "wrench_camera": torch.from_numpy(wrench_camera),
            "gravity_camera": torch.tensor(
                physics.gravity_world, dtype=torch.float32),
            "mass": torch.tensor(physics.mass, dtype=torch.float32),
            "camera_c2w": torch.from_numpy(c2w_out),
            "position": torch.from_numpy(edge_pad(
                position_cam[::SUBSTEPS_PER_TOKEN], left, right)),
            "rotation": torch.from_numpy(edge_pad(
                rotation_cam[::SUBSTEPS_PER_TOKEN], left, right)),
            "velocity": torch.from_numpy(edge_pad(
                velocity_cam[::SUBSTEPS_PER_TOKEN], left, right)),
            "real_mask": torch.from_numpy(real_mask),
            "physics": physics,
            "vertex_force_camera_30hz": torch.from_numpy(force_out),
            "vertex_lever_camera_30hz": torch.from_numpy(lever_out),
            "hand_wrench_camera_30hz": torch.from_numpy(hand_out),
            "environment_wrench_camera_30hz": torch.from_numpy(environment_out),
            "hand_present": torch.from_numpy(episode["episode_hand_present"]),
            "hand_present_10hz": torch.from_numpy(hand_present_10hz),
            "hand_present_native_10hz": torch.from_numpy(
                hold(native_present)),
            "position_30hz": torch.from_numpy(position_30hz),
            "rotation_30hz": torch.from_numpy(rotation_30hz),
            "velocity_30hz": torch.from_numpy(velocity_30hz),
            "real_mask_30hz": torch.from_numpy(mask30),
            "force_valid_30hz": torch.from_numpy(
                valid30 if record.force_supervised
                else np.zeros(total * SUBSTEPS_PER_TOKEN, bool)),
            "force_supervision_valid": torch.tensor(record.force_supervised),
            "rollout_eligible": torch.tensor(record.rollout_eligible),
            "physics_30hz": physics,
            "gravity_camera_30hz": torch.from_numpy(gravity_30hz.copy()),
            "contact_target_10hz": torch.from_numpy(contact_10hz),
            "proximity_target_10hz": torch.from_numpy(proximity_10hz),
            "contact_valid_10hz": torch.from_numpy(contact_valid_10hz),
            "proximity_valid_10hz": torch.from_numpy(proximity_valid_10hz),
            # Native (crop-override-free) contact targets/validity for
            # EVALUATION scoring — see _row_targets.  Training losses keep
            # reading contact_target_10hz.
            "contact_target_native_10hz": torch.from_numpy(
                contact_native_10hz),
            "contact_valid_native_10hz": torch.from_numpy(
                contact_valid_native_10hz),
            "dataset_mode": ("articulated_trajectory"
                             if record.topology == TOPOLOGY_ARTICULATED
                             else "rigid_trajectory"),
            "episode_index": record.episode_index,
            "view_id": record.view_id,
            "left_padding": left,
            "right_padding": right,
            # Which hand's crop backs the input frames (0=left, 1=right,
            # -1=full view); external single-hand predictors (HACO) need it
            # to map their per-crop output back onto (hand, vertex) axes.
            "crop_side": torch.tensor(
                -1 if crop_side is None else int(crop_side),
                dtype=torch.int8),
            # Canonical label-timeline frame per token (edge-padded), for
            # visualizers that need to line panels up with the token axis.
            "source_frames": torch.from_numpy(np.asarray(
                edge_pad(real_frames.astype(np.int64), left, right))),
            "geometry_solver_rows": torch.from_numpy(
                edge_pad(geometry_rows, left, right)),
            "force_solver_rows": torch.from_numpy(
                edge_pad(substep_rows[::SUBSTEPS_PER_TOKEN], left, right)),
            "target_object_index": record.object_idx,
        }
        if self.supervise_hand_visibility:
            target, valid = self._hand_visibility_targets(
                record, episode, geometry_rows, real_frames, camera_rows,
                c2w_view[camera_rows], crop_side, content_hw)
            result["hand_visibility_target_10hz"] = torch.from_numpy(hold(target))
            result["hand_visibility_valid_10hz"] = torch.from_numpy(
                hold(valid) & real_mask[:, None])
        if getattr(self, "random_source_clip_origin", False):
            result["force_prediction_valid_30hz"] = torch.from_numpy(
                hold(prediction_valid).reshape(-1))
        if images_by_hand is not None:
            result["images_by_hand"] = images_by_hand
        if self.vertex_force_frame == VERTEX_LOCAL_FRAME:
            result.update(self._vertex_local(
                episode, substep_rows, left, available, total,
                present, rotate, physics.mass, repeat_edge, lever_rows=world_lever))
        if bool(self.cfg.get("deduplicate_batch_frames", False)):
            absolute = edge_pad(
                self._video_positions(record, real_frames), left, right)
            result["frame_indices"] = torch.from_numpy(
                absolute.astype(np.int64, copy=False))
            result["video_path"] = str(record.video_path)
        if record.topology == TOPOLOGY_RIGID and len(labels['object_ids']) > 1:
            if bool(episode.get('body_ready', False)):
                body = {key: episode[f'body_{key}'] for key in (
                    'position', 'rotation', 'velocity', 'environment', 'gravity', 'owner', 'valid')}
                attach_multi_body(result, body, labels, substep_rows, boundary_rows,
                    rotate, camera_reference, left, available)
                if crop_side is not None:
                    # The full union labels include an off-crop hand. Until
                    # its reference force is explicitly modeled, preserve
                    # direct supervision and keep this rollout masked.
                    result['rollout_eligible'] = torch.tensor(False)
            else:
                attach_single_body(result, verified=False)
                result['rollout_eligible'] = torch.tensor(False)
            result['rigid_body_recovery_report'] = str(episode.get('body_recovery_report', 'unavailable'))
        else:
            attach_single_body(result)
            result['rigid_body_recovery_report'] = 'native single-body reference'
        if plan.force_intervals_valid is not None and float(labels["solver_fps"]) != 30.0:
            # The rigid rollout still consumes three 30 Hz intervals per
            # visual step. Native RGB can train geometry/point force on
            # other solver clocks without entering that integration path.
            result["rollout_eligible"] = torch.tensor(False)
        return result

    def _static_window(self, record: "LabelRunRecord",
                       row: int) -> dict[str, Any]:
        """One independent static frame stacked ``sequence_length`` times.

        Static episodes (mow / obman) have no tokens and no temporal
        coupling — every row is its own physical problem.  A window is the
        SAME frame and its labels repeated across the whole sequence axis
        (real_mask all-true, no padding), so the model sees a frozen scene
        and supervises every step on the same row.  Hand-crop input and
        supervision follow the same rule as trajectory windows: the crop
        hand keeps its labels, the other hand is supervised as zeros.
        """
        def tile(values, count):
            return np.repeat(np.asarray(values)[None], count, axis=0)

        episode = self._labels_cache[(record.episode_index,)]
        c2w_view = self._c2w_cache[(record.episode_index, record.view_id)][
            "c2w"]
        labels = self.index_by_episode[record.episode_index]
        row = int(np.clip(int(row), 0, int(record.real_length) - 1))
        total = self.sequence_length
        steps = total * SUBSTEPS_PER_TOKEN
        real_frames = record.source_frames[row:row + 1]
        camera_rows = self._camera_rows(record, real_frames)
        camera_reference = c2w_view[int(camera_rows[0])]
        rotate = camera_reference[:3, :3]
        object_idx = int(episode["static_object_index"][row])
        row_valid = bool(episode["force_valid"][row])
        physics = self._physics(labels, max(0, object_idx), rotate)
        physics = replace(physics, gravity_world=tuple(
            rotate.T @ episode["static_gravity_world"][row]))

        native_present = self._native_evaluable_present(
            record, episode["present"][row])
        crop_side = self._handcrop_input_side(
            record, episode["present"][row][None, :])
        present, _absent = self._crop_presence_override(
            episode["present"][row], crop_side)
        target_rows = self._row_targets(
            episode, np.asarray([row], np.int64), crop_side)
        contact, proximity, contact_valid, proximity_valid, \
            contact_native, contact_valid_native = (
                value[0] for value in target_rows)

        force_row = (episode["force"][row] @ rotate) * physics.mass
        lever_row = episode["lever"][row] @ rotate
        # Wrench targets are 6-vectors (force||torque); rotate both parts,
        # matching the trajectory path's rot6 layout (T, S, 6).
        def rot6_row(six: np.ndarray) -> np.ndarray:
            return np.concatenate(
                (six[:3] @ rotate, six[3:] @ rotate)).astype(np.float32)
        hand_row = rot6_row(episode["hand_wrench"][row])
        environment_row = rot6_row(episode["environment_wrench"][row])
        if crop_side is not None:
            force_row[1 - crop_side] = 0.0
            lever_row[1 - crop_side] = 0.0

        def steps_tiled(values):
            return tile(values, total)[:, None].repeat(
                SUBSTEPS_PER_TOKEN, axis=1).astype(np.float32)

        force_out = steps_tiled(force_row)
        lever_out = steps_tiled(lever_row)
        hand_out = steps_tiled(hand_row)
        environment_out = steps_tiled(environment_row)

        position_cam = (episode["position"][row]
                        - camera_reference[:3, 3]) @ rotate
        rotation_cam = rotate.T @ episode["rotation"][row].reshape(3, 3)
        velocity_cam = episode["velocity"][row].copy()
        velocity_cam[:3] = episode["velocity"][row][:3] @ rotate
        velocity_cam[3:] = episode["velocity"][row][3:] @ rotate
        position_30hz = tile(position_cam, steps + 1).astype(np.float32)
        rotation_30hz = tile(rotation_cam, steps + 1).astype(np.float32)
        velocity_30hz = tile(velocity_cam, steps + 1).astype(np.float32)

        wrench_row = episode["contact_wrench"][row]
        wrench_camera = np.zeros((total, 6), np.float32)
        wrench_camera[:, :3] = wrench_row[None, :3] @ rotate
        wrench_camera[:, 3:] = wrench_row[None, 3:] @ rotate

        frames = self._read_input_frames(record, real_frames, crop_side)
        both_hand_frames = self._read_both_hand_frames(
            record, real_frames, native_present)
        content_hw = tuple(np.asarray(frames[0]).shape[:2])
        frame = self._maybe_photometric(
            letterbox_square(np.asarray(frames[0], np.uint8),
                             int(self.cfg["image_size"])),
            content_hw)
        images = self._to_model_images(tile(frame, total))
        images_by_hand = None
        if both_hand_frames is not None:
            dual_images = []
            for stream in both_hand_frames:
                dual_frame = letterbox_square(
                    np.asarray(stream[0], np.uint8), int(self.cfg["image_size"]))
                dual_images.append(self._to_model_images(tile(dual_frame, total)))
            images_by_hand = torch.stack(dual_images)
        # c2w_view[camera_rows] already carries its row axis: (1, 4, 4).
        c2w_out = np.repeat(c2w_view[camera_rows], total,
                            axis=0).astype(np.float32)

        result: dict[str, Any] = {
            "images": images,
            "wrench_camera": torch.from_numpy(wrench_camera),
            "gravity_camera": torch.tensor(
                physics.gravity_world, dtype=torch.float32),
            "mass": torch.tensor(physics.mass, dtype=torch.float32),
            "camera_c2w": torch.from_numpy(c2w_out),
            # 10 Hz pose keys carry the T+1 boundary states, matching the
            # trajectory path's boundary_rows slicing ([::substeps] of
            # T*S+1 rows).
            "position": torch.from_numpy(tile(position_cam, total + 1)),
            "rotation": torch.from_numpy(tile(rotation_cam, total + 1)),
            "velocity": torch.from_numpy(tile(velocity_cam, total + 1)),
            "real_mask": torch.ones(total, dtype=torch.bool),
            "physics": physics,
            "vertex_force_camera_30hz": torch.from_numpy(force_out),
            "vertex_lever_camera_30hz": torch.from_numpy(lever_out),
            "hand_wrench_camera_30hz": torch.from_numpy(hand_out),
            "environment_wrench_camera_30hz": torch.from_numpy(environment_out),
            "hand_present": torch.from_numpy(episode["episode_hand_present"]),
            "hand_present_10hz": torch.from_numpy(tile(present, total)),
            "hand_present_native_10hz": torch.from_numpy(
                tile(native_present, total)),
            "position_30hz": torch.from_numpy(position_30hz),
            "rotation_30hz": torch.from_numpy(rotation_30hz),
            "velocity_30hz": torch.from_numpy(velocity_30hz),
            "real_mask_30hz": torch.ones(steps, dtype=torch.bool),
            "force_valid_30hz": torch.from_numpy(
                np.full(steps, bool(episode["force_valid"][row]),
                        dtype=bool)),
            "force_supervision_valid": torch.tensor(True),
            "rollout_eligible": torch.tensor(record.rollout_eligible and row_valid),
            "physics_30hz": physics,
            "contact_target_10hz": torch.from_numpy(tile(contact, total)),
            "proximity_target_10hz": torch.from_numpy(tile(proximity, total)),
            "contact_valid_10hz": torch.from_numpy(tile(contact_valid, total)),
            "proximity_valid_10hz": torch.from_numpy(
                tile(proximity_valid, total)),
            "contact_target_native_10hz": torch.from_numpy(
                tile(contact_native, total)),
            "contact_valid_native_10hz": torch.from_numpy(
                tile(contact_valid_native, total)),
            "dataset_mode": "static_frame",
            "episode_index": record.episode_index,
            "view_id": record.view_id,
            "left_padding": 0,
            "right_padding": 0,
            # Same contract as the rigid_trajectory window (see there).
            "crop_side": torch.tensor(
                -1 if crop_side is None else int(crop_side),
                dtype=torch.int8),
            # Same row tiled across the sequence axis, matching the labels.
            "source_frames": torch.from_numpy(
                np.full(total, int(real_frames[0]), np.int64)),
            "geometry_solver_rows": torch.full((total,), row, dtype=torch.int64),
            "force_solver_rows": torch.full((total,), row, dtype=torch.int64),
            "target_object_index": object_idx,
        }
        if self.supervise_hand_visibility:
            target, valid = self._hand_visibility_targets(
                record, episode, np.asarray([row]), real_frames, camera_rows,
                c2w_view[camera_rows], crop_side, content_hw)
            result["hand_visibility_target_10hz"] = torch.from_numpy(tile(target[0], total))
            result["hand_visibility_valid_10hz"] = torch.from_numpy(tile(valid[0], total))
        if getattr(self, "random_source_clip_origin", False):
            prediction_valid = torch.zeros_like(result["force_valid_30hz"])
            prediction_valid[::SUBSTEPS_PER_TOKEN] = result["force_valid_30hz"][::SUBSTEPS_PER_TOKEN]
            result["force_prediction_valid_30hz"] = prediction_valid
        if images_by_hand is not None:
            result["images_by_hand"] = images_by_hand
        if self.vertex_force_frame == VERTEX_LOCAL_FRAME:
            # Same row tiled: every (t, substep) reads the identical source
            # row, so the rigid vertex-local builder applies unchanged.
            result.update(self._vertex_local(
                episode, np.full(total * SUBSTEPS_PER_TOKEN, row, np.int64),
                0, total, total, tile(present, total), rotate,
                physics.mass, True))
        if bool(self.cfg.get("deduplicate_batch_frames", False)):
            result["frame_indices"] = torch.from_numpy(np.full(
                total, int(self._video_positions(record, real_frames)[0]),
                np.int64))
            result["video_path"] = str(record.video_path)
        attach_single_body(result)
        result['rigid_body_recovery_report'] = 'native independent static reference'
        return result

    def _physics(self, labels: dict, object_idx: int, rotate: np.ndarray) -> Physics:
        gravity_world = np.asarray(labels["gravity_world"], np.float64)
        if not labels.get("object_mass_kg"):
            # Contact-only episodes carry no object: unit placeholder keeps
            # the sample's physics fields well-formed (all force-side
            # losses are masked off for them anyway).
            return Physics(
                mass=1.0, com_body=(0.0, 0.0, 0.0),
                inertia_body=np.eye(3),
                gravity_world=tuple(rotate.T @ gravity_world),
                dt=1.0 / 30.0, substeps=int(labels["substeps"]))
        inertia = np.asarray(
            labels["object_inertia_body"][object_idx], np.float64)
        if inertia.size != 9 or not np.all(
                np.linalg.eigvalsh(inertia.reshape(3, 3)) > 0):
            # Articulated conversions store a zero joint placeholder: their
            # episodes never enter the rollout, so the identity keeps the
            # Physics contract valid for the mass/gravity readouts.
            inertia = np.eye(3)
        return Physics(
            mass=float(labels["object_mass_kg"][object_idx]),
            com_body=tuple(np.asarray(
                labels["object_com_body"][object_idx], np.float64)),
            inertia_body=inertia.reshape(3, 3),
            gravity_world=tuple(rotate.T @ gravity_world),
            # Backend convention: dt is the SUBSTEP size and one
            # rollout_interval advances dt*substeps.  A 30 Hz row spacing
            # with substeps=4 therefore needs dt=1/(30*4)=1/120 — feeding
            # dt=1/30 here made every transition advance 4x too fast and
            # the rollout diverged from the GT pose it was derived from.
            dt=1.0 / (float(labels.get("solver_fps", 30.0))
                      * max(1, int(labels.get("substeps", 4)))),
            substeps=int(labels["substeps"]),
        )

    def _vertex_local(self, episode, substep_rows, left,
                      available, total, present, rotate, mass, repeat_edge, *, lever_rows=None):
        if self.training_data_fast_path:
            return self._aligned_vertex_axes(
                episode, substep_rows, left, available, total, present,
                rotate, repeat_edge, lever_rows=lever_rows)
        world_force = (episode["force"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN, 2, MANO_VERTICES, 3) * mass)
        world_vertices = episode["lever"][substep_rows].reshape(
            available, SUBSTEPS_PER_TOKEN, 2, MANO_VERTICES, 3)
        local_force = np.zeros(
            (total, SUBSTEPS_PER_TOKEN, 2, MANO_VERTICES, 3), np.float32)
        yz_camera = np.empty(
            (total, SUBSTEPS_PER_TOKEN, 2, MANO_VERTICES, 2, 3), np.float32)
        for side, name in enumerate(("left", "right")):
            y_axis, z_axis = canonical_vertex_local_axes(name)
            canonical = np.stack((y_axis, z_axis), axis=-2)
            yz_camera[:, :, side] = canonical            # (total,3,778,2,3)
            token_present = present[:, side]
            if not bool(token_present.any()):
                continue
            y_world, z_world = vertex_local_axes(
                world_vertices[:, :, side], name)
            converted = force_world_to_local(
                world_force[:, :, side], y_world, z_world)
            converted_yz = axes_world_to_camera(y_world, z_world, rotate)
            local_force[left:left + available, :, side][token_present] = \
                converted[token_present]
            yz_camera[left:left + available, :, side][token_present] = \
                converted_yz[token_present]
        if repeat_edge:
            if left:
                local_force[:left] = local_force[left]
                yz_camera[:left] = yz_camera[left]
            if total - left - available:
                local_force[left + available:] = local_force[left + available - 1]
                yz_camera[left + available:] = yz_camera[left + available - 1]
        return {
            "vertex_force_local_30hz": torch.from_numpy(local_force),
            "vertex_local_yz_camera_30hz": torch.from_numpy(yz_camera),
        }

    def _aligned_vertex_axes(self, episode, rows, left, available, total,
                             present, rotate, repeat_edge, *, lever_rows=None):
        """S=1 consumer axes; legacy/raw-local readers keep the full path."""
        selected = np.asarray(rows).reshape(available, SUBSTEPS_PER_TOKEN)[:, 0]
        unique, first, inverse = np.unique(selected, return_index=True, return_inverse=True)
        yz_camera = np.empty((total, 1, 2, MANO_VERTICES, 2, 3), np.float32)
        for side, name in enumerate(("left", "right")):
            y_axis, z_axis = canonical_vertex_local_axes(name)
            yz_camera[:, :, side] = np.stack((y_axis, z_axis), axis=-2)
            token_present = present[:, side]
            if not (bool(episode["force_geometry_available"])
                    and bool(token_present.any())):
                continue
            world_vertices = (episode["lever"][unique, side] if lever_rows is None
                              else lever_rows[first * SUBSTEPS_PER_TOKEN, side])
            y_world, z_world = vertex_local_axes(world_vertices, name)
            converted = axes_world_to_camera(y_world, z_world, rotate)[inverse]
            yz_camera[left:left + available, 0, side][token_present] = \
                converted[token_present]
        if repeat_edge:
            if left:
                yz_camera[:left] = yz_camera[left]
            if total - left - available:
                yz_camera[left + available:] = yz_camera[left + available - 1]
        return {"vertex_local_yz_camera_30hz": torch.from_numpy(yz_camera)}


class SequentialLabelSampler(Sampler):
    """DDP-aware sequential (stream, window) sampler with per-view phases.

    Order-level randomness (train split, all epoch-seeded and therefore
    reproducible/rank-consistent): the by-video record groups may be
    permuted every epoch (groups stay contiguous — same video handles,
    same seek pattern), and the window order inside a record may be
    shuffled (all windows of a record share one video handle).
    """

    def __init__(self, dataset: LabelStreamDataset, drop_last: bool = False, *,
                 seed: int | None = None):
        self.dataset = dataset
        self.drop_last = bool(drop_last)
        self.seed = int(getattr(dataset, "window_seed", 42)
                        if seed is None else seed)
        self._epoch = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.rank = max(0, min(self.rank, max(self.world_size - 1, 0)))
        self.order = sorted(
            range(len(dataset.records)),
            key=lambda index: (
                str(dataset.records[index].video_path),
                dataset.records[index].video_segment_start
                + int(dataset.records[index].source_frames[0]),
                dataset.records[index].episode_index,
                dataset.records[index].view_id,
            ),
        )
        # Contiguous runs of records that read the same video file.
        groups: list[list[int]] = []
        for index in self.order:
            path = str(self.dataset.records[index].video_path)
            if groups and str(self.dataset.records[groups[-1][0]].video_path) == path:
                groups[-1].append(index)
            else:
                groups.append([index])
        self.video_groups = groups

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _phase(self, record: LabelRunRecord, stream_index: int) -> int:
        if self.dataset.window_mode != "phase_offset" \
                or self.dataset.window_stride <= 1:
            return 0
        key = (f"{self.seed}:{self._epoch}:{record.episode_index}:"
               f"{record.view_id}:{stream_index}")
        return int(zlib.crc32(key.encode()) % self.dataset.window_stride)

    def _build_order(self) -> list[tuple[int, ...]]:
        rng = random.Random((self.seed << 20) ^ self._epoch)
        groups = list(self.video_groups)
        if getattr(self.dataset, "shuffle_video_order", False) and len(groups) > 1:
            rng.shuffle(groups)
        order: list[tuple[int, int]] = []
        shuffle_windows = getattr(self.dataset, "shuffle_window_order", False)
        for group in groups:
            for stream_index in group:
                record = self.dataset.records[stream_index]
                random_origin = (getattr(self.dataset, "random_source_clip_origin", False)
                                 and record.topology != TOPOLOGY_STATIC)
                phase = 0 if random_origin else self._phase(record, stream_index)
                windows = [(stream_index, start) for start in
                           self.dataset.window_starts(record, phase)]
                if random_origin:
                    windows = [(stream_index, start, (self._epoch << 32) | zlib.crc32(
                        f"{self.seed}:{record.episode_index}:{record.view_id}:"
                        f"{stream_index}:{start}".encode()))
                               for stream_index, start in windows]
                if shuffle_windows and len(windows) > 1:
                    rng.shuffle(windows)
                order.extend(windows)
        return order

    def __iter__(self):
        order = self._build_order()
        per_rank = math.ceil(len(order) / self.world_size)
        if self.drop_last:
            order = order[:per_rank * self.world_size]
        elif len(order) < per_rank * self.world_size and order:
            order = order + order[:per_rank * self.world_size - len(order)]
        begin = self.rank * per_rank
        return iter(order[begin:begin + per_rank])

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.world_size)
