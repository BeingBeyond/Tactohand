"""Lightning data module for the labels-v5 TactoHand substep task."""
from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import core.data.processed_compat as processed_contract
from core.data.video import PreparedVideoReader as VideoReader
from core.tactile_labels.decode import load_label_index, load_label_stats
from training.data.dataset import episode_training_role
from training.data.dataset import LabelStreamDataset, SequentialLabelSampler


class CombinedLabelDataset(Dataset):
    """Present several labels-v5 datasets as one stream collection."""

    def __init__(self, datasets: list[LabelStreamDataset]):
        self.datasets = list(datasets)
        self.records = [record for dataset in self.datasets
                        for record in dataset.records]
        self._owners = {
            id(record): (dataset, local_index)
            for dataset in self.datasets
            for local_index, record in enumerate(dataset.records)}
        first = self.datasets[0]
        self.stream_offsets = np.cumsum(
            [0] + [len(dataset.records) for dataset in self.datasets[:-1]]).tolist()
        for name in ("window_seed", "window_mode", "window_stride"):
            setattr(self, name, getattr(first, name))

    @property
    def video_paths(self) -> list[Path]:
        return list(dict.fromkeys(
            path for dataset in self.datasets for path in dataset.video_paths))

    def attach_video_reader(self, reader: VideoReader, video_ids: dict) -> None:
        for dataset in self.datasets:
            dataset.attach_video_reader(reader, video_ids)

    def window_count(self, record, phase: int = 0) -> int:
        dataset, _ = self._owners[id(record)]
        return dataset.window_count(record, phase)

    def window_starts(self, record, phase: int = 0) -> list[int]:
        dataset, _ = self._owners[id(record)]
        return dataset.window_starts(record, phase)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    @staticmethod
    def _child_address(index, local):
        # The third combined index is the metric-padding flag. An optional
        # fourth value carries the training clip's native origin draw.
        address = (local, int(index[1]))
        return address + (int(index[3]),) if len(index) > 3 else address

    def __getitem__(self, index):
        if isinstance(index, tuple):
            record = self.records[int(index[0])]
            dataset, local = self._owners[id(record)]
            result = dataset[self._child_address(index, local)]
            result["metric_sample_valid"] = torch.tensor(
                bool(index[2]) if len(index) > 2 else True)
            result["dataset_name"] = dataset.dataset_name
            return result
        offset = int(index)
        for dataset in self.datasets:
            if offset < len(dataset):
                result = dataset[offset]
                result["dataset_name"] = dataset.dataset_name
                return result
            offset -= len(dataset)
        raise IndexError(index)

    def __getitems__(self, indices):
        if len(indices) < 2 or not all(isinstance(index, tuple) for index in indices):
            return [self[index] for index in indices]
        owners = [self._owners[id(self.records[int(index[0])])] for index in indices]
        dataset = owners[0][0]
        if any(child is not dataset for child, _ in owners) or not hasattr(dataset, "__getitems__"):
            return [self[index] for index in indices]
        samples = dataset.__getitems__([
            self._child_address(index, local) for (_, local), index in zip(owners, indices)])
        for sample, index in zip(samples, indices):
            sample["metric_sample_valid"] = torch.tensor(bool(index[2]) if len(index) > 2 else True)
            sample["dataset_name"] = dataset.dataset_name
        return samples


class HomogeneousLabelBatchSampler(Sampler):
    """Batch within one dataset child, then shard complete batches over DDP."""

    def __init__(self, dataset: CombinedLabelDataset, batch_size: int,
                 dataset_sampling_power: float = 1.0, *,
                 dataset_sampling: str = "power",
                 dataset_weights: Optional[dict[str, float]] = None,
                 shuffle_dataset_order: bool = True,
                 epoch_window_fraction: float = 1.0,
                 seed: int = 0, pad_final_batch: bool = True):
        self.dataset, self.batch_size, self.epoch = dataset, int(batch_size), 0
        self.dataset_sampling = str(dataset_sampling)
        if self.dataset_sampling not in {"power", "sqrt", "uniform"}:
            raise ValueError("dataset_sampling must be power, sqrt, or uniform")
        self.dataset_sampling_power = float(dataset_sampling_power)
        if not 0.0 <= self.dataset_sampling_power <= 1.0:
            raise ValueError("dataset_sampling_power must be in [0, 1]")
        self.dataset_weights = dict(dataset_weights) if dataset_weights else None
        self.shuffle_dataset_order = bool(shuffle_dataset_order)
        # Uniform per-epoch quota scale (mixture ratios untouched): the
        # rotating per-epoch start walks each child's fresh slice, so
        # ceil(1/fraction) consecutive epochs cover every window once.
        self.epoch_window_fraction = float(epoch_window_fraction)
        if not 0.0 < self.epoch_window_fraction <= 1.0:
            raise ValueError("epoch_window_fraction must be in (0, 1]")
        self.seed = int(seed)
        self.pad_final_batch = bool(pad_final_batch)
        self.world_size = (torch.distributed.get_world_size()
                           if torch.distributed.is_available()
                           and torch.distributed.is_initialized()
                           else int(os.environ.get("WORLD_SIZE", "1")))
        self.rank = (torch.distributed.get_rank()
                     if torch.distributed.is_available()
                     and torch.distributed.is_initialized()
                     else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))))
        self._child_samplers: list[SequentialLabelSampler] | None = None
        self._batch_cache_key: tuple[int, int] | None = None
        self._batch_cache: list[list[tuple]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[tuple]]:
        key = self.epoch, self.world_size
        if self._batch_cache_key == key:
            return self._batch_cache
        # Lightning can query length and iterate more than once per epoch.
        # Child video groups depend only on the prepared dataset records.
        if self._child_samplers is None:
            self._child_samplers = [
                SequentialLabelSampler(child, seed=child.window_seed)
                for child in self.dataset.datasets]
        child_samples = []
        for sampler, offset in zip(self._child_samplers, self.dataset.stream_offsets):
            sampler.set_epoch(self.epoch)
            samples = []
            for address in sampler._build_order():
                combined = (offset + address[0], address[1])
                if len(address) > 2:
                    combined += (True, address[2])
                samples.append(combined)
            child_samples.append(samples)
        sizes = np.asarray([len(samples) for samples in child_samples], np.float64)
        active = sizes > 0
        target_samples = np.zeros(len(sizes), dtype=np.int64)
        if bool(active.any()):
            base = np.zeros_like(sizes)
            if self.dataset_sampling == "uniform":
                base[active] = 1.0
            elif self.dataset_sampling == "sqrt":
                base[active] = np.sqrt(sizes[active])
            else:
                base[active] = sizes[active] ** self.dataset_sampling_power
            exact = sizes.sum() * base / base.sum()
            if self.dataset_weights:
                multiplier = np.ones_like(sizes)
                for index, child in enumerate(self.dataset.datasets):
                    if active[index]:
                        multiplier[index] = float(self.dataset_weights.get(
                            getattr(child, "dataset_name", ""), 1.0))
                if multiplier[active].min() <= 0:
                    raise ValueError("dataset_weights values must be positive")
                exact = exact * multiplier
            exact = exact * self.epoch_window_fraction
            target_samples = np.floor(exact).astype(np.int64)
            remainder = int(round(exact.sum() - target_samples.sum()))
            if remainder:
                order = np.argsort(-(exact - target_samples))
                target_samples[order[:remainder]] += 1
        blocks: list[list[list[tuple]]] = []
        for samples, target_count in zip(child_samples, target_samples.tolist()):
            if not samples or target_count <= 0:
                continue
            child_batches = []
            start = (self.epoch * target_count) % len(samples)
            for begin in range(0, target_count, self.batch_size):
                batch = [
                    samples[(start + index) % len(samples)]
                    for index in range(begin, min(begin + self.batch_size, target_count))]
                if batch:
                    if self.pad_final_batch:
                        batch.extend([batch[-1]] * (self.batch_size - len(batch)))
                    child_batches.append(batch)
            if child_batches:
                target = math.ceil(len(child_batches) / self.world_size) * self.world_size
                padding = target - len(child_batches)
                if padding:
                    # ``padding`` can exceed the number of source batches
                    # (for example one batch needs seven replicas at world
                    # size eight).  A plain ``child_batches[:padding]`` then
                    # under-fills the block and gives later ranks one fewer
                    # batch, desynchronizing DDP at the epoch boundary.
                    source_batches = list(child_batches)
                    padded_batches = []
                    for index in range(padding):
                        batch = source_batches[index % len(source_batches)]
                        padded_batches.append([
                            (sample[0], sample[1], False, *sample[3:]) for sample in batch])
                    child_batches.extend(padded_batches)
                if len(child_batches) % self.world_size:
                    raise RuntimeError(
                        "DDP batch padding did not produce a world-size "
                        f"multiple: batches={len(child_batches)} "
                        f"world_size={self.world_size}")
                blocks.append(child_batches)
        # Dataset-level shuffle: permute the ORDER of the per-dataset blocks
        # each epoch.  Blocks stay contiguous, so the by-video locality inside
        # a block (video cache reuse) is untouched — the permutation itself
        # costs nothing.  Seeded by (seed, epoch): identical on every DDP
        # rank and reproducible across resumes.
        if self.shuffle_dataset_order and len(blocks) > 1:
            random.Random((self.seed << 20) ^ self.epoch).shuffle(blocks)
        self._batch_cache = [batch for block in blocks for batch in block]
        self._batch_cache_key = key
        return self._batch_cache

    def __iter__(self):
        batches = self._batches()[self.rank::self.world_size]
        return (batch.copy() for batch in batches)

    def __len__(self) -> int:
        return math.ceil(len(self._batches()) / self.world_size)


class SubstepForceDataModule(pl.LightningDataModule):
    """labels-v5 datamodule; the dataset roots themselves are the cache."""

    dataset_class = LabelStreamDataset

    def __init__(
        self,
        *,
        root: str = "",
        processed_root: Optional[str] = None,
        labels_root: Optional[str] = None,
        datasets: Optional[list[str]] = None,
        val_datasets: Optional[list[str]] = None,
        val_split_by_dataset: Optional[dict[str, str]] = None,
        split_manifest_file: Optional[str] = None,
        split_manifest_files_by_dataset: Optional[dict[str, str]] = None,
        training_statistics_file: Optional[str] = None,
        val_views_by_dataset: Optional[dict[str, list[str]]] = None,
        object_allowlist_file: Optional[str] = None,
        contact_only_datasets: Optional[list[str]] = None,
        handcrop_datasets: Optional[list[str]] = None,
        handcrop_both_hands: bool = False,
        target_fps_by_dataset: Optional[dict[str, str]] = None,
        target_fps: float = 10.0,
        sequence_length: int = 32,
        image_size: int = 256,
        video_backend: str = "torchcodec",
        video_cache_size: int = 48,
        torchcodec_seek_mode: str = "exact",
        torchcodec_num_ffmpeg_threads: int = 4,
        padding_mode: str = "static_edge",
        padding_placement: str = "right",
        padding_loss_weight: float = 1.0,
        train_split: str = "train",
        val_split: str = "test",
        views: Optional[list[str]] = None,
        train_max_episodes: Optional[int] = None,
        val_max_episodes: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 6,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sampler_mode: str = "sequential_by_video",
        dataset_sampling_power: float = 1.0,
        dataset_sampling: str = "power",
        dataset_weights: Optional[dict[str, float]] = None,
        shuffle_dataset_order: bool = True,
        window_stride: int = 16,
        train_window_mode: str = "phase_offset",
        train_clip_origin: str = "random_source_frame",
        val_clip_origin: str = "random_source_frame",
        native_timeline_root: Optional[str] = None,
        val_window_mode: str = "fixed",
        episode_subject_filter: str = "",
        window_sampling_seed: int = 42,
        window_include_tail: bool = True,
        uint8_images: bool = False,
        deduplicate_batch_frames: bool = False,
        target_mode: str = "applied",
        limit_val_batches: Optional[float] = None,
        # rows_rms (legacy): whole-row point-force RMS — deflated by the
        # 70-90% exactly-zero vertices and inflated by solver-blowup episodes.
        # nonzero_rms: RMS over nonzero clamped per-vertex magnitudes
        # (label_stats keys written by
        # the dataset preparation stage), so the
        # head's typical nonzero target sits near 1 in normalized units.
        point_force_scale_source: str = "rows_rms",
        # Per-epoch window quota scale applied uniformly to every dataset
        # (mixture ratios unchanged).  Validation always stays full-size so
        # epoch metrics stay comparable across fraction changes.
        epoch_window_fraction: float = 1.0,
        point_episode_cache_size: int = 2,
        substep_episode_cache_size: int = 2,
        episode_cache_bytes: int = 512 << 20,
        training_data_fast_path: bool = False,
        window_scoped_geometry_datasets: Sequence[str] | None = None,
        supervise_hand_visibility: bool = False,
        vertex_force_frame: str = "camera",
        predict_both_hands: bool = False,
        proximity_tau: float = 10.0,
        pad_label_mode: str = "repeat_edge",
        # Train-split randomness (val always keeps canonical order/frames):
        # per-epoch video-group permutation + within-record window shuffle
        # (both zero-cost for the LRU video-handle cache), and a clip-wise
        # brightness/contrast/saturation jitter applied at model resolution.
        shuffle_video_order: bool = True,
        shuffle_window_order: bool = True,
        photometric_augment: bool = True,
        rigid_body_reference_root: Optional[str] = None,
        require_rigid_body_references: bool = False,
        rollout_exclude_episodes: Optional[dict[str, list[int]]] = None,
    ) -> None:
        super().__init__()
        if int(sequence_length) != 32:
            raise ValueError("the formal experiment is fixed to 32 input frames")
        if float(padding_loss_weight) != 1.0:
            raise ValueError("padding and real frames must have equal loss weight (1.0)")
        if target_fps_by_dataset:
            raise ValueError("All datasets must use the model's configured input cadence")
        self.dataset_cfg = {
            "root": root, "target_fps": float(target_fps),
            "sequence_length": int(sequence_length),
            "image_size": int(image_size),
            "video_backend": video_backend,
            "video_cache_size": int(video_cache_size),
            "torchcodec_seek_mode": str(torchcodec_seek_mode),
            "torchcodec_num_ffmpeg_threads": max(1, int(torchcodec_num_ffmpeg_threads)),
            "padding": {"mode": padding_mode, "placement": padding_placement,
                        "loss_weight": float(padding_loss_weight)},
            "views": list(views) if views is not None else None,
            "window_stride": int(window_stride),
            "train_window_mode": str(train_window_mode),
            "train_clip_origin": str(train_clip_origin),
            "val_clip_origin": str(val_clip_origin),
            "native_timeline_root": native_timeline_root,
            "val_window_mode": str(val_window_mode),
            "episode_subject_filter": str(episode_subject_filter),
            "window_sampling_seed": int(window_sampling_seed),
            "window_include_tail": bool(window_include_tail),
            "uint8_images": bool(uint8_images),
            "deduplicate_batch_frames": bool(deduplicate_batch_frames),
            "handcrop_both_hands": bool(handcrop_both_hands),
            "episode_cache_bytes": int(episode_cache_bytes),
            "training_data_fast_path": bool(training_data_fast_path),
            "supervise_hand_visibility": bool(supervise_hand_visibility),
            "window_scoped_geometry_datasets": list(window_scoped_geometry_datasets or ()),
            "vertex_force_frame": str(vertex_force_frame),
            "predict_both_hands": bool(predict_both_hands),
            "proximity_tau": float(proximity_tau),
            "pad_label_mode": str(pad_label_mode),
            "shuffle_video_order": bool(shuffle_video_order),
            "shuffle_window_order": bool(shuffle_window_order),
            "photometric_augment": bool(photometric_augment),
            "object_allowlist_file": (str(object_allowlist_file)
                                      if object_allowlist_file else None),
            "contact_only_datasets": (list(contact_only_datasets)
                                      if contact_only_datasets else None),
        }
        self.dataset_cfg["split_manifest_file"] = split_manifest_file
        self.split_manifest_files_by_dataset = dict(split_manifest_files_by_dataset or {})
        if self.split_manifest_files_by_dataset:
            from training.data.split_manifest import load_manifest

            for name, path in self.split_manifest_files_by_dataset.items():
                if name not in load_manifest(path)["datasets"]:
                    raise ValueError(f"{path}: missing split mapping for {name}")
        self.dataset_cfg['rigid_body_reference_root'] = rigid_body_reference_root
        self.dataset_cfg['require_rigid_body_references'] = bool(require_rigid_body_references)
        self.dataset_cfg['rollout_exclude_episodes'] = dict(rollout_exclude_episodes or {})
        self.training_statistics_file = training_statistics_file
        if training_statistics_file and not split_manifest_file:
            raise ValueError("training_statistics_file requires a split_manifest_file")
        self.val_views_by_dataset = dict(val_views_by_dataset or {})
        self.labels_root = labels_root
        self.processed_root = processed_root
        # Per-dataset video-source switch: children in this set read the
        # hand-crop stream (label_data.py video_source="handcrop"); every
        # other child keeps full view videos.  Injected per child in
        # _build_dataset — the single-root path (no `datasets`) stays full.
        self.handcrop_datasets = frozenset(
            str(name) for name in (handcrop_datasets or ()))
        self.dataset_names = list(datasets) if datasets else []
        self.val_dataset_names = list(val_datasets) if val_datasets else None
        if (self.dataset_names or self.val_dataset_names) and not self.processed_root:
            raise ValueError(
                "Set PROCESSED_ROOT or data.processed_root to the directory containing "
                "<dataset>_lerobot children when data.datasets/val_datasets is configured.")
        self.val_split_by_dataset = dict(val_split_by_dataset or {})
        if deduplicate_batch_frames and not uint8_images:
            raise ValueError("frame deduplication requires uint8_images=true")
        self.train_split, self.val_split = train_split, val_split
        self.train_max_episodes = train_max_episodes
        self.val_max_episodes = val_max_episodes
        self.batch_size, self.num_workers = int(batch_size), int(num_workers)
        self.prefetch_factor, self.pin_memory = int(prefetch_factor), bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.sampler_mode = str(sampler_mode)
        self.dataset_sampling_power = float(dataset_sampling_power)
        self.dataset_sampling = str(dataset_sampling)
        self.dataset_weights = dict(dataset_weights) if dataset_weights else None
        self.shuffle_dataset_order = bool(shuffle_dataset_order)
        self.window_sampling_seed = int(window_sampling_seed)
        self.train_batch_sampler: HomogeneousLabelBatchSampler | SequentialLabelSampler | None = None
        if str(target_mode) != "applied":
            raise ValueError("substep training requires applied hand-wrench targets")
        self.target_mode = "applied"
        self.point_force_scale_source = str(point_force_scale_source)
        if self.point_force_scale_source not in {"rows_rms", "nonzero_rms"}:
            raise ValueError(
                "point_force_scale_source must be rows_rms|nonzero_rms, got "
                f"{point_force_scale_source!r}")
        self.epoch_window_fraction = float(epoch_window_fraction)
        if not 0.0 < self.epoch_window_fraction <= 1.0:
            raise ValueError(
                f"epoch_window_fraction must be in (0, 1], got "
                f"{epoch_window_fraction!r}")
        self.limit_val_batches = limit_val_batches
        self.train_ds: LabelStreamDataset | CombinedLabelDataset | None = None
        self.val_ds: LabelStreamDataset | CombinedLabelDataset | None = None
        self._video_reader: VideoReader | None = None
        self.wrench_scale: np.ndarray | None = None
        self.point_force_scale: float | None = None
        del point_episode_cache_size, substep_episode_cache_size  # legacy knobs

    def setup(self, stage: Optional[str] = None) -> None:
        del stage
        if self.train_ds is None:
            self.train_ds = self._build_dataset(
                self.train_split, self.train_max_episodes, for_validation=False)
        val_enabled = self.limit_val_batches is None \
            or float(self.limit_val_batches) != 0.0
        if val_enabled and self.val_ds is None:
            self.val_ds = self._build_dataset(
                self.val_split, self.val_max_episodes, for_validation=True)
        datasets = [dataset for dataset in (self.train_ds, self.val_ds)
                    if dataset is not None]
        if self._video_reader is None:
            paths = list(dict.fromkeys(
                path for dataset in datasets for path in dataset.video_paths))
            video_ids = {path: index for index, path in enumerate(paths)}
            self._video_reader = VideoReader(
                {index: path for path, index in video_ids.items()},
                cache_size=int(self.dataset_cfg["video_cache_size"]),
                backend=str(self.dataset_cfg["video_backend"]),
                image_mode="rgb",
                torchcodec_seek_mode=str(
                    self.dataset_cfg["torchcodec_seek_mode"]),
                torchcodec_num_ffmpeg_threads=int(
                    self.dataset_cfg["torchcodec_num_ffmpeg_threads"]))
            for dataset in datasets:
                dataset.attach_video_reader(self._video_reader, video_ids)
        stats = self._collect_stats(self.train_ds)
        self.wrench_scale = stats["wrench_scale"]
        assert self.wrench_scale is not None
        self.point_force_scale = stats["point_force_rms_n"]
        print(
            f"[labels-v5 DM] train_streams={len(self.train_ds.records)} "
            f"train_windows={len(self.train_ds)} "
            f"val_streams={len(self.val_ds.records) if self.val_ds else 0} "
            f"val_windows={len(self.val_ds) if self.val_ds else 0} "
            f"scale={self.wrench_scale.tolist()} "
            f"force_rms={self.point_force_scale:.6g} "
            f"({self.point_force_scale_source})", flush=True)

    def _collect_stats(self, dataset: Dataset) -> dict[str, Any]:
        if self.training_statistics_file:
            from training.data.split_manifest import manifest_sha256
            stats = json.loads(Path(self.training_statistics_file).read_text())
            if stats["split_sha256"] != manifest_sha256(self.dataset_cfg["split_manifest_file"]):
                raise ValueError("Training statistics belong to another split manifest")
            # JSON loses ndarray/scalar types. Restore the same contract as
            # the published-statistics branch before setup and batch collation.
            scale = np.asarray(stats["wrench_scale"], dtype=np.float32)
            rms = float(stats["point_force_rms_n"])
            if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("Training wrench_scale must contain six finite positive values")
            if not math.isfinite(rms) or rms <= 0:
                raise ValueError("Training point_force_rms_n must be finite and positive")
            return dict(stats, wrench_scale=scale, point_force_rms_n=rms)
        children = getattr(dataset, "datasets", [dataset])
        scales, squared, count = [], 0.0, 0
        nonzero_squared, nonzero_count = 0.0, 0
        for child in children:
            stats = child.label_stats
            rows = float(stats.get("n_rows", 0.0))
            rms = stats.get("point_force_rms_n")
            scale = stats.get("wrench_scale")
            # Contact-only sources carry no force statistics: they
            # contribute contact/proximity supervision, not force scaling.
            if rms is None or scale is None or rows <= 0:
                continue
            scales.append(np.asarray(scale, np.float64))
            rms = float(rms)
            squared += rows * rms * rms
            count += rows
            if self.point_force_scale_source == "nonzero_rms":
                nonzero_rms = stats.get("point_force_nonzero_rms_n")
                nonzero_rows = stats.get("point_force_nonzero_count")
                if nonzero_rms is None or not nonzero_rows:
                    raise ValueError(
                        f"{getattr(child, 'dataset_name', '?')}: "
                        "point_force_scale_source=nonzero_rms but label_stats "
                        "lacks point_force_nonzero_rms_n; use a prepared dataset "
                        "with nonzero-force statistics or select rows_rms"
                    )
                nonzero_squared += float(nonzero_rms) ** 2 * int(nonzero_rows)
                nonzero_count += int(nonzero_rows)
        if not scales:
            raise ValueError(
                "no force-capable dataset in the mixture: point-force "
                "scaling is undefined")
        point_force_scale = (
            float(np.sqrt(nonzero_squared / max(nonzero_count, 1)))
            if self.point_force_scale_source == "nonzero_rms"
            else float(np.sqrt(squared / max(count, 1))))
        return {
            # One global scale for the mixture: the element-wise median of the
            # per-dataset robust scales (exact for a single dataset).
            "wrench_scale": np.median(np.stack(scales), axis=0).astype(np.float32),
            "point_force_rms_n": max(point_force_scale, 1e-6),
        }

    def _build_dataset(self, split, max_episodes, *,
                       for_validation: bool | None = None) -> LabelStreamDataset | CombinedLabelDataset:
        use_val_datasets = for_validation
        if for_validation is None:
            for_validation = split != self.train_split
            use_val_datasets = for_validation and split == self.val_split
        dataset_cfg = dict(self.dataset_cfg, for_validation=for_validation)
        names = (self.val_dataset_names
                 if use_val_datasets and self.val_dataset_names is not None
                 else self.dataset_names)
        if not names:
            return self.dataset_class(dataset_cfg, split, max_episodes)
        if not self.processed_root:
            raise ValueError(
                "Set PROCESSED_ROOT or data.processed_root to the directory containing "
                "<dataset>_lerobot children when data.datasets is configured.")
        # episode_cache_bytes is a PER-DATASET ceiling inside each
        # LabelStreamDataset, and forked DataLoader workers each fill their
        # own copies: n_datasets x budget x num_workers.  The full mixture
        # at the 512MB default demanded ~47GB of cache alone on 4 workers
        # and got OOM-killed on cgroup-limited boxes (2026-09-14).  Split
        # the budget across children so each PROCESS keeps one global
        # ceiling; 16MB per child still covers several decoded episodes.
        per_child_cache = max(
            int(self.dataset_cfg.get("episode_cache_bytes", 512 << 20))
            // max(1, len(names)), 16 << 20)
        child_args = []
        for name in names:
            root_name = name if name.endswith("_lerobot") else f"{name}_lerobot"
            child_cfg = dict(dataset_cfg)
            child_cfg["dataset_name"] = name
            child_cfg["episode_cache_bytes"] = per_child_cache
            child_cfg["root"] = str(Path(self.processed_root) / root_name)
            split_manifest = self.split_manifest_files_by_dataset.get(name.removesuffix("_lerobot"))
            if split_manifest is not None:
                child_cfg["split_manifest_file"] = split_manifest
            child_cfg["video_source"] = ("handcrop" if name in self.handcrop_datasets
                                         else "full")
            if self.labels_root:
                child_cfg["labels_root"] = str(
                    Path(self.labels_root) / root_name)
            requested_split = (self.val_split_by_dataset.get(name, split)
                               if names is self.val_dataset_names else split)
            if names is self.val_dataset_names and name in self.val_views_by_dataset:
                child_cfg["views"] = list(self.val_views_by_dataset[name])
            child_args.append((name, child_cfg, requested_split))
        # Bound concurrent constructors per rank to avoid CPU/IO contention
        # when every DDP rank initializes the same collection.
        init_workers = int(os.environ.get('V2C_INIT_WORKERS', '2'))
        if init_workers < 1:
            raise ValueError('V2C_INIT_WORKERS must be positive')
        with ThreadPoolExecutor(max_workers=min(init_workers, len(child_args))) as pool:
            children = [
                child for child in pool.map(
                    lambda args: self._build_child(
                        *args, max_episodes=max_episodes), child_args)
                if child is not None]
        if not children:
            raise ValueError(f"no {split!r} samples in configured datasets {names}")
        return children[0] if len(children) == 1 else CombinedLabelDataset(children)

    def _build_child(self, name: str, child_cfg: dict,
                     split: str, max_episodes) -> Optional[LabelStreamDataset]:
        labels_root = Path(child_cfg.get("labels_root") or child_cfg["root"])
        try:
            if processed_contract.is_processed(labels_root):
                # Admission only needs two scalar fields, never physical
                # groups, tokens or reconstructed valid runs.
                reader = processed_contract.reader_for(labels_root)
                unified = reader.manifest.get('contracts', {}).get('unified_data')
                if ((unified and not reader.has_modality('tactile')) or
                        (not unified and 'tactile_context' not in reader.manifest['modalities'])):
                    index_rows = []
                else:
                    table = processed_contract.payload_table(labels_root, 'tactile_context',
                        ['topology', 'has_force'])
                    index_rows = table.group_by(['topology', 'has_force']).aggregate([]).to_pylist()
            else:
                index_rows = load_label_index(labels_root)
        except FileNotFoundError as error:
            print(f"[labels-v5 DM] SKIP {name}: {error}", flush=True)
            return None
        # Admission policy lives in ONE place: episode_training_role
        # (also drives the reader's record builders).  Mixed-topology
        # datasets (hoi4d) keep their eligible subset — LabelStreamDataset
        # skips unsupported episodes and reports the split; only datasets
        # with no eligible episode at all are dropped here.
        roles = {episode_training_role(row) for row in index_rows}
        roles.discard("skip")
        topologies = {str(row["topology"]) for row in index_rows}
        if not roles:
            print(
                f"[labels-v5 DM] SKIP {name}: topologies {sorted(topologies)} "
                "have no supported contact/force supervision",
                flush=True)
            return None
        if roles == {"contact_only"}:
            print(
                f"[labels-v5 DM] CONTACT-ONLY {name}: contact/proximity "
                "supervision, force losses masked",
                flush=True)
        child = self.dataset_class(child_cfg, split, max_episodes)
        child.dataset_name = name
        child.label_stats = load_label_stats(labels_root)
        return child

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        from training.data.rigid_scene import BODY_AXIS, collate_body_field
        result: dict[str, Any] = {}
        for key in batch[0]:
            if key in BODY_AXIS:
                result[key] = collate_body_field(key, [item[key] for item in batch])
                continue
            if (key == "images" and self.dataset_cfg.get("training_data_fast_path", False)
                    and self.dataset_cfg["deduplicate_batch_frames"]):
                # Gather only the frames the consumer will use, rather than
                # stacking [B,T,C,H,W] before discarding repeated frames.
                result[key] = None
                continue
            values = [item[key] for item in batch]
            result[key] = (torch.stack(values) if isinstance(values[0], torch.Tensor)
                           else values)
        if self.dataset_cfg["deduplicate_batch_frames"]:
            images = result["images"]
            frame_indices = result.pop("frame_indices")
            video_paths = result.pop("video_path")
            batch_size, time = frame_indices.shape
            first_position: dict[tuple[str, int], int] = {}
            unique_positions, inverse = [], []
            for batch_index, indices in enumerate(frame_indices.tolist()):
                video_path = str(video_paths[batch_index])
                for time_index, frame_index in enumerate(indices):
                    key = (video_path, frame_index)
                    if key not in first_position:
                        first_position[key] = len(unique_positions)
                        unique_positions.append(batch_index * time + time_index)
                    inverse.append(first_position[key])
            if images is None:
                result["images"] = torch.stack([
                    batch[position // time]["images"][position % time]
                    for position in unique_positions])
            else:
                flat = images.reshape(batch_size * time, *images.shape[2:])
                result["images"] = flat.index_select(
                    0, torch.tensor(unique_positions, dtype=torch.long))
            result["image_inverse"] = torch.tensor(
                inverse, dtype=torch.long).reshape(batch_size, time)
        assert self.wrench_scale is not None and self.point_force_scale is not None
        result["wrench_scale"] = torch.from_numpy(self.wrench_scale.copy())
        result["point_force_scale"] = torch.tensor(
            self.point_force_scale, dtype=torch.float32)
        return result

    def _loader(self, dataset, shuffle: bool, *, pad_final_batch: bool = True) -> DataLoader:
        if isinstance(dataset, CombinedLabelDataset):
            if self.sampler_mode != "sequential_by_video":
                raise ValueError("multi-dataset training requires sequential sampling")
            sampler = HomogeneousLabelBatchSampler(
                dataset, self.batch_size,
                self.dataset_sampling_power if shuffle else 1.0,
                dataset_sampling=self.dataset_sampling if shuffle else "power",
                dataset_weights=self.dataset_weights if shuffle else None,
                # Val keeps the canonical child order and the FULL epoch
                # quota; train may permute the dataset blocks every epoch
                # (seeded, rank-consistent) and scale its quota down.
                shuffle_dataset_order=self.shuffle_dataset_order and shuffle,
                epoch_window_fraction=(self.epoch_window_fraction
                                       if shuffle else 1.0),
                seed=self.window_sampling_seed, pad_final_batch=pad_final_batch)
            if shuffle:
                self.train_batch_sampler = sampler
            return DataLoader(
                dataset, batch_sampler=sampler,
                num_workers=self.num_workers, collate_fn=self._collate,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None)
        if self.sampler_mode != "sequential_by_video":
            raise ValueError("labels-v5 supports sequential_by_video sampling only")
        sampler = SequentialLabelSampler(dataset, drop_last=False)
        if shuffle and getattr(dataset, "random_source_clip_origin", False):
            self.train_batch_sampler = sampler
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler,
            num_workers=self.num_workers, collate_fn=self._collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=False)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self, *, pad_final_batch: bool = True):
        return self._loader(self.val_ds, False, pad_final_batch=pad_final_batch) if self.val_ds is not None else None

    def teardown(self, stage: Optional[str] = None) -> None:
        del stage
        if self._video_reader is not None:
            self._video_reader.close()
            self._video_reader = None
