"""Typed Hydra schemas for current TactoHand and HOPE data."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class SubstepForceDataConfig:
    """Three 30 Hz force targets for every 10 Hz video token (the tactohand task).

    Reads the labels-v5 contract (docs/labels_v5.md) directly from the dataset
    roots; no external stream/runtime caches exist anymore.
    """

    _target_: str = "training.data.datamodule.SubstepForceDataModule"

    root: str = ""
    processed_root: Optional[str] = None
    # Optional override pointing at a directory holding only meta/label_* +
    # label streams (transitional preview layout); defaults to each root.
    labels_root: Optional[str] = None
    datasets: Optional[list[str]] = None
    val_datasets: Optional[list[str]] = None
    val_split_by_dataset: dict[str, str] = field(default_factory=dict)
    split_manifest_file: Optional[str] = None
    split_manifest_files_by_dataset: dict[str, str] = field(default_factory=dict)
    training_statistics_file: Optional[str] = None
    rigid_body_reference_root: Optional[str] = None
    require_rigid_body_references: bool = False
    # Dataset name -> episodes kept under force/contact supervision but
    # excluded from multi-body rollout (no rigid reference required).
    rollout_exclude_episodes: dict[str, list[int]] = field(default_factory=dict)
    val_views_by_dataset: dict[str, list[str]] = field(default_factory=dict)
    # Optional JSON file mapping dataset name -> asset display names
    # ({"gigahands": ["flute_scan", ...]}); label runs whose target object
    # resolves (core.tactile_labels.assets join) outside the list are
    # dropped at record-build time.  Datasets without an entry are
    # unfiltered.  Human-picked tracking-quality curation lives outside
    # git so re-picking never touches the training code.
    object_allowlist_file: Optional[str] = None
    # Datasets whose runs train contact/proximity only: force supervision
    # and rollout are masked exactly like natively contact-only episodes
    # (topology "none"), while the rest of the mixture keeps force.
    contact_only_datasets: Optional[list[str]] = None
    # Datasets whose network input is the per-hand crop stream
    # (meta/cropped_video_observations.parquet + handcropped_videos/) instead
    # of the full view video: each window shows the dominant-present hand's
    # crop, the off-crop hand is supervised present-with-zeros, and roots
    # without the crop stream fail at dataset init.  Reader semantics live
    # in training/data/dataset.py (_handcrop_input_side /
    # _crop_presence_override); the review visualizer mirrors them.
    handcrop_datasets: Optional[list[str]] = None
    # Eval-only: expose synchronized left/right crop streams so a
    # single-hand baseline can be assembled into a scene-level prediction.
    handcrop_both_hands: bool = False
    target_fps: float = 10.0
    sequence_length: int = 32
    image_size: int = 256
    video_backend: str = "torchcodec"
    video_cache_size: int = 96
    # TorchCodec decode knobs (ignored by other backends). cfr_timestamp
    # uses explicit time lookup only on verified published CFR full videos;
    # unversioned/crop streams retain exact ordinal access. Bare approximate
    # ordinal access is not the formal training path.
    # num_ffmpeg_threads is per open handle, so total decode threads scale
    # as num_workers x this.
    torchcodec_seek_mode: str = "exact"
    torchcodec_num_ffmpeg_threads: int = 4
    padding_mode: str = "static_edge"
    padding_placement: str = "right"
    padding_loss_weight: float = 1.0
    train_split: str = "train"
    val_split: str = "test"
    # None means every synchronized exo view (8 views for canonical DexYCB).
    views: Optional[list[str]] = None
    train_max_episodes: Optional[int] = None
    val_max_episodes: Optional[int] = None
    batch_size: int = 1
    num_workers: int = 12
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    sampler_mode: str = "sequential_by_video"
    dataset_sampling_power: float = 1.0
    # Per-epoch dataset quota for training batches: power (size**power),
    # sqrt (size**0.5), or uniform (equal per dataset). dataset_weights is an
    # orthogonal per-dataset multiplier on that quota (missing names default
    # to 1.0): 1 keeps the quota, 0.1 shrinks that dataset's windows per
    # epoch and shortens the epoch; a rotating per-epoch start still walks
    # the whole dataset across epochs. Validation always stays
    # size-proportional so global metrics are unbiased.
    dataset_sampling: str = "power"
    dataset_weights: Optional[dict[str, float]] = None
    # Train-split order/photometric randomness (val keeps canonical
    # order and raw frames): per-epoch video-group permutation inside each
    # dataset (groups contiguous -> LRU handle cache unaffected),
    # within-record window shuffle, and clip-wise brightness/contrast/
    # saturation jitter at model resolution with letterbox borders
    # restored.
    shuffle_video_order: bool = True
    shuffle_window_order: bool = True
    photometric_augment: bool = True
    # Per-epoch permutation of the per-dataset batch blocks (train only).
    # Blocks stay contiguous so by-video cache locality is untouched; the
    # order is seeded by window_sampling_seed + epoch (rank-consistent,
    # reproducible). Validation keeps the canonical child order.
    shuffle_dataset_order: bool = True
    # Window-grid indexing: every epoch traverses all generated windows;
    # training may shift each view's start grid by a deterministic phase.
    window_stride: int = 16
    train_window_mode: str = "phase_offset"
    train_clip_origin: str = "random_source_frame"
    val_clip_origin: str = "random_source_frame"
    native_timeline_root: Optional[str] = None
    val_window_mode: str = "fixed"
    # Keep only episodes whose meta/episodes subject_id matches (case-
    # insensitive), e.g. "s05" reproduces EgoPHI's official ARCTIC
    # held-out-subject protocol.  Empty (default) = no filtering.
    episode_subject_filter: str = ""
    window_sampling_seed: int = 42
    window_include_tail: bool = True
    # Optional exact input-pipeline optimization: retain decoded pixels as
    # uint8 and encode repeated frames from overlapping windows only once per
    # batch.  Enabled selectively by data options that use sequential windows.
    uint8_images: bool = False
    deduplicate_batch_frames: bool = False
    # Physical labels remain applied/contact wrench; this chooses the head
    # parameterization and target used for normalization.
    target_mode: str = "applied"
    # Point-force head scale caliber: rows_rms (legacy whole-row RMS, deflated
    # by the exact-zero vertices and inflated by solver-blowup episodes) or
    # nonzero_rms (post-clamp nonzero-magnitude RMS from label_stats, written
    # by the dataset preparation stage).
    point_force_scale_source: str = "rows_rms"
    # Uniform per-epoch window quota scale (mixture ratios unchanged): 0.5
    # halves every dataset's per-epoch share and the epoch wall-clock; the
    # rotating per-epoch start walks fresh slices so consecutive epochs
    # still cover every window.  Raise max_epochs by 1/fraction to keep the
    # total data budget.  Validation always stays full-size.
    epoch_window_fraction: float = 1.0
    vertex_force_frame: str = "camera"  # camera | vertex_local
    predict_both_hands: bool = False
    # Proximity encoding is a task hyperparameter over the stored signed
    # distance: proximity = exp(-tau * clip(d, 0, 0.25)).
    proximity_tau: float = 10.0
    # Padding supervision: repeat_edge re-uses the nearest real token's
    # labels on pad frames (edge-held like the images) so static tails
    # stay in the loss; zero is the legacy unsupervised-padding mode.
    pad_label_mode: str = "repeat_edge"
    # Decoded-episode LRU budget per dataloader worker.
    episode_cache_bytes: int = 512 << 20
    training_data_fast_path: bool = False
    window_scoped_geometry_datasets: list[str] = field(default_factory=list)
    supervise_hand_visibility: bool = False
    # Legacy labels-v4 knobs accepted for config compatibility, unused.
    point_episode_cache_size: int = 2
    substep_episode_cache_size: int = 2

SUBSTEP_FORCE_SCHEMA_NAME = "substep_force_schema"
TASK_TO_DATA_SCHEMA = {
    "tactohand_wrench": (SubstepForceDataConfig,),
    "hope_baseline": (SubstepForceDataConfig,),
}


def register_data_schemas() -> None:
    cs = ConfigStore.instance()
    cs.store(group="data", name=SUBSTEP_FORCE_SCHEMA_NAME, node=SubstepForceDataConfig)


def validate_task_data_pairing(cfg: DictConfig) -> None:
    task = str(cfg.get("task", "tactohand_wrench") or "tactohand_wrench")
    accepted = TASK_TO_DATA_SCHEMA.get(task)
    if not accepted:
        raise ValueError(f"Unsupported task {task!r}; expected {sorted(TASK_TO_DATA_SCHEMA)}")
    actual = OmegaConf.get_type(cfg.data)
    if actual not in accepted:
        raise ValueError(
            f"task={task} received data schema {getattr(actual, '__name__', actual)!r}; "
            "use data=tactohand for TactoHand/HOPE")
