"""Standalone evaluation harness for TactoHand runs.

Reconstructs the model + validation datamodule from a complete YAML config or
a training run's ``.hydra/config.yaml``, runs inference over the selected
validation datasets, and reduces the selected metric families per dataset.

Design goals:

* parity — the forward path mirrors ``_shared_substep`` exactly (same
  ``_physical_outputs`` call with the gate logits, same clamped targets),
  so contact/proximity/force numbers reconcile with training TensorBoard;
* extensibility — metrics come from the registry (``--metrics`` selectors);
  adding a metric never touches this file;
* provenance — every report embeds run id, checkpoint, git rev and the
  eval-side knobs used.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from training.evaluation.registry import available_families, build_families
from training.evaluation.view import BatchView, MetricSpec
from training.evaluation.tasks import build_model, build_view, load_prediction_config
from training.evaluation.targets import _native_contact_target, _native_contact_valid

# Allow OmegaConf types in checkpoint loading (PyTorch 2.6+ security).
try:
    from omegaconf.base import Container, ContainerMetadata, Node
    from omegaconf.nodes import ValueNode

    torch.serialization.add_safe_globals([
        DictConfig, ListConfig, Container, ContainerMetadata, Node, ValueNode
    ])
except ImportError:  # pragma: no cover
    pass

PROXIMITY_CLIP_M = 0.25  # core.tactile_labels.schema.PROXIMITY_CLIP_M
_DECODE_EPS = 1e-6


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:
        return "unknown"


def resolve_checkpoint(run_dir: Optional[Path], ckpt: str) -> Path:
    """Accept ``last``/monitor-name prefixes or an explicit path."""
    ckpt_path = Path(ckpt).expanduser()
    if ckpt_path.is_file():
        return ckpt_path.resolve()
    if run_dir is None or ckpt_path.is_absolute():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}; --config requires an explicit --ckpt file")
    candidates = sorted((run_dir / "checkpoints").glob(f"{ckpt}*.ckpt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        available = [p.name for p in (run_dir / "checkpoints").glob("*.ckpt")]
        raise FileNotFoundError(
            f"no checkpoint matching {ckpt!r} in {run_dir}/checkpoints; "
            f"available: {available}")
    raise FileNotFoundError(
        f"ambiguous checkpoint prefix {ckpt!r}: "
        f"{[p.name for p in candidates]}")






def build_external_view(outputs: Dict[str, torch.Tensor],
                        batch: Dict[str, Any],
                        fallback_dataset: Optional[str] = None,
                        tau: float = 10.0,
                        fps: float = 10.0) -> BatchView:
    """BatchView from an external predictor (HACO, ...).

    Only ``contact_prob`` [B,T,H,V] is required in ``outputs``.  Gate/cond
    stages and all force/wrench/proximity slots are mirrors or zeros: the
    contact family reads real numbers from ``contact_prob`` vs the shared
    GT, while every other family scores a trivial predictor and reports
    whatever that yields — visible in the report, never silently dropped.
    """
    device = torch.device("cpu")
    b = int(batch["real_mask"].shape[0])
    t = int(batch["real_mask"].shape[1])
    if "contact_prob" in outputs:
        contact_prob = outputs["contact_prob"].float()
    else:
        # Proximity-only baselines (interfield) carry no contact field:
        # zeros keep the view well-formed (contact metrics then read as a
        # trivial all-zero predictor — evaluate with --metrics proximity).
        contact_prob = torch.zeros(b, t, 2, 778)
    real = batch["real_mask"].to(device).bool().unsqueeze(-1)
    if "metric_sample_valid" in batch:
        sample_valid = batch["metric_sample_valid"].to(device).bool()
        real = real & sample_valid.view(-1, 1, 1)
    # 0.5-cap gate so cond_prob == contact_prob: the contact family's
    # P/R/F1 read contact_prob directly either way, and the gate/cond
    # diagnostics stay well-defined (degenerate, flat 1).
    gate_prob = torch.ones(b, t, 1, device=device)
    cond_prob = contact_prob
    target = _native_contact_target(batch, device)
    valid = _native_contact_valid(batch, device) & real
    gate_target = valid & (target.sum(dim=-1) > 0)

    if "proximity_prob" in outputs:
        # Real proximity fields (interfield): decoded with the shared
        # encoding so the proximity family scores model vs GT.
        proximity_prob = outputs["proximity_prob"].float()
    else:
        # GT mirror: contact-only baselines carry no proximity field; the
        # proximity family then scores a trivial predictor (visible in the
        # report, never silently dropped).
        proximity_prob = batch["proximity_target_10hz"].to(device).float()
    proximity_target = batch["proximity_target_10hz"].to(device).float()
    proximity_valid = (batch["proximity_valid_10hz"].to(device).bool()
                       & real).unsqueeze(-1)
    # Crop-side gating for handcrop windows: the off-crop hand is a label
    # convention (forced present, GT distance = inf -> target pinned at the
    # 250 mm saturation) that trained models learn and external baselines
    # have never seen — an InterField cannot output beyond its 100 mm clamp.
    # Score external proximity fields on the crop-side hand only.
    crop_side = batch.get("crop_side")
    if "proximity_prob" in outputs and crop_side is not None:
        hand_valid = torch.ones(b, t, 2, 1, dtype=torch.bool)
        for i, side in enumerate(
                crop_side.to("cpu").long().reshape(-1).tolist()):
            if side >= 0:
                hand_valid[i, 1 - side] = False
        proximity_valid = proximity_valid & hand_valid
    sat = math.exp(-tau * PROXIMITY_CLIP_M)
    # Decode with the SAME tau that encoded the fields: the run's label
    # pipeline (and the interfield adapter) encode prob = exp(-tau*d) with
    # cfg.data.proximity_tau — a hardcoded 10 here would silently rescale
    # both sides whenever a run picks another tau (20 on the final rounds).
    distance_mm_hat = -torch.log(proximity_prob.clamp(min=_DECODE_EPS)) \
        / tau * 1000.0
    # GT decoding saturates at the clip bound; exact zeros (absent-hand
    # present-with-zeros supervision) map to the same bound.
    distance_mm_gt = -torch.log(
        proximity_target.clamp(min=sat, max=1.0)) / tau * 1000.0

    source_frames = batch.get("source_frames")
    if source_frames is None:
        source_frames = -torch.arange(
            t, device=device, dtype=torch.float32).expand(b, -1)
    else:
        source_frames = source_frames.to(device)

    force = torch.zeros_like(batch["vertex_force_camera_30hz"])
    target_force = batch["vertex_force_camera_30hz"].to(device).float()
    force_gate = batch.get("force_prediction_valid_30hz", batch["force_valid_30hz"]).to(device).bool()
    force_gate = force_gate.reshape(b, force_gate.shape[1], -1)
    # Parity with build_view: score only the image-aligned substep 0.
    aligned = torch.zeros_like(force_gate)
    aligned[:, :, 0] = True
    real30 = batch["real_mask_30hz"].to(device).bool().reshape_as(force_gate)
    force_mask = force_gate & real30 & aligned
    magnitude = target_force.norm(dim=-1)
    active = torch.zeros_like(magnitude, dtype=torch.bool)

    hand = torch.zeros_like(batch["hand_wrench_camera_30hz"])
    target_hand = batch["hand_wrench_camera_30hz"].to(device).float()

    names = batch.get("dataset_name")
    if names is None:
        if fallback_dataset is None:
            raise ValueError(
                "batch carries no dataset_name and no fallback configured")
        names = [fallback_dataset]
    unique = set(map(str, names))

    return BatchView(
        dataset_name=next(iter(unique)),
        contact_prob=contact_prob,
        cond_prob=cond_prob,
        gate_prob=gate_prob,
        contact_target=target,
        contact_mask=valid.unsqueeze(-1),
        gate_target=gate_target,
        proximity_prob=proximity_prob,
        proximity_target=proximity_target,
        proximity_mask=proximity_valid,
        distance_mm_hat=distance_mm_hat,
        distance_mm_gt=distance_mm_gt,
        source_frames=source_frames,
        real_tokens=real.squeeze(-1),
        force_pred=force,
        force_target=target_force,
        force_mask=force_mask,
        active_mask=active,
        hand_pred=hand,
        hand_target=target_hand,
        force_supervised=bool(batch.get(
            "force_supervision_valid", torch.tensor(True)).bool().all()),
        batch_size=int(batch["real_mask"].shape[0]),
        tau_per_m=tau,
        fps=fps,
    )




class EvalHarness:
    """Run selected metric families over selected val datasets."""

    def __init__(self, run_dir: Optional[Path], checkpoint: Path,
                 datasets: List[str], metric_selectors: List[str],
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 device: str = "cuda:0", limit_batches: int = 0,
                 precision: Optional[str] = None,
                 views: Optional[List[str]] = None,
                 visualize_windows: int = 0,
                 viz_dir: Optional[Path] = None,
                 split: str = "val",
                 split_by_dataset: Optional[Dict[str, str]] = None,
                shard_index: int = 0, shard_count: int = 1,
                predictor: str = "tactohand",
                input_source: Optional[str] = None,
                subject_filter: Optional[str] = None,
                 viz_modalities: Optional[List[str]] = None,
                 config_path: Optional[Path] = None,
                 processed_root: Optional[Path] = None,
                 labels_root: Optional[Path] = None,
                 rigid_body_reference_root: Optional[Path] = None) -> None:
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("Select --datasets or provide data.val_datasets in the config")
        self.metric_selectors = list(metric_selectors)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.limit_batches = int(limit_batches)
        self.precision = precision
        self.views = list(views) if views else None
        self.visualize_windows = int(visualize_windows)
        self.viz_dir = viz_dir
        self.split = str(split)
        self.split_by_dataset = dict(split_by_dataset or {})
        # process-level sharding: keep only every shard_count-th batch
        # (indices mod shard_count == shard_index) so concurrent
        # processes split the eval; per-dataset metrics are computed
        # over the shard only — recombine by running the shards' frames
        # as one collection or treating each shard as a split sample.
        self.shard_index = int(shard_index)
        self.shard_count = max(int(shard_count), 1)
        # Episode subject filter (e.g. 's05' reproduces EgoPHI's official
        # ARCTIC held-out-subject split); '' = no filtering.
        self.subject_filter = str(subject_filter or "")
        # Prediction source: 'tactohand' = the run checkpoint; 'haco' = an
        # external HACO release model scored on the same GT (hand-crop
        # inputs forced on, see _build_loader).
        self.predictor_kind = str(predictor)
        # Input override: None = native behavior (full views for the run
        # checkpoint, hand crops for external single-hand predictors);
        # 'handcrop' = force the crop stream even for the run checkpoint
        # (baseline-aligned-input ablation).
        external = self.predictor_kind != "tactohand"
        if input_source is None:
            self.force_handcrop = external
        elif str(input_source) == "handcrop":
            self.force_handcrop = True
        elif str(input_source) == "full":
            self.force_handcrop = False
        else:
            raise ValueError(f"unknown input_source {input_source!r}")
        # Rendered modalities default to whatever --metrics selected; an
        # explicit --viz-modalities narrows or extends that (e.g. render
        # proximity panels while computing contact metrics only).
        self.viz_modalities = (
            [m for m in viz_modalities if m]
            if viz_modalities
            else None)

        if self.shard_count > 1:
            self._extra_config = {"shard": f"{self.shard_index}/{self.shard_count}"}
        else:
            self._extra_config = {}
        if config_path is None:
            if run_dir is None:
                raise ValueError("Provide config_path or run_dir")
            config_path = run_dir / ".hydra" / "config.yaml"
        self.config_path = config_path.expanduser().resolve()
        self.cfg = load_prediction_config(self.config_path)
        if processed_root is not None:
            OmegaConf.update(self.cfg, "data.processed_root", str(processed_root.expanduser().resolve()), merge=False)
        if labels_root is not None:
            OmegaConf.update(self.cfg, "data.labels_root", str(labels_root.expanduser().resolve()), merge=False)
        if rigid_body_reference_root is None and os.environ.get("TACTOHAND_RIGID_REFERENCE_ROOT"):
            rigid_body_reference_root = Path(os.environ["TACTOHAND_RIGID_REFERENCE_ROOT"])
        if rigid_body_reference_root is not None:
            OmegaConf.update(self.cfg, "data.rigid_body_reference_root",
                             str(rigid_body_reference_root.expanduser().resolve()), merge=False)
        else:
            self._resolve_default_rigid_references()

    def _resolve_default_rigid_references(self) -> None:
        from core.data import processed_compat

        protocol = Path(__file__).resolve().parents[2] / "artifacts/training_protocol/rebuttal_v1"
        reference_roots = {
            "meta/index/manifest.native_timeline_20261112_sealed.json":
                protocol / "native_timeline_20261112/rigid_references_sealed",
            "meta/index/manifest.native_timeline_cleanup_20261112.json":
                protocol / "native_timeline_cleanup_20261112/rigid_references",
            "meta/index/manifest.tactile_compact_20261112.json":
                protocol / "tactile_compact_20261112/rigid_references",
        }
        configured = self.cfg.data.get("rigid_body_reference_root")
        defaults = {protocol / "rigid_references", *reference_roots.values()}
        if configured and Path(configured).resolve() not in defaults:
            return
        prepared = self.cfg.data.get("labels_root") or self.cfg.data.get("processed_root")
        if not prepared:
            return
        root = Path(prepared) / f"{self.datasets[0].removesuffix('_lerobot')}_lerobot"
        if not (root / "meta/info.json").is_file() or not processed_compat.is_processed(root):
            return
        reader = processed_compat.reader_for(root)
        reference_root = reference_roots.get(reader.manifest_path)
        if reference_root is None:
            return
        OmegaConf.update(self.cfg, "data.rigid_body_reference_root",
                         str(reference_root), merge=False)

    # -- construction -----------------------------------------------------

    def _phase(self, label: str, started: float) -> float:
        now = time.perf_counter()
        print(f"[eval-tactohand] {label}: {now - started:.1f}s", flush=True)
        return now

    def _build_model(self):
        return build_model(self.cfg, self.checkpoint, self.device, self._phase)

    def _build_loader(self):
        data_cfg = self.cfg.data
        configured = list(data_cfg.get("val_datasets", []) or [])
        unknown = [d for d in self.datasets
                   if d not in configured
                   and d not in (data_cfg.get("datasets") or [])]
        if unknown:
            raise ValueError(
                f"datasets {unknown} are neither in the run's val_datasets "
                f"({configured}) nor its training datasets "
                f"({data_cfg.get('datasets')}); refusing to guess their "
                f"val splits")
        OmegaConf.update(self.cfg, "data.val_datasets", self.datasets,
                         merge=False)
        if self.split_by_dataset:
            # Explicit NAME=SPLIT overrides (already merged over the run
            # mapping in the val branch below); a dataset only present in
            # the training mix needs one to be evaluable.
            mapping = dict(data_cfg.get("val_split_by_dataset", {}) or {})
            mapping.update(self.split_by_dataset)
            data_cfg.val_split_by_dataset = mapping
        if self.batch_size is not None:
            OmegaConf.update(self.cfg, "data.batch_size", self.batch_size,
                             merge=False)
        if self.num_workers is not None:
            OmegaConf.update(self.cfg, "data.num_workers", self.num_workers,
                             merge=False)
        if self.views is not None:
            # Eval-side only: restricts every selected dataset to these
            # view_ids (record build just skips other views).
            OmegaConf.update(self.cfg, "data.views", self.views, merge=False)
        if self.split != "val":
            # Train-split inspection: retarget the val loader at the run's
            # train split.  val_split_by_dataset must go (it would remap
            # arctic/dexycb back to their per-dataset val splits), and the
            # reader's train-only photometric jitter + window shuffling must
            # be off so inspection sees the clean deterministic inputs.
            OmegaConf.update(self.cfg, "data.val_split", self.split,
                             merge=False)
            OmegaConf.update(self.cfg, "data.val_split_by_dataset", {},
                             merge=False)
            OmegaConf.update(self.cfg, "data.photometric_augment", False,
                             merge=False)
            OmegaConf.update(self.cfg, "data.shuffle_window_order", False,
                             merge=False)
            OmegaConf.update(self.cfg, "data.shuffle_video_order", False,
                             merge=False)
        elif self.split_by_dataset:
            # Per-dataset split override (e.g. hocap on its official test
            # split instead of the run's configured "all").  Merged over the
            # run config so unspecified datasets keep their own mapping.
            mapping = dict(data_cfg.get("val_split_by_dataset", {}) or {})
            mapping.update(self.split_by_dataset)
            OmegaConf.update(self.cfg, "data.val_split_by_dataset", mapping,
                             merge=False)
        OmegaConf.update(self.cfg, "data.episode_subject_filter",
                         self.subject_filter, merge=False)
        if self.force_handcrop:
            # Single-hand predictors (and --input-source handcrop) consume
            # hand crops, not full views: force the selected datasets onto
            # the hand-crop video stream for this eval regardless of the
            # training mix.
            handcrop = set(data_cfg.get("handcrop_datasets") or [])
            handcrop.update(self.datasets)
            OmegaConf.update(self.cfg, "data.handcrop_datasets",
                             sorted(handcrop), merge=False)
        if self.predictor_kind in {"haco", "contactprompt"}:
            # HACO is a single-hand crop model.  Scene-level evaluation runs
            # it independently on both benchmark-provided crops and assembles
            # the two outputs; native hand presence still decides the fixed
            # scoring denominator for one-hand datasets.
            OmegaConf.update(self.cfg, "data.handcrop_both_hands", True,
                             merge=False, force_add=True)
            # The eval-only [hand,time,...] stream is intentionally kept
            # simple; primary-stream dedup remains unchanged elsewhere.
            OmegaConf.update(self.cfg, "data.deduplicate_batch_frames", False,
                             merge=False)
        datamodule = hydra.utils.instantiate(self.cfg.data)
        self._skip_train_build(datamodule, strict_stats=self.run_dir is None)
        t0 = time.perf_counter()
        datamodule.setup("validate")
        t0 = self._phase("datamodule setup", t0)
        # Standalone reports score each window once, including short batches.
        loader = datamodule.val_dataloader(pad_final_batch=False)
        from core.data import processed_compat
        self._extra_config["point_force_scale_n"] = datamodule.point_force_scale
        self._extra_config["prepared_releases"] = {}
        for child in getattr(loader.dataset, "datasets", [loader.dataset]):
            if processed_compat.is_processed(child.labels_root):
                reader = processed_compat.reader_for(child.labels_root)
                self._extra_config["prepared_releases"][child.dataset_name] = {
                    "manifest": reader.manifest_path,
                    "revision": reader.revision,
                    "sampling": ("native_source_frames" if child.native_solver_timeline
                                 else "published_label_timeline"),
                }
        print(f"[eval-tactohand] val windows: {len(loader.dataset)}",
              flush=True)
        return loader

    @staticmethod
    def _skip_train_build(datamodule, *, strict_stats: bool = False) -> None:
        """Pre-set ``train_ds`` so setup() skips the 23-dataset train build.

        setup() needs exactly two things from the train side: per-dataset
        ``label_stats`` (drives the global point-force / wrench scales via
        ``_collect_stats``) and the video paths.  The stats are dataset-level
        aggregates read straight from the labels tree — episode selection
        plays no role — so a stats-only stub reproduces training's scales
        bit-for-bit while skipping window indexing and video readers.
        """
        from core.tactile_labels.decode import load_label_stats

        class _StatsOnlyTrain:
            """Duck-types the two introspections setup() performs on it."""

            def __init__(self, children):
                self.datasets = children
                self.video_paths = []
                self.records = []

            def __len__(self) -> int:
                return 0

            def attach_video_reader(self, *_args, **_kwargs) -> None:
                pass

        labels_root = Path(datamodule.labels_root) if datamodule.labels_root \
            else Path(datamodule.processed_root)
        children = []
        for name in datamodule.dataset_names:
            root_name = name if name.endswith("_lerobot") else f"{name}_lerobot"
            try:
                stats = load_label_stats(labels_root / root_name)
            except FileNotFoundError as error:
                if strict_stats:
                    raise FileNotFoundError(
                        f"Explicit-config evaluation needs the original training mixture's "
                        f"normalization statistics; missing label_stats for {name!r} under "
                        f"{labels_root}. Provide those metadata rather than changing "
                        "data.datasets and silently recomputing a different force scale.") from error
                continue  # mirrors _build_child's SKIP policy
            children.append(type("_Child", (), {
                "dataset_name": name, "label_stats": stats})())
        datamodule.train_ds = _StatsOnlyTrain(children)

    # -- execution ---------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        torch.set_float32_matmul_precision(
            str(self.cfg.training.get("matmul_precision", "highest")) or
            "highest")
        model = self._build_model() if self.predictor_kind == "tactohand" \
            else None
        from training.evaluation.predictors import build_predictor
        predictor = build_predictor(self.predictor_kind, model, self.cfg,
                                    self.datasets, self.device)
        print(f"[eval-tactohand] predictor: {predictor.name}", flush=True)
        loader = self._build_loader()
        tau = float(self.cfg.data.get("proximity_tau", 10.0))
        fps = float(self.cfg.data.get("target_fps", 10.0))

        families: Dict[str, List] = {
            ds: build_families(self.metric_selectors) for ds in self.datasets}
        counts: Dict[str, int] = defaultdict(int)
        batches = 0
        count_viz: Dict[str, int] = defaultdict(int)
        viz = None
        if self.visualize_windows > 0:
            from training.visualization.predictions import ComparisonRenderer, modalities_from_metrics
            modalities = (self.viz_modalities
                          or modalities_from_metrics(self.metric_selectors)
                          or ["proximity"])
            viz = ComparisonRenderer(
                processed_root=self.cfg.data.processed_root,
                labels_root=(self.cfg.data.get("labels_root")
                             or self.cfg.data.processed_root),
                handcrop_datasets=list(
                    self.cfg.data.get("handcrop_datasets") or []),
                device=str(self.device),
                tau=tau,
                modalities=modalities,
                video_backend=(str(self.cfg.data.get("video_backend"))
                               or None))
            print(f"[eval-tactohand] viz modalities: {modalities}",
                  flush=True)

        amp_dtype = None
        precision = (self.precision
                     or str(self.cfg.training.get("precision", "32")))
        if self.device.type == "cuda" and "bf16" in precision \
                and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16

        total_batches: Optional[int] = None
        try:
            total_batches = len(loader)  # type: ignore[arg-type]
        except Exception:
            total_batches = None
        if self.limit_batches > 0 and total_batches is not None:
            total_batches = min(total_batches, self.limit_batches)

        pbar = None
        if tqdm is not None:
            pbar = tqdm(loader, total=total_batches, desc="eval", unit="batch",
                        dynamic_ncols=True, smoothing=0.1,
                        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                    "[{elapsed}<{remaining}, {rate_fmt}]"))
        elif not self.limit_batches:
            print(f"[eval-tactohand] evaluating (batches={total_batches})",
                  flush=True)

        with torch.no_grad():
            data_wait = forward_s = metrics_s = 0.0
            fetch_started = time.perf_counter()
            batch_iter = loader if pbar is None else pbar
            for batch_idx, batch_cpu in enumerate(batch_iter):
                if self.shard_count > 1 and (
                        batch_idx % self.shard_count != self.shard_index):
                    continue
                if self.limit_batches and batch_idx >= self.limit_batches:
                    break
                data_wait += time.perf_counter() - fetch_started
                batch = {
                    k: (v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v)
                    for k, v in batch_cpu.items()
                }
                if self.predictor_kind != "tactohand" and len(
                        self.datasets) == 1 and "dataset_name" not in batch:
                    # Single-child loaders omit dataset_name; external
                    # predictors need it to pick their per-dataset assets.
                    batch["_fallback_dataset"] = self.datasets[0]
                forward_started = time.perf_counter()
                with torch.autocast(device_type=self.device.type,
                                    dtype=amp_dtype, enabled=amp_dtype
                                    is not None):
                    outputs = predictor.predict_batch(batch)
                predictions = outputs.get("predictions")
                forward_s += time.perf_counter() - forward_started
                if predictions is not None:
                    force, hand, contact = predictor.physical_outputs(predictions, batch)
                else:
                    # External baselines carry no force/wrench heads: feed
                    # zero fields so _physical_outputs-free view assembly
                    # still binds the wrench slots (wrench families then
                    # score against a trivial predictor — reported, not
                    # hidden, per the paper's dash-not-imputed rule the
                    # families themselves decide what to emit).
                    force = torch.zeros_like(
                        batch["vertex_force_camera_30hz"])
                    hand = torch.zeros_like(batch["hand_wrench_camera_30hz"])
                    contact = None
                metrics_started = time.perf_counter()
                if predictions is not None:
                    view = predictor.metric_view(predictions, batch,
                                      force.float(), hand.float(), tau, fps,
                                      fallback_dataset=(
                                          self.datasets[0]
                                          if len(self.datasets) == 1 else None))
                else:
                    # External predictor path: contact probs come straight
                    # from the predictor dict; gate/cond/proximity slots
                    # degrade to the deployed-prob decomposition the
                    # contact family needs (or None-masked, per family).
                    view = build_external_view(outputs, batch,
                                               fallback_dataset=(
                                                   self.datasets[0]
                                                   if len(self.datasets) == 1
                                                   else None),
                                               tau=tau, fps=fps)
                if contact is not None and "rollout" \
                        in self.metric_selectors:
                    # Compute only when the rollout family is selected: the
                    # FP64 implicit-midpoint rollout is pure overhead
                    # otherwise.  Attaches per-(horizon, key) sums for the
                    # family to pool; external predictors carry no wrench
                    # head, so they never produce rollout numbers.
                    from training.evaluation.families_rollout import compute_rollout_sums
                    view.extras["rollout"] = compute_rollout_sums(
                        model, contact, batch, force=force)
                for family in families[view.dataset_name]:
                    family.update(view)
                counts[view.dataset_name] += view.batch_size
                batches += 1
                metrics_s += time.perf_counter() - metrics_started
                if pbar is not None:
                    forward_ms = 1000.0 * forward_s / batches
                    pbar.set_postfix_str(
                        f"{view.dataset_name} · {sum(counts.values())} samples"
                        f" · fwd {forward_ms:.0f} ms/batch")
                elif batches % 50 == 0 or batches == 1:
                    print(f"[eval-tactohand] batch {batches}/{total_batches}: "
                          f"data_wait={data_wait:.1f}s forward={forward_s:.1f}s "
                          f"metrics={metrics_s:.1f}s", flush=True)
                if viz is not None and count_viz[view.dataset_name] \
                        < self.visualize_windows:
                    # Batches are dataset-homogeneous; render in-batch
                    # samples until the per-dataset quota is met.
                    quota = self.visualize_windows - count_viz[view.dataset_name]
                    # Window count, NOT images.shape[0]: with
                    # deduplicate_batch_frames the images tensor holds the
                    # batch's UNIQUE frames ([unique, C, H, W]), which
                    # exceeds the sample count and made the render loop
                    # index past per-sample fields (episode_index, ...).
                    batch_size = len(batch["episode_index"])
                    for i in range(min(quota, batch_size)):
                        self._render_window(view, batch, i, viz)
                        count_viz[view.dataset_name] += 1
                fetch_started = time.perf_counter()
            if pbar is not None:
                pbar.close()
        print(f"[eval-tactohand] loop done: batches={batches} "
              f"data_wait={data_wait:.1f}s forward={forward_s:.1f}s "
              f"metrics={metrics_s:.1f}s", flush=True)
        if viz is not None:
            viz.write_index(self._viz_root())
        # Anti-silent-degradation: predictors that gate frames (egophi's
        # valid-frame filter) report how many frames actually reached the
        # forward.  A low coverage means most frames were scored as empty
        # predictions — the report stays green while the eval is hollow
        # (2026-09-22 hocap handcrop lesson).
        coverage = predictor.coverage() if hasattr(predictor, "coverage") \
            else None
        if coverage is not None:
            self._predictor_coverage = coverage
            pct = 100 * coverage["coverage"]
            print(f"[eval-tactohand] predictor forward coverage: "
                  f"{coverage['gnn_frames']:.0f}/{coverage['offered_frames']:.0f}"
                  f" frames ({pct:.1f}%), "
                  f"{coverage['skipped_windows']:.0f} windows skipped",
                  flush=True)
            if pct < 50:
                print(f"[eval-tactohand] WARNING: predictor forward coverage "
                      f"{pct:.1f}% — MOST frames were scored as EMPTY "
                      f"predictions; the metrics below are NOT a faithful "
                      f"eval (cf. the 2026-09-22 hocap handcrop run)",
                      flush=True)
            elif pct < 95:
                print(f"[eval-tactohand] note: {100 - pct:.1f}% of frames "
                      f"were gated out (no object metadata/mask) and scored "
                      f"as empty predictions", flush=True)
        report = self._report(families, counts, batches, started)
        return report

    def _viz_root(self) -> Path:
        # Namespace by checkpoint + split so a second eval (other split,
        # checkpoint, or metric mix) can never overwrite earlier renders.
        base = self.viz_dir if self.viz_dir is not None \
            else ((self.run_dir / "eval") if self.run_dir is not None
                  else Path.cwd() / "outputs" / "eval") / "viz"
        return Path(base) / (
            f"{self.checkpoint.stem.split('-')[0]}_{self.split}")

    def _render_window(self, view, batch: Dict[str, Any], sample_index: int,
                       viz) -> None:
        """Render one GT-vs-pred window; failures never abort the eval."""
        try:
            records = viz.render(
                view, batch, sample_index, view.dataset_name,
                self._viz_root() / view.dataset_name)
            for record in records:
                print(f"[eval-tactohand] viz {view.dataset_name}/"
                      f"{record['modality']} ep{record['episode']} -> "
                      f"{record['sheet']}", flush=True)
        except Exception as error:  # noqa: BLE001 — viz is best-effort
            import traceback
            print(f"[eval-tactohand] viz FAILED ({view.dataset_name}): "
                  f"{error!r}", flush=True)
            traceback.print_exc()

    # -- reporting -----------------------------------------------------------

    def _report(self, families, counts, batches, started) -> Dict[str, Any]:
        specs = {}
        for family in next(iter(families.values()), []):
            specs.update({s.key: s for s in family.specs()})
        per_dataset: Dict[str, Dict[str, Any]] = {}
        paper: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
        for ds, family_list in families.items():
            block: Dict[str, Any] = {}
            for family in family_list:
                for key, value in family.compute().items():
                    spec = specs.get(key, MetricSpec(key, ""))
                    block[key] = {
                        "value": float(value),
                        "unit": spec.unit,
                        "note": spec.note,
                    }
                    if spec.paper_table:
                        table = paper.setdefault(spec.paper_table, {})
                        table.setdefault(ds, {})[
                            spec.paper_column or key] = float(value)
            per_dataset[ds] = block
        return {
            "run": str(self.run_dir) if self.run_dir is not None else None,
            "run_name": self.run_dir.name if self.run_dir is not None else self.config_path.stem,
            "config_path": str(self.config_path),
            "checkpoint": str(self.checkpoint),
            "predictor": self.predictor_kind,
            "git_rev": _git_rev(),
            "created": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.perf_counter() - started, 1),
            "config": {
                "datasets": self.datasets,
                "metrics": self.metric_selectors,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "limit_batches": self.limit_batches,
                "precision": self.precision,
                "views": self.views,
                "split": self.split, **getattr(self, "_extra_config", {}),
                "split_by_dataset": self.split_by_dataset or None,
                "processed_root": self.cfg.data.get("processed_root"),
                "labels_root": self.cfg.data.get("labels_root"),
                "rigid_body_reference_root": self.cfg.data.get("rigid_body_reference_root"),
                "normalization_stats_policy": ("require_original_training_mixture" if self.run_dir is None
                                               else "legacy_skip_missing"),
            },
            "counts": {"batches": batches,
                       "samples_per_dataset": dict(counts)},
            "predictor_coverage": getattr(self, "_predictor_coverage", None),
            "per_dataset": per_dataset,
            "paper": {
                "primary_metric": "contact/f1_macro",
                "auxiliary_metrics": [
                    "contact/f1_micro",
                    "contact/localization_f1",
                    "contact/localization_f1_haco",
                ],
                "tables": {k: v for k, v in paper.items()},
            } if "contact" in self.metric_selectors else {
                "tables": {k: v for k, v in paper.items()},
            },
        }


def write_report(report: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def format_console(report: Dict[str, Any]) -> str:
    lines = []
    for ds, block in sorted(report["per_dataset"].items()):
        lines.append(f"\n[{ds}]  ({report['counts']['samples_per_dataset'].get(ds, 0)} samples)")
        for key in sorted(block):
            entry = block[key]
            value = entry["value"]
            text = f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
            unit = f" {entry['unit']}" if entry["unit"] else ""
            note = f"   # {entry['note']}" if entry["note"] else ""
            lines.append(f"  {key:<34} {text}{unit}{note}")
    tables = report.get("paper", {}).get("tables", report.get("paper", {}))
    if tables:
        lines.append("\npaper-table view:")
        for table, ds_map in sorted(tables.items()):
            for ds, cols in sorted(ds_map.items()):
                cells = "  ".join(
                    f"{k}={v:.4f}" for k, v in sorted(cols.items()))
                lines.append(f"  {table} [{ds}]  {cells}")
    return "\n".join(lines)
