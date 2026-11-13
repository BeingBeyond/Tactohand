"""Lightning module for the tactohand 30 Hz vertex-force experiment.

Single architecture (the former ``query_substep`` mode): a per-frame DINOv3
backbone, a temporal feature encoder, and a query head predicting the 30 Hz
vertex-force field of the substep aligned with each 10 Hz token's image
(substep 0; label rows take the leading edge of every substep interval, so
substep 0 shares the token's timestamp). The other two substeps of the 30 Hz
axis are detached GT stitches (``_physical_outputs``), so the wrench
aggregation, physics rollout and metrics keep their 30 Hz axes while every
learned output — and the rollout's gradient-bearing intervals — correspond
one-to-one with a visual frame. Contact/proximity auxiliary supervision and
the GT-pose rollout loss are unchanged in shape. The earlier
``direct``/``query`` architecture lineage was removed 2026-09-10.
"""
from __future__ import annotations

import math
import time

import numpy as np
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from core.physics.articulated_torch import ArticulatedState, ArticulatedTorchBackend
from core.physics.backends import BatchedTorchBackend, RigidState
from core.physics.math3d import matrix_to_quat, quat_to_matrix
from training.losses.haco_losses import VCBLoss
from training.runtime.schedules import warmup_cosine_multiplier
from core.data.resources import resolve_resource_path
from core.physics.wrench import aggregate_vertex_forces, gravity_wrench
from training.tactohand_wrench.nn.network import QueryForceWrenchNetwork
from core.geometry.force_frames import CAMERA_FRAME, clamp_force_magnitude, VERTEX_LOCAL_FRAME, force_local_to_camera, validate_vertex_force_frame
from training.tactohand_wrench.wrench.metrics import (
    PoseLossWindows, rollout_metrics, rollout_metrics_and_pose_loss,
)
from training.tactohand_wrench.wrench.scene import (
    scene_rollout_frame_mask, scene_rollout_inputs, scene_rollout_prediction_mask,
)
from training.tactohand_wrench.wrench.targets import validate_target_mode




def _optional_positive(value: Any, name: str) -> float | None:
    """Config helper: None stays off, otherwise require a positive finite."""
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number or null")
    return result


class SubstepForceModel(pl.LightningModule):
    """Predict the image-aligned 30 Hz force field from every 10 Hz token."""

    def __init__(self, cfg, *, initialize_losses: bool = True):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": __import__("omegaconf").OmegaConf.to_container(cfg, resolve=True)})
        model_cfg = cfg.model.wrench
        self.network = QueryForceWrenchNetwork(dict(model_cfg))
        self.target_mode = validate_target_mode(
            str(cfg.data.get("target_mode", "applied")))
        self.vertex_force_frame = validate_vertex_force_frame(
            cfg.data.get("vertex_force_frame", CAMERA_FRAME))
        loss_cfg = cfg.loss.get("wrench", {})
        self.loss_type = str(loss_cfg.get("type", "l2")).lower()
        if self.loss_type != "l2" or self.target_mode != "applied":
            raise ValueError("tactohand training requires L2 and applied hand-wrench targets")
        self.point_force_weight = float(loss_cfg.get("point_force_weight", 0.0))
        self.wrench_force_weight = float(loss_cfg.get("wrench_force_weight", 1.0))
        self.wrench_torque_weight = float(loss_cfg.get("wrench_torque_weight", 1.0))
        # Magnitude clamps (Newton / N*m): solver-blowup labels reach 100-2400 N
        # on <0.1% of vertices and one such target outweighs the whole batch in
        # squared error.  Direction-preserving rescale on BOTH the GT target and
        # the predicted physical force; the prediction-side clamp also bounds
        # the rollout inputs (see _physical_outputs).  None = legacy unclamped.
        self.point_force_clamp_n = _optional_positive(
            loss_cfg.get("point_force_clamp_n"), "loss.wrench.point_force_clamp_n")
        self.wrench_force_clamp_n = _optional_positive(
            loss_cfg.get("wrench_force_clamp_n"), "loss.wrench.wrench_force_clamp_n")
        self.wrench_torque_clamp_nm = _optional_positive(
            loss_cfg.get("wrench_torque_clamp_nm"), "loss.wrench.wrench_torque_clamp_nm")
        # Active/zero split for the point-force loss: per-vertex force fields
        # are ~91-96% zero entries, which pins a plain frame-level MSE at the
        # all-zero predictor floor (train loss flat at ~0.027 N^2 for 8
        # epochs in the 20260914 run).  The active term carries saturating
        # CB per-vertex weights (clamp caps the rare-class ratio) over a
        # Huber shape; the zero term keeps phantom forces suppressed.
        self.force_active_zero_split = bool(
            loss_cfg.get("force_active_zero_split", False))
        self.force_active_threshold_n = float(
            loss_cfg.get("force_active_threshold_n", 0.05))
        self.force_zero_weight = float(loss_cfg.get("force_zero_weight", 0.1))
        self.force_active_huber_delta_n = float(
            loss_cfg.get("force_active_huber_delta_n", 0.05))
        # Raw local-frame diagnostic path (overfit probes): the force head's
        # vertex-local components are the supervised signal themselves — no
        # engagement gate, no local->camera rotation, plain MSE against the
        # reader's vertex_force_local_30hz targets.  Isolates whether the
        # gate/loss-shape/rotation pipeline is what pins the head at the
        # conditional-mean plateau (20260918_222424).
        self.raw_local_mse = bool(loss_cfg.get("raw_local_mse", False))
        if cfg.data.get("training_data_fast_path", False) and (
                int(model_cfg.get("output_substeps", 1)) != 1 or self.raw_local_mse):
            raise ValueError(
                "training_data_fast_path requires output_substeps=1 and "
                "raw_local_mse=false; disable it for legacy/diagnostic models")
        if self.raw_local_mse:
            if self.vertex_force_frame != VERTEX_LOCAL_FRAME:
                raise ValueError(
                    "loss.wrench.raw_local_mse requires "
                    "data.vertex_force_frame=vertex_local")
            if self.force_active_zero_split:
                raise ValueError(
                    "loss.wrench.raw_local_mse and force_active_zero_split "
                    "are mutually exclusive")
        if self.force_active_zero_split and initialize_losses:
            self._load_force_vertex_weights(
                str(loss_cfg.get(
                    "force_statistics_path",
                    "core/resources/tactohand/force_statistics_v5.npz")),
                beta=float(loss_cfg.get("force_cb_beta", 0.9999)),
                count_floor=float(loss_cfg.get("force_cb_clamp", 100.0)))
        weights = (self.point_force_weight, self.wrench_force_weight,
                   self.wrench_torque_weight)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("loss.wrench weights must be finite and non-negative")
        if sum(weights) == 0.0:
            # Contact/proximity-only baseline (experiment=tactohand_contact_only):
            # the aux head carries the objective, and the zero-weighted force
            # terms still keep the force head's parameters in the loss graph.
            # (point_force > 0 with both wrench weights at 0 — the vertex-only
            # ablation — is allowed the same way: the wrench terms stay in the
            # graph at zero weight.)
            if not self.network.predict_contact_proximity:
                raise ValueError(
                    "loss.wrench weights are all zero but predict_contact_proximity"
                    " is disabled — the objective would be empty")
        # A frozen backbone (model.wrench.freeze_backbone=true) is allowed for
        # probes: its params carry requires_grad=False, so plain DDP excludes
        # them from the reducer entirely and the graph stays connected through
        # the trainable heads.
        backbone_trainable = [
            parameter.requires_grad for parameter in self.network.backbone.parameters()]
        if any(backbone_trainable) and not all(backbone_trainable):
            raise RuntimeError(
                "backbone is partially frozen — freeze all of it or none")
        self.backbone_frozen = not all(backbone_trainable)
        # output_substeps=3 is the pre-aligned-substep layout (all three 30 Hz
        # substeps predicted per token, trained through 2026-09-19).  New
        # training must use 1, but rebuilds of archived S=3 checkpoints (eval
        # harness) stay loadable: _physical_outputs restores the legacy
        # full-substep path for them.
        self.legacy_substeps3 = self.network.output_substeps == 3
        if (self.network.output_mode != "vertex_force"
                or not (self.legacy_substeps3
                        or self.network.output_substeps == 1)):
            raise ValueError(
                "vertex_force output requires output_substeps 1 (aligned"
                " substep + GT stitching) or 3 (legacy checkpoints, eval"
                " rebuilds only)")
        rollout = cfg.loss.get("rollout", {})
        self.rollout_weight = float(rollout.get("weight", 0.0))
        self.rollout_horizon_10hz = int(rollout.get("horizon_10hz", 4))
        self.rollout_position_scale = float(rollout.get("position_scale_m", 0.01))
        self.rollout_rotation_scale = float(rollout.get("rotation_scale_rad", 0.1))
        self.rollout_position_weight = float(rollout.get("position_weight", 1.0))
        self.rollout_rotation_weight = float(rollout.get("rotation_weight", 1.0))
        self.rollout_joint_weight = float(rollout.get("joint_weight", 1.0))
        self.rollout_start_epoch = int(rollout.get("start_epoch", 0))
        self.rollout_warmup_epochs = int(rollout.get("warmup_epochs", 0))
        self.rollout_graph_extra_body_slots = int(rollout.get("graph_extra_body_slots", 0))
        self.rollout_graph_flat_substep = bool(rollout.get("graph_flat_substep", False))
        if self.rollout_weight < 0 or self.rollout_horizon_10hz < 1:
            raise ValueError("rollout weight must be nonnegative and horizon positive")
        if self.rollout_graph_extra_body_slots < 0 or self.rollout_graph_extra_body_slots == 1:
            raise ValueError("rollout graph_extra_body_slots must be 0 or at least 2")
        if self.rollout_start_epoch < 0 or self.rollout_warmup_epochs < 0:
            raise ValueError("rollout start and warmup epochs must be nonnegative")
        if min(self.rollout_position_scale, self.rollout_rotation_scale) <= 0:
            raise ValueError("rollout position/rotation scales must be positive")
        if min(self.rollout_position_weight, self.rollout_rotation_weight,
               self.rollout_joint_weight) < 0:
            raise ValueError("rollout component weights must be nonnegative")
        if (self.rollout_position_weight + self.rollout_rotation_weight
                + self.rollout_joint_weight == 0):
            raise ValueError("at least one rollout pose component must be enabled")
        auxiliary = cfg.loss.get("contact_proximity", {})
        self.auxiliary_enabled = bool(self.network.predict_contact_proximity)
        self.contact_weight = float(auxiliary.get("contact_weight", 0.0))
        self.contact_gate_weight = float(auxiliary.get("gate_weight", 0.25))
        self.proximity_weight = float(auxiliary.get("proximity_weight", 0.0))
        self.hand_visibility_weight = float(cfg.loss.get("hand_visibility", {}).get("weight", 0.0))
        if self.network.visibility_head is not None:
            if not math.isfinite(self.hand_visibility_weight) or self.hand_visibility_weight <= 0:
                raise ValueError("hand visibility prediction requires a positive finite loss weight")
            if not cfg.data.get("supervise_hand_visibility", False):
                raise ValueError("hand visibility prediction requires data.supervise_hand_visibility")
            # [full/crop, left/right, TP/FP/FN/valid]
            self.register_buffer("_hand_visibility_counts", torch.zeros(2, 2, 4, dtype=torch.float64),
                                 persistent=False)
        elif self.hand_visibility_weight != 0:
            raise ValueError("hand visibility loss requires model.wrench.predict_hand_visibility")
        # Structural gate sharing: physical force outputs are multiplied by
        # the engagement gate probability (sigmoid(gate) * raw_force).  The
        # 20mm pre-contact frames carry exactly-zero force targets (verified
        # over labels-v5), so gating never drops nonzero supervision; the
        # force losses still run on the full force_valid population.
        self.force_gate_enabled = bool(auxiliary.get("force_gate", True))
        self.vcb_loss = None
        if self.auxiliary_enabled and initialize_losses:
            if min(self.contact_weight, self.proximity_weight) <= 0:
                raise ValueError("contact and proximity weights must be positive")
            haco = cfg.loss.haco
            self.vcb_loss = VCBLoss(
                contact_data_path=haco.contact_data_path,
                v_regressor_336_path=haco.v_regressor_336_path,
                v_regressor_84_path=haco.v_regressor_84_path,
                j_regressor=torch.zeros(21, 778),
                total_epochs=int(cfg.training.max_epochs),
                beta=float(haco.get("beta", 0.9999)), v_type="mesh",
                contact_statistics_path=haco.get("contact_statistics_path"))
        self._metric_state_names: dict[str, str] = {}
        metric_groups = ["train", "val"] + [
            f"val/{name}" for name in (cfg.data.get("val_datasets") or [])
        ]
        for index, group in enumerate(metric_groups):
            buffer_name = f"_contact_metric_state_{index}"
            self.register_buffer(buffer_name, torch.zeros(11, dtype=torch.float64),
                                 persistent=False)
            self._metric_state_names[group] = buffer_name
        # Physical RMSE and rollout metrics follow the same buffered pattern
        # as the contact states.  Per-batch conditional self.log(sync_dist)
        # makes the collective count rank-dependent — mixed validation
        # datasets, active-force coverage and rollout-window availability all
        # differ per rank — which desynchronizes the NCCL collective order
        # and deadlocks the run at the first validation epoch (2026-09-14
        # 1node_8gpu_20260914_211514).  Buffers accumulate locally; one
        # allreduce per group at epoch end keeps the sequence uniform.
        self._rmse_state_names: dict[str, str] = {}
        self._rollout_state_names: dict[str, str] = {}
        self._rollout_loss_state_names: dict[str, str] = {}
        self._rollout_metric_horizons = list(map(int, cfg.metrics.get(
            "tensorboard_rollout_horizons", cfg.metrics.rollout_horizons)))
        self._rollout_metric_keys = list(cfg.metrics.get(
            "tensorboard_rollout_metrics",
            ["position_ade_m", "rotation_ade_rad", "explosion_rate"]))
        for index, group in enumerate(metric_groups):
            rmse_name = f"_rmse_metric_state_{index}"
            # [point_sq, point_n, active_sq, active_n,
            #  force_sq, force_n, torque_sq, torque_n]
            self.register_buffer(rmse_name, torch.zeros(8, dtype=torch.float64),
                                 persistent=False)
            self._rmse_state_names[group] = rmse_name
            rollout_name = f"_rollout_metric_state_{index}"
            # [..., horizon, key, (weighted_sum, weight)]
            self.register_buffer(
                rollout_name,
                torch.zeros(len(self._rollout_metric_horizons),
                            len(self._rollout_metric_keys), 2,
                            dtype=torch.float64),
                persistent=False)
            self._rollout_state_names[group] = rollout_name
            loss_name = f"_rollout_loss_state_{index}"
            # [loss_sum, position_sum, rotation_sum, valid_windows,
            #  attempted_batches, failed_batches]
            self.register_buffer(loss_name, torch.zeros(6, dtype=torch.float64),
                                 persistent=False)
            self._rollout_loss_state_names[group] = loss_name
        # DDP broadcasts ordinary buffers from rank 0 before each forward.
        # These are rank-local sums, reduced explicitly at epoch end; a
        # forward broadcast would overwrite the other ranks' accumulated data.
        self._ddp_params_and_buffers_to_ignore = [
            name for names in (self._metric_state_names, self._rmse_state_names,
                               self._rollout_state_names, self._rollout_loss_state_names)
            for name in names.values()
        ]
        if self.network.visibility_head is not None:
            self._ddp_params_and_buffers_to_ignore.append("_hand_visibility_counts")
        tensorboard = cfg.logging.get("tensorboard", {})
        # Keep rank-local sums outside registered buffers: DDP broadcasts
        # buffers before forward, which would overwrite other ranks' sums.
        self._step_loss_sums: torch.Tensor | None = None
        self._metric_level = str(tensorboard.get("level", "core")).lower()
        if self._metric_level not in {"core", "diagnostic", "profile"}:
            raise ValueError("logging.tensorboard.level must be core, diagnostic, or profile")
        profiling = cfg.get("profiling", {})
        self._timing_enabled = bool(profiling.get("enabled", False))
        self._timing_sync_cuda = bool(profiling.get("sync_cuda", True))
        self._timing_last = {
            "transfer_s": 0.0,
            "forward_loss_s": 0.0,
            "backbone_forward_s": 0.0,
            "temporal_forward_loss_s": 0.0,
            "backward_s": 0.0,
            "optimizer_step_s": 0.0,
        }
        self._timing_backward_start = 0.0

    def forward(self, images, image_inverse=None, compute_force=True):
        return self.network(
            images, image_inverse=image_inverse, compute_force=compute_force)

    @staticmethod
    def _cudagraph_mark_step_begin() -> None:
        """Tell compiled CUDA graphs that outputs from the prior batch are dead."""
        compiler = getattr(torch, "compiler", None)
        mark_step = getattr(compiler, "cudagraph_mark_step_begin", None)
        if mark_step is not None:
            mark_step()

    def on_train_batch_start(self, batch, batch_idx) -> None:
        del batch, batch_idx
        self._cudagraph_mark_step_begin()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0) -> None:
        del batch, batch_idx, dataloader_idx
        self._cudagraph_mark_step_begin()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move tensors without asking Lightning to mutate frozen Physics records."""
        del dataloader_idx
        self._timing_sync()
        started = time.perf_counter()
        result = {
            key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        result.pop("_rigid_body_owner_range_validated", None)
        for key in ("force_supervision_valid", "rollout_eligible"):
            flag = batch.get(key)
            if isinstance(flag, torch.Tensor) and flag.device.type == "cpu":
                # These decisions are rank-local and used in Python. Reduce
                # the CPU inputs while the asynchronous transfers are queued.
                result[f"_{key}_all"] = bool(flag.bool().all())
        force_valid = batch.get("force_valid_30hz")
        if (result.get("_force_supervision_valid_all") is False
                and isinstance(force_valid, torch.Tensor)
                and force_valid.device.type == "cpu"):
            result["_empty_force_supervision"] = (
                not bool(batch["force_supervision_valid"].any())
                and not bool(force_valid.any()))
        if (self.rollout_weight > 0 and "rigid_body_position_30hz" in batch
                and result.get("_force_supervision_valid_all")
                and result.get("_rollout_eligible_all")
                and set(batch.get("dataset_mode", ())) in ({"rigid_trajectory"}, {"static_frame"})
                and batch["rigid_body_valid_30hz"].device.type == "cpu"):
            owner = batch.get("vertex_object_index_30hz")
            if (isinstance(owner, torch.Tensor) and owner.device.type == "cpu"
                    and owner.dtype == torch.int16 and owner.numel()):
                # Range-check the immutable integer IDs before the forward,
                # so force attribution does not synchronize CUDA mid-step.
                ids = owner.numpy()
                bodies = batch["rigid_body_position_30hz"].shape[2]
                if ids.min() >= -1 and ids.max() < bodies:
                    result["_rigid_body_owner_range_validated"] = True
            _, valid = scene_rollout_frame_mask(batch, torch.device("cpu"))
            horizon = 3 * self.rollout_horizon_10hz
            if horizon <= valid.shape[1]:
                mask = valid.unfold(1, horizon, 1).all(-1)
                prediction = scene_rollout_prediction_mask(batch, torch.device("cpu"))
                if prediction is not None:
                    mask &= prediction.unfold(1, horizon, 1).any(-1)
                # The supervision mask is already on the host. Compute its
                # count before the forward to avoid synchronizing CUDA there.
                result["_rigid_rollout_windows"] = (
                    mask.to(device, non_blocking=True), int(mask.sum()))
        self._timing_sync()
        self._timing_last["transfer_s"] = time.perf_counter() - started
        return result

    def _timing_sync(self) -> None:
        if self._timing_enabled and self._timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_latest_timing_stats(self) -> dict[str, float]:
        return dict(self._timing_last)

    def _reset_contact_metric_states(self, prefix: str) -> None:
        for state_names in (self._metric_state_names, self._rmse_state_names,
                            self._rollout_state_names, self._rollout_loss_state_names):
            for group, name in state_names.items():
                if group == prefix or group.startswith(prefix + "/"):
                    getattr(self, name).zero_()

    def on_train_epoch_start(self) -> None:
        self._step_loss_sums = None
        self._reset_contact_metric_states("train")
        # Drive the batch sampler's epoch: without this the per-epoch window
        # phases, cycling offsets and dataset-block permutation never fire.
        datamodule = getattr(self.trainer, "datamodule", None)
        sampler = getattr(datamodule, "train_batch_sampler", None)
        if sampler is not None:
            sampler.set_epoch(int(self.current_epoch))

    def on_validation_epoch_start(self) -> None:
        self._reset_contact_metric_states("val")
        if self.network.visibility_head is not None:
            self._hand_visibility_counts.zero_()

    def _update_contact_metric_state(self, group, tp, fp, fn, abs_error, count,
                                     gate_tp, gate_fp, gate_fn,
                                     cond_tp, cond_fp, cond_fn):
        if group not in self._metric_state_names:
            # This only supports configured validation datasets so all ranks
            # own states in the same order before DDP starts.
            raise KeyError(f"unregistered contact metric group {group!r}")
        state = getattr(self, self._metric_state_names[group])
        state.add_(torch.stack((tp, fp, fn, abs_error, count,
                                gate_tp, gate_fp, gate_fn,
                                cond_tp, cond_fp, cond_fn)).to(state))

    def _log_contact_metric_states(self, prefix: str) -> None:
        for group, name in self._metric_state_names.items():
            if not (group == prefix or group.startswith(prefix + "/")):
                continue
            state = getattr(self, name).clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.SUM)
            tp, fp, fn, abs_error, count = state[:5]
            gate_tp, gate_fp, gate_fn = state[5:8]
            cond_tp, cond_fp, cond_fn = state[8:11]
            # Dead-group guard: a group that never received an update (the
            # per-dataset groups when a single-dataset val loader omits
            # dataset_name, or per-dataset splits with no valid labels)
            # would otherwise log all-zero curves next to the real global
            # ones.  The allreduce above already made `count` rank-uniform,
            # so skipping stays collective-order-safe.
            if float(count) <= 0.0:
                continue
            precision = tp / (tp + fp).clamp_min(1)
            recall = tp / (tp + fn).clamp_min(1)
            f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
            proximity_mae = abs_error / count.clamp_min(1)
            base = f"epoch/metrics/{group}"
            # Counts were summed before forming ratios, so every rank now has
            # the same global metric. Explicit distributed logging prevents
            # Lightning's warning; its mean reduction leaves the value unchanged.
            log_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{base}/contact/precision", precision, **log_kwargs)
            self.log(f"{base}/contact/recall", recall, **log_kwargs)
            self.log(f"{base}/contact/f1", f1, **log_kwargs)
            self.log(f"{base}/proximity/mae", proximity_mae, **log_kwargs)
            # contact/* above is the deployed product probability
            # sigmoid(gate) * sigmoid(vertex logit); gate/* and contact_cond/*
            # are the per-stage diagnostics.
            self.log(f"{base}/gate/precision",
                     gate_tp / (gate_tp + gate_fp).clamp_min(1), **log_kwargs)
            self.log(f"{base}/gate/recall",
                     gate_tp / (gate_tp + gate_fn).clamp_min(1), **log_kwargs)
            self.log(f"{base}/gate/f1",
                     2 * gate_tp / (2 * gate_tp + gate_fp + gate_fn).clamp_min(1),
                     **log_kwargs)
            self.log(f"{base}/contact_cond/f1",
                     2 * cond_tp / (2 * cond_tp + cond_fp + cond_fn).clamp_min(1),
                     **log_kwargs)

    def on_train_epoch_end(self) -> None:
        self._log_contact_metric_states("train")
        self._log_rmse_states("train")
        self._log_rollout_states("train")
        self._log_rollout_loss_states("train")

    def on_validation_epoch_end(self) -> None:
        self._log_contact_metric_states("val")
        self._log_rmse_states("val")
        self._log_rollout_states("val")
        self._log_rollout_loss_states("val")
        if self.network.visibility_head is not None:
            counts = self._hand_visibility_counts.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts)
            for view, view_counts in zip(("full", "handcrop"), counts):
                for side, (tp, fp, fn, valid) in zip(("left", "right"), view_counts):
                    group = f"epoch/metrics/val/hand_visibility/{view}/{side}"
                    metrics = {"precision": tp / (tp + fp).clamp_min(1),
                               "recall": tp / (tp + fn).clamp_min(1),
                               "f1": 2 * tp / (2 * tp + fp + fn).clamp_min(1),
                               "valid_frames": valid}
                    for name, value in metrics.items():
                        self.log(f"{group}/{name}", value.float(),
                                 on_step=False, on_epoch=True, sync_dist=False)

    def _hand_visibility_loss(self, logits, batch, prefix):
        logits = logits.float()
        target = batch["hand_visibility_target_10hz"].to(device=logits.device, dtype=torch.bool)
        valid = batch["hand_visibility_valid_10hz"].to(device=logits.device, dtype=torch.bool)
        if prefix == "val" and "metric_sample_valid" in batch:
            valid = valid & batch["metric_sample_valid"].to(logits.device).bool().view(-1, 1, 1)
        entries = F.binary_cross_entropy_with_logits(
            torch.where(valid, logits, 0.0), target.to(logits.dtype), reduction="none")
        loss = torch.where(valid, entries, 0.0).sum() / valid.sum().clamp_min(1)
        if prefix == "val":
            with torch.no_grad():
                crop = batch["crop_side"].to(logits.device).ge(0).view(-1, 1, 1)
                predicted = logits >= 0
                for index, is_crop in enumerate((False, True)):
                    selected = valid & (crop == is_crop)
                    tp = (selected & predicted & target).sum((0, 1))
                    fp = (selected & predicted & ~target).sum((0, 1))
                    fn = (selected & ~predicted & target).sum((0, 1))
                    self._hand_visibility_counts[index].add_(
                        torch.stack((tp, fp, fn, selected.sum((0, 1))), dim=-1))
        return loss

    def _log_rollout_loss_states(self, prefix: str) -> None:
        """Pool valid windows; skipped batches never masquerade as zero error.

        Every rank reduces every group, including ranks without rollout
        samples. No data-dependent distributed collectives occur in a step.
        """
        if self.rollout_weight <= 0:
            return
        for group, name in self._rollout_loss_state_names.items():
            if not (group == prefix or group.startswith(prefix + "/")):
                continue
            state = getattr(self, name).clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.SUM)
            count = state[3]
            for index, component in enumerate(("rollout", "rollout_position", "rollout_rotation")):
                self.log(f"loss/{group}/{component}", state[index] / count.clamp_min(1),
                         on_step=False, on_epoch=True, sync_dist=False)
            for index, key in ((3, "valid_windows"), (4, "attempted_batches"),
                               (5, "failed_batches")):
                self.log(f"epoch/rollout_supervision/{group}/{key}", state[index],
                         on_step=False, on_epoch=True, sync_dist=False)

    def _accumulate_rollout_loss(self, prefix, batch, losses, windows, attempted, failed):
        groups = [prefix]
        names = set(map(str, batch.get("dataset_name", [])))
        if prefix == "val" and len(names) == 1:
            groups.append(f"val/{next(iter(names))}")
        for group in groups:
            name = self._rollout_loss_state_names.get(group)
            if name is None:
                continue
            state = getattr(self, name)
            state[:3].add_(torch.stack([value.detach() for value in losses]).to(state) * windows)
            state[3] += windows
            state[4] += int(attempted)
            state[5] += int(failed)

    def _log_rmse_states(self, prefix: str) -> None:
        """Pool the per-batch RMSE accumulators and log once per group.

        The allreduce happens before any emptiness decision, so every rank
        takes the same branch and the collective sequence stays uniform.
        """
        log_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True)
        for group, name in self._rmse_state_names.items():
            if not (group == prefix or group.startswith(prefix + "/")):
                continue
            state = getattr(self, name).clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.SUM)
            (point_sq, point_n, active_sq, active_n,
             force_sq, force_n, torque_sq, torque_n) = state.unbind()
            base = f"epoch/metrics/{group}/force"
            if point_n > 0:
                self.log(f"{base}/point_rmse_n",
                         torch.sqrt(point_sq / point_n), **log_kwargs)
            if active_n > 0:
                self.log(f"{base}/point_active_rmse_n",
                         torch.sqrt(active_sq / active_n), **log_kwargs)
            if force_n > 0:
                self.log(f"{base}/resultant_rmse_n",
                         torch.sqrt(force_sq / force_n), **log_kwargs)
            if torque_n > 0:
                self.log(f"{base}/torque_rmse_nm",
                         torch.sqrt(torque_sq / torque_n), **log_kwargs)

    def _log_rollout_states(self, prefix: str) -> None:
        """Pool the per-batch rollout accumulators; one mean per metric."""
        for group, name in self._rollout_state_names.items():
            if group != prefix:
                continue
            state = getattr(self, name).clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.SUM)
            for h_index, horizon_10hz in enumerate(self._rollout_metric_horizons):
                for k_index, key in enumerate(self._rollout_metric_keys):
                    total, weight = state[h_index, k_index].unbind()
                    if weight > 0:
                        self.log(
                            f"epoch/rollout/{group}/h{horizon_10hz}_{key}",
                            total / weight, on_step=False, on_epoch=True,
                            sync_dist=True)

    @staticmethod
    def _apply_force_gate(force: torch.Tensor, gate_logits: torch.Tensor):
        """Multiply the aligned-substep force magnitudes by sigmoid(gate).

        ``gate_logits`` is [B, T, hands] at 10 Hz; ``force`` is the head's
        [B, T, hands, V, 3] single-substep field, so the gate broadcasts
        over vertices and components.
        """
        gate = torch.sigmoid(gate_logits.to(force.dtype))
        return force * gate.unsqueeze(-1).unsqueeze(-1)

    def _physical_outputs(self, output, batch, gate_logits=None, *, point_target=None):
        """Scale the head's aligned-substep force, stitch GT into the rest.

        The head emits [B, T, H, V, 3] for substep 0 — the 30 Hz frame
        sharing the token image's timestamp. The returned hybrid
        [B, T, S, H, V, 3] carries detached, identically clamped GT vertex
        forces at substeps 1..S-1: the 30 Hz wrench aggregation, rollout
        and metrics keep their physical axes, the losses mask back to
        substep 0, and the rollout's non-aligned intervals replay GT
        forces by construction (GT-damped rollout — compounding error can
        no longer explode through intervals the 10 Hz input cannot see).
        """
        force = output.float() * batch["point_force_scale"]
        if self.legacy_substeps3:
            # Pre-aligned-substep checkpoints rebuild with S=3 outputs: run
            # the retired full-substep path (gate/rotation/presence/clamp
            # broadcast over all substeps, no GT stitching) so archived runs
            # stay evaluable.  Never used by new training (S=1 enforced).
            if gate_logits is not None and self.force_gate_enabled:
                gate = torch.sigmoid(gate_logits.to(force.dtype))
                force = force * gate.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
            if self.vertex_force_frame == VERTEX_LOCAL_FRAME:
                force = force_local_to_camera(
                    force, batch["vertex_local_yz_camera_30hz"])
            if "hand_present_10hz" in batch:
                present = batch["hand_present_10hz"].to(force.device).bool()
                force = force * present[:, :, None, :, None, None]
            if self.point_force_clamp_n is not None:
                force = clamp_force_magnitude(force, self.point_force_clamp_n)
            return force, *self._aggregate(force, batch)[1:]
        if self.raw_local_mse:
            # Diagnostic path: force stays in the head's vertex-local
            # parameterization — no engagement gate, no local->camera
            # rotation, no presence masking. GT stitches stay local too,
            # so the wrench aggregation below is physically meaningless in
            # this mode (structurally zero-weighted in probe configs) but
            # every (force, hand, contact) shape matches the formal path.
            if self.point_force_clamp_n is not None:
                force = clamp_force_magnitude(force, self.point_force_clamp_n)
        else:
            if gate_logits is not None and self.force_gate_enabled:
                force = self._apply_force_gate(force, gate_logits)
            if self.vertex_force_frame == VERTEX_LOCAL_FRAME:
                # Substep-0 axes: the aligned frame's posed local basis.
                force = force_local_to_camera(
                    force, batch["vertex_local_yz_camera_30hz"][:, :, 0])
            if "hand_present_10hz" in batch:
                present = batch["hand_present_10hz"].to(force.device).bool()
                force = force * present[:, :, :, None, None]
            if self.point_force_clamp_n is not None:
                # Symmetric with the GT clamp in _point_force_loss: bounds
                # what the head can emit in Newtons AND what the rollout
                # integrates (the aggregated wrench of clamped vertex
                # forces is bounded).
                force = clamp_force_magnitude(force, self.point_force_clamp_n)
        if point_target is None:
            point_target = self._clamped_point_target(batch, force, local=self.raw_local_mse)
        prediction_valid = batch.get("force_prediction_valid_30hz")
        if prediction_valid is not None:
            point_target = point_target.detach()
            aligned = prediction_valid.to(force.device).reshape(
                force.shape[0], force.shape[1], -1)[:, :, 0].bool()
            force = torch.where(
                aligned[..., None, None, None], force, point_target[:, :, 0])
        force = torch.cat((force.unsqueeze(2), point_target[:, :, 1:]), dim=2)
        return self._aggregate(force, batch)

    def _aggregate(self, force, batch):
        """Deterministic vertex-force -> (force, hand wrench, net contact)."""
        if "vertex_joint_row_camera_30hz" in batch:
            lever = batch["vertex_lever_camera_30hz"].to(force)
            joint_row = batch["vertex_joint_row_camera_30hz"].to(force)
            hand = torch.cat((
                force.sum(-2), torch.linalg.cross(lever, force).sum(-2),
                (joint_row * force).sum(dim=(-2, -1)).unsqueeze(-1)), -1).sum(-2)
        elif "vertex_generalized_map_camera_30hz" in batch:
            mapping = batch["vertex_generalized_map_camera_30hz"].to(force)
            hand = torch.einsum("...hvkc,...hvc->...k", mapping, force)
        else:
            lever = batch["vertex_lever_camera_30hz"]
            if force.ndim == 6:
                force_for_wrench = force.flatten(-3, -2)
                lever = lever.flatten(-3, -2)
            else:
                force_for_wrench = force
            hand = aggregate_vertex_forces(force_for_wrench, lever)
        contact = hand + batch["environment_wrench_camera_30hz"].to(hand)
        return force, hand, contact

    @staticmethod
    def _gate_target(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Engagement per (batch, time, hand): any positive vertex, on valid slots.

        Single source of truth for the stage-1 gate label and for restricting
        the per-vertex VCB loss to engaged hand-frames (conditional form).
        """
        return valid & (target.sum(dim=-1) > 0)

    def _auxiliary_losses(self, predictions, batch):
        assert self.vcb_loss is not None
        logits = predictions["contact_logits_10hz"]
        target = batch["contact_target_10hz"].to(logits)
        # Validity flags alone gate the contact/proximity losses: with
        # pad_label_mode=repeat_edge the dataset marks edge-held pad labels
        # valid, so static tails stay supervised; zero-mode pads are invalid.
        contact_valid = batch["contact_valid_10hz"].to(logits.device).bool()
        gate_logits = predictions["contact_gate_10hz"].to(logits)
        gate_target = self._gate_target(target, contact_valid)
        # Stage 1: plain BCE on the near-balanced engaged/not problem; the
        # no-contact majority (and absent-hand zero supervision) lives here.
        safe_gate_logits = torch.where(contact_valid, gate_logits, 0.0)
        gate_entries = F.binary_cross_entropy_with_logits(
            safe_gate_logits, gate_target.to(gate_logits.dtype), reduction="none")
        gate_loss = torch.where(contact_valid, gate_entries, 0.0).sum() / contact_valid.sum().clamp_min(1)
        # Stage 2: VCB on the conditional population (engaged hand-frames
        # only), so the class statistics match the artifact's conditional
        # distribution instead of the no-contact-flooded marginal.
        batch_size, time, hands, vertices = logits.shape
        contact_loss = self.vcb_loss(
            logits.permute(0, 2, 1, 3).reshape(batch_size * hands * time, vertices),
            target.permute(0, 2, 1, 3).reshape(batch_size * hands * time, vertices),
            int(self.current_epoch),
            gate_target.permute(0, 2, 1).reshape(-1))

        proximity = predictions["proximity_10hz"]
        proximity_target = batch["proximity_target_10hz"].to(proximity)
        proximity_valid = (
            batch["proximity_valid_10hz"].to(proximity.device).bool()
        ).unsqueeze(-1)
        # Invalid labels may use NaN as an explicit missing-value sentinel.
        # Mask before squaring: NaN * 0 is still NaN under IEEE arithmetic.
        error = torch.where(
            proximity_valid, proximity - proximity_target, torch.zeros_like(proximity))
        proximity_loss = error.square().sum() / proximity_valid.sum().mul(
            vertices).clamp_min(1)
        return gate_loss, contact_loss, proximity_loss

    @staticmethod
    def _frame_gate(batch, like: torch.Tensor,
                    key: str = "force_valid_30hz") -> torch.Tensor:
        """Per-(b,t,s) validity bool reshaped to ``(B, T, S)``.

        The 30 Hz batch arrays always carry ``T * S`` rows along dim 0 for
        the token axis of ``like`` (dim 1), so S is recovered from the row
        count rather than assumed from ``like``'s rank.
        """
        prediction_key = "force_prediction_valid_30hz"
        if key == "force_valid_30hz" and prediction_key in batch:
            key = prediction_key
        gate = batch.get(key, batch["real_mask_30hz"]).to(like.device).bool()
        return gate.reshape(gate.shape[0], like.shape[1], -1)

    @staticmethod
    def _aligned_frame_mask(gate: torch.Tensor) -> torch.Tensor:
        """Zero the frame gate on the GT-stitched substeps (s > 0).

        The head only predicts substep 0 — the 30 Hz frame sharing the
        token image's timestamp — so losses and metrics must not read the
        stitched entries at s > 0: prediction equals target there and any
        leakage would report exact-zero error and dilute every term.
        """
        aligned = torch.zeros_like(gate)
        aligned[:, :, 0] = True
        return gate & aligned

    def _clamped_point_target(self, batch, like: torch.Tensor,
                              local: bool = False) -> torch.Tensor:
        """GT vertex forces, magnitude-clamped to match the prediction side.

        ``local=True`` reads the reader's vertex-local targets instead of the
        camera-frame ones (raw_local_mse diagnostic path); both are produced
        by the same reader, so keys must exist regardless of frame config.
        """
        key = ("vertex_force_local_30hz" if local
               else "vertex_force_camera_30hz")
        target = batch[key].to(like)
        if self.point_force_clamp_n is None:
            return target
        return clamp_force_magnitude(target, self.point_force_clamp_n)

    def _load_force_vertex_weights(self, path: str, *, beta: float,
                                   count_floor: float) -> None:
        """Per-vertex active-class CB weights from the force statistics npz.

        Same saturating class-balanced form as VCBLoss, with ``count_floor``
        clamping the normalized per-1000 counts so the rare end (dorsal
        vertices with <0.1% active rate) cannot explode the weight ratio.
        """
        artifact = np.load(resolve_resource_path(path), allow_pickle=False)
        if int(artifact["schema_version"]) != 1:
            raise ValueError(f"unsupported force statistics schema in {path}")
        counts = torch.from_numpy(
            np.asarray(artifact["nonzero_counts"], dtype=np.float64))
        slots = float(artifact["supervised_slots"])
        if tuple(counts.shape) != (778,) or slots <= 0:
            raise ValueError(f"malformed force statistics in {path}")
        probability = (counts / slots).clamp(0.0, 1.0)
        per_cls = torch.stack(
            (1000.0 * (1.0 - probability), 1000.0 * probability),
            dim=1).clamp_min(count_floor)
        per_cls = per_cls / per_cls.sum(dim=1, keepdim=True) * 1000.0
        effective = 1.0 - beta ** per_cls
        weights = (1.0 - beta) / effective
        weights = weights / weights.sum(dim=1, keepdim=True)
        self.register_buffer(
            "force_vertex_weights", weights[:, 1].float(), persistent=False)

    def _point_force_loss(self, force, batch, *, point_target=None):
        """Per-vertex force loss on the image-aligned substep (s = 0).

        ``force`` is the hybrid [B, T, S, H, V, 3] field from
        _physical_outputs; the mask intersects the per-frame validity with
        the aligned substep so the GT-stitched s > 0 entries never
        contribute (prediction equals target there, which would dilute the
        loss and fake the metrics).

        force_active_zero_split=False keeps the legacy plain MSE (aligned
        substep only).  With the split on, the loss decomposes into a
        per-vertex-weighted Huber over entries whose clamped target
        magnitude exceeds the active threshold (contact-bearing vertices;
        the VCB-analogous rare class) plus ``force_zero_weight`` times the
        plain MSE over the remaining entries.  The raw_local_mse
        diagnostic path supervises the head's vertex-local parameterization
        against the reader's local targets under the same aligned mask.
        """
        frame_mask = SubstepForceModel._frame_gate(batch, force)[:, :, 0]
        if point_target is None:
            point_target = self._clamped_point_target(batch, force, local=self.raw_local_mse)
        force = force[:, :, 0]
        target = point_target[:, :, 0]
        if self.raw_local_mse or not self.force_active_zero_split:
            # Native RGB clips may have no frame aligned with a valid force
            # label. The empty sum stays graph-connected and finite.
            error = force[frame_mask] - target[frame_mask]
            return error.square().mean() if error.numel() else error.sum()
        magnitude = target.norm(dim=-1)
        for _ in range(magnitude.ndim - frame_mask.ndim):
            frame_mask = frame_mask.unsqueeze(-1)
        active = frame_mask & (magnitude > self.force_active_threshold_n)
        zero = frame_mask & ~active
        err = force - target
        vertex_weights = self.force_vertex_weights.view(
            [1] * (magnitude.ndim - 1) + [778]).to(err.dtype)
        huber = F.huber_loss(
            err, torch.zeros_like(err),
            delta=self.force_active_huber_delta_n,
            reduction="none").mean(dim=-1)
        weight_mask = vertex_weights * active
        active_weight = weight_mask.sum()
        active_term = (huber * weight_mask).sum() / torch.where(active.any(), active_weight, 1.0)
        zero_entries = zero.unsqueeze(-1).expand(err.shape)
        zero_term = ((err.square() * zero_entries).sum()
                     / zero_entries.sum().clamp_min(1))
        return active_term + self.force_zero_weight * zero_term

    def _wrench_losses(self, hand, batch):
        """Scene-total wrench MSE on the aligned substep, split into
        resultant force and torque.

        Masked by per-frame force validity AND restricted to substep 0:
        the hybrid's s > 0 hand wrenches aggregate GT vertex forces, so
        their error against the stored target is ~zero by construction and
        would only dilute the reported value.  Contact-only samples in a
        mixed batch carry placeholder zero wrench targets that must never
        read as supervision.  GT force/torque resultants are
        magnitude-clamped (solver-blowup rows reach hundreds of N / N*m on
        <0.1% of frames); the predicted side stays raw — its inputs are
        already clamped vertex forces, and an over-bound prediction must
        keep its full gradient.
        """
        target = batch["hand_wrench_camera_30hz"].to(hand)
        if (self.wrench_force_clamp_n is not None
                or self.wrench_torque_clamp_nm is not None):
            target = target.clone()
            if self.wrench_force_clamp_n is not None:
                target[..., :3] = clamp_force_magnitude(
                    target[..., :3], self.wrench_force_clamp_n)
            if self.wrench_torque_clamp_nm is not None:
                target[..., 3:] = clamp_force_magnitude(
                    target[..., 3:], self.wrench_torque_clamp_nm)
        error = hand - target
        gate = SubstepForceModel._aligned_frame_mask(
            SubstepForceModel._frame_gate(batch, error))
        weight = gate.to(hand.dtype)[..., None]
        frames = gate.sum().clamp_min(1.0)        # one entry per (b,t,s)
        return ((error[..., :3].square() * weight).sum() / (frames * 3),
                (error[..., 3:].square() * weight).sum() / (frames * 3))

    def _rollout_weight_now(self):
        if self.current_epoch < self.rollout_start_epoch:
            return 0.0
        if self.rollout_warmup_epochs <= 0:
            return self.rollout_weight
        progress = min(
            float(self.current_epoch - self.rollout_start_epoch)
            / self.rollout_warmup_epochs,
            1.0,
        )
        return self.rollout_weight * progress

    @staticmethod
    def _rollout_window_mask(batch, horizon):
        valid = batch["real_mask_30hz"].bool()
        if "force_valid_30hz" in batch:
            valid = valid & batch["force_valid_30hz"].to(valid.device).bool()
        for key in ("rollout_eligible", "force_supervision_valid"):
            if key in batch:
                valid = valid & batch[key].to(valid.device).bool().reshape(-1, 1)
        if horizon > valid.shape[1]:
            raise ValueError("rollout loss horizon exceeds the 30 Hz sequence")
        windows = valid.unfold(1, horizon, 1).all(-1)
        prediction = batch.get("force_prediction_valid_30hz")
        if prediction is not None:
            windows &= prediction.to(valid.device).bool().unfold(1, horizon, 1).any(-1)
        return windows

    @torch.autocast(device_type="cuda", enabled=False)
    def _gt_pose_rollout_loss(self, predicted_contact, batch, *, window_mask=None,
                              valid_windows=None, check_finite_with_solver=False):
        """GT-damped rollout: the aligned substep carries the prediction,
        the other two substeps of every token replay GT forces (stitched in
        _physical_outputs); native clips can also replay GT at unaligned
        image slots. Every step is supervised against the stored GT pose.
        At most ~1/3 of the intervals bear gradient, so the loss value
        and its scale are NOT comparable with the retired full-prediction
        rollout."""
        modes = set(map(str, batch.get("dataset_mode", ["rigid_trajectory"])))
        if len(modes) != 1:
            raise ValueError(
                f"a training batch must have one dynamics mode, got {sorted(modes)}")
        if modes == {"articulated_trajectory"}:
            return self._articulated_pose_rollout_loss(predicted_contact, batch)
        # The rigid-body backend uses torch.linalg.solve, which CUDA does not
        # implement for BF16, and normalized-mass physics reaches mass-matrix
        # condition numbers near 1e8.  Keep this differentiable branch in
        # FP64 (the backend derives every dtype from this tensor) while the
        # visual network continues to use mixed precision; results are cast
        # back to FP32 at the return.
        predicted_contact = predicted_contact.double()
        horizon = 3 * self.rollout_horizon_10hz
        transition_count = predicted_contact.shape[1]
        if horizon > transition_count:
            raise ValueError("rollout loss horizon exceeds the 30 Hz sequence")
        starts = transition_count - horizon + 1
        if window_mask is None:
            window_mask = SubstepForceModel._rollout_window_mask(batch, horizon)
        if valid_windows is None:
            valid_windows = int(window_mask.sum())
        if valid_windows == 0:
            zero = (predicted_contact.sum() * 0.0).float()
            return zero, zero, zero
        position = batch["position_30hz"].to(predicted_contact)
        rotation = batch["rotation_30hz"].to(predicted_contact)
        velocity = batch["velocity_30hz"].to(predicted_contact)
        implementation = str(self.cfg.loss.rollout.get("implementation", "compiled_substep"))
        integrator = str(self.cfg.metrics.get("rollout_integrator", "implicit_midpoint"))
        backend = BatchedTorchBackend(
            batch["physics_30hz"], predicted_contact,
            integrator=integrator,
            implicit_iterations=int(self.cfg.metrics.get("implicit_iterations", 6)),
            midpoint_implementation="compiled_fused3" if implementation == "graphed_fused3" else implementation,
        )
        predicted_net = predicted_contact + gravity_wrench(
            batch["physics_30hz"], predicted_contact, batch.get("gravity_camera_30hz"),
            mass=backend.mass)
        if (implementation == "graphed_fused3" and integrator == "implicit_midpoint"
                and predicted_net.is_cuda and predicted_net.requires_grad
                and getattr(self, "training", True) and not torch.cuda.is_current_stream_capturing()):
            from core.physics.rollout_fast import RigidRolloutGraph

            batch_size = int(self.cfg.get("data", {}).get("batch_size", predicted_net.shape[0]))
            body_slots = batch.get("_rigid_body_slots", 1)
            extra_slots = getattr(self, "rollout_graph_extra_body_slots", 0)
            if (body_slots in (1, extra_slots)
                    and predicted_net.shape[0] == batch_size * body_slots):
                # Keep one graph for the common single-body shape and at
                # most one more for an explicitly selected multi-body shape.
                graph_name = "_rigid_rollout_graph" if body_slots == 1 else "_multi_body_rollout_graph"
                graph = getattr(self, graph_name, None)
                if graph is None:
                    graph = RigidRolloutGraph(
                        flat_substep=getattr(self, "rollout_graph_flat_substep", False))
                    setattr(self, graph_name, graph)
                result = graph.run(backend, predicted_net, position, rotation, velocity,
                                   window_mask, horizon=horizon,
                                   position_scale=self.rollout_position_scale,
                                   rotation_scale=self.rollout_rotation_scale)
                if result is not None:
                    position_loss, rotation_loss, failed = result
                    losses = SubstepForceModel._combine_rigid_rollout_losses(self, position_loss, rotation_loss)
                    if check_finite_with_solver:
                        failed = failed | ~torch.isfinite(losses[0])
                    backend.check_solve_errors(failed)
                    return losses
        predicted_state = RigidState(
            position[:, :starts], matrix_to_quat(rotation[:, :starts]),
            velocity[:, :starts, :3], velocity[:, :starts, 3:],
        )
        position_terms, rotation_terms = [], []
        for step in range(horizon):
            predicted_state = backend.rollout_interval(
                predicted_state, predicted_net[:, step:step + starts])
            target_slice = slice(step + 1, step + 1 + starts)
            position_terms.append(
                ((predicted_state.x_com - position[:, target_slice]).norm(dim=-1)
                 / self.rollout_position_scale).square())
            # SO(3) chordal loss is exactly zero for identical rotations and
            # avoids the acos/geodesic numerical floor at the teacher target.
            # 0.5 ||R_s-R_t||_F^2 = 4 sin^2(theta/2) ~= theta^2.
            predicted_rotation = quat_to_matrix(predicted_state.quat_xyzw)
            rotation_terms.append(
                0.5 * ((predicted_rotation - rotation[:, target_slice])
                       / self.rollout_rotation_scale).square().sum((-1, -2)))
        # Keep the shape fixed. Boolean indexing compacts on the GPU and
        # synchronizes with the host to allocate the variable-length output.
        mask = window_mask.unsqueeze(-1)
        denominator = valid_windows * horizon
        position_loss = torch.where(mask, torch.stack(position_terms, -1), 0.0).sum() / denominator
        rotation_loss = torch.where(mask, torch.stack(rotation_terms, -1), 0.0).sum() / denominator
        losses = SubstepForceModel._combine_rigid_rollout_losses(self, position_loss, rotation_loss)
        # Check all Newton solves, and optionally the loss, in one transfer.
        # On failure the caller discards the entire graph before backward.
        backend.check_solve_errors(~torch.isfinite(losses[0]) if check_finite_with_solver else None)
        return losses

    def _combine_rigid_rollout_losses(self, position_loss, rotation_loss):
        total = (
            self.rollout_position_weight * position_loss
            + self.rollout_rotation_weight * rotation_loss
        ) / (self.rollout_position_weight + self.rollout_rotation_weight)
        return total.float(), position_loss.float(), rotation_loss.float()

    def _articulated_pose_rollout_loss(self, predicted_contact, batch):
        """Roll seven-axis trajectories, including the articulated joint."""
        # FP64 for the same conditioning reasons as the rigid branch: the
        # 7x7 articulated mass matrix couples a ~1 kg normalized base with
        # child effective inertias down to ~1e-5.
        predicted_contact = predicted_contact.double()
        horizon = 3 * self.rollout_horizon_10hz
        transition_count = predicted_contact.shape[1]
        if horizon > transition_count:
            raise ValueError("rollout loss horizon exceeds the articulated sequence")
        starts = transition_count - horizon + 1
        mask = batch["real_mask_30hz"].bool().unfold(1, horizon, 1).all(-1)
        if not bool(mask.any()):
            zero = (predicted_contact.sum() * 0.0).float()
            return zero, zero, zero
        c2w = batch["articulated_c2w_30hz"].to(predicted_contact)
        # Stored articulated arrays contain states (and may include additional
        # source-rate states), while the head predicts exactly one wrench per
        # 30 Hz transition. Align camera rotations to those transitions.
        rotation_c2w = c2w[:, :transition_count, :3, :3]
        world = predicted_contact.clone()
        world[..., :3] = torch.einsum(
            "bti,btji->btj", predicted_contact[..., :3], rotation_c2w)
        world[..., 3:6] = torch.einsum(
            "bti,btji->btj", predicted_contact[..., 3:6], rotation_c2w)
        target_position = batch["articulated_base_position_30hz"].to(world)
        target_rotation = batch["articulated_base_rotation_30hz"].to(world)
        target_angle = batch["articulated_joint_angle_30hz"].to(world)
        target_velocity = batch["articulated_velocity_30hz"].to(world)
        position_terms, rotation_terms, joint_terms = [], [], []
        for batch_index, physics in enumerate(batch["physics_30hz"]):
            backend = ArticulatedTorchBackend(
                list(physics.bodies), physics.child_zero_transform,
                fps=physics.fps, substeps=physics.substeps,
                gravity_world=physics.gravity_world, dtype=world.dtype,
                device=world.device)
            sample_position, sample_rotation, sample_joint = [], [], []
            for start in range(starts):
                state = ArticulatedState(
                    target_position[batch_index, start],
                    target_rotation[batch_index, start],
                    target_angle[batch_index, start],
                    target_velocity[batch_index, start])
                positions, rotations, joints = [], [], []
                for step in range(horizon):
                    state = backend.rollout_interval(
                        state, world[batch_index, start + step])
                    target = start + step + 1
                    positions.append(((state.base_position_world
                                       - target_position[batch_index, target]).norm()
                                      / self.rollout_position_scale).square())
                    rotations.append(0.5 * ((state.base_rotation_world
                                             - target_rotation[batch_index, target])
                                            / self.rollout_rotation_scale).square().sum())
                    delta = state.joint_angle - target_angle[batch_index, target]
                    wrapped = torch.atan2(torch.sin(delta), torch.cos(delta))
                    joints.append((wrapped / self.rollout_rotation_scale).square())
                sample_position.append(torch.stack(positions))
                sample_rotation.append(torch.stack(rotations))
                sample_joint.append(torch.stack(joints))
            position_terms.append(torch.stack(sample_position))
            rotation_terms.append(torch.stack(sample_rotation))
            joint_terms.append(torch.stack(sample_joint))
        position_loss = torch.stack(position_terms)[mask].mean()
        rotation_loss = torch.stack(rotation_terms)[mask].mean()
        joint_loss = torch.stack(joint_terms)[mask].mean()
        total_weight = (self.rollout_position_weight
                        + self.rollout_rotation_weight + self.rollout_joint_weight)
        total = (self.rollout_position_weight * position_loss
                 + self.rollout_rotation_weight * rotation_loss
                 + self.rollout_joint_weight * joint_loss) / total_weight
        angular_weight = self.rollout_rotation_weight + self.rollout_joint_weight
        angular_loss = (self.rollout_rotation_weight * rotation_loss
                        + self.rollout_joint_weight * joint_loss) / max(angular_weight, 1e-12)
        return total.float(), position_loss.float(), angular_loss.float()

    def _log_physical_metrics(self, prefix, force, hand, batch, *, point_target=None):
        """Accumulate physical RMSE metrics into the rank-local buffer.

        Buffered instead of per-batch self.log: the active-entry condition
        differs per rank on mixed validation data, and per-batch sync_dist
        logging would desynchronize the NCCL collective order.
        """
        # The dataset stores the real-frame mask on the flattened 30 Hz axis
        # ([B, 96]), while predictions retain the token/substep axes
        # ([B, 32, 3, ...]).  Restore that layout and restrict to the
        # aligned substep: the s > 0 stitched entries would report
        # exact-zero error against their own GT.
        mask30 = batch["real_mask_30hz"].bool()
        if "force_prediction_valid_30hz" in batch:
            mask30 = mask30 & batch["force_prediction_valid_30hz"].to(mask30.device).bool()
        mask = mask30.reshape(mask30.shape[0], force.shape[1], -1)[:, :, 0]
        # Same clamped targets the losses train on: unclamped artifact rows
        # (up to 2400 N) would spike the RMSE against any bounded prediction.
        # The raw_local_mse diagnostic path trains in vertex-local space, so
        # its point RMSE is measured against the vertex-local targets too.
        if point_target is None:
            point_target = self._clamped_point_target(batch, force, local=self.raw_local_mse)
        target_force = point_target[:, :, 0]
        target_hand = batch["hand_wrench_camera_30hz"][:, :, 0].to(hand)
        if (self.wrench_force_clamp_n is not None
                or self.wrench_torque_clamp_nm is not None):
            target_hand = target_hand.clone()
            if self.wrench_force_clamp_n is not None:
                target_hand[..., :3] = clamp_force_magnitude(
                    target_hand[..., :3], self.wrench_force_clamp_n)
            if self.wrench_torque_clamp_nm is not None:
                target_hand[..., 3:] = clamp_force_magnitude(
                    target_hand[..., 3:], self.wrench_torque_clamp_nm)
        error = (force[:, :, 0] - target_force).square()
        point_mask = mask.view(mask.shape + (1,) * (error.ndim - mask.ndim))
        # Active-only RMSE: the plain RMSE is 91-96% zero-vs-zero entries and
        # barely moves; this tracks the entries the split loss trains on.
        active_mask = (point_mask
                       & (target_force.norm(dim=-1)
                          > self.force_active_threshold_n).unsqueeze(-1))
        hand_error = (hand[:, :, 0] - target_hand).square()
        hand_error = torch.where(mask.unsqueeze(-1), hand_error, 0.0)
        state = getattr(self, self._rmse_state_names[prefix])
        frames = mask.sum()
        updates = (
            torch.where(point_mask, error, 0.0).sum(), frames * math.prod(error.shape[2:]),
            (error * active_mask).sum(), active_mask.sum().float() * 3.0,
            hand_error[..., :3].sum(), frames * 3,
            hand_error[..., 3:].sum(), frames * (hand_error.shape[-1] - 3),
        )
        state.add_(torch.stack(updates).to(state.dtype))

    def _log_30hz_rollout(self, prefix, contact, batch, force=None, *, values=None):
        """Accumulate rollout metrics into the rank-local buffer.

        ``contact`` is the GT-damped hybrid (aligned substep predicted, the
        other two replaying GT), so these metrics score the model's
        physics consistency at 1/3 duty cycle and are NOT comparable with
        the retired full-prediction rollout numbers — nor with inference
        time, where no GT exists.

        Buffered instead of per-batch self.log: valid-window counts and the
        articulated skip differ per rank on mixed validation data, and
        per-batch sync_dist logging would desynchronize the NCCL collective
        order.
        """
        if not SubstepForceModel._batch_flag(batch, "rollout_eligible"):
            return
        if set(map(str, batch.get("dataset_mode", []))) == {"articulated_trajectory"}:
            return
        if values is None:
            if force is not None:
                scene = scene_rollout_inputs(force, batch)
                if scene is not None:
                    contact, batch = scene
            values, _ = self._compute_30hz_rollout(contact, batch)
        state = getattr(self, self._rollout_state_names[prefix])
        for h_index, horizon_10hz in enumerate(self._rollout_metric_horizons):
            metrics = values[str(3 * horizon_10hz)]["real_only"]
            count = int(metrics["valid_window_count"])
            if not count:
                continue
            for k_index, key in enumerate(self._rollout_metric_keys):
                if key not in metrics:
                    continue
                weight = float(
                    metrics.get("window_count", count)
                    if key == "explosion_rate"
                    else count
                )
                state[h_index, k_index, 0] += metrics[key] * weight
                state[h_index, k_index, 1] += weight

    def _compute_30hz_rollout(self, contact, batch, *, pose_loss=None):
        # rollout_metrics operates on the physical 30 Hz transition axis.
        contact = contact.reshape(contact.shape[0], -1, 6)
        rollout_batch = {
            "position": batch["position_30hz"],
            "rotation": batch["rotation_30hz"],
            "velocity": batch["velocity_30hz"],
            "real_mask": batch["real_mask_30hz"].bool() & batch["force_valid_30hz"].bool(),
            "physics": batch["physics_30hz"],
        }
        if "gravity_camera_30hz" in batch:
            rollout_batch["gravity"] = batch["gravity_camera_30hz"]
        if "force_prediction_valid_30hz" in batch:
            rollout_batch["prediction_mask"] = batch["force_prediction_valid_30hz"]
        # CUDA linalg.solve has no BF16 implementation, and normalized-mass
        # physics reaches condition numbers near 1e8.  Reporting rollout is
        # intentionally FP64 even when Lightning's surrounding step is under
        # bf16 autocast, matching the rollout-loss precision.
        training_implementation = self.cfg.loss.rollout.get("implementation", "reference")
        implementation = (
            "compiled_fused3"
            if training_implementation in {"compiled_fused3", "graphed_fused3"}
            else "reference"
        )
        with torch.autocast(device_type=contact.device.type, enabled=False):
            args = (
                contact.double(), rollout_batch,
                [3 * value for value in self._rollout_metric_horizons],
                ["real_only"],
                float(self.cfg.metrics.explosion_linear_speed_m_s),
                float(self.cfg.metrics.explosion_angular_speed_rad_s),
                str(self.cfg.metrics.get("rollout_integrator", "implicit_midpoint")),
                int(self.cfg.metrics.get("implicit_iterations", 6)),
            )
            if pose_loss is None:
                return rollout_metrics(*args, midpoint_implementation=implementation), None
            return rollout_metrics_and_pose_loss(
                *args, midpoint_implementation=implementation, pose_loss=pose_loss)

    def _log_step_loss_metrics(self, metrics, batch_size, batch_idx):
        """One packed reduction per optimizer update, including the epoch tail.

        Step curves are sample-weighted means over the accumulation window;
        epoch metrics retain Lightning's existing independent accumulation.
        Every rank enters on the same batch indices, even on contact-only
        data. Detached sums never participate in the training graph.
        """
        values = torch.stack([value.detach() for value in metrics.values()]).double()
        packed = torch.cat((values * batch_size, values.new_tensor([batch_size])))
        if self._step_loss_sums is None:
            self._step_loss_sums = packed
        else:
            self._step_loss_sums.add_(packed)
        completed = batch_idx + 1
        if (completed % self.trainer.accumulate_grad_batches != 0
                and completed != self.trainer.num_training_batches):
            return
        state = self._step_loss_sums
        assert state is not None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.SUM)
        means = state[:-1] / state[-1].clamp_min(1)
        for index, name in enumerate(metrics):
            # Already global; another sync_dist would undo the savings.
            self.log(name, means[index].float(), on_step=True, on_epoch=False,
                     sync_dist=False, batch_size=batch_size,
                     prog_bar=name == "step/objective/total")
        self._step_loss_sums = None

    @staticmethod
    def _batch_flag(batch, key):
        cached = batch.get(f"_{key}_all")
        if cached is not None:
            return cached
        value = batch.get(key)
        return value is None or bool(value.bool().all())

    def _shared_substep(self, batch, prefix, batch_idx):
        force_supervised = SubstepForceModel._batch_flag(batch, "force_supervision_valid")
        # Always run the force head.  Skipping it on contact-only batches
        # leaves the head's parameters without gradients; plain DDP (no
        # find_unused_parameters) then fails at the first such batch, and
        # find_unused_parameters=True instead wedged the 8-GPU runs: ranks
        # with different data presented different unused-parameter sets,
        # so the reducer skipped different gradient buckets per rank and
        # desynchronized the NCCL collective order at the epoch-end flush
        # (2026-09-14/15 hangs; see experiment=tactohand strategy note).
        predictions = self(
            batch["images"], batch.get("image_inverse"),
            compute_force=True)
        output = predictions.get("force_30hz") if isinstance(predictions, dict) else predictions
        if output is None:
            raise KeyError("network did not produce force_30hz")
        reference = output
        zero = reference.sum() * 0.0
        output = output.float()
        if (self.auxiliary_enabled and self.force_active_zero_split
                and batch.get("_empty_force_supervision") is True):
            # Still connect the force head to the objective for zero
            # gradients and identical DDP participation on every rank.
            force_zero = output.sum() * 0.0
            point_force_loss = wrench_force_loss = wrench_torque_loss = force_zero
            force = hand = contact = point_target = None
        else:
            point_target = self._clamped_point_target(batch, output, local=self.raw_local_mse)
            force, hand, contact = self._physical_outputs(
                output, batch, gate_logits=predictions.get(
                    "contact_gate_10hz"), point_target=point_target)
            point_force_loss = self._point_force_loss(force, batch, point_target=point_target)
            wrench_force_loss, wrench_torque_loss = self._wrench_losses(hand, batch)
        # A zero-weight gate (not a Python skip) keeps the head in the loss
        # graph on contact-only batches: the masked losses are exactly zero
        # there, so semantics are unchanged while every rank uses the same
        # parameters.
        supervision = 1.0 if force_supervised else 0.0
        effective_weight = self._rollout_weight_now()
        rollout_loss = zero.new_zeros(())
        position_loss = zero.new_zeros(())
        rotation_loss = zero.new_zeros(())
        rollout_windows = 0
        rollout_attempted = False
        rollout_failed = False
        rollout_metric_values = None
        report_rollout = False
        if force_supervised:
            maximum = (self.cfg.metrics.train_max_batches if prefix == "train"
                       else self.cfg.metrics.val_max_batches)
            report_rollout = maximum is None or batch_idx < int(maximum)
        # The sampler is dataset-homogeneous, not topology-homogeneous:
        # ARCTIC can put rigid and articulated samples in one batch. The
        # rigid rollout backend cannot consume that mixed physical layout.
        # Keep force/auxiliary supervision for the entire batch, and admit
        # rollout only when every sample is eligible and the mode is uniform.
        rollout_modes = set(map(str, batch.get("dataset_mode", ["rigid_trajectory"])))
        rollout_eligible = (SubstepForceModel._batch_flag(batch, "rollout_eligible")
            and rollout_modes in ({"rigid_trajectory"}, {"static_frame"}))
        if (force_supervised and self.rollout_weight > 0 and rollout_eligible
                and (prefix == "val" or effective_weight > 0)):
            assert contact is not None
            # Rigid samples carry 6D wrenches; articulated samples append one
            # generalized joint-force component. Preserve that physical axis.
            predicted_flat = contact.reshape(
                contact.shape[0], -1, contact.shape[-1])
            rollout_batch = batch
            scene = scene_rollout_inputs(force, batch)
            if scene is not None:
                predicted_flat, rollout_batch = scene
            zero = predicted_flat.sum() * 0.0
            rollout_attempted = True
            prepared_windows = batch.get("_rigid_rollout_windows") if scene is not None else None
            if prepared_windows is None:
                window_mask = self._rollout_window_mask(rollout_batch, 3 * self.rollout_horizon_10hz)
                rollout_windows = int(window_mask.sum())
            else:
                window_mask, rollout_windows = prepared_windows
            fused_rollout = self.cfg.loss.rollout.get("implementation") in {"compiled_fused3", "graphed_fused3"}
            shared_validation = (prefix == "val" and report_rollout and rollout_windows > 0
                                 and not torch.is_grad_enabled() and fused_rollout)
            checked_with_solver = fused_rollout and rollout_windows > 0 and not shared_validation
            try:
                if shared_validation:
                    request = PoseLossWindows(
                        3 * self.rollout_horizon_10hz, window_mask, rollout_windows,
                        self.rollout_position_scale, self.rollout_rotation_scale)
                    rollout_metric_values, losses = self._compute_30hz_rollout(
                        predicted_flat, rollout_batch, pose_loss=request)
                    assert losses is not None
                    rollout_loss, position_loss, rotation_loss = self._combine_rigid_rollout_losses(*losses)
                else:
                    rollout_loss, position_loss, rotation_loss = self._gt_pose_rollout_loss(
                        predicted_flat, rollout_batch, window_mask=window_mask,
                        valid_windows=rollout_windows, check_finite_with_solver=checked_with_solver)
                if not checked_with_solver and not bool(torch.isfinite(rollout_loss)):
                    # Near-singular systems can yield NaN/Inf without raising
                    # LinAlgError on some CUDA paths.
                    rollout_loss, position_loss, rotation_loss = zero, zero, zero
                    rollout_windows = 0
                    rollout_failed = True
                elif prefix == "train":
                    # Step-level logs are train-only: the val loop also runs
                    # this branch (for the epoch rollout metrics), and its
                    # on_step writes would land at val-batch-index "steps"
                    # (0..n_val_batches), polluting the train step curve.
                    # Rank-local logging: deliberately no sync_dist, a
                    # per-batch conditional allreduce would desynchronize the
                    # NCCL collective order on multi-rank runs.
                    self.log("step/loss/rollout_position", position_loss,
                             on_step=True, on_epoch=False)
                    self.log("step/loss/rollout_rotation", rotation_loss,
                             on_step=True, on_epoch=False)
            except torch._C._LinAlgError:
                # Extreme force predictions can make the implicit midpoint
                # integrator's system exactly singular; the rollout is a
                # auxiliary signal, so skip the batch rather
                # than crash the distributed run.
                rollout_loss, position_loss, rotation_loss = zero, zero, zero
                rollout_windows = 0
                rollout_failed = True
        if self.rollout_weight > 0:
            self._accumulate_rollout_loss(
                prefix, batch, (rollout_loss, position_loss, rotation_loss),
                rollout_windows, rollout_attempted, rollout_failed)
        loss = (
            supervision * (
                self.point_force_weight * point_force_loss
                + self.wrench_force_weight * wrench_force_loss
                + self.wrench_torque_weight * wrench_torque_loss)
            + effective_weight * rollout_loss
        )
        gate_loss = zero.new_zeros(())
        contact_loss = zero.new_zeros(())
        proximity_loss = zero.new_zeros(())
        if self.auxiliary_enabled:
            gate_loss, contact_loss, proximity_loss = self._auxiliary_losses(
                predictions, batch)
            loss = (loss + self.contact_gate_weight * gate_loss
                    + self.contact_weight * contact_loss
                    + self.proximity_weight * proximity_loss)
        if self.network.visibility_head is not None:
            visibility_loss = self._hand_visibility_loss(
                predictions["hand_visibility_logits_10hz"], batch, prefix)
            loss = loss + self.hand_visibility_weight * visibility_loss
            self.log(f"epoch/objective/{prefix}/hand_visibility", visibility_loss,
                     on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=batch["real_mask"].shape[0])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "non-finite 30 Hz force objective: "
                f"point_force={float(point_force_loss.detach())}, "
                f"wrench_force={float(wrench_force_loss.detach())}, "
                f"wrench_torque={float(wrench_torque_loss.detach())}, "
                f"rollout={float(rollout_loss.detach())}"
            )
        batch_size = batch["real_mask"].shape[0]
        metric_prefix = prefix
        if prefix == "val" and "dataset_name" in batch:
            dataset_names = set(map(str, batch["dataset_name"]))
            if len(dataset_names) != 1:
                raise ValueError(
                    f"validation batches must be dataset-homogeneous, got {dataset_names}")
            metric_prefix = f"val/{next(iter(dataset_names))}"
        if prefix == "train":
            step_metrics = {
                "step/objective/total": loss,
                "step/loss/point_force": point_force_loss,
                "step/loss/wrench_force": wrench_force_loss,
                "step/loss/wrench_torque": wrench_torque_loss,
            }
            if self.auxiliary_enabled:
                step_metrics.update({
                    "step/loss/contact_gate": gate_loss,
                    "step/loss/contact": contact_loss,
                    "step/loss/proximity": proximity_loss,
                })
            step_metrics.update({
                "step/weighted/point_force": self.point_force_weight * point_force_loss,
                "step/weighted/wrench_force": self.wrench_force_weight * wrench_force_loss,
                "step/weighted/wrench_torque": self.wrench_torque_weight * wrench_torque_loss,
            })
            self._log_step_loss_metrics(step_metrics, batch_size, batch_idx)
        self.log(f"epoch/objective/{prefix}/total", loss, on_step=False, on_epoch=True,
                 prog_bar=prefix == "val", sync_dist=True, batch_size=batch_size)
        # Logged unconditionally (rank-invariant collective count; the
        # masked losses are exactly zero on contact-only batches, and the
        # zero batch_size keeps them out of the epoch mean).
        force_log = dict(on_step=False, on_epoch=True, sync_dist=True,
                         batch_size=batch_size if force_supervised else 0)
        self.log(f"loss/{prefix}/point_force", point_force_loss, **force_log)
        self.log(f"loss/{prefix}/wrench_force", wrench_force_loss, **force_log)
        self.log(f"loss/{prefix}/wrench_torque", wrench_torque_loss, **force_log)
        if self.auxiliary_enabled:
            self.log(f"epoch/objective/{prefix}/contact_gate", gate_loss,
                     on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=batch_size)
            self.log(f"epoch/objective/{prefix}/contact", contact_loss,
                     on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=batch_size)
            self.log(f"epoch/objective/{prefix}/proximity", proximity_loss,
                     on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=batch_size)
            with torch.no_grad():
                real = batch["real_mask"].to(reference.device).bool().unsqueeze(-1)
                if "metric_sample_valid" in batch:
                    sample_valid = batch["metric_sample_valid"].to(
                        reference.device).bool().view(-1, 1, 1)
                    real = real & sample_valid
                valid = batch["contact_valid_10hz"].to(reference.device).bool() & real
                target = batch["contact_target_10hz"].to(reference.device).bool()
                gate_target = self._gate_target(target, valid)
                gate_prob = torch.sigmoid(predictions["contact_gate_10hz"].to(reference))
                cond_prob = torch.sigmoid(
                    predictions["contact_logits_10hz"].to(reference))
                predicted = (gate_prob.unsqueeze(-1) * cond_prob) > 0.5
                cond_predicted = cond_prob > 0.5
                gate_predicted = gate_prob > 0.5
                mask = valid.unsqueeze(-1)
                tp = (predicted & target & mask).sum().float()
                fp = (predicted & ~target & mask).sum().float()
                fn = (~predicted & target & mask).sum().float()
                cond_tp = (cond_predicted & target & mask).sum().float()
                cond_fp = (cond_predicted & ~target & mask).sum().float()
                cond_fn = (~cond_predicted & target & mask).sum().float()
                gate_tp = (gate_predicted & gate_target).sum().float()
                gate_fp = (gate_predicted & ~gate_target).sum().float()
                gate_fn = (~gate_predicted & gate_target).sum().float()
                proximity_valid = (
                    batch["proximity_valid_10hz"].to(reference.device).bool()
                    & real).unsqueeze(-1)
                proximity_abs_error = (
                    (predictions["proximity_10hz"]
                     - batch["proximity_target_10hz"].to(reference.device)).abs()
                    * proximity_valid).sum()
                proximity_count = proximity_valid.sum().mul(
                    predictions["proximity_10hz"].shape[-1]
                )
                self._update_contact_metric_state(
                    metric_prefix, tp, fp, fn,
                    proximity_abs_error, proximity_count.float(),
                    gate_tp, gate_fp, gate_fn, cond_tp, cond_fp, cond_fn)
                if metric_prefix != prefix:
                    # Per-dataset group: also accumulate the global group so
                    # epoch/metrics/val/contact/f1 aggregates across all
                    # validation datasets instead of reading zeros.
                    self._update_contact_metric_state(
                        prefix, tp, fp, fn,
                        proximity_abs_error, proximity_count.float(),
                        gate_tp, gate_fp, gate_fn, cond_tp, cond_fp, cond_fn)
        # rollout_weight is rank-invariant; the zero defaults keep the call
        # count (and thus the sync_dist collective order) identical on every
        # rank even when this batch did not compute a rollout.
        if self.rollout_weight > 0:
            # Raw losses are window-pooled at epoch end. This weighted term
            # remains a batch-weighted contribution to the actual objective.
            self.log(f"epoch/objective/{prefix}/rollout_weighted",
                     effective_weight * rollout_loss, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
            if prefix == "train":
                self.log("epoch/schedule/rollout_weight", effective_weight,
                         on_step=False, on_epoch=True, sync_dist=True,
                         batch_size=batch_size)
        if force_supervised:
            with torch.no_grad():
                self._log_physical_metrics(
                    metric_prefix, force.detach(), hand.detach(), batch,
                    point_target=point_target)
                if report_rollout:
                    self._log_30hz_rollout(
                        prefix, contact.detach(), batch, force=force.detach(),
                        values=rollout_metric_values)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_substep(batch, "train", batch_idx)

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self._timing_enabled:
            self._timing_sync()
            self._timing_backward_start = time.perf_counter()
        return super().on_before_backward(loss)

    def on_after_backward(self) -> None:
        if self._timing_enabled:
            self._timing_sync()
            self._timing_last["backward_s"] = (
                time.perf_counter() - self._timing_backward_start
            )
        return super().on_after_backward()

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        if not self._timing_enabled:
            return super().optimizer_step(*args, **kwargs)
        positional = list(args)
        closure = kwargs.get("optimizer_closure")
        closure_index = None
        if closure is None and len(positional) >= 4 and callable(positional[3]):
            closure_index, closure = 3, positional[3]
        closure_s = 0.0

        def timed_closure():
            nonlocal closure_s
            assert closure is not None
            self._timing_sync()
            closure_started = time.perf_counter()
            result = closure()
            self._timing_sync()
            closure_s += time.perf_counter() - closure_started
            return result

        if closure is not None:
            if closure_index is None:
                kwargs["optimizer_closure"] = timed_closure
            else:
                positional[closure_index] = timed_closure
        self._timing_sync()
        started = time.perf_counter()
        super().optimizer_step(*positional, **kwargs)
        self._timing_sync()
        self._timing_last["optimizer_step_s"] += max(
            0.0, time.perf_counter() - started - closure_s
        )

    def validation_step(self, batch, batch_idx):
        self._shared_substep(batch, "val", batch_idx)

    def configure_optimizers(self):
        opt = self.cfg.training.optimizer
        backbone = list(self.network.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        head = [parameter for parameter in self.parameters()
                if parameter.requires_grad and id(parameter) not in backbone_ids]
        optimizer_options = {}
        if bool(opt.get("fused", False)):
            optimizer_options.update(fused=True, foreach=False)
        elif opt.get("foreach") is False:
            optimizer_options.update(fused=False, foreach=False)
        optimizer = torch.optim.AdamW(
            [{"params": backbone, "lr": float(opt.backbone_lr)},
             {"params": head, "lr": float(opt.lr)}],
            weight_decay=float(opt.weight_decay), betas=tuple(opt.betas),
            **optimizer_options)
        if optimizer_options:
            def apply_execution_options(_optimizer, state_dict):
                # Apply the selected kernel before PyTorch casts Adam's
                # step tensors. Keep the saved moments, LR and schedule.
                groups = [dict(group, **optimizer_options) for group in state_dict["param_groups"]]
                return dict(state_dict, param_groups=groups)

            optimizer.register_load_state_dict_pre_hook(apply_execution_options)
        scheduler_cfg = self.cfg.training.scheduler
        if str(scheduler_cfg.get("name", "none")).lower() != "cosine":
            return optimizer
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        min_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.1))
        warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0.0))
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError("training.scheduler.min_lr_ratio must be in [0, 1]")
        if not math.isfinite(warmup_epochs) or warmup_epochs < 0.0:
            raise ValueError("training.scheduler.warmup_epochs must be finite and nonnegative")
        max_epochs = max(1, int(self.cfg.training.max_epochs))
        explicit_steps = scheduler_cfg.get("warmup_steps", None)
        if explicit_steps is not None:
            warmup_steps = min(total_steps, int(explicit_steps))
        else:
            warmup_steps = min(
                total_steps, int(round(total_steps * warmup_epochs / max_epochs))
            )

        def cosine(step: int) -> float:
            # LambdaLR applies lambda(0) during construction.
            return warmup_cosine_multiplier(
                step, total_steps, warmup_steps, min_ratio
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
