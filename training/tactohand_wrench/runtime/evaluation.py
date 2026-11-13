"""TactoHand checkpoint, physical-output and metric-view adapter."""
from __future__ import annotations
import time
import math
from typing import Any, Dict, Optional
import torch
from training.evaluation.view import BatchView
from training.evaluation.targets import _native_contact_target, _native_contact_valid
from .checkpoints import checkpoint_config, load_checkpoint_state, load_prediction_config
PROXIMITY_CLIP_M = 0.25
_DECODE_EPS = 1e-6

def build_view(model, predictions, batch, force, hand, tau: float,
               fps: float, fallback_dataset: Optional[str] = None
               ) -> BatchView:
    """Normalize one batch's outputs into the metric-family contract.

    ``fallback_dataset`` names the loader's single dataset for the
    single-child case, where the datamodule skips the CombinedLabelDataset
    wrapper and batches carry no ``dataset_name`` field.
    """
    device = force.device
    real = batch["real_mask"].to(device).bool().unsqueeze(-1)  # [B,T,1]
    if "metric_sample_valid" in batch:
        sample_valid = batch["metric_sample_valid"].to(device).bool()
        real = real & sample_valid.view(-1, 1, 1)

    gate_prob = torch.sigmoid(predictions["contact_gate_10hz"].float())
    cond_prob = torch.sigmoid(predictions["contact_logits_10hz"].float())
    contact_prob = gate_prob.unsqueeze(-1) * cond_prob
    # Scoring targets: native (crop-override-free) when available — see
    # _native_contact_target above.  Training losses keep reading
    # contact_target_10hz.
    target = _native_contact_target(batch, device)
    valid = _native_contact_valid(batch, device) & real
    gate_target = valid & (target.sum(dim=-1) > 0)

    proximity_prob = predictions["proximity_10hz"].float()
    proximity_target = batch["proximity_target_10hz"].to(device).float()
    proximity_valid = (batch["proximity_valid_10hz"].to(device).bool()
                       & real).unsqueeze(-1)          # [B,T,H,1]
    sat = math.exp(-tau * PROXIMITY_CLIP_M)
    distance_mm_hat = (-torch.log(proximity_prob.clamp_min(_DECODE_EPS))
                       / tau * 1000.0)
    # GT decoding saturates at the clip bound; exact zeros (absent-hand
    # present-with-zeros supervision) map to the same bound.
    distance_mm_gt = (-torch.log(proximity_target.clamp(min=sat, max=1.0))
                      / tau * 1000.0)

    source_frames = batch.get("source_frames")
    if source_frames is None:
        # No canonical timeline (static datasets): accel triples never gate on.
        source_frames = -torch.arange(
            batch["real_mask"].shape[1], device=device,
            dtype=torch.float32).expand(batch["real_mask"].shape[0], -1)
    else:
        source_frames = source_frames.to(device)

    target_force = model._clamped_point_target(batch, force)
    target_hand = batch["hand_wrench_camera_30hz"].to(hand)
    if (model.wrench_force_clamp_n is not None
            or model.wrench_torque_clamp_nm is not None):
        from core.geometry.force_frames import clamp_force_magnitude
        target_hand = target_hand.clone()
        if model.wrench_force_clamp_n is not None:
            target_hand[..., :3] = clamp_force_magnitude(
                target_hand[..., :3], model.wrench_force_clamp_n)
        if model.wrench_torque_clamp_nm is not None:
            target_hand[..., 3:] = clamp_force_magnitude(
                target_hand[..., 3:], model.wrench_torque_clamp_nm)

    force_gate = batch.get("force_prediction_valid_30hz", batch["force_valid_30hz"]).to(device).bool()
    force_gate = force_gate.reshape(force.shape[0], force.shape[1],
                                    force.shape[2])
    # Score only the image-aligned substep 0: ``force`` is the GT-damped
    # hybrid whose s > 0 entries are GT stitches — exact-zero error there
    # would dilute every force/wrench family.
    aligned = torch.zeros_like(force_gate)
    aligned[:, :, 0] = True
    real30 = batch["real_mask_30hz"].to(device).bool().reshape_as(force_gate)
    force_mask = force_gate & real30 & aligned   # [B,T,S], s = 0 only
    magnitude = target_force.norm(dim=-1)                # [B,T,S,H,V]
    active = (force_mask.unsqueeze(-1).unsqueeze(-1)
              & (magnitude > model.force_active_threshold_n))  # [B,T,S,H,V]

    names = batch.get("dataset_name")
    if names is None:
        if fallback_dataset is None:
            raise ValueError(
                "batch carries no dataset_name and no fallback configured "
                "(single-dataset loaders bypass the Combined wrapper)")
        names = [fallback_dataset]
    unique = set(map(str, names))
    if len(unique) != 1:
        raise ValueError(f"eval batches must be dataset-homogeneous, got {unique}")

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
        force_pred=force.detach(),
        force_target=target_force.detach(),
        force_mask=force_mask,
        active_mask=active,
        hand_pred=hand.detach(),
        hand_target=target_hand.detach(),
        force_supervised=bool(batch.get(
            "force_supervision_valid", torch.tensor(True)).bool().all()),
        batch_size=int(batch["real_mask"].shape[0]),
        tau_per_m=tau,
        fps=fps,
    )


def build_model(cfg, checkpoint, device, phase):
    t0 = time.perf_counter()
    task = str(cfg.get("task", "tactohand_wrench"))
    if task != "tactohand_wrench":
        raise ValueError(f"unsupported task for this harness: {task}")
    t_imp = time.perf_counter()
    from training.tactohand_wrench.model import SubstepForceModel
    t0 = phase("model module imports", t_imp)
    t0 = time.perf_counter()
    # The training config builds a DINOv3-initialized backbone (a ~43s
    # safetensors read); the checkpoint's state_dict overwrites every
    # one of those weights right after, so eval builds the same module
    # graph with random init instead.  timm's module structure does not
    # depend on the pretrained flag.
    cfg = checkpoint_config(cfg)
    # Metric-only evaluation does not consume training class priors or
    # reduced-mesh loss assets. Their buffers are nonpersistent; prediction
    # parameters and checkpoint keys remain identical.
    model = SubstepForceModel(cfg, initialize_losses=False)
    t0 = phase("model construct", t0)
    state_dict = load_checkpoint_state(checkpoint)
    if not any(key.startswith("network.") for key in state_dict):
        state_dict = {f"network.{key}": value for key, value in state_dict.items()}
    t0 = phase(f"checkpoint load ({checkpoint.name})", t0)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the run config model "
            f"(missing={len(missing)}, unexpected={len(unexpected)}); "
            f"first missing: {missing[:3]}, first unexpected: "
            f"{unexpected[:3]}")
    t0 = phase("state_dict load", t0)
    model.to(device).eval()
    phase("move to device", t0)
    return model


class TactoHandPredictor:
    """The run's own checkpoint model; the historical forward path."""

    name = "tactohand"

    def __init__(self, model) -> None:
        self.model = model

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        predictions = self.model(
            batch["images"], batch.get("image_inverse"), compute_force=True)
        gate = torch.sigmoid(predictions["contact_gate_10hz"].float())
        cond = torch.sigmoid(predictions["contact_logits_10hz"].float())
        return {
            "contact_prob": gate.unsqueeze(-1) * cond,
            "gate_prob": gate,
            "cond_prob": cond,
            "predictions": predictions,
        }


    def physical_outputs(self, predictions, batch):
        output = predictions.get("force_30hz")
        if output is None:
            raise KeyError("network did not produce force_30hz")
        return self.model._physical_outputs(
            output.float(), batch,
            gate_logits=predictions.get("contact_gate_10hz"))

    def metric_view(self, predictions, batch, force, hand, tau, fps,
                    fallback_dataset=None):
        return build_view(self.model, predictions, batch, force, hand,
                          tau, fps, fallback_dataset)
