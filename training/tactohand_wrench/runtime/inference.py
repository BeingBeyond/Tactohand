"""Label-free prediction and explicit, optional physical force decoding."""
from __future__ import annotations

import math

import numpy as np
from omegaconf import DictConfig
import torch

from core.geometry.force_frames import CAMERA_FRAME, VERTEX_LOCAL_FRAME, clamp_force_magnitude, force_local_to_camera, validate_vertex_force_frame


def validate_inference_config(cfg: DictConfig) -> None:
    model = cfg.model.wrench
    if (str(model.get("output_mode")) != "vertex_force"
            or int(model.get("hands", 1)) != 2
            or not model.get("predict_contact_proximity", False)):
        raise ValueError("Standalone inference requires the current two-hand contact/proximity/vertex-force model")
    if int(model.get("output_substeps", 1)) != 1:
        raise ValueError("Standalone inference supports output_substeps=1; use evaluation for legacy S=3 checkpoints")
    if cfg.loss.get("wrench", {}).get("raw_local_mse", False):
        raise ValueError("Standalone inference does not support raw_local_mse diagnostic configs")


def validate_images(images: np.ndarray, cfg: DictConfig,
                    normalized: bool = False) -> torch.Tensor:
    if images.ndim == 4:
        images = images[None]
    size = int(cfg.data.image_size)
    if (images.ndim != 5 or images.shape[2:] != (3, size, size)
            or min(images.shape[:2]) < 1):
        raise ValueError(f"images must have shape [B,T,3,{size},{size}] or [T,3,{size},{size}]")
    if normalized:
        if not np.issubdtype(images.dtype, np.floating) or not np.isfinite(images).all():
            raise ValueError("--normalized requires finite, already normalized floating-point images")
        images = images.astype(np.float32)
    elif images.dtype != np.uint8:
        raise ValueError("Default images must be RGB uint8; use --normalized only for training-normalized floats")
    return torch.from_numpy(np.ascontiguousarray(images))


def decode_predictions(output: dict[str, torch.Tensor], cfg: DictConfig,
                       geometry: dict[str, np.ndarray] | None = None,
                       ) -> dict[str, torch.Tensor]:
    """Decode image-aligned outputs; never stitch GT forces or run rollout."""
    validate_inference_config(cfg)
    force = output["force_30hz"].float()
    if force.ndim != 5 or force.shape[-3:] != (2, 778, 3):
        raise ValueError("Standalone inference requires two-hand vertex-force outputs")
    gate = output["contact_gate_10hz"].float().sigmoid()
    conditional = output["contact_logits_10hz"].float().sigmoid()
    proximity = output["proximity_10hz"].float()
    tau = float(cfg.data.get("proximity_tau", 10.0))
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("data.proximity_tau must be positive and finite")
    force_gate = bool(cfg.loss.get("contact_proximity", {}).get("force_gate", True))
    result = {
        "contact_prob": gate.unsqueeze(-1) * conditional,
        "contact_gate_prob": gate,
        "contact_cond_prob": conditional,
        "proximity_prob": proximity,
        "distance_m": -torch.log(proximity.clamp_min(1e-6)) / tau,
        "force_raw_10hz": force,
        "force_normalized_10hz": (force * gate[..., None, None] if force_gate else force),
    }
    if "hand_visibility_logits_10hz" in output:
        result["hand_visibility_prob_10hz"] = output["hand_visibility_logits_10hz"].float().sigmoid()
    frame = validate_vertex_force_frame(cfg.data.get("vertex_force_frame", CAMERA_FRAME))
    if geometry is None:
        return result
    if "point_force_scale" not in geometry:
        raise ValueError("Geometry must supply the checkpoint's training point_force_scale; no default physical scale is inferred")
    scale = torch.as_tensor(geometry["point_force_scale"], device=force.device, dtype=force.dtype)
    if (scale.numel() not in (1, force.shape[0])
            or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError("point_force_scale must be a positive finite scalar or one value per batch item")
    physical = force * scale.reshape(-1, 1, 1, 1, 1)
    if force_gate:
        physical = physical * gate[..., None, None]
    if frame == VERTEX_LOCAL_FRAME:
        if "vertex_local_yz_camera_10hz" not in geometry:
            raise ValueError("Local force decoding requires vertex_local_yz_camera_10hz [B,T,2,778,2,3]")
        axes = torch.as_tensor(geometry["vertex_local_yz_camera_10hz"], device=force.device, dtype=force.dtype)
        if axes.shape != (*force.shape[:-1], 2, 3) or not torch.isfinite(axes).all():
            raise ValueError("Local axes must be finite [B,T,2,778,2,3] arrays")
        y_axis, z_axis = axes.unbind(-2)
        if (not torch.allclose(axes.norm(dim=-1), torch.ones_like(axes[..., 0]), atol=1e-3, rtol=0)
                or not torch.allclose((y_axis * z_axis).sum(-1), torch.zeros_like(y_axis[..., 0]), atol=1e-3, rtol=0)):
            raise ValueError("Local y/z axes must be orthonormal in camera coordinates")
        physical = force_local_to_camera(physical, axes)
    if "hand_present_10hz" in geometry:
        present = torch.as_tensor(geometry["hand_present_10hz"], device=force.device)
        if present.shape != force.shape[:3] or not ((present == 0) | (present == 1)).all():
            raise ValueError("hand_present_10hz must be boolean/0-1 [B,T,2]")
        physical = physical * present[..., None, None]
    limit = cfg.loss.get("wrench", {}).get("point_force_clamp_n")
    if limit is not None:
        limit = float(limit)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("point_force_clamp_n must be positive and finite or null")
        physical = clamp_force_magnitude(physical, limit)
    result["vertex_force_camera_n_10hz"] = physical
    return result
