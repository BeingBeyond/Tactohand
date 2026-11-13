"""The tactohand force network: per-frame backbone + temporal fusion + query heads."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import timm


class TemporalFeatureEncoder(nn.Module):
    """Temporal encoder shared by the matched wrench-query variants."""

    def __init__(self, dim: int, *, layers: int, heads: int, dropout: float,
                 max_seq_len: int = 128) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        time = features.shape[1]
        if time > self.position.shape[1]:
            raise ValueError("sequence is longer than max_seq_len")
        return self.norm(self.encoder(features + self.position[:, :time]))


class ForceQueryHead(nn.Module):
    """Existing task-query decoder/fusion pattern, with one force-task query.

    A single semantic query produces either 6 wrench values or 778x3 forces;
    these are not 778 cross-attention queries.
    """

    def __init__(self, dim: int, output_dim: int, *, heads: int, layers: int,
                 dropout: float = 0.0, zero_output: bool = True):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            dim, heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", norm_first=True, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.output = nn.Linear(dim, output_dim)
        # A dense head otherwise sums 778 independent random outputs into a
        # very large initial wrench.  Both parameterizations start from the
        # same zero hand-wrench prediction for a fair optimization comparison.
        if zero_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, temporal):
        batch, time, _ = temporal.shape
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(query, temporal).expand(-1, time, -1)
        return self.output(self.fusion(torch.cat((temporal, decoded), -1)))


class FrameContactProximityHead(nn.Module):
    """Predict two-hand vertex contact and proximity at input-frame rate."""

    def __init__(self, dim, *, heads, layers, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(2, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            dim, heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", norm_first=True, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.fusion = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
            for _ in range(2)
        ])
        self.output = nn.ModuleList([nn.Linear(dim, 2 * 778) for _ in range(2)])
        # Stage-1 engagement gate: one logit per (frame, hand) on the same
        # fused task features as the per-vertex heads.
        self.gate_output = nn.ModuleList([
            nn.Linear(dim, 1) for _ in range(2)])

    def forward(self, temporal):
        batch, time, _ = temporal.shape
        decoded = self.decoder(
            self.queries.unsqueeze(0).expand(batch, -1, -1), temporal)
        values = []
        gates = []
        for index in range(2):
            task = decoded[:, index:index + 1].expand(-1, time, -1)
            fused = self.fusion[index](torch.cat((temporal, task), -1))
            values.append(self.output[index](fused).reshape(batch, time, 2, 778))
            gates.append(self.gate_output[index](fused))
        return {
            "contact_logits_10hz": values[0],
            "proximity_10hz": values[1].sigmoid(),
            "contact_gate_10hz": torch.cat(gates, dim=-1),
        }


class QueryForceWrenchNetwork(nn.Module):
    """DINOv3 backbone, temporal feature encoder, and query force heads."""

    def __init__(self, cfg):
        super().__init__()
        pretrained_path = cfg.get("pretrained_path")
        if pretrained_path:
            pretrained_path = str(Path(pretrained_path).expanduser().resolve())
            if not Path(pretrained_path).is_file():
                raise FileNotFoundError(
                    f"Local backbone checkpoint not found: {pretrained_path}. "
                    "Set PRETRAINED_PATH or model.wrench.pretrained_path to the matching "
                    "checkpoint; use pretrained=false and pretrained_path=null for scratch/smoke runs.")
        backbone_kwargs = {}
        if cfg.get("backbone_img_size") is not None:
            backbone_kwargs["img_size"] = int(cfg["backbone_img_size"])
        self.backbone = timm.create_model(
            cfg["backbone"],
            # A configured local checkpoint is authoritative and must never
            # trigger one Hugging Face request per DDP rank.
            pretrained=bool(cfg["pretrained"]) and not pretrained_path,
            checkpoint_path=pretrained_path or "",
            num_classes=0,
            **backbone_kwargs,
        )
        if bool(cfg.get("gradient_checkpointing", False)):
            if not hasattr(self.backbone, "set_grad_checkpointing"):
                raise ValueError("requested gradient checkpointing is unsupported by backbone")
            self.backbone.set_grad_checkpointing(True)
        self.frozen_backbone = bool(cfg.get("freeze_backbone", False))
        if self.frozen_backbone:
            self.backbone.requires_grad_(False)
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False)
        dim = int(self.backbone.num_features)
        self.temporal = TemporalFeatureEncoder(
            dim, layers=int(cfg["temporal_layers"]), heads=int(cfg["temporal_heads"]),
            dropout=float(cfg["dropout"]))
        self.output_mode = str(cfg["output_mode"])
        self.output_substeps = int(cfg.get("output_substeps", 1))
        self.hands = int(cfg.get("hands", 1))
        self.predict_contact_proximity = bool(cfg.get(
            "predict_contact_proximity", False))
        # Single-frame ablation switch (false = no cross-frame information
        # flow anywhere; parameters and output shapes unchanged).
        self.temporal_fusion = bool(cfg.get("temporal_fusion", True))
        if self.output_substeps < 1:
            raise ValueError("output_substeps must be positive")
        if self.output_mode not in {"direct", "vertex_force"}:
            raise ValueError("output_mode must be direct or vertex_force")
        self.query_head = ForceQueryHead(
            dim, 6 if self.output_mode == "direct" else
            self.hands * 778 * 3 * self.output_substeps,
            heads=int(cfg.get("query_heads", 8)), layers=int(cfg.get("query_layers", 2)),
            dropout=float(cfg["dropout"]))
        self.auxiliary_head = None
        if self.predict_contact_proximity:
            if self.output_mode != "vertex_force" or self.hands != 2:
                raise ValueError(
                    "contact/proximity prediction requires hands=2 vertex forces")
            self.auxiliary_head = FrameContactProximityHead(
                dim, heads=int(cfg.get("query_heads", 8)),
                layers=int(cfg.get("query_layers", 2)),
                dropout=float(cfg["dropout"]))
        self.visibility_head = None
        if cfg.get("predict_hand_visibility", False):
            self.visibility_head = ForceQueryHead(
                dim, 2, heads=int(cfg.get("query_heads", 8)), layers=1,
                dropout=float(cfg["dropout"]), zero_output=False)

    def forward_features(self, features, compute_force=True):
        batch, time = features.shape[0], features.shape[1]
        if not self.temporal_fusion:
            # Single-frame ablation: every frame passes through the full
            # network independently, so all three cross-frame attention
            # paths degenerate at once ([B,T,D] -> [B*T,1,D] and back).
            features = features.reshape(batch * time, 1, features.shape[-1])
        temporal = self.temporal(features)
        auxiliary = self.auxiliary_head(temporal) if self.auxiliary_head is not None else {}
        if self.visibility_head is not None:
            auxiliary["hand_visibility_logits_10hz"] = self.visibility_head(temporal)
        if auxiliary and not self.temporal_fusion:
            auxiliary = {key: value.reshape(batch, time, *value.shape[2:])
                         for key, value in auxiliary.items()}
        if not compute_force:
            if not auxiliary:
                raise ValueError("cannot skip force without an auxiliary output head")
            return auxiliary
        output = self.query_head(temporal)
        if not self.temporal_fusion:
            output = output.reshape(batch, time, output.shape[-1])
        if self.output_mode == "vertex_force":
            output = output.reshape(
                *output.shape[:-1], self.output_substeps, self.hands, 778, 3)
            if self.output_substeps == 1:
                output = output.squeeze(-4)
            if self.hands == 1:
                output = output.squeeze(-3)
        if auxiliary:
            return {"force_30hz": output, **auxiliary}
        return output

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            images = images.float().div(255.0)
            images = (images - self.image_mean) / self.image_std
        return images

    def _backbone_forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = self._prepare_images(images)
        if self.frozen_backbone:
            # Skip autograd graph construction entirely: the backbone's output
            # never requires grad, so the backward pass stops at the temporal
            # encoder's input.
            with torch.no_grad():
                return self.backbone(pixels)
        return self.backbone(pixels)

    def forward(
        self, images: torch.Tensor, image_inverse: torch.Tensor | None = None,
        compute_force: bool = True,
    ) -> torch.Tensor:
        if image_inverse is None:
            if images.ndim != 5:
                raise ValueError("images must have shape [batch, time, channel, height, width]")
            batch, time = images.shape[:2]
            flat_images = images.reshape(batch * time, *images.shape[2:])
            features = self._backbone_forward(flat_images)
            features = features.reshape(batch, time, -1)
        else:
            if images.ndim != 4 or image_inverse.ndim != 2:
                raise ValueError(
                    "deduplicated images and inverse must have shapes [unique,C,H,W] and [B,T]")
            features = self._backbone_forward(images)
            features = features[image_inverse]
        return self.forward_features(features, compute_force)
