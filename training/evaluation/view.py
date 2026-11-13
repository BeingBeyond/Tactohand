"""Eval-view contract shared by metric families.

A :class:`BatchView` is the normalized per-batch record the harness builds
once from the model outputs and the label batch; metric families consume it
without touching the model or the raw batch again.  Keeping the decode and
mask logic in one place makes metric definitions one-liners and guarantees
every family gates on the same validity rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BatchView:
    """Decoded predictions + targets + validity masks for one batch.

    Shapes follow the trajectory-window convention: ``B`` windows, ``T``
    tokens (10 Hz), ``S`` substeps (30 Hz force axis), ``H`` hands,
    ``V`` MANO vertices.  Proximity/contact live on the token axis;
    forces/wrenches on the substep axis.

    Masks are all boolean and already broadcastable to their tensor:
    ``contact_mask`` [B,T,H,1], ``proximity_mask`` [B,T,H,1],
    ``force_mask`` [B,T,S], ``active_mask`` [B,T,S,H,V].
    """

    dataset_name: str

    # --- contact (10 Hz) -------------------------------------------------
    contact_prob: torch.Tensor          # [B,T,H,V] deployed sigmoid(gate)*sigmoid(logit)
    cond_prob: torch.Tensor             # [B,T,H,V] stage-2 conditional probability
    gate_prob: torch.Tensor             # [B,T,H]   stage-1 engagement probability
    contact_target: torch.Tensor        # [B,T,H,V] bool
    contact_mask: torch.Tensor          # [B,T,H,1] valid & real & sample-valid
    gate_target: torch.Tensor           # [B,T,H]   bool (engagement on valid slots)

    # --- proximity (10 Hz) -----------------------------------------------
    proximity_prob: torch.Tensor        # [B,T,H,V] sigmoid output in [0,1]
    proximity_target: torch.Tensor      # [B,T,H,V] exp(-tau*clip(d,0,d_max))
    proximity_mask: torch.Tensor        # [B,T,H,1] valid & real & sample-valid
    distance_mm_hat: torch.Tensor       # [B,T,H,V] decoded -ln(p)/tau*1000
    distance_mm_gt: torch.Tensor        # [B,T,H,V] same decode on the target
    source_frames: torch.Tensor         # [B,T] canonical label-timeline frame
    real_tokens: torch.Tensor           # [B,T] bool

    # --- force (30 Hz substeps) ------------------------------------------
    force_pred: torch.Tensor            # [B,T,S,H,V,3] camera frame, clamped N
    force_target: torch.Tensor          # [B,T,S,H,V,3] clamped GT N
    force_mask: torch.Tensor            # [B,T,S] force_valid & real substeps
    active_mask: torch.Tensor           # [B,T,S,H,V] target norm > threshold
    hand_pred: torch.Tensor             # [B,T,S,K] aggregated hand wrench
    hand_target: torch.Tensor           # [B,T,S,K] clamped GT wrench
    force_supervised: bool

    # --- bookkeeping -------------------------------------------------------
    batch_size: int = 0
    tau_per_m: float = 20.0
    fps: float = 10.0
    extras: dict = field(default_factory=dict)


@dataclass
class MetricSpec:
    """Static description of one metric line in the report."""

    key: str                # e.g. "contact/f1_macro"
    unit: str               # "", "mm", "m/s^2", "N", "N.m", "deg"
    paper_table: Optional[str] = None   # e.g. "tab:contact_estimation"
    paper_column: Optional[str] = None  # e.g. "f1"
    note: str = ""
    higher_is_better: bool = False
