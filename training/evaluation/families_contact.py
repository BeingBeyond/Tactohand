"""Contact family: dense-contact macro/micro F1 (paper Table 2).

A "frame" is one (window, token, hand) unit: a 778-dim predicted-contact
vector scored against the 778-dim GT mask. The two paper metrics are:

* macro F1: mean per-frame vertex-set F1 over the active union (GT or
  prediction non-empty). Correct empty/empty frames are excluded; misses and
  empty-GT false alarms enter as zero;
* micro F1: pooled vertex TP/FP/FN over every valid frame-hand unit.

Both metrics penalize contact misses and no-contact hallucinations without
rewarding abundant true-negative empty frames.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from training.evaluation.registry import MetricFamily, Sums, register
from training.evaluation.view import MetricSpec

THRESHOLD = 0.5


def set_contact_threshold(value: float) -> None:
    """Set the deployed binarisation threshold for THIS eval run.

    Each predictor is scored at its own official operating point:
    TactoHand's deployed sigmoid rule is 0.5; the EgoPHI release
    evaluates hands at 0.3 (its ``HAND_CONTACT_THRESHOLD``).
    ``build_predictor`` sets this before each predictor runs.
    """
    global THRESHOLD
    THRESHOLD = float(value)


def _frame_prf(predicted: torch.Tensor, target: torch.Tensor
               ) -> tuple[float, float, float]:
    """P, R, F1 for one frame's 778-dim boolean pair, empty-GT rule."""
    gt_any = bool(target.any())
    pred_any = bool(predicted.any())
    tp = int((predicted & target).sum())
    if not gt_any:
        # Empty GT: the frame only tests false positives.  F1 = 1 when the
        # prediction is also empty, else 0.  P and R are undefined here and
        # enter neither mean (the F1 rule carries the whole signal).
        return 0.0, 0.0, (1.0 if not pred_any else 0.0)
    p = tp / max(int(predicted.sum()), 1)
    r = tp / max(int(target.sum()), 1)
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


@register("contact")
class ContactFamily(MetricFamily):
    def __init__(self) -> None:
        self._sums = Sums()
        self._sums.add("frame/n", 0.0)
        self._sums.add("frame/p_sum", 0.0)
        self._sums.add("frame/p_n", 0.0)
        self._sums.add("frame/r_sum", 0.0)
        self._sums.add("frame/r_n", 0.0)
        self._sums.add("frame/f1_haco_sum", 0.0)
        self._sums.add("frame/f1_haco_n", 0.0)
        self._sums.add("frame/empty_n", 0.0)
        self._sums.add("frame/empty_fp", 0.0)
        self._sums.add("frame/f1_gt_sum", 0.0)
        self._sums.add("frame/f1_gt_n", 0.0)

    @staticmethod
    def specs() -> List[MetricSpec]:
        return [
            MetricSpec("contact/precision", "", note="diagnostic only"),
            MetricSpec("contact/recall", "", note="diagnostic only"),
            MetricSpec("contact/precision_micro", "", "tab:contact_estimation",
                       "precision", note="pooled vertex TP/(TP+FP)"),
            MetricSpec("contact/recall_micro", "", "tab:contact_estimation",
                       "recall", note="pooled vertex TP/(TP+FN)"),
            MetricSpec("contact/frames", "",
                       note="frames scored (valid, GT or empty under the "
                            "empty-prediction rule)"),
            MetricSpec("contact/f1_macro", "", "tab:contact_estimation",
                       "macro_f1", note="PRIMARY: per-frame vertex-set F1 "
                       "over GT-or-prediction active frame-hand units"),
            MetricSpec("contact/f1_micro", "", "tab:contact_estimation",
                       "micro_f1", note="SECONDARY: pooled vertex TP/FP/FN "
                       "over all valid frame-hand units"),
            MetricSpec("contact/localization_f1", "",
                       note="auxiliary: per-frame F1 over contact-bearing "
                            "frames only (empty-GT frames excluded)"),
            MetricSpec("contact/localization_f1_haco", "",
                       note="auxiliary HACO-release variant: zero-prediction "
                            "frames excluded from the mean (diagnostic only)"),
            MetricSpec("contact/empty_fp_rate", "", "tab:contact_estimation",
                       "empty_fp_rate",
                       note="share of no-contact frames with >=1 predicted "
                            "contact vertex (lower is better)"),
            MetricSpec("contact/count", "",
                       note="pooled GT-positive vertex count (diagnostic)"),
            MetricSpec("contact/gate_f1", "",
                       note="stage-1 engagement-gate F1 (diagnostic)"),
            MetricSpec("contact/cond_f1", "",
                       note="stage-2 conditional contact F1 (diagnostic)"),
        ]

    def update(self, view) -> None:
        # Vectorised per-frame statistics: every quantity that the former
        # per-frame Python loop accumulated is computed here as one [N,778]
        # tensor reduction (N = batch*tokens*hands), bit-for-bit identical
        # semantics without the per-frame host-device syncs.
        predicted = view.contact_prob > THRESHOLD          # [B,T,H,V]
        target = view.contact_target                        # [B,T,H,V] bool
        mask = view.contact_mask.bool()                     # [B,T,H,1]
        b_n, t_n, h_n = predicted.shape[:3]
        flat_pred = predicted.reshape(b_n * t_n * h_n, -1)  # [N,778]
        flat_tgt = target.reshape(b_n * t_n * h_n, -1)
        # A frame is processed when at least one vertex slot is valid
        # (the former loop's `if not flat_val[frame]: continue`).
        flat_val = mask.reshape(b_n * t_n * h_n, -1)        # [N,1]
        processed = flat_val.any(-1)                        # [N]

        gt_any = flat_tgt.any(-1) & processed               # [N]
        pred_any = flat_pred.any(-1) & processed
        tp = (flat_pred & flat_tgt).sum(-1).float()         # [N]
        pred_cnt = flat_pred.sum(-1).float()
        tgt_cnt = flat_tgt.sum(-1).float()
        p_f = tp / pred_cnt.clamp_min(1.0)
        r_f = tp / tgt_cnt.clamp_min(1.0)
        f1_f = torch.where((p_f + r_f) > 0,
                           2.0 * p_f * r_f / torch.clamp(p_f + r_f, min=1e-12),
                           torch.zeros_like(p_f))
        # Empty-GT frames only test false positives: F1 = 1 iff prediction
        # is also empty (former _frame_prf empty-GT rule).
        f1_f = torch.where(gt_any, f1_f, (~pred_any).float())

        # Scored-frame counter. Kept separately because shard merging sums
        # numerators and denominators before recomputing metrics.
        self._sums.add("frame/n", processed.sum())
        # Alternate, valid protocol definitions retained as diagnostics:
        # GT-contact frames only and the HACO release convention.
        self._sums.add("frame/p_sum", (p_f * gt_any).sum())
        self._sums.add("frame/p_n", gt_any.sum())
        self._sums.add("frame/r_sum", (r_f * gt_any).sum())
        self._sums.add("frame/r_n", gt_any.sum())
        self._sums.add("frame/f1_gt_sum", (f1_f * gt_any).sum())
        self._sums.add("frame/f1_gt_n", gt_any.sum())
        # HACO-protocol variant: their eval (compute_contact_metrics) leaves
        # precision/f1 undefined when a frame has NO predicted positives, so
        # such frames are excluded from the mean instead of counting as F1=0.
        haco_frames = gt_any & pred_any
        self._sums.add("frame/f1_haco_sum", (f1_f * haco_frames).sum())
        self._sums.add("frame/f1_haco_n", haco_frames.sum())
        # empty-frame false-alarm rate: frames with >=1 predicted contact
        # vertex although GT is all-negative
        self._sums.add("frame/empty_n", (processed & ~gt_any).sum())
        self._sums.add("frame/empty_fp",
                       (processed & ~gt_any & pred_any).sum())
        # pooled diagnostics (legacy behaviour, not the paper numbers)
        self._sums.add("contact/pooled_tp",
                       (predicted & target & mask).sum())
        self._sums.add("contact/pooled_fp",
                       (predicted & ~target & mask).sum())
        self._sums.add("contact/pooled_fn",
                       (~predicted & target & mask).sum())
        gate_predicted = view.gate_prob > 0.5
        self._sums.add("contact/gate_tp",
                       (gate_predicted & view.gate_target).sum())
        self._sums.add("contact/gate_fp",
                       (gate_predicted & ~view.gate_target).sum())
        self._sums.add("contact/gate_fn",
                       (~gate_predicted & view.gate_target).sum())
        cond_predicted = view.cond_prob > 0.5
        cond_mask = mask.expand_as(view.cond_prob)
        self._sums.add("contact/cond_tp",
                       (cond_predicted & target & cond_mask).sum())
        self._sums.add("contact/cond_fp",
                       (cond_predicted & ~target & cond_mask).sum())
        self._sums.add("contact/cond_fn",
                       (~cond_predicted & target & cond_mask).sum())

    def compute(self) -> Dict[str, float]:
        n = self._sums.sum("frame/n")
        empty_n = self._sums.sum("frame/empty_n")
        tp = self._sums.sum("contact/pooled_tp")
        fp = self._sums.sum("contact/pooled_fp")
        fn = self._sums.sum("contact/pooled_fn")
        # Additive accumulators: --shard runs each see a disjoint window
        # subset, so a merger can sum these across shard reports and
        # recompute every metric above losslessly (official numbers must
        # never be averaged over shards).
        out = {
            f"contact/agg/{key}": float(self._sums.sum(key)) for key in (
                "frame/n",
                "frame/p_sum", "frame/p_n", "frame/r_sum", "frame/r_n",
                "frame/f1_gt_sum", "frame/f1_gt_n",
                "frame/f1_haco_sum", "frame/f1_haco_n",
                "frame/empty_n", "frame/empty_fp",
                "contact/pooled_tp", "contact/pooled_fp", "contact/pooled_fn",
                "contact/gate_tp", "contact/gate_fp", "contact/gate_fn",
                "contact/cond_tp", "contact/cond_fp", "contact/cond_fn")
        }
        out.update({
            "contact/precision": self._sums.sum("frame/p_sum") / max(
                self._sums.sum("frame/p_n"), 1.0),
            "contact/recall": self._sums.sum("frame/r_sum") / max(
                self._sums.sum("frame/r_n"), 1.0),
            "contact/frames": n,
            "contact/precision_micro": tp / max(tp + fp, 1.0),
            "contact/recall_micro": tp / max(tp + fn, 1.0),
            "contact/f1_macro": (
                self._sums.sum("frame/f1_gt_sum")
                / max(self._sums.sum("frame/f1_gt_n")
                      + self._sums.sum("frame/empty_fp"), 1.0)),
            "contact/f1_micro": (
                2.0 * tp / max(2.0 * tp + fp + fn, 1.0)),
            # Auxiliary protocol: GT-contact frames only.
            "contact/localization_f1": self._sums.sum("frame/f1_gt_sum") / max(
                self._sums.sum("frame/f1_gt_n"), 1.0),
            "contact/localization_f1_haco": (
                self._sums.sum("frame/f1_haco_sum")
                / max(self._sums.sum("frame/f1_haco_n"), 1.0)
            ) if self._sums.sum("frame/f1_haco_n") > 0 else float("nan"),
            "contact/empty_fp_rate": (
                self._sums.sum("frame/empty_fp")
                / max(empty_n, 1.0)) if empty_n > 0 else 0.0,
            "contact/count": tp + fp + fn,
        })
        for stage in ("gate", "cond"):
            s_tp = self._sums.sum(f"contact/{stage}_tp")
            s_fp = self._sums.sum(f"contact/{stage}_fp")
            s_fn = self._sums.sum(f"contact/{stage}_fn")
            out[f"contact/{stage}_f1"] = (
                2.0 * s_tp / max(2.0 * s_tp + s_fp + s_fn, 1.0))
        return out
