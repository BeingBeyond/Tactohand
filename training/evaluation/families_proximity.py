"""Proximity family: dense distance field (paper Table 1).

Three views of the same field:

* ``proximity/mae_normalized`` — parity with the training-time
  ``epoch/metrics/val/<ds>/proximity/mae`` (normalized exp encoding; NOT
  comparable across ``proximity_tau`` changes).
* ``proximity/ade_mm`` — Table 1's ADE: decode both prediction and target
  back to millimetres via ``d = -ln(p)/tau`` and take the mean absolute
  difference.  The GT encoding clips distance at ``PROXIMITY_CLIP_M`` so
  decoded targets saturate at the clip bound; entries beyond it still enter
  the mean (they carry real signal: "farther than clip").
* ``proximity/accel_m_s2`` — Table 1's Accel: temporal smoothness of the
  *predicted* field, second central difference over consecutive real
  tokens, scaled by fps^2, in m/s^2.  GT-side accel is reported alongside
  as the label noise floor.

Units: tau is per metre (``data.proximity_tau``), fps is the token rate
(``data.target_fps``).
"""
from __future__ import annotations

from typing import Dict, List

from training.evaluation.registry import MetricFamily, Sums, register
from training.evaluation.view import MetricSpec


@register("proximity")
class ProximityFamily(MetricFamily):
    def __init__(self) -> None:
        self._sums = Sums()
        self._fps2: float | None = None

    @staticmethod
    def specs() -> List[MetricSpec]:
        return [
            MetricSpec("proximity/mae_normalized", "",
                       note="parity with training proximity/mae (normalized)"),
            MetricSpec("proximity/ade_mm", "mm",
                       "tab:proximity_estimation", "ade"),
            MetricSpec("proximity/ade_clamp100_mm", "mm",
                       note="ARCTIC-protocol ADE: pred and GT both clamped "
                            "at the InterField output range (100 mm)"),
            MetricSpec("proximity/accel_m_s2", "m/s^2",
                       "tab:proximity_estimation", "accel",
                       note="smoothness of the predicted field"),
            MetricSpec("proximity/accel_err_m_s2", "m/s^2",
                       note="official ARCTIC protocol: |second difference "
                            "of (pred - GT)| — acceleration ERROR of the "
                            "error signal, not smoothness"),
            MetricSpec("proximity/accel_gt_m_s2", "m/s^2",
                       note="GT-side accel (label noise floor)"),
            MetricSpec("proximity/count", "",
                       note="supervised (frame,hand,vertex) slots"),
        ]

    def update(self, view) -> None:
        self._fps2 = view.fps ** 2
        mask = view.proximity_mask.bool().expand_as(view.proximity_prob)
        err = (view.proximity_prob - view.proximity_target).abs()
        self._sums.add("proximity/abs_normalized", (err * mask).sum(),
                       mask.sum())
        ade = (view.distance_mm_hat - view.distance_mm_gt).abs()
        self._sums.add("proximity/ade_mm", (ade * mask).sum(), mask.sum())
        # ARCTIC-protocol variant: InterField's targets saturate at its
        # 100 mm training clamp, so its far-field error is bounded by the
        # output range rather than by GT (up to 250 mm here).  Clamping
        # both sides restores the range the baseline was trained on.
        ade100 = (view.distance_mm_hat.clamp(0.0, 100.0)
                  - view.distance_mm_gt.clamp(0.0, 100.0)).abs()
        self._sums.add("proximity/ade_clamp100_mm",
                       (ade100 * mask).sum(), mask.sum())
        self._sums.add("proximity/count", mask.sum())
        self._accumulate_accel(view)

    def _accumulate_accel(self, view) -> None:
        """Second central difference over consecutive real tokens, |.| mean.

        Tokens are the uniform time axis (1/fps apart); ``source_frames``
        only gates monotonicity — strictly increasing within the triple
        excludes edge-padded (repeated) slots.  Source-frame strides are
        non-constant on retokenized datasets (e.g. arctic alternates +2/+3),
        so no fixed stride is required.
        """
        valid = view.proximity_mask.bool().expand_as(view.distance_mm_hat)
        frames = view.source_frames
        increasing = ((frames[:, 1:-1] > frames[:, :-2])
                      & (frames[:, 2:] > frames[:, 1:-1]))
        real3 = (view.real_tokens[:, :-2] & view.real_tokens[:, 1:-1]
                 & view.real_tokens[:, 2:])
        token_gate = (increasing & real3)[:, :, None, None]
        for series, key in ((view.distance_mm_hat, "proximity/accel_m_s2"),
                            (view.distance_mm_gt, "proximity/accel_gt_m_s2")):
            d2 = ((series[:, :-2] - 2.0 * series[:, 1:-1] + series[:, 2:])
                  / 1000.0)  # mm -> m
            gate = (valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]) & token_gate
            self._sums.add(key, (d2.abs() * gate).sum(), gate.sum())
        # Official ARCTIC accel protocol (eval_modules.compute_error_accel):
        # the second difference of the ERROR signal (pred - GT) — penalises
        # temporal jitter relative to GT, not the field's own motion.  A
        # perfectly tracking prediction scores ~0 here while its raw
        # smoothness (above) stays at the GT dynamics level.
        err = view.distance_mm_hat - view.distance_mm_gt
        d2e = ((err[:, :-2] - 2.0 * err[:, 1:-1] + err[:, 2:]) / 1000.0).abs()
        gate = (valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]) & token_gate
        self._sums.add("proximity/accel_err_m_s2",
                       (d2e * gate).sum(), gate.sum())

    def compute(self) -> Dict[str, float]:
        scale = self._fps2 if self._fps2 is not None else 1.0
        return {
            "proximity/mae_normalized":
                self._sums.mean("proximity/abs_normalized"),
            "proximity/ade_mm": self._sums.mean("proximity/ade_mm"),
            "proximity/ade_clamp100_mm":
                self._sums.mean("proximity/ade_clamp100_mm"),
            "proximity/accel_m_s2":
                self._sums.mean("proximity/accel_m_s2") * scale,
            "proximity/accel_err_m_s2":
                self._sums.mean("proximity/accel_err_m_s2") * scale,
            "proximity/accel_gt_m_s2":
                self._sums.mean("proximity/accel_gt_m_s2") * scale,
            "proximity/count": self._sums.sum("proximity/count"),
        }
