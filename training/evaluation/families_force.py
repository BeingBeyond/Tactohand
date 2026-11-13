"""Force family: vertex forces + aggregate wrench (paper Table 3).

Parity targets: ``epoch/metrics/val/<ds>/force/*`` RMSE lines in training
TensorBoard (same clamped targets, same validity gating: ``force_valid_30hz``
& real substeps, force-supervised episodes only).

Reported metrics:

* point MAE / RMSE (per-component, all supervised entries) and active-only
  variants (target magnitude above ``force_active_threshold_n`` — the rare
  class the split loss trains on);
* Force EPE: per-vertex vector endpoint error ``||f_pred - f_gt||``;
* Angle: direction error in degrees on active entries;
* Net force / torque errors of the aggregated hand wrench vs the clamped
  GT wrench columns (per-component MAE and RMSE).

Paper-table columns supplied for ``tab:force-baselines``: vertex_mae, epe,
angle, net_force, torque. The table's Rollout err. and Explode columns come
from the rollout family (``families_rollout``), not from here.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from training.evaluation.registry import MetricFamily, Sums, register
from training.evaluation.view import MetricSpec


@register("force")
class ForceFamily(MetricFamily):
    def __init__(self) -> None:
        self._sums = Sums()

    @staticmethod
    def specs() -> List[MetricSpec]:
        return [
            MetricSpec("force/point_mae_n", "N", "tab:force-baselines",
                       "vertex_mae",
                       note="per-component MAE over supervised entries"),
            MetricSpec("force/point_rmse_n", "N",
                       note="parity with training point_rmse_n"),
            MetricSpec("force/active_mae_n", "N", "tab:force-baselines",
                       "active_mae",
                       note="per-component MAE, active entries only"),
            MetricSpec("force/active_rmse_n", "N",
                       note="parity with training point_active_rmse_n"),
            MetricSpec("force/epe_n", "N",
                       note="vector endpoint error, all supervised entries"),
            MetricSpec("force/epe_active_n", "N",
                       note="vector endpoint error, active entries only"),
            MetricSpec("force/angle_deg", "deg", "tab:force-baselines",
                       "angle", note="direction error on active entries"),
            MetricSpec("force/net_force_mae_n", "N",
                       note="per-component MAE, substep level"),
            MetricSpec("force/net_force_rmse_n", "N"),
            MetricSpec("force/torque_mae_nm", "N.m",
                       note="per-component MAE, substep level"),
            MetricSpec("force/torque_rmse_nm", "N.m",
                       note="parity with training torque_rmse_nm"),
            MetricSpec("force/count_entries", "",
                       note="supervised vertex-substep entries"),
            MetricSpec("force/count_active", ""),
            MetricSpec("force/count_substeps", ""),
        ]

    def update(self, view) -> None:
        if not view.force_supervised:
            return
        vm5 = view.force_mask.bool()[:, :, :, None, None]      # [B,T,S,1,1]
        am5 = view.active_mask.bool()                          # [B,T,S,H,V]
        err_vec = view.force_pred - view.force_target          # [B,T,S,H,V,3]
        err_sq = err_vec.square()

        m3 = vm5.unsqueeze(-1).expand_as(err_vec)
        a3 = am5.unsqueeze(-1).expand_as(err_vec)
        n3 = m3.sum()
        na3 = a3.sum()
        hands, vertices = view.force_pred.shape[3], view.force_pred.shape[4]
        nv = vm5.sum() * (hands * vertices)
        nav = am5.sum()
        self._sums.add("force/abs_sum", (err_vec.abs() * m3).sum(), n3)
        self._sums.add("force/sq_sum", (err_sq * m3).sum(), n3)
        self._sums.add("force/active_abs_sum", (err_vec.abs() * a3).sum(), na3)
        self._sums.add("force/active_sq_sum", (err_sq * a3).sum(), na3)
        self._sums.add("force/count_entries", nv)
        self._sums.add("force/count_active", nav)

        epe = err_vec.norm(dim=-1)                             # [B,T,S,H,V]
        self._sums.add("force/epe_sum", (epe * vm5).sum(), nv)
        self._sums.add("force/epe_active_sum", (epe * am5).sum(), nav)

        self._accumulate_angle(view, am5)
        self._accumulate_wrench(view)

    def _accumulate_angle(self, view, active_v) -> None:
        f = view.force_pred[active_v]
        g = view.force_target[active_v]
        fn = f.norm(dim=-1).clamp_min(1e-8)
        gn = g.norm(dim=-1).clamp_min(1e-8)
        cos = (f * g).sum(-1) / (fn * gn)
        angle = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
        self._sums.add("force/angle_sum", angle.sum(), angle.numel())

    def _accumulate_wrench(self, view) -> None:
        substep_mask = view.force_mask.bool()
        n_sub = substep_mask.sum()
        pred, target = view.hand_pred, view.hand_target
        force_err = (pred[..., :3] - target[..., :3]).abs()
        torque_err = (pred[..., 3:6] - target[..., 3:6]).abs()
        gate3 = substep_mask.unsqueeze(-1)
        for key, err in (("force/net_force", force_err),
                         ("force/torque", torque_err)):
            self._sums.add(f"{key}_mae_sum", (err * gate3).sum(),
                           n_sub * 3.0)
            self._sums.add(f"{key}_sq_sum", (err.square() * gate3).sum(),
                           n_sub * 3.0)
        self._sums.add("force/count_substeps", n_sub)

    def compute(self) -> Dict[str, float]:
        s = self._sums
        return {
            "force/point_mae_n": s.mean("force/abs_sum"),
            "force/point_rmse_n": math.sqrt(s.mean("force/sq_sum")),
            "force/active_mae_n": s.mean("force/active_abs_sum"),
            "force/active_rmse_n": math.sqrt(s.mean("force/active_sq_sum")),
            "force/epe_n": s.mean("force/epe_sum"),
            "force/epe_active_n": s.mean("force/epe_active_sum"),
            "force/angle_deg": s.mean("force/angle_sum"),
            "force/net_force_mae_n": s.mean("force/net_force_mae_sum"),
            "force/net_force_rmse_n":
                math.sqrt(s.mean("force/net_force_sq_sum")),
            "force/torque_mae_nm": s.mean("force/torque_mae_sum"),
            "force/torque_rmse_nm": math.sqrt(s.mean("force/torque_sq_sum")),
            "force/count_entries": s.sum("force/count_entries"),
            "force/count_active": s.sum("force/count_active"),
            "force/count_substeps": s.sum("force/count_substeps"),
        }
