"""Rollout family: GT-initialized open-loop pose rollout from predicted forces.

Parity target: ``epoch/rollout/val/<ds>/h*`` lines in training TensorBoard.
The scored input is the same GT-damped hybrid wrench ``_physical_outputs``
returns (aligned substep predicted, the other substeps replaying GT), so
eval numbers are directly comparable with the training-time validation
rollout — and equally NOT comparable with the retired full-prediction
numbers, nor with inference time where no GT exists.

Protocol, identical to ``SubstepForceModel._log_30hz_rollout``:

* rigid-body rollout in camera frame from GT init state, FP64 with autocast
  disabled (CUDA ``linalg.solve`` has no BF16 path and normalized-mass
  physics reaches condition numbers near 1e8);
* implicit midpoint integrator; horizons from ``cfg.metrics.rollout_horizons``
  (1/2/4/8 tokens at 10 Hz here) expressed as 3/6/12/24 transitions on the
  physical 30 Hz axis — richer than the TB-slim ``tensorboard_rollout_*``
  override, whose h4 lines stay directly comparable;
* ``real_only`` windows only — padded transitions would score GT-vs-GT;
* articulated-only batches are skipped (the rigid rollout does not model
  joints), matching the training-side gate;
* exploded windows (state speed past the cfg thresholds, or non-finite
  state) are excluded from the ADE/FDE means and reported through
  ``explosion_rate`` instead.

Tactile eval computes metrics only where a predictor exposes forces, so
this family carries no data under the external predictors (haco, egophi,
contactprompt) — reported as absent, not zero.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from training.evaluation.registry import MetricFamily, Sums, register
from training.evaluation.view import BatchView, MetricSpec

# (metric key, unit, note) reported per horizon; rollout_metrics also emits
# velocity/omega ADE and rotation_fde_rad — the FDE rotation is reported in
# degrees (converted below), the rest are deliberately not reported.
_ROLLOUT_KEYS = (
    ("position_ade_m", "m",
     "mean displacement error over the horizon window"),
    ("position_fde_m", "m",
     "displacement error at the horizon end"),
    ("rotation_ade_rad", "rad",
     "mean rotation error over the horizon window"),
    ("rotation_ade_deg", "deg", "same as rotation_ade_rad, in degrees"),
    ("explosion_rate", "",
     "share of windows whose rollout diverged or went non-finite"),
)
_REPORTED_KEYS = {key for key, _, _ in _ROLLOUT_KEYS}
_RAD_TO_DEG = 57.29577951308232

# Horizons listed in the static spec grid; keys for horizons outside a run's
# cfg.metrics.rollout_horizons still print via the report's fallback spec.
_SPEC_HORIZONS = (1, 2, 4, 8)


def compute_rollout_sums(
    model, contact: torch.Tensor, batch: Dict[str, Any], force=None,
) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
    """Per-(horizon, subset, key) ``(weighted_sum, weight)`` pairs.

    Mirrors the accumulation semantics of ``_log_30hz_rollout``: error keys
    are weighted by valid-window count, ``explosion_rate`` by total window
    count. Windows are counted even when every rollout in a batch diverges.
    """
    if not bool(batch.get("rollout_eligible", torch.tensor(True)).bool().all()):
        return {}
    if set(map(str, batch.get("dataset_mode", []))) == {"articulated_trajectory"}:
        return {}
    # Parity with the training-side gate: rollout accumulates only on
    # force-supervised batches (contact-only episodes mask force losses,
    # so their (zero) force fields would pollute the rollout).
    if not bool(batch.get("force_supervision_valid",
                          torch.tensor(True)).bool().all()):
        return {}
    if force is not None:
        from training.tactohand_wrench.wrench.scene import scene_rollout_inputs
        scene = scene_rollout_inputs(force, batch)
        if scene is not None:
            contact, batch = scene
    transitions = contact.reshape(contact.shape[0], -1, 6)
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
    metrics_cfg = model.cfg.metrics
    horizons = [int(h) for h in metrics_cfg.get(
        "rollout_horizons", model._rollout_metric_horizons)]
    from training.tactohand_wrench.wrench.metrics import rollout_metrics
    with torch.autocast(device_type=transitions.device.type, enabled=False):
        values = rollout_metrics(
            transitions.double(), rollout_batch,
            [3 * horizon for horizon in horizons],
            ["real_only"],
            float(metrics_cfg.explosion_linear_speed_m_s),
            float(metrics_cfg.explosion_angular_speed_rad_s),
            str(metrics_cfg.get("rollout_integrator", "implicit_midpoint")),
            int(metrics_cfg.get("implicit_iterations", 6)),
            midpoint_implementation=str(metrics_cfg.get("rollout_evaluation_implementation", "reference")),
        )
    sums: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    for horizon in horizons:
        metrics = values[str(3 * horizon)]["real_only"]
        count = int(metrics["window_count"])
        if not count:
            continue
        valid = int(metrics["valid_window_count"])
        for key, value in metrics.items():
            if key in ("window_count", "valid_window_count"):
                sums[(str(horizon), "real_only", key)] = (float(value), 1.0)
                continue
            if key != "explosion_rate" and not valid:
                continue
            weight = float(count if key == "explosion_rate" else valid)
            sums[(str(horizon), "real_only", key)] = (
                float(value) * weight, weight)
    return sums


@register("rollout")
class RolloutFamily(MetricFamily):
    """Open-loop rollout error vs horizon (ADE/FDE) plus explosion rate."""

    def __init__(self) -> None:
        self._sums = Sums()

    @staticmethod
    def specs() -> List[MetricSpec]:
        specs: List[MetricSpec] = []
        for horizon in _SPEC_HORIZONS:
            for key, unit, note in _ROLLOUT_KEYS:
                spec = MetricSpec(
                    f"rollout/h{horizon}_{key}", unit,
                    note=f"horizon {horizon} tokens: {note}")
                if horizon == 4 and key == "position_ade_m":
                    # Table 3's Rollout err. column: position ADE at 0.4 s.
                    spec.paper_table = "tab:force-baselines"
                    spec.paper_column = "rollout"
                if horizon == 4 and key == "rotation_ade_deg":
                    spec.paper_table = "tab:force-baselines"
                    spec.paper_column = "rollout_angle"
                specs.append(spec)
            specs.append(MetricSpec(
                f"rollout/h{horizon}_rotation_fde_deg", "deg",
                note=f"horizon {horizon} tokens: rotation error at the "
                     f"horizon end"))
            specs.append(MetricSpec(
                f"rollout/h{horizon}_windows", "",
                note=f"horizon {horizon} tokens: total real_only windows"))
            specs.append(MetricSpec(
                f"rollout/h{horizon}_valid_windows", "",
                note=f"horizon {horizon} tokens: non-divergent real_only windows"))
        return specs

    def update(self, view: BatchView) -> None:
        rollout = view.extras.get("rollout")
        if not rollout:
            return
        for (horizon, subset, key), (total, weight) in rollout.items():
            if subset != "real_only":
                continue
            if key == "window_count":
                self._sums.add(f"rollout/h{horizon}_windows", total, 1.0)
            elif key == "valid_window_count":
                self._sums.add(f"rollout/h{horizon}_valid_windows", total, 1.0)
            elif key == "rotation_fde_rad":
                self._sums.add(f"rollout/h{horizon}_rotation_fde_deg",
                               float(total) * _RAD_TO_DEG, weight)
            elif key in _REPORTED_KEYS:
                self._sums.add(f"rollout/h{horizon}_{key}", total, weight)

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key in self._sums.keys():
            if key.endswith("_windows"):
                metrics[key] = self._sums.sum(key)
            else:
                metrics[key] = self._sums.mean(key)
        return metrics
