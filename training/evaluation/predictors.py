"""Pluggable prediction sources for the eval harness.

A predictor turns one batch's input frames into contact probabilities on the
standard (hand, vertex) axes, so external baselines (HACO, ...) can be
scored against the same GT and masks as TactoHand.  The default
``TactoHandPredictor`` delegates to the checkpoint's own model and is
bit-identical to the historical in-loop forward.

Contract (``predict_batch``):
* input  — the harness batch dict (device-resident tensors); consumers may
  rely on ``images`` [B,T,3,H,W] (float0-1 or normalized per run config)
  and ``crop_side`` [B] int8 (0=left crop, 1=right crop, -1=full view).
* output — dict with ``contact_prob`` [B,T,H,V] float32 in [0, 1] on the
  ``cpu`` device; absent outputs (e.g. a baseline with no proximity head)
  are simply omitted, and the corresponding metric families then report
  nothing for them.
"""
from __future__ import annotations

from typing import Any, Dict

import torch




def build_predictor(kind: str, model, cfg, datasets, device: torch.device) -> Any:
    """Current predictor plus on-demand research adapters from source checkouts."""
    if kind == "tactohand":
        from training.evaluation import families_contact
        families_contact.set_contact_threshold(0.5)
        from training.evaluation.tasks import task_adapter
        return task_adapter(str(cfg.get("task", "tactohand_wrench"))).TactoHandPredictor(model)
    try:
        from training.evaluation.predictors_research import build_research_predictor
    except ModuleNotFoundError as error:
        if error.name != "training.evaluation.predictors_research":
            raise
        raise RuntimeError(
            f"Predictor {kind!r} is a research workflow, excluded from the current-model "
            "distribution. Run it from the research source checkout with its external assets."
        ) from None
    return build_research_predictor(kind, model, cfg, datasets, device)


def __getattr__(name: str):
    if name == "TactoHandPredictor":
        from training.evaluation.tasks import task_adapter
        return task_adapter("tactohand_wrench").TactoHandPredictor
    # Preserve the old source import location for the HACO research adapter.
    if name == "HacoPredictor":
        from training.evaluation.predictors_research import HacoPredictor
        return HacoPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
