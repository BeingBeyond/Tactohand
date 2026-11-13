"""Lazy task assembly boundary for the shared evaluator."""
from __future__ import annotations
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def task_adapter(task: str):
    if task == "tactohand_wrench":
        from training.tactohand_wrench.runtime import evaluation
        return evaluation
    raise ValueError(f"unsupported task for this harness: {task}")


def load_prediction_config(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction config not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError("Prediction config must be a mapping")
    return task_adapter(str(cfg.get("task", "tactohand_wrench"))).load_prediction_config(path)


def build_model(cfg, checkpoint, device, phase):
    return task_adapter(str(cfg.get("task", "tactohand_wrench"))).build_model(
        cfg, checkpoint, device, phase)


def build_view(model, predictions, batch, force, hand, tau, fps, fallback_dataset=None):
    # Retained public contract; the current native predictor owns its decoder.
    return task_adapter("tactohand_wrench").build_view(
        model, predictions, batch, force, hand, tau, fps, fallback_dataset)
