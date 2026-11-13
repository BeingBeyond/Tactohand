"""Shared explicit config/checkpoint loading for inference and evaluation."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from zipfile import is_zipfile

from omegaconf import DictConfig, OmegaConf
import torch


def load_prediction_config(path: Path | str) -> DictConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction config not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig) or OmegaConf.select(cfg, "model.wrench") is None:
        raise ValueError(
            "Use a complete composed TactoHand YAML config, such as an exported "
            "'tactohand-train --cfg job --resolve experiment=... data=tactohand' "
            "or the run's .hydra/config.yaml; a Hydra group YAML alone is insufficient.")
    if cfg.get("task", "tactohand_wrench") != "tactohand_wrench":
        raise ValueError("Prediction config must select task=tactohand_wrench")
    return cfg


def checkpoint_config(cfg: DictConfig) -> DictConfig:
    """Rebuild the checkpoint graph without downloading initialization weights."""
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    assert isinstance(result, DictConfig)
    OmegaConf.update(result, "model.wrench.pretrained", False, merge=False)
    OmegaConf.update(result, "model.wrench.pretrained_path", None, merge=False)
    return result


def load_checkpoint_state(path: Path | str) -> dict[str, torch.Tensor]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Evaluation consumes model weights, so leave optimizer tensors on disk.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=is_zipfile(path))
    if isinstance(checkpoint, Mapping):
        checkpoint = checkpoint.get("state_dict", checkpoint)
    if (not isinstance(checkpoint, Mapping) or not checkpoint
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                   for k, v in checkpoint.items())):
        raise ValueError("Expected a tensor state_dict or a checkpoint containing state_dict")
    return dict(checkpoint)


def load_network(cfg: DictConfig, checkpoint: Path | str,
                 device: str | torch.device = "cpu"):
    from training.tactohand_wrench.nn.network import QueryForceWrenchNetwork

    effective = checkpoint_config(cfg)
    model_cfg = OmegaConf.to_container(effective.model.wrench, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError("model.wrench must be a mapping")
    if not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    network = QueryForceWrenchNetwork(model_cfg)
    state = load_checkpoint_state(checkpoint)
    if any(key.startswith("network.") for key in state):
        # Current Lightning model's prediction parameters all live in network.
        # Loss/metric state outside it is not part of the inference network.
        state = {key.removeprefix("network."): value for key, value in state.items()
                 if key.startswith("network.")}
    network.load_state_dict(state, strict=True)
    return network.to(device).eval()
