"""Callback for random stride offset sampling."""
from __future__ import annotations

import random
from typing import Any

import pytorch_lightning as pl


class RandomStrideOffsetCallback(pl.Callback):
    """Randomly set stride_offset at the start of each training epoch.

    This callback works with HOIDataset to enable random frame sampling.
    When clip_stride > 1 in single-frame mode, different offsets sample
    different frames:
    - offset=0: frames [0, stride, 2*stride, ...]
    - offset=1: frames [1, stride+1, 2*stride+1, ...]
    - etc.

    Usage in config:
        profiling:
          enabled: true  # or training callback config

    Or add to training callbacks manually.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # Get the dataset from the datamodule
        datamodule = trainer.datamodule
        if datamodule is None:
            return

        # Handle MixedHOIDataset
        train_ds = getattr(datamodule, "train_ds", None)
        if train_ds is None:
            return

        # Get clip_stride from dataset
        datasets = getattr(train_ds, "_datasets", [])
        if not datasets:
            return

        clip_stride = getattr(datasets[0], "clip_stride", 1)
        if clip_stride <= 1:
            return  # No stride, no need for random offset

        # Generate random offset
        new_offset = self._rng.randint(0, clip_stride - 1)

        # Set offset on all sub-datasets
        for ds in datasets:
            if hasattr(ds, "set_stride_offset"):
                ds.set_stride_offset(new_offset)

        # Log the new offset
        if trainer.is_global_zero:
            print(f"[RandomStrideOffset] epoch={trainer.current_epoch} offset={new_offset}")