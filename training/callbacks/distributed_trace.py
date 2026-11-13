"""Low-overhead per-rank breadcrumbs and stack dumps for DDP stalls."""
from __future__ import annotations

import faulthandler
import os
import time
from typing import Any

import pytorch_lightning as pl
import torch


class DistributedTraceCallback(pl.Callback):
    """Print every distributed stage boundary and periodically dump stacks."""

    def __init__(self, stack_dump_interval_s: int = 120) -> None:
        super().__init__()
        self.stack_dump_interval_s = max(10, int(stack_dump_interval_s))
        self._started = time.monotonic()

    @staticmethod
    def _rank(trainer: pl.Trainer) -> str:
        return (
            f"global_rank={trainer.global_rank} local_rank="
            f"{os.environ.get('LOCAL_RANK', '0')}"
        )

    def _emit(self, trainer: pl.Trainer, event: str, **fields: Any) -> None:
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        memory = ""
        if torch.cuda.is_available():
            memory = (
                f" cuda_alloc_gib={torch.cuda.memory_allocated() / 2**30:.3f}"
                f" cuda_reserved_gib={torch.cuda.memory_reserved() / 2**30:.3f}"
            )
        print(
            f"[DDP_TRACE] t={time.monotonic() - self._started:.3f}s "
            f"{self._rank(trainer)} event={event}"
            f"{(' ' + details) if details else ''}{memory}",
            flush=True,
        )

    def setup(self, trainer, pl_module, stage) -> None:
        del pl_module
        self._emit(trainer, "setup", stage=stage, pid=os.getpid())
        faulthandler.enable()
        faulthandler.dump_traceback_later(
            self.stack_dump_interval_s, repeat=True)

    def teardown(self, trainer, pl_module, stage) -> None:
        del pl_module
        faulthandler.cancel_dump_traceback_later()
        self._emit(trainer, "teardown", stage=stage)

    def on_before_batch_transfer(self, trainer, pl_module, batch, dataloader_idx):
        del pl_module, batch
        self._emit(trainer, "before_batch_transfer", dataloader_idx=dataloader_idx)

    def on_after_batch_transfer(self, trainer, pl_module, batch, dataloader_idx):
        del pl_module, batch
        self._emit(trainer, "after_batch_transfer", dataloader_idx=dataloader_idx)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del pl_module, batch
        self._emit(trainer, "train_batch_start", batch_idx=batch_idx,
                   global_step=trainer.global_step)

    def on_before_backward(self, trainer, pl_module, loss):
        del pl_module, loss
        self._emit(trainer, "before_backward", global_step=trainer.global_step)

    def on_after_backward(self, trainer, pl_module):
        del pl_module
        self._emit(trainer, "after_backward", global_step=trainer.global_step)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        del pl_module, optimizer
        self._emit(trainer, "before_optimizer_step", global_step=trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del pl_module, outputs, batch
        self._emit(trainer, "train_batch_end", batch_idx=batch_idx,
                   global_step=trainer.global_step)

    def on_validation_start(self, trainer, pl_module):
        del pl_module
        self._emit(trainer, "validation_start", global_step=trainer.global_step)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0,
    ):
        del pl_module, batch
        self._emit(trainer, "validation_batch_start", batch_idx=batch_idx,
                   dataloader_idx=dataloader_idx)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ):
        del pl_module, outputs, batch
        self._emit(trainer, "validation_batch_end", batch_idx=batch_idx,
                   dataloader_idx=dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        del pl_module
        self._emit(trainer, "validation_epoch_end_callback_enter")

    def on_validation_end(self, trainer, pl_module):
        del pl_module
        self._emit(trainer, "validation_end", global_step=trainer.global_step)

    def on_exception(self, trainer, pl_module, exception):
        del pl_module
        self._emit(trainer, "exception", exception=repr(exception))
        faulthandler.dump_traceback(all_threads=True)
        faulthandler.cancel_dump_traceback_later()
