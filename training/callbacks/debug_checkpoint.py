from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class DebugModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with explicit debug prints for save decisions."""

    def __init__(self, *args: Any, debug_name: str = "checkpoint", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.debug_name = debug_name

    def _debug(self, trainer: Any, message: str) -> None:
        pass

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return f"{float(value.detach().cpu().item()):.6f}"
            return f"tensor(shape={tuple(value.shape)})"
        return repr(value)

    def _debug_monitor_candidates(self, trainer: Any) -> dict[str, Any]:
        monitor_candidates = self._monitor_candidates(trainer)
        keys = sorted(str(k) for k in monitor_candidates.keys())
        self._debug(trainer, f"callback_metrics keys={keys}")
        if self.monitor is None:
            self._debug(trainer, "monitor=None")
        elif self.monitor in monitor_candidates:
            self._debug(
                trainer,
                f"monitor {self.monitor}={self._format_metric_value(monitor_candidates[self.monitor])}",
            )
        else:
            self._debug(trainer, f"monitor {self.monitor} is missing from callback_metrics")
        return monitor_candidates

    def _should_skip_saving_checkpoint(self, trainer: Any) -> bool:
        should_skip = super()._should_skip_saving_checkpoint(trainer)
        self._debug(
            trainer,
            "skip_check="
            f"{should_skip} fast_dev_run={bool(trainer.fast_dev_run)} "
            f"state={trainer.state.fn} sanity_checking={trainer.sanity_checking} "
            f"last_global_step_saved={self._last_global_step_saved} current_global_step={trainer.global_step}",
        )
        return should_skip

    def _save_last_if_stale(self, trainer: Any) -> None:
        """Write last.ckpt even when the monitored metric did not improve.

        Upstream ModelCheckpoint only refreshes last.ckpt at a save step where
        a top-k checkpoint was also written (``_last_global_step_saved ==
        trainer.global_step``).  On a plateau this drops every epoch's weights,
        so a run that is later killed mid-epoch keeps only the stale best --
        a failure mode that previously lost 40 epochs of training progress.
        """
        if not self.save_last or self._should_skip_saving_checkpoint(trainer):
            return
        if self._last_global_step_saved == trainer.global_step:
            return  # super() already refreshed last.ckpt at this step
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self._debug(
            trainer,
            "on_train_epoch_end entered "
            f"epoch={trainer.current_epoch} should_save_on_train_epoch_end="
            f"{self._should_save_on_train_epoch_end(trainer)} every_n_epochs={self._every_n_epochs}",
        )
        self._debug_monitor_candidates(trainer)
        super().on_train_epoch_end(trainer, pl_module)
        # Safe here: the epoch's save hooks are done, so refreshing last.ckpt
        # cannot cause a later top-k save to be skipped.
        self._save_last_if_stale(trainer)
        self._debug(
            trainer,
            "on_train_epoch_end exit "
            f"best_model_path={self.best_model_path!r} last_model_path={self.last_model_path!r}",
        )

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        self._debug(
            trainer,
            "on_validation_end entered "
            f"epoch={trainer.current_epoch} should_save_on_train_epoch_end="
            f"{self._should_save_on_train_epoch_end(trainer)} defer_save_until_validation="
            f"{self._defer_save_until_validation}",
        )
        self._debug_monitor_candidates(trainer)
        super().on_validation_end(trainer, pl_module)
        # When validation runs at train-epoch end, upstream defers the whole
        # save (top-k best + last) to on_train_epoch_end.  Backfilling
        # last.ckpt here would set _last_global_step_saved and make that hook
        # skip the top-k best save entirely -- exactly why no best_*.ckpt for
        # the primary monitor ever appeared in the 20260914 tactohand runs.
        if not self._should_save_on_train_epoch_end(trainer):
            self._save_last_if_stale(trainer)
        self._debug(
            trainer,
            "on_validation_end exit "
            f"best_model_path={self.best_model_path!r} last_model_path={self.last_model_path!r}",
        )

    def _save_topk_checkpoint(self, trainer: Any, monitor_candidates: dict[str, Any]) -> None:
        if self.save_top_k == 0:
            self._debug(trainer, "skip _save_topk_checkpoint because save_top_k=0")
            return
        if self.monitor is not None and self.monitor not in monitor_candidates:
            self._debug(
                trainer,
                f"skip _save_topk_checkpoint because monitor {self.monitor!r} is absent",
            )
        else:
            current = monitor_candidates.get(self.monitor) if self.monitor is not None else None
            self._debug(
                trainer,
                "_save_topk_checkpoint called "
                f"monitor={self.monitor!r} current={self._format_metric_value(current)} "
                f"save_top_k={self.save_top_k} mode={self.mode}",
            )
        super()._save_topk_checkpoint(trainer, monitor_candidates)
        self._debug(
            trainer,
            "_save_topk_checkpoint finished "
            f"best_model_path={self.best_model_path!r} best_model_score={self._format_metric_value(self.best_model_score)}",
        )

    def _save_last_checkpoint(self, trainer: Any, monitor_candidates: dict[str, Any]) -> None:
        if not self.save_last:
            self._debug(trainer, "skip _save_last_checkpoint because save_last is disabled")
            return
        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
        self._debug(trainer, f"_save_last_checkpoint target={filepath!r}")
        super()._save_last_checkpoint(trainer, monitor_candidates)
        self._debug(trainer, f"_save_last_checkpoint finished last_model_path={self.last_model_path!r}")

    def _save_checkpoint(self, trainer: Any, filepath: str) -> None:
        self._debug(trainer, f"_save_checkpoint writing filepath={filepath!r}")
        super()._save_checkpoint(trainer, filepath)
        self._debug(trainer, f"_save_checkpoint done filepath={filepath!r}")

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        self._debug(trainer, f"on_exception type={type(exception).__name__} message={exception}")
        super().on_exception(trainer, pl_module, exception)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        self._debug(
            trainer,
            "on_train_end entered "
            f"save_last={self.save_last} last_checkpoint_saved={self._last_checkpoint_saved!r}",
        )
        self._debug_monitor_candidates(trainer)
        super().on_train_end(trainer, pl_module)
        self._debug(
            trainer,
            "on_train_end exit "
            f"best_model_path={self.best_model_path!r} last_model_path={self.last_model_path!r}",
        )
