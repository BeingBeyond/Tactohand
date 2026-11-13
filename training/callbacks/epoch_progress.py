from __future__ import annotations

import time
from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer


class EpochProgressCallback(Callback):
    """Print epoch/step progress with timing — works in non-TTY log files."""

    def __init__(self, log_every_n_steps: int = 50) -> None:
        self._log_every = max(1, int(log_every_n_steps))
        self._epoch_start: float = 0.0
        self._train_start: float = 0.0
        self._prev_step: int = 0
        self._prev_time: float = 0.0
        self._prev_batch_idx: int = 0
        self._last_batch_end: float = 0.0
        self._data_wait_s: float = 0.0
        self._batch_start: float = 0.0

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 0 or seconds == float("inf"):
            return "unknown"
        seconds_i = int(round(seconds))
        hours, rem = divmod(seconds_i, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _num_training_batches(trainer: Trainer) -> int | None:
        total = getattr(trainer, "num_training_batches", None)
        if total is None:
            return None
        try:
            total_i = int(total)
        except (TypeError, ValueError, OverflowError):
            return None
        return total_i if total_i > 0 else None

    @staticmethod
    def _append_metric(fields: list[str], metrics: Any, label: str,
                       *keys: str) -> None:
        """Append the first available metric, with legacy-name fallbacks."""
        for key in keys:
            if key in metrics:
                fields.append(f"{label}={float(metrics[key]):.5g}")
                return

    @classmethod
    def _metric_summary(cls, metrics: Any, prefix: str) -> list[str]:
        fields: list[str] = []
        cls._append_metric(
            fields, metrics, "loss", f"epoch/objective/{prefix}/total",
            f"loss/{prefix}", f"loss/{prefix}_epoch")
        cls._append_metric(
            fields, metrics, "point_rmse_n",
            f"epoch/metrics/{prefix}/point_force_rmse_n",
            f"force/{prefix}/point_rmse_n")
        cls._append_metric(
            fields, metrics, "force_rmse_n",
            f"epoch/metrics/{prefix}/resultant_force_rmse_n",
            f"wrench/{prefix}/force_rmse_n", f"wrench/{prefix}/force_rmse")
        cls._append_metric(
            fields, metrics, "torque_rmse_nm",
            f"epoch/metrics/{prefix}/resultant_torque_rmse_nm",
            f"wrench/{prefix}/torque_rmse_nm", f"wrench/{prefix}/torque_rmse")
        for horizon in (1, 4):
            cls._append_metric(
                fields, metrics, f"h{horizon}_pos_m",
                f"epoch/rollout/{prefix}/h{horizon}_position_ade_m",
                f"rollout/{prefix}/h{horizon}/position_ade_m")
            cls._append_metric(
                fields, metrics, f"h{horizon}_rot_rad",
                f"epoch/rollout/{prefix}/h{horizon}_rotation_ade_rad",
                f"rollout/{prefix}/h{horizon}/rotation_ade_rad")
        cls._append_metric(
            fields, metrics, "h4_explosion",
            f"epoch/rollout/{prefix}/h4_explosion_rate",
            f"rollout/{prefix}/h4/explosion_rate")
        return fields

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank == 0:
            self._train_start = time.time()
            print(f"[PROGRESS] Training started at {time.strftime('%H:%M:%S')} | {trainer.max_epochs} epochs")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank == 0:
            self._epoch_start = time.time()
            self._prev_step = trainer.global_step
            self._prev_time = time.time()
            self._prev_batch_idx = 0
            self._last_batch_end = time.perf_counter()
            print(f"[PROGRESS] ━━ Epoch {trainer.current_epoch} started at {time.strftime('%H:%M:%S')}")

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, batch, batch_idx
        if trainer.global_rank == 0:
            now = time.perf_counter()
            self._data_wait_s = max(0.0, now - self._last_batch_end)
            self._batch_start = now

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        if trainer.global_rank != 0:
            return
        batch_compute_s = max(0.0, time.perf_counter() - self._batch_start)
        self._last_batch_end = time.perf_counter()
        # Lightning increments global_step after on_train_batch_end. Use the
        # one-based batch completion index here so the first batch is visible.
        step = trainer.global_step
        completed_batch = int(batch_idx) + 1
        if completed_batch % self._log_every == 0:
            now = time.time()
            elapsed = now - self._epoch_start
            total_elapsed = now - self._train_start
            current_batch = int(batch_idx) + 1
            total_batches = self._num_training_batches(trainer)
            display_step = step + 1
            recent_steps = max(1, display_step - self._prev_step)
            recent_batches = current_batch - self._prev_batch_idx
            recent_dt = now - self._prev_time
            step_speed = recent_steps / recent_dt if recent_dt > 0 else 0.0
            batch_speed = recent_batches / recent_dt if recent_dt > 0 else 0.0
            avg_batch_speed = current_batch / elapsed if elapsed > 0 else 0.0

            progress_str = ""
            eta_str = ""
            if total_batches is not None:
                pct = 100.0 * float(current_batch) / float(total_batches)
                remaining = max(0, total_batches - current_batch)
                speed_for_eta = batch_speed if batch_speed > 0 else avg_batch_speed
                eta = remaining / speed_for_eta if speed_for_eta > 0 else float("inf")
                epoch_total = total_batches / avg_batch_speed if avg_batch_speed > 0 else float("inf")
                progress_str = f" | batch={current_batch}/{total_batches} ({pct:.2f}%)"
                eta_str = (
                    f" | eta={self._format_duration(eta)}"
                    f" | epoch_total≈{self._format_duration(epoch_total)}"
                )
            # Loss from logged metrics
            logs = trainer.callback_metrics
            loss_str = ""
            if "train/total_loss" in logs:
                loss_str = f" loss={float(logs['train/total_loss']):.4f}"
            elif "step/objective/total" in logs:
                loss_str = f" loss={float(logs['step/objective/total']):.4f}"
            elif "loss/train" in logs:
                loss_str = f" loss={float(logs['loss/train']):.4f}"
            elif "loss/train_step" in logs:
                loss_str = f" loss={float(logs['loss/train_step']):.4f}"
            elif "train/loss" in logs:
                loss_str = f" loss={float(logs['train/loss']):.4f}"
            elif "train/loss_step" in logs:
                loss_str = f" loss={float(logs['train/loss_step']):.4f}"
            elif "train_loss" in logs:
                loss_str = f" loss={float(logs['train_loss']):.4f}"
            lr_str = ""
            if trainer.optimizers and trainer.optimizers[0].param_groups:
                group_lrs = [float(group["lr"]) for group in trainer.optimizers[0].param_groups]
                lr_str = " lr=" + "/".join(f"{lr:.2e}" for lr in group_lrs)
            print(f"[PROGRESS]   Epoch {trainer.current_epoch} step={display_step} | "
                  f"speed={step_speed:.1f} step/s"
                  f"{progress_str} | elapsed={self._format_duration(elapsed)}"
                  f"{eta_str}{loss_str}{lr_str}"
                  f" | data_wait={self._data_wait_s:.3f}s"
                  f" compute={batch_compute_s:.3f}s", flush=True)
            self._prev_step = display_step
            self._prev_time = now
            self._prev_batch_idx = current_batch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # ``callback_metrics`` lazily materializes sync_dist epoch reductions.
        # Every rank must enter those collectives from the same callback, even
        # though only rank zero prints the resulting summary.
        metrics = trainer.callback_metrics
        if trainer.global_rank == 0:
            epoch_time = time.time() - self._epoch_start
            total_time = time.time() - self._train_start
            summary = self._metric_summary(metrics, "train")
            metric_text = (" | " + " ".join(summary)) if summary else ""
            print(f"[PROGRESS] ━━ Epoch {trainer.current_epoch} done | "
                  f"epoch_time={epoch_time:.0f}s | total_time={total_time:.0f}s "
                  f"({total_time/3600:.1f}h){metric_text}")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        del pl_module
        # See ``on_train_epoch_end``: materialize distributed metrics on every
        # rank before applying the rank-zero-only presentation policy.
        metrics = trainer.callback_metrics
        if trainer.global_rank != 0 or trainer.sanity_checking:
            return
        fields = self._metric_summary(metrics, "val")
        if fields:
            print(
                f"[VALIDATION] epoch={trainer.current_epoch} " + " ".join(fields),
                flush=True,
            )
