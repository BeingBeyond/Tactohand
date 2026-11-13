"""Batch-level timing profiler callback for Lightning training."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable

import pytorch_lightning as pl


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Iterable[float], mean_value: float) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return math.sqrt(sum((v - mean_value) ** 2 for v in vals) / len(vals))


def _percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = int(round((len(vals) - 1) * q))
    return vals[max(0, min(idx, len(vals) - 1))]


class BatchTimingProfilerCallback(pl.Callback):
    """Profiles per-step timing and prints rolling bottleneck analysis."""

    def __init__(
        self,
        print_interval_steps: int = 50,
        window_size: int = 50,
        warmup_steps: int = 5,
        log_to_logger: bool = True,
    ) -> None:
        super().__init__()
        self.print_interval_steps = max(1, int(print_interval_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.log_to_logger = bool(log_to_logger)

        self._window_size = max(5, int(window_size))
        self._stage_windows: Dict[str, Deque[float]] = {}
        self._batch_start_time: float = 0.0  # 独立记录 batch 开始时间
        self._last_batch_end_time: float | None = None
        self._step_count = 0
        self._last_gap_s = 0.0

    def _stage_window(self, stage: str) -> Deque[float]:
        if stage not in self._stage_windows:
            self._stage_windows[stage] = deque(maxlen=self._window_size)
        return self._stage_windows[stage]

    def _record(self, stage: str, seconds: float) -> None:
        self._stage_window(stage).append(max(0.0, float(seconds)))

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        now = time.perf_counter()
        self._batch_start_time = now  # 独立记录 batch 开始时间
        if self._last_batch_end_time is None:
            self._last_gap_s = 0.0
        else:
            self._last_gap_s = max(0.0, now - self._last_batch_end_time)
            self._record("inter_batch_gap", self._last_gap_s)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        now = time.perf_counter()

        # 独立计算 batch_total_s，不依赖 Model 的 timing
        batch_total_s = max(0.0, now - self._batch_start_time)
        self._record("batch_total_s", batch_total_s)

        self._last_batch_end_time = now
        self._step_count += 1

        if hasattr(pl_module, "get_latest_timing_stats"):
            stats = pl_module.get_latest_timing_stats()  # type: ignore[attr-defined]
        else:
            stats = {}
        # 不再从 Model 获取 batch_total_s，其他阶段仍然从 Model 获取
        for key in (
            "transfer_s", "forward_loss_s", "backbone_forward_s",
            "temporal_forward_loss_s", "backward_s", "optimizer_step_s",
        ):
            if key in stats:
                self._record(key, float(stats[key]))

        if self._step_count <= self.warmup_steps:
            return
        if self._step_count % self.print_interval_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        self._print_and_log_summary(trainer, pl_module)

    def _stage_summary(self, stage: str) -> Dict[str, float]:
        window = list(self._stage_windows.get(stage, []))
        mean_s = _mean(window)
        std_s = _std(window, mean_s)
        cv = (std_s / mean_s) if mean_s > 1e-12 else 0.0
        return {
            "mean_s": mean_s,
            "std_s": std_s,
            "p95_s": _percentile(window, 0.95),
            "max_s": max(window) if window else 0.0,
            "cv": cv,
        }

    def _print_and_log_summary(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        target_stages = (
            "batch_total_s",
            "inter_batch_gap",
            "transfer_s",
            "forward_loss_s",
            "backbone_forward_s",
            "temporal_forward_loss_s",
            "backward_s",
            "optimizer_step_s",
        )
        summaries = {stage: self._stage_summary(stage) for stage in target_stages}
        batch_total = summaries["batch_total_s"]["mean_s"]

        breakdown = {}
        for stage in ("inter_batch_gap", "transfer_s", "forward_loss_s", "backward_s", "optimizer_step_s"):
            stage_mean = summaries[stage]["mean_s"]
            breakdown[stage] = (100.0 * stage_mean / batch_total) if batch_total > 1e-12 else 0.0

        bottleneck_stage = max(
            ("inter_batch_gap", "transfer_s", "forward_loss_s", "backward_s", "optimizer_step_s"),
            key=lambda k: summaries[k]["mean_s"],
        )
        bottleneck_pct = breakdown[bottleneck_stage]

        print("")
        print("=" * 104)
        print(
            f"[Timing] step={self._step_count} window={self._window_size} "
            f"bottleneck={bottleneck_stage} ({bottleneck_pct:.1f}%)"
        )
        print("-" * 104)
        print(
            f"{'Stage':<22} {'Mean(ms)':>10} {'Std(ms)':>10} {'CV':>8} "
            f"{'P95(ms)':>10} {'Max(ms)':>10} {'%Batch':>9}"
        )
        print("-" * 104)
        for stage in (
            "inter_batch_gap", "transfer_s", "forward_loss_s",
            "backbone_forward_s", "temporal_forward_loss_s",
            "backward_s", "optimizer_step_s",
        ):
            s = summaries[stage]
            stage_pct = (
                100.0 * s["mean_s"] / batch_total if batch_total > 1e-12 else 0.0
            )
            print(
                f"{stage:<22} {s['mean_s'] * 1000:>10.2f} {s['std_s'] * 1000:>10.2f} "
                f"{s['cv']:>8.2f} {s['p95_s'] * 1000:>10.2f} {s['max_s'] * 1000:>10.2f} "
                f"{stage_pct:>8.1f}%"
            )
        total_s = summaries["batch_total_s"]
        print("-" * 104)
        print(
            f"{'batch_total_s':<22} {total_s['mean_s'] * 1000:>10.2f} "
            f"{total_s['std_s'] * 1000:>10.2f} {total_s['cv']:>8.2f} "
            f"{total_s['p95_s'] * 1000:>10.2f} {total_s['max_s'] * 1000:>10.2f} {100.0:>8.1f}%"
        )

        if summaries["inter_batch_gap"]["cv"] >= 0.30:
            print("[Timing] jitter warning: inter_batch_gap CV >= 0.30 (likely dataloader/decode variability).")
        if summaries["transfer_s"]["cv"] >= 0.30:
            print("[Timing] jitter warning: transfer_s CV >= 0.30 (likely host->device or pinned-memory pressure).")
        print("=" * 104)

        if self.log_to_logger and trainer.logger is not None:
            global_step = trainer.global_step
            metrics = {
                "timing/window/bottleneck_pct": bottleneck_pct,
                "timing/window/batch_total_ms": total_s["mean_s"] * 1000.0,
                "timing/window/batch_total_cv": total_s["cv"],
                "timing/window/inter_batch_gap_ms": summaries["inter_batch_gap"]["mean_s"] * 1000.0,
                "timing/window/inter_batch_gap_cv": summaries["inter_batch_gap"]["cv"],
                "timing/window/transfer_ms": summaries["transfer_s"]["mean_s"] * 1000.0,
                "timing/window/forward_loss_ms": summaries["forward_loss_s"]["mean_s"] * 1000.0,
                "timing/window/backbone_forward_ms": summaries["backbone_forward_s"]["mean_s"] * 1000.0,
                "timing/window/temporal_forward_loss_ms": summaries["temporal_forward_loss_s"]["mean_s"] * 1000.0,
                "timing/window/backward_ms": summaries["backward_s"]["mean_s"] * 1000.0,
                "timing/window/optimizer_step_ms": summaries["optimizer_step_s"]["mean_s"] * 1000.0,
            }
            trainer.logger.log_metrics(metrics, step=global_step)
