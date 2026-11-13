"""Training entrypoint for TactoHand and the supported HOPE baseline."""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Tuple
from zipfile import is_zipfile

# Ensure root package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import warnings

# Set V2C_RUN_VERSION early so both Hydra (config resolution) and the Python
# logger code share the same experiment directory name. Direct invocation
# generates a fresh timestamp when no run version was supplied.
if "V2C_RUN_VERSION" not in os.environ:
    os.environ["V2C_RUN_VERSION"] = time.strftime("%Y%m%d_%H%M%S")

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.io import CheckpointIO
from pytorch_lightning.plugins.precision import MixedPrecision, Precision
from pytorch_lightning.utilities import GradClipAlgorithmType

# Allow OmegaConf types in checkpoint loading (PyTorch 2.6+ security)
try:
    from omegaconf.base import Container, ContainerMetadata, Node
    from omegaconf.nodes import ValueNode
    torch.serialization.add_safe_globals([
        DictConfig, ListConfig, Container, ContainerMetadata, Node, ValueNode
    ])
except ImportError:
    # Fallback for older OmegaConf versions
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

from training.configs import validate_task_data_pairing
from training.callbacks import (
    BatchTimingProfilerCallback,
    DebugModelCheckpoint,
    DistributedTraceCallback,
    EpochProgressCallback,
    RandomStrideOffsetCallback,
)


class _FusedAdamWBF16Precision(MixedPrecision):
    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        clip_val: int | float = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        # BF16 never scales gradients. Lightning's AMP clipping guard also
        # rejects fused AdamW in this mode, where ordinary clipping is safe.
        Precision.clip_gradients(self, optimizer, clip_val, gradient_clip_algorithm)


def _build_datamodule(cfg: DictConfig):
    """Instantiate the typed data pipeline declared by a supported task."""
    if cfg.data is None or not cfg.data.get("_target_"):
        raise ValueError("A supported typed data config with _target_ is required")
    return hydra.utils.instantiate(
        cfg.data, limit_val_batches=cfg.training.get("limit_val_batches", None))



def _build_model(cfg: DictConfig):
    """Build the current TactoHand model or a supported baseline."""
    task = str(cfg.get("task", "tactohand_wrench")).strip().lower()
    if task == "tactohand_wrench":
        from training.tactohand_wrench.model import SubstepForceModel

        return SubstepForceModel(cfg)
    if task == "hope_baseline":
        from training.hope_baseline.model import HopeBaselineModel

        return HopeBaselineModel(cfg)
    raise ValueError(f"Unsupported task: {task}")


def _suppress_known_warnings() -> None:
    """Keep training logs focused by hiding known harmless framework warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r".*Precision bf16-mixed is not supported by the model summary.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*You have overridden `transfer_batch_to_device` in `LightningModule`.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Found .* module\(s\) in eval mode at the start of training.*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*LeafSpec.*deprecated.*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*np\.(bool|object|str).*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The given NumPy array is not writable.*",
        category=UserWarning,
    )


def _resolve_device_settings(train_cfg: DictConfig) -> Tuple[str, Any, str]:
    """Resolve trainer accelerator/devices/strategy from YAML config.

    Priority:
    1) ``training.gpu_ids`` (explicit GPU IDs, e.g. [0, 1])
    2) ``training.devices``
    """
    accelerator = train_cfg.get("accelerator", "gpu")
    devices: Any = train_cfg.get("devices", 1)
    strategy = train_cfg.get("strategy", "auto")

    gpu_ids = train_cfg.get("gpu_ids", None)
    if gpu_ids is not None:
        if isinstance(gpu_ids, ListConfig):
            gpu_ids = list(gpu_ids)
        devices = gpu_ids
        accelerator = "gpu"
        # If user provides multiple explicit IDs and leaves strategy auto,
        # prefer DDP automatically.
        if strategy == "auto" and isinstance(devices, list) and len(devices) > 1:
            strategy = "ddp"
    elif isinstance(devices, ListConfig):
        devices = list(devices)

    return accelerator, devices, strategy


def _load_checkpoint_weights_only(
    model: pl.LightningModule,
    ckpt_path: str | Path,
    local_rank: int,
    exclude_key_prefixes: Any = None,
) -> None:
    """Initialize model weights from a Lightning checkpoint without trainer state."""
    ckpt_path = Path(str(ckpt_path))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"resume_weights_only checkpoint not found: {ckpt_path}")

    # The optimizer tensors are usually larger than the model. Map zip
    # checkpoints so copying model weights never reads the optimizer storage.
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False,
                            mmap=is_zipfile(ckpt_path))
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if exclude_key_prefixes is None:
        exclude_key_prefixes = []
    elif isinstance(exclude_key_prefixes, (str, Path)):
        exclude_key_prefixes = [str(exclude_key_prefixes)]
    elif isinstance(exclude_key_prefixes, ListConfig):
        exclude_key_prefixes = list(exclude_key_prefixes)
    exclude_key_prefixes = tuple(str(prefix) for prefix in exclude_key_prefixes if str(prefix))
    skipped_keys = []
    if exclude_key_prefixes:
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(exclude_key_prefixes):
                skipped_keys.append(key)
                continue
            filtered_state_dict[key] = value
        state_dict = filtered_state_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if local_rank == 0:
        print(
            "[train] loaded checkpoint weights only"
            f" path={ckpt_path}"
            f" excluded_prefixes={list(exclude_key_prefixes)}"
            f" skipped_keys={len(skipped_keys)}"
            f" missing_keys={len(missing)}"
            f" unexpected_keys={len(unexpected)}"
        )
        if skipped_keys:
            print(f"[train] skipped_keys sample={skipped_keys[:20]}")
        if missing:
            print(f"[train] missing_keys sample={list(missing)[:20]}")
        if unexpected:
            print(f"[train] unexpected_keys sample={list(unexpected)[:20]}")


def _checkpoint_callbacks(ckpt_cfg: DictConfig, ckpt_dir: Path) -> list[Any]:
    """Primary checkpoint plus additional-monitors checkpoints, deduped.

    Lightning derives ``ModelCheckpoint.state_key`` from (monitor, mode,
    trigger settings); both construction paths here use the default
    triggers, so two entries sharing monitor and mode produce the same
    state_key and Trainer construction dies with "Found more than one
    stateful callback".  A launcher that overrides ``checkpoint.monitor``
    onto a value already listed in the experiment's additional_monitors
    is exactly that collision -- skip the duplicate instead.
    """
    callbacks: list[Any] = []

    # Primary checkpoint: best F1 score (contact) or the task's monitor
    if int(ckpt_cfg.save_top_k) != 0 or bool(ckpt_cfg.save_last):
        primary = DebugModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            save_last=ckpt_cfg.save_last,
            filename=ckpt_cfg.get(
                "filename", "best_f1-epoch={epoch:02d}-val_f1_mean={val/f1_mean:.4f}"
            ),
            auto_insert_metric_name=False,
            enable_version_counter=bool(ckpt_cfg.get("enable_version_counter", True)),
            debug_name="primary",
        )
        seed = ckpt_cfg.get("best_val_seed")
        if seed is not None:
            best_path = ckpt_dir.resolve() / "best_val.ckpt"
            score = torch.tensor(float(seed))
            if (primary.monitor != "epoch/objective/val/total" or primary.mode != "min"
                    or primary.save_top_k != 1 or primary.filename != "best_val"
                    or not best_path.is_file() or not bool(torch.isfinite(score))):
                raise ValueError("Invalid seed for the total-validation best checkpoint")
            state = primary.state_dict()
            state.update(best_model_path=str(best_path), best_model_score=score,
                         best_k_models={str(best_path): score},
                         kth_best_model_path=str(best_path), kth_value=score)
            primary.load_state_dict(state)
        callbacks.append(primary)

    # Additional checkpoints for different criteria
    seen_ckpt_keys = {(str(ckpt_cfg.monitor), str(ckpt_cfg.mode))}
    for extra in ckpt_cfg.get("additional_monitors", []):
        if int(extra.save_top_k) == 0:
            continue
        ckpt_key = (str(extra.monitor), str(extra.mode))
        if ckpt_key in seen_ckpt_keys:
            print(
                "[checkpoint] skipping duplicate additional monitor"
                f" monitor={extra.monitor!r} mode={extra.mode!r}"
                " (same monitor/mode as an existing checkpoint)",
                flush=True,
            )
            continue
        seen_ckpt_keys.add(ckpt_key)
        callbacks.append(DebugModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor=extra.monitor,
            mode=extra.mode,
            save_top_k=extra.save_top_k,
            filename=extra.get(
                "filename",
                f"epoch={{epoch:02d}}-{str(extra.monitor).replace('/', '_')}={{{extra.monitor}:.4f}}",
            ),
            save_last=False,
            auto_insert_metric_name=False,
            enable_version_counter=bool(ckpt_cfg.get("enable_version_counter", True)),
            debug_name=str(extra.monitor),
        ))
    return callbacks


def _train_impl(cfg: DictConfig) -> None:
    _suppress_known_warnings()
    t_start = time.time()
    print(f"[TIMING] Config parsed: {time.time()-t_start:.2f}s")

    resume_ckpt = cfg.get("resume_from", None)
    train_cfg = cfg.training
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Improve Tensor Core utilization on modern NVIDIA GPUs.
    matmul_precision = cfg.training.get("matmul_precision", None)
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Input shape is mostly static (256x256 clips), so benchmark helps.
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Seed
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    print(f"[TIMING] Seed set: {time.time()-t_start:.2f}s")

    # DataModule
    t_dm = time.time()
    datamodule = _build_datamodule(cfg)
    print(f"[TIMING] DataModule created: {time.time()-t_dm:.2f}s")

    # Model
    t_model = time.time()
    model = _build_model(cfg)
    print(f"[TIMING] Model created: {time.time()-t_model:.2f}s")

    if resume_ckpt and bool(cfg.get("resume_weights_only", False)):
        _load_checkpoint_weights_only(
            model,
            resume_ckpt,
            local_rank,
            cfg.get("resume_exclude_key_prefixes", []),
        )
        resume_ckpt = None

    # Optional module-level torch.compile for compute-bound workloads.
    compile_cfg = train_cfg.get("compile", {})
    if bool(compile_cfg.get("enabled", False)):
        if not hasattr(torch, "compile"):
            if local_rank == 0:
                print("[compile] torch.compile unavailable on this runtime, skipped.")
        else:
            compile_mode = str(compile_cfg.get("mode", "reduce-overhead"))
            compile_backend = compile_cfg.get("backend", "auto")
            compile_fullgraph = bool(compile_cfg.get("fullgraph", False))
            compile_dynamic = bool(compile_cfg.get("dynamic", False))
            dynamic_modules = []
            for module_name in map(str, compile_cfg.get("dynamic_modules", [])):
                if module_name.endswith(".*"):
                    parent_name = module_name[:-2]
                    children = list(model.get_submodule(parent_name).named_children())
                    if not children:
                        raise ValueError(f"dynamic compile group has no child modules: {module_name}")
                    dynamic_modules.extend(f"{parent_name}.{name}" for name, _ in children)
                else:
                    dynamic_modules.append(module_name)
            compile_modules = dict.fromkeys((*map(str, compile_cfg.get("modules", [])), *dynamic_modules))
            compile_kwargs = {
                "mode": compile_mode,
                "fullgraph": compile_fullgraph,
            }
            if compile_backend not in (None, "", "auto"):
                compile_kwargs["backend"] = str(compile_backend)

            compiled = []
            for module_name in compile_modules:
                owner = model
                parts = str(module_name).split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part, None)
                    if owner is None:
                        break
                module = getattr(owner, parts[-1], None) if owner is not None else None
                if isinstance(module, torch.nn.Module):
                    module_kwargs = dict(compile_kwargs, dynamic=compile_dynamic or module_name in dynamic_modules)
                    # Prefer the in-place Module.compile API so checkpoint
                    # keys remain stable (no ``_orig_mod`` wrapper prefix).
                    if hasattr(module, "compile"):
                        module.compile(**module_kwargs)
                    else:
                        setattr(owner, parts[-1], torch.compile(module, **module_kwargs))
                    compiled.append(str(module_name))
            if local_rank == 0:
                print(
                    "[compile] enabled"
                    f" modules={compiled} mode={compile_mode}"
                    f" backend={compile_backend}"
                )

    # Logger
    # V2C_RUN_VERSION is set at module level (and exported by run_train_with_logs.sh
    # before spawning DDP workers).  All processes share the same version string so
    # TensorBoard, WandB, Hydra and checkpoints land in the same experiment directory.
    log_cfg = cfg.logging
    verbosity = str(log_cfg.get("verbosity", "normal")).lower()
    verbosity_rank = {"quiet": 0, "normal": 1, "verbose": 2, "debug": 3, "trace": 4}
    if verbosity not in verbosity_rank:
        raise ValueError("logging.verbosity must be quiet, normal, verbose, debug, or trace")
    version = os.environ.get("V2C_RUN_VERSION") or time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(log_cfg.save_dir) / version
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        print(f"[train] experiment_dir={experiment_dir}")

    # Build loggers list
    loggers = []

    # TensorBoard is the compact trend store; console cadence is independent.
    tensorboard_cfg = log_cfg.get("tensorboard", {})
    tb_logger = TensorBoardLogger(
        save_dir=str(experiment_dir),
        name="",
        version="",
        max_queue=1,
        flush_secs=int(tensorboard_cfg.get("flush_secs", 10)),
    )
    if local_rank == 0 and str(cfg.get("task", "")) in {"tactohand_wrench", "hope_baseline"}:
        if str(cfg.get("task", "")) == "hope_baseline":
            from training.hope_baseline.tensorboard_layout import custom_scalar_layout
        else:
            from training.tactohand_wrench.runtime.tensorboard import custom_scalar_layout

        tb_logger.experiment.add_custom_scalars(custom_scalar_layout())
    loggers.append(tb_logger)

    # WandB (enabled via logging.wandb.enabled: true)
    wandb_cfg = log_cfg.get("wandb", {})
    if bool(wandb_cfg.get("enabled", False)):
        wandb_logger = WandbLogger(
            project=log_cfg.get("project", "video2contact"),
            name=log_cfg.get("name", None),
            id=version,
            save_dir=str(experiment_dir),
            log_model=bool(wandb_cfg.get("log_model", False)),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)

    # Resolve device settings early for trainer initialization.
    accelerator, devices, strategy = _resolve_device_settings(train_cfg)

    # Checkpoint directory: use logger's log_dir
    ckpt_dir = Path(loggers[0].log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = []
    ckpt_cfg = cfg.checkpoint

    trace_cfg = cfg.get("diagnostics", {}).get("distributed_trace", {})
    if bool(trace_cfg.get("enabled", False)):
        callbacks.append(DistributedTraceCallback(
            stack_dump_interval_s=int(trace_cfg.get("stack_dump_interval_s", 120))))
        print(
            "[DDP_TRACE] enabled"
            f" stack_dump_interval_s={int(trace_cfg.get('stack_dump_interval_s', 120))}",
            flush=True,
        )

    # Epoch progress callback (always enabled — prints timing to log file)
    callbacks.append(
        EpochProgressCallback(
            log_every_n_steps=int(
                log_cfg.get("console", {}).get(
                    "progress_every_n_steps", log_cfg.get("log_every_n_steps", 50)
                )
            ),
        )
    )

    callbacks.extend(_checkpoint_callbacks(ckpt_cfg, ckpt_dir))

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Device stats monitor (GPU utilization, memory, etc.)
    device_stats_cfg = cfg.logging.get("device_stats", {})
    if device_stats_cfg.get("enabled", False):
        callbacks.append(DeviceStatsMonitor())

    # Batch timing profiler callback (for bottleneck/jitter analysis)
    profiling_cfg = cfg.get("profiling", {})
    if bool(profiling_cfg.get("enabled", False)):
        callbacks.append(
            BatchTimingProfilerCallback(
                print_interval_steps=int(profiling_cfg.get("print_interval_steps", log_cfg.log_every_n_steps)),
                window_size=int(profiling_cfg.get("window_size", 50)),
                warmup_steps=int(profiling_cfg.get("warmup_steps", 5)),
                log_to_logger=bool(profiling_cfg.get("log_to_logger", True)),
            )
        )
        if local_rank == 0:
            print(
                "[Timing] Batch timing profiler enabled"
                f" (interval={int(profiling_cfg.get('print_interval_steps', log_cfg.log_every_n_steps))},"
                f" window={int(profiling_cfg.get('window_size', 50))},"
                f" sync_cuda={bool(profiling_cfg.get('sync_cuda', True))})"
            )

    # Random stride offset callback (for single-frame mode with stride > 1)
    sampler_cfg = cfg.data.get("sampler", {})
    if bool(sampler_cfg.get("random_stride_offset", False)):
        callbacks.append(
            RandomStrideOffsetCallback(
                seed=int(cfg.get("seed", 42)),
            )
        )
        if local_rank == 0:
            print("[RandomStrideOffset] Enabled - will randomize frame sampling offset each epoch")

    # Trainer
    limit_val_batches_cfg = train_cfg.get("limit_val_batches", None)
    validation_disabled = False
    if limit_val_batches_cfg is not None:
        try:
            validation_disabled = float(limit_val_batches_cfg) == 0.0
        except (TypeError, ValueError):
            validation_disabled = False

    trainer_kwargs = dict(
        max_epochs=train_cfg.max_epochs,
        accelerator=accelerator,
        devices=devices,
        num_nodes=int(train_cfg.get("num_nodes", 1)),
        strategy=strategy,
        precision=train_cfg.precision,
        gradient_clip_val=train_cfg.gradient_clip_val,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
        log_every_n_steps=int(
            tensorboard_cfg.get("scalar_every_n_steps", log_cfg.log_every_n_steps)
        ),
        num_sanity_val_steps=0 if validation_disabled else int(train_cfg.get("num_sanity_val_steps", 2)),
        callbacks=callbacks,
        logger=loggers,
    )
    plugins: list[CheckpointIO | Precision] = []
    if bool(ckpt_cfg.get("stream_local_writes", False)):
        from training.runtime.checkpoint_io import StreamingCheckpointIO

        plugins.append(StreamingCheckpointIO())
    if (cfg.get("task", "tactohand_wrench") == "tactohand_wrench"
            and train_cfg.precision == "bf16-mixed"
            and bool(train_cfg.optimizer.get("fused", False))):
        device = str(accelerator)
        if device == "auto":
            device = "gpu" if torch.cuda.is_available() else "cpu"
        plugins.append(_FusedAdamWBF16Precision(
            "bf16-mixed", device="cuda" if device == "gpu" else device))
        del trainer_kwargs["precision"]
    if plugins:
        trainer_kwargs["plugins"] = plugins
    # Optional debug/smoke knobs (kept opt-in, no behavior change by default).
    for optional_key in ("max_steps", "limit_train_batches", "limit_val_batches"):
        if optional_key in train_cfg and train_cfg.get(optional_key) is not None:
            trainer_kwargs[optional_key] = train_cfg.get(optional_key)

    sampler_mode = str(cfg.data.get("sampler", {}).get("mode", "random")).strip().lower()
    wrench_sampler_mode = str(cfg.data.get("sampler_mode", "random")).strip().lower()
    if sampler_mode == "locality_blocked" or (
        str(cfg.get("task", "")) == "tactohand_wrench"
        and wrench_sampler_mode == "sequential_by_video"
    ):
        trainer_kwargs["use_distributed_sampler"] = False

    # Validation scheduling (configurable):
    # - Epoch mode (default): run validation every N epochs via
    #   ``training.check_val_every_n_epoch``.
    # - Step mode (optional): if ``training.val_check_interval_steps > 0``,
    #   run validation every N train steps and disable epoch-based checks.
    val_check_interval_steps = int(train_cfg.get("val_check_interval_steps", 0))
    check_val_every_n_epoch = int(train_cfg.get("check_val_every_n_epoch", 1))
    trainer_kwargs["check_val_every_n_epoch"] = max(1, check_val_every_n_epoch)
    if val_check_interval_steps > 0:
        trainer_kwargs["val_check_interval"] = val_check_interval_steps

    trainer = pl.Trainer(
        **trainer_kwargs,
    )
    print(f"[TIMING] Trainer created: {time.time()-t_start:.2f}s")

    if local_rank == 0 and verbosity_rank[verbosity] >= verbosity_rank["debug"]:
        print(f"[DEBUG] check_val_every_n_epoch={trainer.check_val_every_n_epoch}")
        print(f"[DEBUG] val_check_interval={trainer.val_check_interval}")
        print(f"[DEBUG] num_sanity_val_steps={trainer.num_sanity_val_steps}")
        print(
            "[DEBUG] use_distributed_sampler="
            f"{trainer._accelerator_connector.use_distributed_sampler}"
        )

    # When using step-based validation, disable epoch-based checks AFTER
    # Trainer init so that sanity validation still runs.
    if val_check_interval_steps > 0:
        trainer.check_val_every_n_epoch = None
        print(
            "[DEBUG] Step-based validation active: "
            f"validation/checkpoint only every {val_check_interval_steps} train steps; "
            "epoch-end validation disabled"
        )
    else:
        print(
            "[DEBUG] Epoch-based validation active: "
            f"validation/checkpoint every {trainer.check_val_every_n_epoch} epoch(s)"
        )

    # Suppress known false-positive warnings:
    # 1) PL steps OneCycleLR scheduler before optimizer on accumulation batches — cosmetic,
    #    schedule is correct because estimated_stepping_batches already accounts for accumulation.
    # 2) DDP AccumulateGrad stream mismatch — known PyTorch/DDP interaction, no correctness impact.
    warnings.filterwarnings(
        "ignore",
        message=".*lr_scheduler.step.*before.*optimizer.step.*",
        category=UserWarning,
    )
    if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    print(f"[TIMING] Starting trainer.fit() at: {time.time()-t_start:.2f}s")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)


@hydra.main(version_base="1.3", config_path=str(Path(__file__).resolve().parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    validate_task_data_pairing(cfg)
    _train_impl(cfg)


if __name__ == "__main__":
    main()
