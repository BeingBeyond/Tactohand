"""Standalone TactoHand evaluation CLI.

Evaluates a checkpoint using a run directory or an explicit complete YAML
config, with selectable metric families and JSON/console reports.

Examples:
    tactohand-eval --config model.yaml --ckpt model.ckpt \
        --processed-root /path/to/processed --datasets dexycb

    python training/eval_tactohand.py \
        --run logs/tactohand_1node_8gpu_20260917_020343 \
        --ckpt last --datasets arctic

    python training/eval_tactohand.py --run <run> --ckpt best_val_contact_f1 \
        --datasets arctic dexycb --metrics contact proximity \
        --limit-batches 5            # fast smoke

    # Only the proximity family on arctic's ego_view records:
    python training/eval_tactohand.py --run <run> --ckpt last \
        --datasets arctic --views ego_view --metrics proximity

``--ckpt`` accepts ``last``, a checkpoint filename prefix (e.g.
``best_val_contact_f1``), or an explicit path. With ``--config``, an actual
checkpoint file is required. Datasets and split rules come from the config.
``--metrics`` selects registry families (space- or
comma-separated; ``all`` selects everything registered; ``--list-metrics``
prints the available names).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from training.evaluation.harness import EvalHarness, format_console, resolve_checkpoint, write_report
from training.evaluation.registry import available_families
from training.tactohand_wrench.runtime.checkpoints import load_prediction_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", type=Path,
                        help="training run directory (contains .hydra/config.yaml)")
    source.add_argument("--config", type=Path,
                        help="complete composed YAML config; requires an explicit --ckpt file")
    parser.add_argument("--processed-root", type=Path,
                        help="override the config's processed dataset root")
    parser.add_argument("--labels-root", type=Path,
                        help="optional labels tree override")
    parser.add_argument("--rigid-body-reference-root", type=Path,
                        help="offline rigid references for the selected prepared release")
    parser.add_argument("--ckpt", default="last",
                        help="checkpoint: 'last', a filename prefix in "
                             "checkpoints/, or an explicit path")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subset of the run's val_datasets "
                             "(default: all of them)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help=f"metric families (default: all; available: "
                             f"{available_families()})")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="stop after N batches (smoke runs); 0 = full")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="process only every N-th batch starting at I "
                             "(e.g. --shard 0/2 and --shard 1/2 in two "
                             "concurrent processes shard the eval; the "
                             "report marks the shard so runs never "
                             "overwrite each other)")
    parser.add_argument("--precision", default=None,
                        help="override autocast precision (e.g. bf16-mixed, 32)")
    parser.add_argument("--views", nargs="+", default=None,
                        help="restrict every selected dataset to these "
                             "view_ids (e.g. --views ego_view; multiview "
                             "datasets build records only for these views)")
    parser.add_argument("--visualize", type=int, default=0, metavar="N",
                        help="render a GT-vs-prediction comparison video + "
                             "contact sheet for the first N windows per "
                             "dataset (proximity panels: GT | PRED | |err|)")
    parser.add_argument("--viz-dir", type=Path, default=None,
                        help="viz output root (default: <run>/eval/viz; "
                             "renders land under <viz-dir>/<ckpt>_<split>/)")
    parser.add_argument("--viz-modalities", nargs="+", default=None,
                        choices=("proximity", "contact", "force"),
                        help="which comparison panels to render (default: "
                             "inferred from --metrics)")
    parser.add_argument("--split", default="val",
                        choices=("val", "train"),
                        help="'val' (default): the run's per-dataset val "
                             "splits; 'train': evaluate on the training "
                             "split (photometric jitter and window/video "
                             "shuffling disabled for clean inputs)")
    parser.add_argument("--split-by-dataset", nargs="+", default=None,
                        metavar="NAME=SPLIT",
                        help="per-dataset val-split override, e.g. "
                             "hocap_multiview=test; datasets not listed "
                             "keep the run config's mapping")
    parser.add_argument("--predictor", nargs="+", default=["tactohand"],
                        help="one or more prediction sources, evaluated in "
                        "sequence with per-predictor output subfolders: "
                        "'tactohand' (the run checkpoint, default), "
                        "'haco' (external HACO release model on "
                        "hand-crop inputs; forced handcrop video source "
                        "for the selected datasets), 'contactprompt' "
                        "(training-free MLLM baseline on hand-crop "
                        "inputs), 'egophi' (zero-shot EgoPHI on hand-crop "
                        "inputs), or 'interfield' / 'interfield_sf' / "
                        "'interfield_lstm' (ARCTIC InterField FieldSF / "
                        "FieldLSTM proximity baselines; proximity-only; "
                        "research adapters require the full source checkout). "
                        "Non-tactohand predictors skip the "
                        "run checkpoint entirely.")
    parser.add_argument("--input-source", default=None,
                        choices=("full", "handcrop"),
                        help="override the input frames for the tactohand "
                             "predictor: 'handcrop' evaluates on hand-crop "
                             "crops (baseline-aligned inputs; forces the "
                             "handcrop video stream like --predictor haco), "
                             "'full' keeps the run's native inputs. Default: "
                             "full for tactohand, handcrop for external "
                             "predictors.")
    parser.add_argument("--subject-filter", default=None, metavar="SUBJ",
                        help="evaluate only episodes whose meta/episodes "
                             "subject_id matches (e.g. 's05' reproduces "
                             "EgoPHI's official ARCTIC held-out-subject "
                             "split). Default: no filtering.")
    parser.add_argument("--output", type=Path, default=None,
                        help="output ROOT directory: each eval creates a "
                             "timestamped subfolder inside it holding the "
                             "report JSON and (if --visualize was passed) the "
                             "viz renders. Default: <run>/eval/")
    parser.add_argument("--list-metrics", action="store_true",
                        help="print registered metric families and exit")
    args = parser.parse_args()
    if not args.list_metrics and args.run is None and args.config is None:
        parser.error("provide --run or --config")
    return args


def main() -> None:
    args = parse_args()
    # This is a SINGLE-PROCESS eval tool: sharding is --shard's job.  Stale
    # DDP env from training launchers (WORLD_SIZE/RANK/LOCAL_RANK) would
    # silently make HomogeneousLabelBatchSampler hand every eval process
    # only 1/world_size of the batches (2026-09-22: a WORLD_SIZE=8 pod
    # yielded 6 of 46 arctic batches and starved two shards to zero).
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                "ROLE_RANK", "ROLE_NAME", "LOCAL_WORLD_SIZE"):
        os.environ.pop(var, None)
    if args.list_metrics:
        for name in available_families():
            print(name)
        return
    if args.run is not None:
        args.run = args.run.expanduser().resolve()
    config_path = (args.config.expanduser().resolve() if args.config is not None
                   else args.run / ".hydra" / "config.yaml")
    configured = load_prediction_config(config_path)

    shard_index, shard_count = 0, 1
    if args.shard is not None:
        head, _, tail = str(args.shard).partition("/")
        shard_index, shard_count = int(head), int(tail)
        if not (0 <= shard_index < shard_count):
            raise SystemExit(f"--shard expects I/N with 0<=I<N, got {args.shard}")
    metric_selectors = (args.metrics if args.metrics and args.metrics != ["all"]
                        else available_families())
    # Accept comma-separated selectors too: --metrics contact,proximity
    metric_selectors = [s for chunk in metric_selectors
                        for s in str(chunk).split(",") if s]
    split_by_dataset = None
    if args.split_by_dataset:
        split_by_dataset = {}
        for chunk in args.split_by_dataset:
            for pair in str(chunk).split(","):
                name, _, value = pair.partition("=")
                if not name or not value:
                    raise SystemExit(
                        f"--split-by-dataset expects NAME=SPLIT, got {pair!r}")
                split_by_dataset[name] = value
    # --output is an eval ROOT: each invocation creates one timestamped
    # subfolder with a per-predictor child holding that model's report JSON
    # and viz renders, so a multi-model eval keeps everything from one run
    # in one place and repeated evals never overwrite each other.
    eval_dir = None
    output_root = args.output
    if args.run is None and output_root is None:
        output_root = Path.cwd() / "outputs" / "eval"
    if output_root is not None:
        # resolve(): some predictors (haco) chdir into their release root
        # during construction — relative output paths must not follow.
        eval_dir = _new_eval_dir(output_root).resolve()
        print(f"[eval-tactohand] output dir={eval_dir}")

    for predictor in args.predictor:
        print(f"[eval-tactohand] ===== predictor: {predictor}")
        checkpoint = resolve_checkpoint(args.run, args.ckpt)
        viz_dir = (eval_dir / predictor / "viz") if eval_dir is not None \
            else args.viz_dir
        harness = EvalHarness(
            run_dir=args.run,
            checkpoint=checkpoint,
            datasets=args.datasets or list(configured.data.get("val_datasets", []) or []),
            metric_selectors=metric_selectors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            limit_batches=args.limit_batches,
            precision=args.precision,
            views=args.views,
            visualize_windows=args.visualize,
            viz_dir=viz_dir,
            split=args.split,
            split_by_dataset=split_by_dataset,
            predictor=predictor,
            input_source=args.input_source,
            shard_index=shard_index, shard_count=shard_count,
            subject_filter=args.subject_filter,
            viz_modalities=args.viz_modalities,
            config_path=config_path,
            processed_root=args.processed_root,
            labels_root=args.labels_root,
            rigid_body_reference_root=args.rigid_body_reference_root,
        )
        print(f"[eval-tactohand] run={args.run}")
        print(f"[eval-tactohand] checkpoint={checkpoint}")
        print(f"[eval-tactohand] datasets={harness.datasets} "
              f"metrics={harness.metric_selectors} device={harness.device}")
        report = harness.run()

        if eval_dir is not None:
            output = eval_dir / predictor / _report_name(report, harness)
        else:
            stem = Path(report["checkpoint"]).stem.split("-")[0]
            # Namespace the default name: the same checkpoint evaluated on
            # different splits / view subsets must not overwrite its report.
            parts = [stem]
            if report["config"].get("split") not in (None, "val"):
                parts.append(f"split_{report['config']['split']}")
            if report["config"].get("views"):
                parts.append("_".join(report["config"]["views"]))
            if report["config"].get("shard"):
                parts.append("shard_"
                             + report["config"]["shard"].replace("/", "of"))
            parts.append("_".join(harness.datasets))
            name = "__".join(parts) + ".json"
            assert args.run is not None
            output = args.run / "eval" / name
        write_report(report, output)
        print(format_console(report))
        print(f"\n[eval-tactohand] report written to {output}")


def _new_eval_dir(root: Path) -> Path:
    """Create ``root/<timestamp>`` (plus a collision-free suffix).

    mkdir is the ATOMIC claim: concurrent evals (e.g. the 8-GPU sharded
    launcher starting within the same second) race through any
    exists()-then-mkdir sequence, so the failed claim retries with the
    next suffix instead of raising (2026-09-22: six of eight arctic
    shards died on FileExistsError).
    """
    import time
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = 0
    while True:
        eval_dir = root / (stamp if suffix == 0 else f"{stamp}_{suffix}")
        try:
            eval_dir.mkdir()
            return eval_dir
        except FileExistsError:
            suffix += 1


def _report_name(report: dict, harness: EvalHarness) -> str:
    """The default report filename, relative to the eval dir."""
    stem = Path(report["checkpoint"]).stem.split("-")[0]
    parts = [stem]
    if report["config"].get("split") not in (None, "val"):
        parts.append(f"split_{report['config']['split']}")
    if report["config"].get("views"):
        parts.append("_".join(report["config"]["views"]))
    if report["config"].get("shard"):
        parts.append("shard_"
                     + report["config"]["shard"].replace("/", "of"))
    parts.append("_".join(harness.datasets))
    return "__".join(parts) + ".json"


def _configured_val_datasets(run_dir: Path) -> list[str]:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return list(cfg.data.get("val_datasets", []) or [])


if __name__ == "__main__":
    main()
