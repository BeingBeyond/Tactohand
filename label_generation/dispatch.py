#!/usr/bin/env python3
"""One schema-driven entrypoint for all Part3 wrench-label datasets.

This is the migration entrypoint.  It owns dataset discovery and the public
CLI; topology-specific dynamics remain internal backends until the generic QP
matrix builder replaces the two legacy backend implementations.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.physics.unified_schema import infer_capabilities, write_canonical_label  # noqa: E402
from label_generation.stage_pipeline import (  # noqa: E402
    acquire_cpu_stage2_lease, acquire_gpu_stage1_lease,
)


def run_gpu_stage1(command: list[str], *, dataset: str, episode: int) -> None:
    """Run one articulated Stage 1 under the per-GPU pipeline limiter."""
    begin = time.perf_counter()
    with acquire_gpu_stage1_lease() as lease:
        print({"event": "stage1_gpu_start", "dataset": dataset, "episode": episode,
               "slot": None if lease.path is None else str(lease.path)}, flush=True)
        subprocess.run(command, check=True)
    print({"event": "stage1_gpu_end", "dataset": dataset, "episode": episode,
           "wall_s": time.perf_counter() - begin}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topology-override", choices=("rigid6d", "floating_base_revolute7d"),
                        help="episode-level topology selected by the unified scheduler")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view-id", default=None,
                        help="canonical observation view; used by rigid backends")
    parser.add_argument("--d-off-m", type=float, default=0.020,
                        help="rigid signed-distance candidate cutoff")
    parser.add_argument("--frames", type=int, default=0,
                        help="0 means the full valid interval; otherwise a centered smoke window")
    parser.add_argument("--frame-start", type=int, default=0,
                        help="first independent frame for static sharded backends")
    parser.add_argument("--frame-stop", type=int,
                        help="exclusive independent-frame stop for static sharded backends")
    parser.add_argument("--force-temporal-regularizer", type=float,
                        help="override the topology-specific contact-force temporal weight")
    parser.add_argument("--articulated-window-size", type=int, default=int(
        os.environ.get("ARCTIC_STAGE2_WINDOW_SIZE", "32")))
    parser.add_argument("--articulated-overlap", type=int, default=int(
        os.environ.get("ARCTIC_STAGE2_OVERLAP", "4")))
    parser.add_argument(
        "--sequence-parts", type=int, default=1,
        help="split full rigid Stage-2 sequence solves into equal timeline parts",
    )
    parser.add_argument("--robust-stage2", action="store_true")
    parser.add_argument("--refine-window-size", type=int, default=64)
    parser.add_argument("--refine-overlap", type=int, default=16)
    parser.add_argument("--per-frame-time-limit-s", type=float, default=30.0)
    parser.add_argument("--window-time-limit-s", type=float, default=60.0)
    parser.add_argument(
        "--normal-filter-candidates", action="store_true",
        help="opt-in contact_generation-compatible normal filtering for rigid candidates",
    )
    parser.add_argument("--normal-filter-safe-dist-m", type=float, default=0.001)
    parser.add_argument("--normal-filter-angle-deg", type=float, default=90.0)
    parser.add_argument("--stored-contact-gated-distance", action="store_true",
                        help="use the dataset-specific standard/geometry contact policy")
    parser.add_argument(
        "--coherent-force-activation", action="store_true",
        help="stabilize multi-object contact ownership and require a coherent MANO patch",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frames:
        parser.error("Short-window solver removed; use --frames 0 for the final full-interval solver")
    capability = infer_capabilities(args.root)
    topology = args.topology_override or capability.topology
    if topology == "unsupported":
        parser.error(f"{capability.dataset}: {capability.reason}")
    scripts = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if topology == "rigid6d":
        raw_output = args.output_dir / "backend" / f"episode_{args.episode_index:06d}.npz"
        # The production worker loads and solves every complete object track.
        static_independent = capability.dataset in ("mow_lerobot", "obman_lerobot")
        if capability.dataset == "mow_lerobot":
            worker = "mow.py"
        elif capability.dataset == "obman_lerobot":
            worker = "obman.py"
        else:
            worker = "rigid.py"
        command = [sys.executable, str(scripts / worker),
                   "--root", str(args.root), "--episode-index", str(args.episode_index),
                   "--output", str(raw_output),
                   "--device", args.device]
        if args.view_id:
            command.extend(("--view-id", args.view_id))
        command.extend(("--d-off-m", str(args.d_off_m)))
        if capability.dataset == "obman_lerobot":
            command.extend(("--frame-start", str(args.frame_start)))
            if args.frame_stop is not None:
                command.extend(("--frame-stop", str(args.frame_stop)))
        if args.force_temporal_regularizer is not None and not static_independent:
            command.extend(("--force-temporal-regularizer",
                            str(args.force_temporal_regularizer)))
        if args.sequence_parts != 1:
            if static_independent:
                parser.error("--sequence-parts is supported only by full rigid multi-object runs")
            command.extend(("--sequence-parts", str(args.sequence_parts)))
        if args.robust_stage2:
            if static_independent:
                parser.error("--robust-stage2 is supported only by full rigid multi-object runs")
            command.extend((
                "--robust-stage2",
                "--refine-window-size", str(args.refine_window_size),
                "--refine-overlap", str(args.refine_overlap),
                "--per-frame-time-limit-s", str(args.per_frame_time_limit_s),
                "--window-time-limit-s", str(args.window_time_limit_s),
            ))
        if args.normal_filter_candidates and not static_independent:
            command.extend((
                "--normal-filter-candidates",
                "--normal-filter-safe-dist-m", str(args.normal_filter_safe_dist_m),
                "--normal-filter-angle-deg", str(args.normal_filter_angle_deg),
            ))
        if args.stored_contact_gated_distance and not static_independent:
            command.append("--stored-contact-gated-distance")
        if args.coherent_force_activation:
            if static_independent:
                parser.error(
                    "--coherent-force-activation is supported only by full rigid multi-object runs")
            command.append("--coherent-force-activation")
    elif capability.dataset == "hoi4d_lerobot":
        if args.sequence_parts != 1 or args.robust_stage2:
            parser.error("sequence partition/refinement options are supported only for rigid datasets")
        stage1_dir = args.output_dir / "stage1"
        stage2_dir = args.output_dir / "stage2"
        stage1_path = stage1_dir / f"episode_{args.episode_index:06d}.npz"
        raw_path = stage2_dir / f"episode_{args.episode_index:06d}.npz"
        command = [sys.executable, str(scripts / "hoi4d_stage1.py"),
                   "--root", str(args.root), "--episode-index", str(args.episode_index),
                   "--output", str(stage1_path), "--device", args.device,
                   "--d-off-m", str(args.d_off_m)]
        if args.view_id:
            command.extend(("--view-id", args.view_id))
        run_gpu_stage1(command, dataset=capability.dataset, episode=args.episode_index)
        command = [sys.executable, str(scripts / "hoi4d.py"),
                   "--input", str(stage1_path), "--output", str(raw_path),
                   "--window-size", str(args.articulated_window_size),
                   "--overlap", str(args.articulated_overlap)]
        if args.force_temporal_regularizer is not None:
            command.extend(("--force-temporal-regularizer", str(args.force_temporal_regularizer)))
    else:
        if args.sequence_parts != 1 or args.robust_stage2:
            parser.error("sequence partition/refinement options are supported only for rigid datasets")
        stage1_dir = args.output_dir / "stage1"
        stage2_dir = args.output_dir / "stage2"
        command = [sys.executable, str(scripts / "arctic_stage1.py"),
                   "--root", str(args.root), "--episodes", str(args.episode_index),
                   "--window", str(args.frames), "--device", args.device,
                   "--output-dir", str(stage1_dir)]
        # Stage1 is independent of the Stage2 window decomposition.  Always
        # reuse a complete v2 Stage1 sidecar, even when Stage2 is being rebuilt.
        run_gpu_stage1(command, dataset=capability.dataset, episode=args.episode_index)
        command = [sys.executable, str(scripts / "arctic.py"),
                   str(stage1_dir / f"episode_{args.episode_index:06d}.npz"),
                   "--output-dir", str(stage2_dir), "--overwrite",
                   "--window-size", str(args.articulated_window_size),
                   "--overlap", str(args.articulated_overlap)]
        if args.force_temporal_regularizer is not None:
            command.extend(("--force-temporal-regularizer",
                            str(args.force_temporal_regularizer)))
    print({"dataset": capability.dataset, "topology": topology, "command": command})
    stage2_lease = (acquire_cpu_stage2_lease()
                    if topology == "floating_base_revolute7d" else None)
    if stage2_lease is not None:
        print({"event": "stage2_cpu_start", "dataset": capability.dataset,
               "episode": args.episode_index,
               "slot": None if stage2_lease.path is None else str(stage2_lease.path)},
              flush=True)
    subprocess.run(command, check=True)
    if topology == "rigid6d":
        raw_path, stage1_path = raw_output, None
    elif capability.dataset != "hoi4d_lerobot":
        raw_path = stage2_dir / f"episode_{args.episode_index:06d}.npz"
        stage1_path = stage1_dir / f"episode_{args.episode_index:06d}.npz"
    shard_suffix = (f"_frames_{args.frame_start:06d}_{args.frame_stop:06d}"
                    if args.frame_stop is not None else "")
    canonical = (args.output_dir / "labels" /
                 f"episode_{args.episode_index:06d}{shard_suffix}.npz")
    write_canonical_label(
        dataset_root=args.root,
        topology=topology,
        raw_path=raw_path,
        output_path=canonical,
        stage1_path=stage1_path,
    )
    if stage2_lease is not None:
        stage2_lease.release()
    print({"status": "ok", "output": str(canonical)})


if __name__ == "__main__":
    main()
