#!/usr/bin/env python3
"""Parallel unified wrench solving with one job per physical sequence.

LeRobot observations contain one row group per camera view, while the episode
table identifies the physical sequence.  This launcher reads both tables,
groups by ``sequence_id``, selects one canonical view, and schedules exactly
one unified solve for each physical sequence.  The worker count controls
process-level parallelism; the underlying topology-specific solvers are
unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
import json
import os
import re
from itertools import islice
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.physics.unified_schema import infer_capabilities  # noqa: E402
from label_generation.adapters.part3_rigid_lerobot import (  # noqa: E402
    HAND_INTERVAL_HEIGHT_POLICY, HOGRASPNET_HAND_COMPLETION_VERSION,
    HOGRASPNET_OBJECT_COMPLETION_VERSION,
)
from label_generation.adapters.trajectory_policy import resolve_rigid_gaussian_sigmas, rigid_gaussian_sigma_s


DATASET_ALIASES = {
    "arctic": "arctic_lerobot",
    "dexycb": "dexycb_lerobot",
    "gigahands": "gigahands_lerobot",
    "h2o3d": "h2o3d_lerobot",
    "h2o": "h2o_lerobot",
    "mow": "mow_lerobot",
    "obman": "obman_lerobot",
    "ho3dv2": "ho3dv2_lerobot",
    "ho3dv3": "ho3dv3_lerobot",
    "hocap": "hocap_multiview_lerobot",
    "hograspnet": "hograspnet_lerobot",
    "hoi4d": "hoi4d_lerobot",
    "hot3d": "hot3d_lerobot",
    "oakink-v1": "oakink-v1_lerobot",
    "oakink-v2": "oakink-v2_lerobot",
    "show3d": "show3d_lerobot",
    "taco": "taco_lerobot",
}


def print_pretty_progress(*, finished: int, total: int,
                          counts: dict[str, int], started_at: float,
                          label: str = "all", running: int = 0) -> None:
    """Render the same episode-completion bar used by single-dataset runs."""
    elapsed = max(time.monotonic() - started_at, 1e-6)
    ratio = finished / max(total, 1)
    width = 32
    filled = min(width, int(width * ratio))
    bar = "━" * filled + "╺" + "─" * max(0, width - filled - 1) if finished < total else "━" * width
    rate = max(finished - counts.get("skipped", 0), 0) / elapsed
    remaining = max(total - finished, 0)
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    eta = "--:--:--" if not np.isfinite(eta_seconds) else time.strftime(
        "%H:%M:%S", time.gmtime(eta_seconds))
    elapsed_text = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    color = "\033[92m" if not counts.get("failed") and not counts.get("partial") else "\033[93m"
    digits = max(4, len(str(total)))
    line = (
        f"\r{color}Force solve {label:<12} [{bar}] "
        f"{finished:{digits}d}/{total:<{digits}d} {ratio:6.2%}\033[0m  "
        f"▶ {running:3d}  "
        f"✓ {counts.get('ok', 0):4d}  ↷ {counts.get('skipped', 0):4d}  "
        f"△ {counts.get('partial', 0):3d}  ✗ {counts.get('failed', 0):3d}  "
        f"{rate:5.2f} seq/s  elapsed {elapsed_text}  ETA {eta}"
    )
    print(line, end="\n" if finished >= total else "", flush=True)


@dataclass(frozen=True)
class Job:
    dataset: str
    root: str
    topology: str
    sequence_id: str
    episode_index: int
    selected_view: str
    view_count: int
    output_dir: str
    device: str
    frames: int
    d_off_m: float
    overwrite: bool
    robust_stage2: bool = False
    refine_window_size: int = 64
    refine_overlap: int = 16
    articulated_window_size: int = 32
    articulated_overlap: int = 4
    normal_filter_candidates: bool = True
    normal_filter_safe_dist_m: float = 0.001
    normal_filter_angle_deg: float = 90.0
    stored_contact_gated_distance: bool = False
    coherent_force_activation: bool = False
    frame_start: int = 0
    frame_stop: int | None = None


def canonical_label_path(job: Job) -> Path:
    suffix = (f"_frames_{job.frame_start:06d}_{job.frame_stop:06d}"
              if job.frame_stop is not None else "")
    return Path(job.output_dir) / "labels" / f"episode_{job.episode_index:06d}{suffix}.npz"


def not_applicable_path(job: Job) -> Path:
    return Path(job.output_dir) / "NOT_APPLICABLE.json"


def terminal_job_output(job: Job) -> bool:
    return valid_canonical_label(canonical_label_path(job), job) or not_applicable_path(job).is_file()


def valid_canonical_label(path: Path, job: Job) -> bool:
    """Return whether *path* is a complete label for this exact job."""
    try:
        with np.load(path, allow_pickle=False) as values:
            required = {
                "canonical_format_version", "dataset", "topology", "episode_index",
                "generalized_contact_wrench", "generalized_fitted_wrench",
                "hand_vertex_force_world", "force_label_valid", "hand_contact_mask",
                "signed_distance_vertices_m", "proximity_vertices",
                "contact_candidate_mask", "contact_geometry_method",
            }
            if not required <= set(values.files):
                return False
            wrench = values["generalized_contact_wrench"]
            fitted = values["generalized_fitted_wrench"]
            force = values["hand_vertex_force_world"]
            valid = values["force_label_valid"]
            version = int(values["canonical_format_version"])
            temporal_mode = str(values["temporal_mode"]) if "temporal_mode" in values.files else "trajectory"
            timeline_valid = (
                (temporal_mode == "independent_frames"
                 and "sample_frame_index" in values.files
                 and len(values["sample_frame_index"]) == len(wrench))
                or (temporal_mode == "trajectory"
                    and "solver_fps" in values.files
                    and float(values["solver_fps"]) > 0
                    and "solver_transition_index" in values.files
                    and np.array_equal(values["solver_transition_index"],
                                       np.arange(len(wrench), dtype=np.int64)))
            )
            expected_length = ((job.frame_stop - job.frame_start)
                               if job.frame_stop is not None else None)
            sequence_table_current = (
                temporal_mode == "independent_frames"
                or job.topology not in {"rigid6d", "floating_base_revolute7d"}
                or (
                    "environment_contact_policy" in values.files
                    and str(values["environment_contact_policy"])
                    == HAND_INTERVAL_HEIGHT_POLICY
                )
            )
            if job.topology in {"rigid6d", "floating_base_revolute7d"}:
                # Smoothing contract: labels must carry the split position/
                # rotation bandwidths currently configured (CLI/env resolved
                # exactly like the solver).  Labels written before the split
                # are stale by definition and get re-solved, which is what
                # makes a sigma change an automatic full re-solve.  Show3D
                # keeps its legacy single-field acceptance so an unchanged
                # rerun does not invalidate its installed generation.
                expected_pos, expected_rot = resolve_rigid_gaussian_sigmas(
                    Path(job.root).name, environ=os.environ)
                smoothing_current = (
                    {"gaussian_sigma_pos_s", "gaussian_sigma_rot_s"} <= set(values.files)
                    and np.isclose(float(values["gaussian_sigma_pos_s"]), expected_pos,
                                   rtol=0.0, atol=1e-12)
                    and np.isclose(float(values["gaussian_sigma_rot_s"]), expected_rot,
                                   rtol=0.0, atol=1e-12)
                ) or (
                    job.topology == "rigid6d"
                    and
                    Path(job.root).name == "show3d_lerobot"
                    and "gaussian_sigma_pos_s" not in values.files
                    and "gaussian_sigma_s" in values.files
                    and np.isclose(
                        float(values["gaussian_sigma_s"]),
                        rigid_gaussian_sigma_s(Path(job.root).name),
                        rtol=0.0, atol=1e-12,
                    )
                )
            else:
                smoothing_current = True
            expected_stage2_window = (
                job.articulated_window_size
                if job.topology == "floating_base_revolute7d"
                else job.refine_window_size
            )
            expected_stage2_overlap = (
                job.articulated_overlap
                if job.topology == "floating_base_revolute7d"
                else job.refine_overlap
            )
            stage2_config_current = (
                temporal_mode == "independent_frames"
                or (
                    "stage2_window_size" in values.files
                    and "stage2_overlap" in values.files
                    and int(values["stage2_window_size"]) == expected_stage2_window
                    and int(values["stage2_overlap"]) == expected_stage2_overlap
                )
            )
            if Path(job.root).name == "arctic_lerobot":
                expected_eps = float(os.environ.get("ARTICULATED_OSQP_EPS", "1e-8"))
                expected_polishing = os.environ.get(
                    "ARTICULATED_OSQP_POLISHING", "1") == "1"
                stage2_config_current = (
                    stage2_config_current
                    and "solver_eps_abs" in values.files
                    and "solver_eps_rel" in values.files
                    and "solver_polishing" in values.files
                    and np.isclose(float(values["solver_eps_abs"]), expected_eps,
                                   rtol=0.0, atol=1e-15)
                    and np.isclose(float(values["solver_eps_rel"]), expected_eps,
                                   rtol=0.0, atol=1e-15)
                    and bool(values["solver_polishing"]) == expected_polishing
                )
            return (
                version == 4 and timeline_valid
                and sequence_table_current and smoothing_current
                and stage2_config_current
                and str(values["dataset"]) == Path(job.root).name
                and str(values["topology"]) == job.topology
                and int(values["episode_index"]) == job.episode_index
                and wrench.ndim == 2 and len(wrench) > 0
                and fitted.shape == wrench.shape
                and force.ndim == 4 and force.shape[:3] == (len(wrench), 2, 778)
                and values["hand_contact_mask"].shape == (len(wrench), 2, 778)
                and values["signed_distance_vertices_m"].shape == (len(wrench), 2, 778)
                and values["proximity_vertices"].shape == (len(wrench), 2, 778)
                and values["contact_candidate_mask"].shape == (len(wrench), 2, 778)
                and valid.shape == (len(wrench),)
                and (expected_length is None or len(wrench) == expected_length)
                and (Path(job.root).name != "arctic_lerobot"
                     or ("force_allocation_policy" in values.files
                         and str(values["force_allocation_policy"])
                         == "strict_hand_qp_infeasible_environment_balance_v6"))
                and (job.topology != "rigid6d"
                     or Path(job.root).name in ("mow_lerobot", "obman_lerobot")
                     or ("force_allocation_policy" in values.files
                         and str(values["force_allocation_policy"])
                         == "hand_contact_qp_direct_environment_balance_v5"))
            )
    except (OSError, EOFError, KeyError, ValueError, TypeError):
        return False


def zero_valid_dynamics_reason(path: Path) -> str | None:
    """Identify structurally valid labels with no usable dynamics transition.

    These arise when a trajectory is shorter than the inverse-dynamics support
    window, or when every object-pose observation run is shorter than that
    window.  Keep genuine Stage-2 infeasibility as a failure.
    """
    try:
        with np.load(path, allow_pickle=False) as values:
            valid = values["force_label_valid"]
            source = Path(str(values["source_result_npz"]))
            if valid.ndim != 1 or len(valid) == 0 or bool(valid.any()):
                return None
        with np.load(source, allow_pickle=False) as backend:
            failures = json.loads(str(backend["invalid_reason_json"]))[
                "strict_infeasible"
            ]
            candidates = backend["contact_candidate_mask"]
            if failures or bool(candidates.any()):
                return None
        return "insufficient_dynamics_support"
    except (OSError, EOFError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                              dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, value: object) -> None:
    """Durably append one scheduler-owned result record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _episode_file(root: Path) -> Path:
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquet under {root / 'meta/episodes'}")
    return files[0]


def discover_jobs(root: Path, dataset: str, output_root: Path, *, frames: int,
                  d_off_m: float, device: str, overwrite: bool,
                  robust_stage2: bool, refine_window_size: int,
                  refine_overlap: int, articulated_window_size: int,
                  articulated_overlap: int,
                  static_shard_size: int, limit: int | None) -> list[Job]:
    capability = infer_capabilities(root)
    episode_path = _episode_file(root)
    episode_schema = set(pq.ParquetFile(episode_path).schema_arrow.names)
    episode_columns = ["episode_index", "sequence_id"]
    if "length" in episode_schema:
        episode_columns.append("length")
    episodes = pq.read_table(episode_path, columns=episode_columns)
    episode_rows = episodes.to_pylist()
    observation_path = root / "meta/observations.parquet"
    columns = ["episode_index", "view_id"]
    schema = set(pq.ParquetFile(observation_path).schema_arrow.names)
    for optional in ("hmr_available", "objects_available", "tactile_available",
                     "hmr_num_rows", "objects_num_rows", "tactile_num_rows"):
        if optional in schema:
            columns.append(optional)
    observations = pq.read_table(observation_path, columns=columns)
    hoi4d_articulated_episodes: set[int] = set()
    if dataset == "hoi4d":
        # HOI4D stores all 890k object rows in one row group.  Read the two
        # identifying columns once; reopening that row group per episode is
        # both incorrect for startup latency and wasteful on NFS.
        object_files = sorted((root / "objects/frames").rglob("*.parquet"))
        for object_file in object_files:
            table = pq.read_table(object_file, columns=["episode_index", "object_instance_id"])
            for episode, instance in zip(table["episode_index"].to_pylist(),
                                         table["object_instance_id"].to_pylist()):
                if "multibody" in str(instance):
                    hoi4d_articulated_episodes.add(int(episode))
    by_episode: dict[int, dict[str, dict[str, object]]] = {}
    arrays = {name: observations[name].to_pylist() for name in columns}
    for row in range(observations.num_rows):
        episode = int(arrays["episode_index"][row])
        view = str(arrays["view_id"][row])
        record = {name: arrays[name][row] for name in columns if name not in ("episode_index", "view_id")}
        by_episode.setdefault(episode, {})[view] = record

    # Keep the first episode row per physical sequence. If a dataset has split
    # rows for multiple views, all of those views are represented in the
    # observation table and only this one job is scheduled.
    selected: dict[str, Job] = {}
    for episode_row in episode_rows:
        episode = int(episode_row["episode_index"])
        sequence_value = episode_row["sequence_id"]
        sequence = str(sequence_value) if sequence_value is not None else f"episode:{episode}"
        if sequence in selected:
            continue
        views = by_episode.get(episode, {})
        if not views:
            raise ValueError(f"episode={episode} sequence={sequence!r} has no observations")
        # A full production solve promises a label for every episode frame, so
        # sidecars shorter than the episode can never satisfy its contract.
        # Smoke windows only need enough aligned context for the loader.
        required_rows = 16 if frames else int(episode_row.get("length") or 16)
        usable = []
        for view, record in views.items():
            required_flags = ["hmr_available", "objects_available"]
            flags = [record[name] for name in required_flags
                     if name in record]
            # Production uses original MANO/object geometry to regenerate
            # signed distance, contact, and proximity. Existing tactile
            # sidecars are neither an input nor a discovery prerequisite.
            required_counts = ["hmr_num_rows", "objects_num_rows"]
            if dataset == "show3d":
                required_counts = ["hmr_num_rows"]
            row_counts = [int(record[name]) for name in required_counts
                          if name in record and record[name] is not None]
            object_rows_ok = (dataset != "show3d"
                              or int(record.get("objects_num_rows", 0) or 0) > 0)
            if (all(value is not False for value in flags) and object_rows_ok
                    and all(count >= required_rows for count in row_counts)):
                usable.append(view)
        if not usable:
            continue
        usable = sorted(usable)
        if capability.topology == "floating_base_revolute7d" and "ego_view" in usable:
            canonical_view = "ego_view"
        else:
            canonical_view = max(usable, key=lambda view: (
                min(int(views[view].get("hmr_num_rows", 0) or 0),
                    int(views[view].get("objects_num_rows", 0) or 0)), view))
        output_dir = output_root / dataset / f"sequence_{len(selected):06d}_episode_{episode:06d}"
        episode_topology = capability.topology
        if dataset == "hoi4d":
            episode_topology = ("floating_base_revolute7d"
                                if episode in hoi4d_articulated_episodes
                                else "rigid6d")
        selected[sequence] = Job(
            dataset=dataset, root=str(root), topology=episode_topology,
            sequence_id=sequence, episode_index=episode, selected_view=canonical_view,
            view_count=len(views), output_dir=str(output_dir), device=device,
            frames=frames, d_off_m=d_off_m, overwrite=overwrite,
            robust_stage2=robust_stage2,
            refine_window_size=refine_window_size,
            refine_overlap=refine_overlap,
            articulated_window_size=articulated_window_size,
            articulated_overlap=articulated_overlap)
        if limit is not None and len(selected) >= limit:
            break
    jobs = list(selected.values())
    if dataset == "obman":
        sharded: list[Job] = []
        lengths = {int(row["episode_index"]): int(row.get("length") or 0)
                   for row in episode_rows}
        for job in jobs:
            length = lengths[job.episode_index]
            for begin in range(0, length, static_shard_size):
                end = min(length, begin + static_shard_size)
                output = Path(job.output_dir) / f"frames_{begin:06d}_{end:06d}"
                sharded.append(replace(job, output_dir=str(output), frame_start=begin,
                                       frame_stop=end))
        jobs = sharded
        if limit is not None:
            jobs = jobs[:limit]
    return jobs


def load_jobs_manifest(path: Path) -> list[Job]:
    """Load and validate an explicit scheduler manifest."""
    jobs: list[Job] = []
    seen_keys: set[tuple[str, int, str, int, int | None]] = set()
    seen_outputs: set[str] = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                value.setdefault("robust_stage2", False)
                value.setdefault("refine_window_size", 64)
                value.setdefault("refine_overlap", 16)
                value.setdefault("articulated_window_size", 32)
                value.setdefault("articulated_overlap", 4)
                value.setdefault("frame_start", 0)
                value.setdefault("frame_stop", None)
                job = Job(**value)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid job manifest row {line_number}: {error}") from error
            key = (job.dataset, job.episode_index, job.selected_view,
                   job.frame_start, job.frame_stop)
            if key in seen_keys:
                raise ValueError(f"duplicate manifest job at row {line_number}: {key}")
            if job.output_dir in seen_outputs:
                raise ValueError(f"duplicate output directory at row {line_number}: {job.output_dir}")
            if not Path(job.root).is_dir():
                raise FileNotFoundError(f"manifest dataset root is missing: {job.root}")
            seen_keys.add(key)
            seen_outputs.add(job.output_dir)
            jobs.append(job)
    if not jobs:
        raise ValueError(f"job manifest is empty: {path}")
    return jobs


def _run_job(job: Job) -> dict[str, object]:
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    if not job.overwrite and terminal_job_output(job):
        if not_applicable_path(job).is_file():
            return {**asdict(job), "status": "skipped", "returncode": 0,
                    "reason": "valid canonical label already exists",
                    "label": str(canonical_label_path(job))}
        marker = json.loads(not_applicable_path(job).read_text())
        return {**asdict(job), "status": "not_applicable", "returncode": 0,
                "reason": marker["reason"], "marker": str(not_applicable_path(job))}
    attempt = 1
    while (output / f"worker.attempt_{attempt:03d}.log").exists():
        attempt += 1
    log_path = output / f"worker.attempt_{attempt:03d}.log"
    if job.overwrite:
        not_applicable_path(job).unlink(missing_ok=True)
    # A Warp process probes/creates primary contexts for every visible CUDA
    # device during initialization.  With many workers this used to multiply
    # the context memory by (workers x GPUs), and 100 workers could OOM before
    # the assigned GPU was even used.  Give each child exactly one physical
    # GPU and address it as logical cuda:0 inside that isolated process.
    child_env = os.environ.copy()
    child_device = job.device
    if job.device.startswith("cuda:"):
        physical_device = job.device.removeprefix("cuda:")
        if not physical_device.isdigit():
            raise ValueError(f"invalid CUDA device {job.device!r}; expected cuda:<index>")
        child_env["CUDA_VISIBLE_DEVICES"] = physical_device
        child_env["BEINGH_PHYSICAL_GPU"] = physical_device
        child_device = "cuda:0"
    command = [sys.executable, str(scripts / "dispatch.py"),
               "--root", job.root, "--episode-index", str(job.episode_index),
               "--output-dir", str(output), "--device", child_device,
               "--frames", str(job.frames), "--d-off-m", str(job.d_off_m),
               "--topology-override", job.topology]
    if job.frame_stop is not None:
        command.extend(("--frame-start", str(job.frame_start),
                        "--frame-stop", str(job.frame_stop)))
    if job.topology == "floating_base_revolute7d":
        command.extend(("--articulated-window-size", str(job.articulated_window_size),
                        "--articulated-overlap", str(job.articulated_overlap)))
    elif (job.robust_stage2 and job.frames == 0
          and Path(job.root).name not in ("mow_lerobot", "obman_lerobot")):
        command.extend(("--robust-stage2",
                        "--refine-window-size", str(job.refine_window_size),
                        "--refine-overlap", str(job.refine_overlap)))
    if job.overwrite:
        command.append("--overwrite")
    static_independent = Path(job.root).name in ("mow_lerobot", "obman_lerobot")
    if job.normal_filter_candidates and not static_independent:
        command.extend(("--normal-filter-candidates",
                        "--normal-filter-safe-dist-m", str(job.normal_filter_safe_dist_m),
                        "--normal-filter-angle-deg", str(job.normal_filter_angle_deg)))
    elif not static_independent:
        command.append("--no-normal-filter-candidates")
    if job.stored_contact_gated_distance and not static_independent:
        command.append("--stored-contact-gated-distance")
    if job.coherent_force_activation:
        command.append("--coherent-force-activation")
    if job.topology == "rigid6d" and job.selected_view:
        command.extend(("--view-id", job.selected_view))
    try:
        with log_path.open("w") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       check=False, text=True, env=child_env)
        label = canonical_label_path(job)
        status = ("ok" if completed.returncode == 0 and valid_canonical_label(label, job)
                  else "failed")
        if status == "ok":
            with np.load(label, allow_pickle=False) as values:
                if not bool(values["force_label_valid"].all()):
                    status = "partial"
        error = None
        if completed.returncode == 0 and status == "failed":
            error = "worker exited successfully but canonical label is missing or invalid"
            reason = zero_valid_dynamics_reason(label)
            if reason is not None:
                marker = {"status": "not_applicable", "reason": reason,
                          "dataset": job.dataset, "episode_index": job.episode_index}
                atomic_write_json(not_applicable_path(job), marker)
                return {**asdict(job), "status": "not_applicable", "returncode": 0,
                        "reason": reason, "marker": str(not_applicable_path(job)),
                        "log": str(log_path), "label": str(label)}
        if status == "failed":
            text = log_path.read_text(errors="replace")
            match = re.search(r"selected object track has only (\d+) aligned frames", text)
            if match and int(match.group(1)) < 2:
                marker = {"status": "not_applicable", "reason": "insufficient_trajectory",
                          "aligned_frames": int(match.group(1)), "dataset": job.dataset,
                          "episode_index": job.episode_index}
                atomic_write_json(not_applicable_path(job), marker)
                return {**asdict(job), "status": "not_applicable", "returncode": 0,
                        "reason": "insufficient_trajectory",
                        "aligned_frames": int(match.group(1)),
                        "marker": str(not_applicable_path(job)), "log": str(log_path)}
            if "has no complete visible rigid-object track" in text:
                marker = {"status": "not_applicable", "reason": "insufficient_trajectory",
                          "dataset": job.dataset, "episode_index": job.episode_index}
                atomic_write_json(not_applicable_path(job), marker)
                return {**asdict(job), "status": "not_applicable", "returncode": 0,
                        "reason": "insufficient_trajectory",
                        "marker": str(not_applicable_path(job)), "log": str(log_path)}
        return {**asdict(job), "status": status, "returncode": completed.returncode,
                "command": command, "assigned_device": job.device,
                "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
                "log": str(log_path), "label": str(label),
                **({"error": error} if error else {})}
    except Exception as error:
        return {**asdict(job), "status": "failed", "returncode": -1,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(), "command": command,
                "assigned_device": job.device,
                "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
                "log": str(log_path)}


def _run_job_in_process(job: Job) -> dict[str, object]:
    """Solve and pack a DexYCB episode inside a persistent pool process."""
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not job.overwrite and valid_canonical_label(canonical_label_path(job), job):
        return {**asdict(job), "status": "skipped", "returncode": 0,
                "reason": "valid canonical label already exists",
                "label": str(canonical_label_path(job))}
    if job.dataset != "dexycb" or job.topology != "rigid6d" or job.frames != 0:
        raise ValueError("in-process mode currently supports full DexYCB rigid jobs only")
    physical_device = job.device.removeprefix("cuda:")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    attempt = 1
    while (output / f"worker.attempt_{attempt:03d}.log").exists():
        attempt += 1
    log_path = output / f"worker.attempt_{attempt:03d}.log"
    raw = output / "backend" / f"episode_{job.episode_index:06d}.npz"
    label = canonical_label_path(job)
    argv = [
        "--root", job.root, "--episode-index", str(job.episode_index),
        "--output", str(raw), "--device", "cuda:0", "--view-id", job.selected_view,
        "--d-off-m", str(job.d_off_m), "--robust-stage2",
        "--refine-window-size", str(job.refine_window_size),
        "--refine-overlap", str(job.refine_overlap),
    ]
    if job.normal_filter_candidates:
        argv += ["--normal-filter-candidates", "--normal-filter-safe-dist-m",
                 str(job.normal_filter_safe_dist_m), "--normal-filter-angle-deg",
                 str(job.normal_filter_angle_deg)]
    else:
        argv.append("--no-normal-filter-candidates")
    if job.stored_contact_gated_distance:
        argv.append("--stored-contact-gated-distance")
    if job.coherent_force_activation:
        argv.append("--coherent-force-activation")
    try:
        raw.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
            from label_generation.rigid import main as solve_rigid
            from core.physics.unified_schema import write_canonical_label
            solve_rigid(argv)
            write_canonical_label(dataset_root=Path(job.root), topology=job.topology,
                                  raw_path=raw, output_path=label, stage1_path=None)
            print({"status": "ok", "output": str(label)}, flush=True)
        status = "ok" if valid_canonical_label(label, job) else "failed"
        if status == "ok":
            with np.load(label, allow_pickle=False) as values:
                if not bool(values["force_label_valid"].all()):
                    status = "partial"
        return {**asdict(job), "status": status, "returncode": 0,
                "assigned_device": job.device, "cuda_visible_devices": physical_device,
                "log": str(log_path), "label": str(label), "execution_mode": "in_process"}
    except Exception as error:
        with log_path.open("a") as stream:
            stream.write(traceback.format_exc())
        return {**asdict(job), "status": "failed", "returncode": -1,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(), "assigned_device": job.device,
                "cuda_visible_devices": physical_device, "log": str(log_path),
                "execution_mode": "in_process"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path,
                        default=Path("data/beingh_pretrain/egocentric_data/_processed"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_ALIASES),
                        default=list(DATASET_ALIASES))
    parser.add_argument("--jobs-manifest", type=Path,
                        help="run exactly these JSONL Job records instead of dataset discovery")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--stage1-prefetch-workers", type=int, default=0,
                        help="extra in-flight jobs reserved for GPU Stage 1")
    parser.add_argument("--output-root", type=Path,
                        default=Path("artifacts/label_solver/unified_multiworker"))
    parser.add_argument("--frames", type=int, default=0,
                        help="0 means full sequence; positive values select a smoke window")
    parser.add_argument("--d-off-m", type=float, default=0.020,
                        help="rigid signed-distance candidate cutoff")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+",
                        help="round-robin jobs across these devices; overrides --device")
    parser.add_argument("--limit", type=int,
                        help="maximum physical sequences per dataset, useful for validation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--robust-stage2", action="store_true",
                        help="use per-frame baseline plus bounded refinement for full rigid runs")
    parser.add_argument("--refine-window-size", type=int, default=64)
    parser.add_argument("--refine-overlap", type=int, default=16)
    parser.add_argument("--articulated-window-size", type=int, default=32)
    parser.add_argument("--articulated-overlap", type=int, default=4)
    parser.add_argument("--static-shard-size", type=int, default=2048,
                        help="independent OBMan frames per recoverable job")
    parser.add_argument("--normal-filter-candidates", action=argparse.BooleanOptionalAction, default=True,
                        help="opt-in contact_generation-compatible normal filtering")
    parser.add_argument("--normal-filter-safe-dist-m", type=float, default=0.001)
    parser.add_argument("--normal-filter-angle-deg", type=float, default=90.0)
    parser.add_argument("--stored-contact-gated-distance", action="store_true",
                        help="DexYCB/HOCAP: query 20 mm geometry only on stored-contact frames")
    parser.add_argument(
        "--coherent-force-activation", action="store_true",
        help="stabilize multi-object contact ownership and require a coherent MANO patch",
    )
    parser.add_argument("--in-process-workers", action="store_true",
                        help="persistent DexYCB workers; requires exactly one CUDA device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream-worker-logs", action="store_true",
                        help="print each completed worker log to scheduler stdout")
    parser.add_argument("--pretty-progress", action="store_true",
                        help="show one scheduler-level progress bar instead of per-job JSON")
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True,
                        help="finish all jobs even when individual jobs fail (default: true)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage1_prefetch_workers < 0:
        parser.error("--stage1-prefetch-workers must be non-negative")
    if args.static_shard_size < 1:
        parser.error("--static-shard-size must be positive")
    for size, overlap, label in (
        (args.refine_window_size, args.refine_overlap, "rigid refinement"),
        (args.articulated_window_size, args.articulated_overlap, "articulated Stage-2"),
    ):
        if size < 1 or overlap < 0 or overlap >= size:
            parser.error(f"{label} requires window_size>=1 and 0<=overlap<window_size")
    jobs: list[Job] = []
    audit = []
    if args.jobs_manifest is not None:
        jobs = load_jobs_manifest(args.jobs_manifest)
        # An explicit repair manifest records the devices used when it was
        # created, but it must remain portable across machines.  If the
        # caller supplies --devices, remap jobs deterministically here just
        # like the discovery path does.  The original jobs.repair.jsonl is
        # never mutated; only this invocation's scheduler manifest is.
        if args.devices:
            jobs = [replace(job, device=args.devices[index % len(args.devices)])
                    for index, job in enumerate(jobs)]
        for dataset in sorted({job.dataset for job in jobs}):
            dataset_jobs = [job for job in jobs if job.dataset == dataset]
            audit.append({"dataset": dataset, "root": dataset_jobs[0].root,
                          "topology": dataset_jobs[0].topology,
                          "jobs": len(dataset_jobs),
                          "physical_sequences": len({job.sequence_id for job in dataset_jobs}),
                          "view_rows_solved_once": sum(job.view_count for job in {
                              job.sequence_id: job for job in dataset_jobs}.values()),
                          "source": "jobs_manifest"})
    else:
        for dataset in args.datasets:
            root = args.processed_root / DATASET_ALIASES[dataset]
            capability = infer_capabilities(root)
            dataset_jobs = discover_jobs(root, dataset, args.output_root, frames=args.frames,
                                         d_off_m=args.d_off_m, device=args.device,
                                         overwrite=args.overwrite,
                                         robust_stage2=args.robust_stage2,
                                         refine_window_size=args.refine_window_size,
                                         refine_overlap=args.refine_overlap,
                                         articulated_window_size=args.articulated_window_size,
                                         articulated_overlap=args.articulated_overlap,
                                         static_shard_size=args.static_shard_size,
                                         limit=args.limit)
            jobs.extend(dataset_jobs)
            topology_counts = {topology: sum(job.topology == topology for job in dataset_jobs)
                               for topology in sorted({job.topology for job in dataset_jobs})}
            audit.append({"dataset": dataset, "root": str(root),
                          "topology": capability.topology,
                          "episode_topologies": topology_counts,
                          "jobs": len(dataset_jobs),
                          "physical_sequences": len({job.sequence_id for job in dataset_jobs}),
                          "view_rows_solved_once": sum(job.view_count for job in {
                              job.sequence_id: job for job in dataset_jobs}.values())})
        devices = args.devices or [args.device]
        jobs = [replace(job, device=devices[index % len(devices)])
                for index, job in enumerate(jobs)]
    jobs = [replace(job, normal_filter_candidates=bool(args.normal_filter_candidates),
                    normal_filter_safe_dist_m=args.normal_filter_safe_dist_m,
                    normal_filter_angle_deg=args.normal_filter_angle_deg)
            for job in jobs]
    if args.stored_contact_gated_distance:
        jobs = [replace(job, stored_contact_gated_distance=True) for job in jobs]
    if args.coherent_force_activation:
        jobs = [replace(job, coherent_force_activation=True) for job in jobs]
    if args.in_process_workers:
        devices = args.devices or [args.device]
        if len(devices) != 1 or args.datasets != ["dexycb"]:
            parser.error("--in-process-workers currently requires --datasets dexycb and one device")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "jobs.jsonl"
    manifest_temporary = manifest.with_suffix(".jsonl.tmp")
    with manifest_temporary.open("w") as stream:
        for job in jobs:
            stream.write(json.dumps(asdict(job), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(manifest_temporary, manifest)
    complete_before_start = sum(
        not job.overwrite and terminal_job_output(job)
        for job in jobs)
    pending_jobs = [job for job in jobs
                    if job.overwrite or not terminal_job_output(job)]
    print(json.dumps({"status": "planned", "jobs": len(jobs),
                      "complete_before_start": complete_before_start,
                      "pending": len(pending_jobs), "audit": audit,
                      "workers": args.workers,
                      "stage1_prefetch_workers": args.stage1_prefetch_workers,
                      "scheduler_processes": args.workers + args.stage1_prefetch_workers,
                      "manifest": str(manifest)}, indent=2))
    if args.dry_run:
        return
    pending_output_dirs = {job.output_dir for job in pending_jobs}
    completed_jobs = [job for job in jobs if job.output_dir not in pending_output_dirs]
    results: list[dict[str, object]] = []
    for job in completed_jobs:
        marker_path = not_applicable_path(job)
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text())
            results.append({**asdict(job), "status": "not_applicable", "returncode": 0,
                            "reason": marker["reason"], "marker": str(marker_path)})
        else:
            results.append({**asdict(job), "status": "skipped", "returncode": 0,
                            "reason": "valid canonical label already exists",
                            "label": str(canonical_label_path(job))})
    journal = args.output_root / "results.jsonl"
    summary_path = args.output_root / "summary.json"
    # The journal is per invocation. Previous attempt logs and labels are retained.
    journal.unlink(missing_ok=True)
    for result in results:
        append_jsonl(journal, result)
    initial = {"status": "running", "planned": len(jobs),
               "complete_before_start": complete_before_start,
               "pending": len(pending_jobs), "workers": args.workers,
               "stage1_prefetch_workers": args.stage1_prefetch_workers,
               "scheduler_processes": args.workers + args.stage1_prefetch_workers,
               "results_journal": str(journal), "skipped": len(completed_jobs)}
    atomic_write_json(summary_path, initial)
    counts = {"ok": 0, "partial": 0,
              "not_applicable": sum(row["status"] == "not_applicable" for row in results),
              "skipped": sum(row["status"] == "skipped" for row in results), "failed": 0}
    progress_started_at = time.monotonic()
    if args.pretty_progress:
        print_pretty_progress(finished=len(completed_jobs), total=len(jobs),
                              counts=counts, started_at=progress_started_at,
                              label=(jobs[0].dataset if len({job.dataset for job in jobs}) == 1
                                     else "all"),
                              running=min(args.workers + args.stage1_prefetch_workers,
                                          len(pending_jobs)))
    run_job = _run_job_in_process if args.in_process_workers else _run_job
    scheduler_processes = args.workers + args.stage1_prefetch_workers
    with ProcessPoolExecutor(max_workers=scheduler_processes) as executor:
        job_iterator = iter(pending_jobs)
        futures = {executor.submit(run_job, job)
                   for job in islice(job_iterator, scheduler_processes)}
        while futures:
            finished, futures = wait(
                futures,
                timeout=1.0 if args.pretty_progress else None,
                return_when=FIRST_COMPLETED,
            )
            if not finished:
                print_pretty_progress(
                    finished=len(results), total=len(jobs), counts=counts,
                    started_at=progress_started_at,
                    label=(jobs[0].dataset if len({job.dataset for job in jobs}) == 1
                           else "all"),
                    running=len(futures),
                )
                continue
            for future in finished:
                result = future.result()
                if args.stream_worker_logs and result.get("log"):
                    log_path = Path(str(result["log"]))
                    print(json.dumps({"event": "worker_log_begin",
                                      "status": result.get("status"),
                                      "log": str(log_path)}), flush=True)
                    try:
                        print(log_path.read_text(errors="replace"), end="", flush=True)
                    except OSError as error:
                        print(f"[unable to read worker log: {error}]", flush=True)
                    print(json.dumps({"event": "worker_log_end",
                                      "log": str(log_path)}), flush=True)
                results.append(result)
                append_jsonl(journal, result)
                counts[str(result["status"])] += 1
                atomic_write_json(summary_path, {**initial, **counts,
                                  "finished_this_run": len(results) - len(completed_jobs),
                                  "remaining_this_run": len(jobs) - len(results)})
                if args.pretty_progress:
                    print_pretty_progress(finished=len(results), total=len(jobs),
                                          counts=counts, started_at=progress_started_at,
                                          label=(jobs[0].dataset if len({job.dataset for job in jobs}) == 1
                                                 else "all"),
                                          running=len(futures))
                else:
                    print(json.dumps({key: result[key] for key in
                                      ("dataset", "sequence_id", "episode_index", "status")},
                                     sort_keys=True), flush=True)
                if result.get("status") == "failed" and not args.keep_going:
                    raise RuntimeError(
                        f"job failed: {result['dataset']} {result['sequence_id']}")
                next_job = next(job_iterator, None)
                if next_job is not None:
                    futures.add(executor.submit(run_job, next_job))
    results.sort(key=lambda row: (str(row["dataset"]), int(row["episode_index"])))
    atomic_write_json(args.output_root / "results.json", results)
    failed = sum(row.get("status") == "failed" for row in results)
    partial = sum(row.get("status") == "partial" for row in results)
    final_status = "ok" if not failed and not partial else "partial"
    final = {**initial, "status": final_status,
             "finished_this_run": len(results) - len(completed_jobs),
             "ok": sum(row.get("status") == "ok" for row in results),
             "partial": partial,
             "not_applicable": sum(row.get("status") == "not_applicable" for row in results),
             "skipped": sum(row.get("status") == "skipped" for row in results),
             "failed": failed,
             "remaining_this_run": 0, "results": str(args.output_root / "results.json")}
    atomic_write_json(summary_path, final)
    print(json.dumps(final, indent=2))
    if failed or partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
