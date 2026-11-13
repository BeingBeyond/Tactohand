#!/usr/bin/env python3
"""Audit the TactoHand DexYCB wrench preflight contract.

All experiment ground truth is read from an installed standard ``_processed``
dataset.  Solver ``outputs/`` directories are rejected deliberately.
"""
from __future__ import annotations
import core.data.processed_compat as processed_contract

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq




REQUIRED_FORCE_COLUMNS = {
    "episode_index",
    "solver_transition_index",
    "tactile_sample_index",
    "solver_timestamp_s",
    "solver_next_timestamp_s",
    "solver_fps",
    "solver_dt_s",
    "topology",
    "generalized_wrench_dim",
    "generalized_contact_wrench",
    "generalized_fitted_wrench",
    "generalized_environment_wrench",
    "object_velocity_before_world",
    "object_velocity_after_world",
    "object_gravity_world",
    "force_label_valid",
}


def validate_processed_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if "outputs" in root.parts:
        raise ValueError(f"solver outputs are not an experiment data source: {root}")
    required = [
        root / "meta/info.json",
        root / "meta/observations.parquet",
        root / "meta/cameras.parquet",
        root / "force_data",
        root / "tactile_data",
        root / "videos",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "root is not a complete installed _processed dataset; missing: "
            + ", ".join(missing)
        )
    if "_processed" not in root.parts:
        raise ValueError(f"expected a standard _processed dataset root, got: {root}")
    return root


def decode_f64(value: Any, *, width: int | None = None) -> np.ndarray:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected binary float64 payload, got {type(value).__name__}")
    array = np.frombuffer(value, dtype=np.float64).copy()
    if width is not None and array.size != width:
        raise ValueError(f"expected {width} float64 values, got {array.size}")
    return array


def decode_f64_rows(value: Any, width: int) -> np.ndarray:
    array = decode_f64(value)
    if array.size == 0 or array.size % width:
        raise ValueError(
            f"expected a non-empty multiple of {width} float64 values, got {array.size}"
        )
    return array.reshape(-1, width)


def wrench_world_to_camera(wrench_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Rotate force and COM torque from world axes into camera axes."""
    wrench = np.asarray(wrench_world, dtype=np.float64)
    transform = np.asarray(c2w, dtype=np.float64)
    if wrench.shape[-1] != 6:
        raise ValueError("wrench_world must end in 6 values")
    if transform.shape[-2:] != (4, 4):
        raise ValueError("c2w must end in shape (4,4)")
    world_to_camera_rotation = np.swapaxes(transform[..., :3, :3], -1, -2)
    force = np.einsum("...ij,...j->...i", world_to_camera_rotation, wrench[..., :3])
    torque = np.einsum("...ij,...j->...i", world_to_camera_rotation, wrench[..., 3:])
    return np.concatenate((force, torque), axis=-1)


def read_episode_rows(root: Path) -> list[dict[str, Any]]:
    if processed_contract.is_processed(root):
        return processed_contract.episode_rows(root)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def split_overlap_report(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    identity_splits: dict[str, set[str]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in episodes:
        split = str(row.get("split", ""))
        identity = str(row.get("sequence_id") or row["episode_index"])
        subject = str(row.get("subject_id") or "")
        identity_splits[identity].add(split)
        if subject:
            subject_splits[subject].add(split)
        split_counts[split] += 1
    identity_overlap = {
        key: sorted(values) for key, values in identity_splits.items() if len(values) > 1
    }
    subject_overlap = {
        key: sorted(values) for key, values in subject_splits.items() if len(values) > 1
    }
    return {
        "split_episode_counts": dict(sorted(split_counts.items())),
        "physical_sequence_overlap_count": len(identity_overlap),
        "physical_sequence_overlap_examples": dict(list(identity_overlap.items())[:10]),
        "subject_identity_overlap_count": len(subject_overlap),
        "subject_identity_overlap_examples": dict(list(subject_overlap.items())[:10]),
    }


def canonical_locator_report(
    episodes: Sequence[dict[str, Any]], observations: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    view_counts: Counter[str] = Counter()
    for row in observations:
        if bool(row.get("force_available")):
            by_episode[int(row["episode_index"])].append(row)
            view_counts[str(row["view_id"])] += 1
    episode_ids = {int(row["episode_index"]) for row in episodes}
    missing = sorted(episode_ids - set(by_episode))
    duplicates = sorted(key for key, values in by_episode.items() if len(values) != 1)
    canonical = [by_episode[key][0] for key in sorted(by_episode) if len(by_episode[key]) == 1]
    return {
        "episode_count": len(episodes),
        "observation_count": len(observations),
        "canonical_force_locator_count": len(canonical),
        "episodes_without_force_locator": missing[:20],
        "episodes_with_nonunique_force_locator": duplicates[:20],
        "canonical_view_counts": dict(sorted(view_counts.items())),
    }, canonical


def read_locator_rows(root: Path, locator: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    path_value = locator.get(f"{prefix}_path")
    if not path_value:
        raise ValueError(
            f"episode {locator['episode_index']} lacks a {prefix} locator in _processed metadata"
        )
    parquet = pq.ParquetFile(root / str(path_value))
    group = int(locator[f"{prefix}_row_group_index"])
    offset = int(locator[f"{prefix}_row_offset"])
    count = int(locator[f"{prefix}_num_rows"])
    table = parquet.read_row_group(group).slice(offset, count)
    if prefix == "force":
        missing = sorted(REQUIRED_FORCE_COLUMNS - set(table.column_names))
        if missing:
            raise ValueError(f"force stream lacks required columns: {missing}")
    return table.to_pylist()


def sample_by_split(
    canonical: Sequence[dict[str, Any]],
    episode_by_id: dict[int, dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for locator in canonical:
        split = str(episode_by_id[int(locator["episode_index"])].get("split", ""))
        if seen[split] < count:
            chosen.append(locator)
            seen[split] += 1
    return chosen
