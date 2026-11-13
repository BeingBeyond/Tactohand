"""Prepared target identity and pose-track reads."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from core.data.inspection import decode_f64_rows, read_locator_rows

def target_instance_id(root: Path, episode_index: int) -> str:
    table = pq.read_table(
        root / "objects/instances.parquet",
        filters=[("episode_index", "=", int(episode_index))],
    ).to_pylist()
    targets = [str(row["object_instance_id"]) for row in table if row.get("role") == "target"]
    if len(targets) != 1:
        raise ValueError(f"episode {episode_index}: expected one role=target, got {targets}")
    return targets[0]
