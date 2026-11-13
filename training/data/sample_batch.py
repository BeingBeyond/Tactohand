"""Build bounded real-data batches for model inspection and rollout validation."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import core.data.processed_compat as processed_contract
from training.data.dataset import LabelStreamDataset
from training.data.datamodule import SubstepForceDataModule

SEQ = 16

def make_batch(name: str, count: int, eligible_only: bool, *,
               processed_root: str | None = None, max_episodes: int = 8,
               video_source: str = "full"):

    processed_root = processed_root or os.environ.get("PROCESSED_ROOT")
    if not processed_root:
        raise ValueError("Pass processed_root or set PROCESSED_ROOT")
    root = Path(processed_root) / f'{name}_lerobot'
    candidates = None
    if processed_contract.is_processed(root):
        context = processed_contract.payload_table(root, 'tactile_context',
            ['episode_index', 'topology', 'has_force', 'has_non_target_force']).to_pylist()
        # Prefer safe rigid episodes even when most of a dataset carries
        # non-target forces. Still audit aggregation if none are eligible.
        context.sort(key=lambda row: not (row['topology'] == 'rigid'
            and row['has_force'] and not row['has_non_target_force']))
        candidates = [int(row['episode_index']) for row in context[:max_episodes]]
    ds = LabelStreamDataset({
        "root": f"{processed_root}/{name}_lerobot",
        "labels_root": f"{processed_root}/{name}_lerobot",
        "split": "all", "sequence_length": SEQ, "image_size": 224,
        "video_backend": "cv2", "video_cache_size": 1,
        "video_source": video_source,
        "vertex_force_frame": "vertex_local",
        "include_episode_indices": candidates,
    }, "all", max_episodes=max_episodes)
    samples, seen = [], set()
    for index, record in sorted(enumerate(ds.records), key=lambda pair: not pair[1].rollout_eligible):
        if eligible_only and not record.rollout_eligible:
            continue
        if not record.force_supervised or record.real_length < SEQ or record.episode_index in seen:
            continue
        seen.add(record.episode_index)
        start = record.real_length // 2 - SEQ // 2
        samples.append(ds[(index, start)])
        if len(samples) >= count:
            break
    ds.close()
    if not samples:
        raise ValueError(f"{name}: no full eligible windows among {max_episodes} episodes")
    dm = object.__new__(SubstepForceDataModule)
    dm.dataset_cfg = {"deduplicate_batch_frames": False}
    dm.wrench_scale = np.ones(6, np.float32)
    dm.point_force_scale = 1.0
    return dm._collate(samples)
