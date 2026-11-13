#!/usr/bin/env python3
"""Exercise HOI4D articulated Stage-1, Stage-2, and canonical packing."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


EPISODE = 215
EXPECTED_FRAMES = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    runner = Path(__file__).resolve().parent / "dispatch.py"
    subprocess.run([
        args.python, str(runner),
        "--root", str(args.dataset_root),
        "--topology-override", "floating_base_revolute7d",
        "--episode-index", str(EPISODE),
        "--output-dir", str(args.work_root),
        "--device", args.device,
        "--articulated-window-size", "32",
        "--articulated-overlap", "4",
    ], check=True)
    label = args.work_root / "labels" / f"episode_{EPISODE:06d}.npz"
    with np.load(label, allow_pickle=False) as values:
        valid = np.asarray(values["force_label_valid"], bool)
        expected_shape = (EXPECTED_FRAMES, 2, 778)
        if len(valid) != EXPECTED_FRAMES or not valid.all():
            raise RuntimeError(
                f"expected {EXPECTED_FRAMES}/{EXPECTED_FRAMES} valid frames, "
                f"got {int(valid.sum())}/{len(valid)}"
            )
        for key in (
            "hand_contact_mask", "signed_distance_vertices_m", "proximity_vertices",
            "contact_candidate_mask", "contact_object_owner",
            "force_candidate_object_owner",
        ):
            if values[key].shape != expected_shape:
                raise RuntimeError(f"{key} has shape {values[key].shape}, expected {expected_shape}")
        if str(values["contact_geometry_method"]) != \
                "original_surface_multiray_inside_normal_filter_v1":
            raise RuntimeError("stale HOI4D contact geometry method")
    print({"status": "ok", "episode": EPISODE, "frames": EXPECTED_FRAMES})


if __name__ == "__main__":
    main()
