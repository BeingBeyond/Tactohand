#!/usr/bin/env python3
"""Run real ARCTIC cases that cover every articulated-v6 allocation branch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


POLICY = "strict_hand_qp_infeasible_environment_balance_v6"
MODEL = "strict_hand_qp_or_complete_environment_wrench_v2"
CASES = {
    # hand-contact annotation gaps while the object is off table
    2: {"frames": 514, "strict_fallback": 0},
    # strict infeasibility with hand + table contact
    115: {"frames": 645, "strict_fallback": 1},
    # strict infeasibility with one-hand/one-part contact while off table
    129: {"frames": 631, "strict_fallback": 23},
}


def check_case(case_dir: Path, episode: int, expected: dict[str, int],
               geometry_revision: str | None = None) -> dict[str, object]:
    canonical_path = case_dir / "labels" / f"episode_{episode:06d}.npz"
    stage2_path = case_dir / "stage2" / f"episode_{episode:06d}.npz"
    with np.load(canonical_path, allow_pickle=False) as canonical:
        valid = np.asarray(canonical["force_label_valid"], bool)
        if len(valid) != expected["frames"] or not valid.all():
            raise RuntimeError(
                f"episode {episode}: expected {expected['frames']} valid frames, "
                f"got {int(valid.sum())}/{len(valid)}"
            )
        if str(canonical["force_allocation_policy"]) != POLICY:
            raise RuntimeError(f"episode {episode}: stale canonical allocation policy")
        if "environment_wrench_world" not in canonical.files:
            raise RuntimeError(f"episode {episode}: missing canonical environment wrench")

    with np.load(stage2_path, allow_pickle=False) as stage2:
        if int(stage2["format_version"]) != 15:
            raise RuntimeError(f"episode {episode}: stale Stage-2 format")
        if str(stage2["environment_model"]) != MODEL:
            raise RuntimeError(f"episode {episode}: stale environment model")
        strict = np.asarray(
            stage2["strict_hand_qp_infeasible_environment_fallback"], bool)
        direct = np.asarray(stage2["direct_environment_balance"], bool)
        target = np.asarray(stage2["original_generalized_contact_wrench"], np.float64)
        environment = np.asarray(stage2["environment_generalized_wrench"], np.float64)
        fitted = np.asarray(stage2["fitted_generalized_wrench"], np.float64)
        # Contact/infeasibility counts depend on hand geometry. The original
        # counts are only a regression oracle before an installed MANO migration.
        # Frame coverage, allocation policy and exact wrench identities remain
        # mandatory for both geometries; record the new counts with its revision.
        if geometry_revision is None and int(strict.sum()) != expected["strict_fallback"]:
            raise RuntimeError(
                f"episode {episode}: expected {expected['strict_fallback']} strict fallbacks, "
                f"got {int(strict.sum())}"
            )
        direct_error = float(np.max(np.abs(environment[direct, :6] - target[direct, :6])))
        fitted_error = float(np.max(np.abs(fitted[direct] - target[direct])))
        if direct_error != 0.0 or fitted_error != 0.0:
            raise RuntimeError(
                f"episode {episode}: non-exact direct balance "
                f"environment={direct_error}, fitted={fitted_error}"
            )
    return {
        "episode": episode,
        "frames": expected["frames"],
        "direct_frames": int(direct.sum()),
        "strict_fallback_frames": int(strict.sum()),
        "legacy_reference_strict_fallback_frames": expected["strict_fallback"],
        "geometry_revision": geometry_revision,
        "max_direct_wrench_error": direct_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=4)
    args = parser.parse_args()
    info = json.loads((args.dataset_root/"meta/info.json").read_text())
    contract = info.get("mano_contract", {})
    geometry_revision = (contract.get("geometry_revision")
                         if contract.get("source_basis") == "stock" else None)
    script = Path(__file__).resolve().parent / "dispatch.py"
    environment = os.environ.copy()
    environment["GAUSSIAN_SIGMA_POS_S"] = "0.075"
    environment["GAUSSIAN_SIGMA_ROT_S"] = "0.15"
    environment["ARTICULATED_OSQP_EPS"] = "1e-6"
    environment["ARTICULATED_OSQP_POLISHING"] = "0"
    rows = []
    for episode, expected in CASES.items():
        case_dir = args.work_root / f"episode_{episode:06d}"
        command = [
            args.python, str(script),
            "--root", str(args.dataset_root),
            "--topology-override", "floating_base_revolute7d",
            "--episode-index", str(episode),
            "--output-dir", str(case_dir),
            "--device", args.device,
            "--frames", "0",
            "--articulated-window-size", str(args.window_size),
            "--articulated-overlap", str(args.overlap),
        ]
        subprocess.run(command, check=True, env=environment)
        rows.append(check_case(case_dir, episode, expected, geometry_revision))
    print({"status": "ok", "policy": POLICY, "cases": rows})


if __name__ == "__main__":
    main()
