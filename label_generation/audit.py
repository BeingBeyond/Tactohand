#!/usr/bin/env python3
"""Validate complete canonical-v4 contact, proximity, and force annotations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

import numpy as np


EXPECTED_METHOD = "original_surface_multiray_inside_normal_filter_v1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def audit_one(record: dict) -> dict:
    status = str(record.get("status", ""))
    if status in {"not_applicable", "skipped"} and not record.get("label"):
        return {"dataset": record.get("dataset"), "episode_index": record.get("episode_index"),
                "status": status, "valid": True, "reason": record.get("reason", "")}
    path = Path(str(record.get("label", "")))
    errors = []
    try:
        with np.load(path, allow_pickle=False) as values:
            required = {
                "canonical_format_version", "hand_contact_mask",
                "signed_distance_vertices_m", "proximity_vertices",
                "contact_candidate_mask", "contact_geometry_method",
                "hand_vertex_force_world", "force_label_valid",
                "generalized_contact_wrench", "generalized_fitted_wrench",
            }
            missing = sorted(required - set(values.files))
            if missing:
                errors.append(f"missing:{','.join(missing)}")
            else:
                version = int(values["canonical_format_version"])
                contact = np.asarray(values["hand_contact_mask"], bool)
                signed = np.asarray(values["signed_distance_vertices_m"], np.float32)
                proximity = np.asarray(values["proximity_vertices"], np.float32)
                candidate = np.asarray(values["contact_candidate_mask"], bool)
                force = np.asarray(values["hand_vertex_force_world"])
                valid = np.asarray(values["force_label_valid"], bool)
                length = len(valid)
                expected = (length, 2, 778)
                if version != 4: errors.append(f"version:{version}")
                if contact.shape != expected: errors.append(f"contact_shape:{contact.shape}")
                if signed.shape != expected: errors.append(f"signed_shape:{signed.shape}")
                if proximity.shape != expected: errors.append(f"proximity_shape:{proximity.shape}")
                if candidate.shape != expected: errors.append(f"candidate_shape:{candidate.shape}")
                if force.shape != (*expected, 3): errors.append(f"force_shape:{force.shape}")
                if not np.isfinite(signed).all(): errors.append("signed_nonfinite")
                if not np.isfinite(proximity).all(): errors.append("proximity_nonfinite")
                if np.any((signed < 0.0) & ~contact): errors.append("inside_without_contact")
                scaled = signed.astype(np.float64) / .004
                exp_negative = np.exp(-np.maximum(scaled, 0.0))
                exp_positive = np.exp(np.minimum(scaled, 0.0))
                expected_proximity = np.where(
                    scaled >= 0.0, 2.0 * exp_negative / (1.0 + exp_negative),
                    2.0 / (1.0 + exp_positive))
                if not np.allclose(proximity, expected_proximity, rtol=2e-5, atol=2e-6):
                    errors.append("proximity_transform_mismatch")
                method = str(values["contact_geometry_method"])
                if method != EXPECTED_METHOD:
                    errors.append(f"geometry_method:{method}")
                if len(values["generalized_contact_wrench"]) != length:
                    errors.append("wrench_length")
                if len(values["generalized_fitted_wrench"]) != length:
                    errors.append("fitted_wrench_length")
                return {
                    "dataset": record.get("dataset"),
                    "episode_index": int(record.get("episode_index", -1)),
                    "status": status, "valid": not errors, "errors": errors,
                    "frames": length, "force_valid_frames": int(valid.sum()),
                    "contact_vertices": int(contact.sum()),
                    "inside_vertices": int((signed < 0.0).sum()),
                    "label": str(path),
                }
    except Exception as error:
        errors.append(f"{type(error).__name__}:{error}")
    return {"dataset": record.get("dataset"), "episode_index": record.get("episode_index"),
            "status": status, "valid": False, "errors": errors, "label": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--require-all-force-valid", action="store_true")
    args = parser.parse_args()
    results_path = args.run / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"terminal results are missing: {results_path}")
    records = json.loads(results_path.read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(audit_one, records, chunksize=8))
    invalid = [row for row in rows if not row["valid"]]
    force_partial = [row for row in rows if row.get("force_valid_frames", 0) < row.get("frames", 0)]
    summary = {
        "status": "ok" if not invalid and (not args.require_all_force_valid or not force_partial)
                  else "failed",
        "run": str(args.run.resolve()), "records": len(rows),
        "valid_records": len(rows) - len(invalid), "invalid_records": len(invalid),
        "partial_force_records": len(force_partial),
        "contact_vertices": sum(row.get("contact_vertices", 0) for row in rows),
        "inside_vertices": sum(row.get("inside_vertices", 0) for row in rows),
        "force_valid_frames": sum(row.get("force_valid_frames", 0) for row in rows),
        "frames": sum(row.get("frames", 0) for row in rows),
        "invalid": invalid, "force_partial": force_partial,
    }
    atomic_json(args.run / "annotation_audit.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
