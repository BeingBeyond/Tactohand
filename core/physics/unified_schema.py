"""Schema-driven discovery for the unified Part3 wrench pipeline.

Every ``*_lerobot`` root is treated as one input family.  Dataset names are
never used to choose physics: topology is inferred from ``objects/parts`` and
hand availability from the canonical observations table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import json
import os
import tempfile

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc


REFERENCE_TOTAL_MASS_KG = 1.0
MASS_NORMALIZATION = {
    "reference_total_mass_kg": REFERENCE_TOTAL_MASS_KG,
    "force_normalization": "per_unit_total_object_mass",
    "specific_force_unit": "N/kg",
    "specific_torque_unit": "N*m/kg",
    "mass_scaling_contract": "all objects use the same 1 kg reference mass",
}


def _atomic_savez_compressed(path: Path, **values: object) -> None:
    """Write an NPZ without exposing a truncated terminal label."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class DatasetCapabilities:
    root: str
    dataset: str
    topology: Literal["rigid6d", "floating_base_revolute7d", "unsupported"]
    hand_slots: int
    has_hmr: bool
    has_tactile: bool
    has_object_tracks: bool
    has_environment_gravity: bool
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_dataset_roots(processed_root: Path) -> list[Path]:
    """Return valid LeRobot dataset roots without a maintained name list."""
    root = Path(processed_root)
    return sorted(path for path in root.glob("*_lerobot")
                  if (path / "meta/observations.parquet").is_file()
                  and (path / "objects/instances.parquet").is_file())


def infer_capabilities(root: Path) -> DatasetCapabilities:
    """Infer the pipeline backend solely from the standardized on-disk schema."""
    root = Path(root)
    observations_path = root / "meta/observations.parquet"
    instances_path = root / "objects/instances.parquet"
    if not observations_path.is_file() or not instances_path.is_file():
        raise FileNotFoundError(f"{root} is not a complete LeRobot object dataset")
    observation_columns = set(pq.ParquetFile(observations_path).schema_arrow.names)
    episode_files = list((root / "meta/episodes").rglob("*.parquet"))
    episode_columns = (set(pq.ParquetFile(episode_files[0]).schema_arrow.names)
                       if episode_files else set())
    has_hmr = {"hmr_available", "hmr_path"} <= observation_columns
    has_tactile = {"tactile_available", "tactile_path"} <= observation_columns
    has_object_tracks = {"objects_available", "objects_path"} <= observation_columns
    parts_path = root / "objects/parts.parquet"
    topology: Literal["rigid6d", "floating_base_revolute7d", "unsupported"] = "rigid6d"
    reason = "no parts table: free rigid object"
    if root.name == "hoi4d_lerobot":
        # HOI4D contains a mixture of rigid and articulated assets in the
        # global parts table.  Its force-label contract deliberately selects
        # the dedicated two-link articulated loader episode by episode.
        topology = "floating_base_revolute7d"
        reason = "dataset-specific two-link articulated HOI4D contract"
    elif parts_path.is_file():
        parts = pq.read_table(parts_path, columns=["object_asset_id", "joint_type"])
        layouts = []
        for asset_id in set(map(str, parts["object_asset_id"].to_pylist())):
            joints = [str(value) for value in parts.filter(
                pc.equal(parts["object_asset_id"], asset_id)
            )["joint_type"].to_pylist()]
            layouts.append((len(joints), sum(value == "revolute" for value in joints)))
        unique_layouts = set(layouts)
        if unique_layouts == {(2, 1)}:
            topology = "floating_base_revolute7d"
            reason = "each asset has two parts with one revolute joint"
        elif unique_layouts == {(1, 0)}:
            reason = "each asset has one fixed part"
        else:
            topology = "unsupported"
            reason = f"unsupported per-asset topologies: {sorted(unique_layouts)}"
    return DatasetCapabilities(
        root=str(root), dataset=root.name, topology=topology, hand_slots=2,
        has_hmr=has_hmr, has_tactile=has_tactile,
        has_object_tracks=has_object_tracks,
        has_environment_gravity="gravity_dir_world" in episode_columns,
        reason=reason,
    )


def _scalar(values: np.lib.npyio.NpzFile, name: str, default: object = "") -> object:
    return values[name].item() if name in values.files else default


def _hand_vertex_field(value: np.ndarray, length: int, name: str) -> np.ndarray:
    """Normalize a geometry label to the public [T,2,778] layout."""
    array = np.asarray(value)
    if array.shape == (length, 1556):
        array = array.reshape(length, 2, 778)
    if array.shape != (length, 2, 778):
        raise ValueError(f"{name} must have shape [T,2,778], got {array.shape}")
    return array


def write_canonical_label(*, dataset_root: Path, topology: str, raw_path: Path,
                          output_path: Path, stage1_path: Path | None = None) -> None:
    """Convert a backend result into the public topology-independent NPZ contract."""
    raw_path, output_path = Path(raw_path), Path(output_path)
    with np.load(raw_path) as raw:
        episode = int(_scalar(raw, "episode_index", -1))
        start, stop = int(_scalar(raw, "start", 0)), int(_scalar(raw, "stop", 0))
        static_independent = Path(dataset_root).name in ("mow_lerobot", "obman_lerobot")
        if topology == "rigid6d":
            wrench = raw["contact_wrench_world"]
            force = raw["vertex_force_world"]
            if force.ndim == 3 and force.shape[1] % 778 == 0:
                force = force.reshape(len(force), force.shape[1] // 778, 778, 3)
            fitted = (raw["fitted_contact_wrench_world"]
                      if "fitted_contact_wrench_world" in raw.files else wrench.copy())
            valid = (raw["force_label_valid"] if "force_label_valid" in raw.files
                     else np.ones(len(wrench), dtype=bool))
            environment = raw["table_force_world"]
            environment_wrench = (raw["environment_wrench_world"]
                                  if "environment_wrench_world" in raw.files else None)
            support_n = raw["table_support_n"]
            active = raw["table_contact_active"]
            position = raw["x_com_world"]
            partition_metadata = {
                key: raw[key]
                for key in (
                    "sequence_parts", "sequence_partition_boundaries",
                    "sequence_partition_policy", "stage2_solve_policy",
                    "refine_window_size", "refine_overlap", "per_frame_time_limit_s",
                    "window_time_limit_s", "refined_frame_mask", "force_activation_policy",
                )
                if key in raw.files
            }
            geometry_metadata = {
                "hand_contact_mask": _hand_vertex_field(
                    raw["hand_contact_mask"], len(wrench), "hand_contact_mask").astype(bool),
                "signed_distance_vertices_m": _hand_vertex_field(
                    raw["signed_distance_vertices_m"], len(wrench),
                    "signed_distance_vertices_m").astype(np.float32),
                "proximity_vertices": _hand_vertex_field(
                    raw["proximity_vertices"], len(wrench),
                    "proximity_vertices").astype(np.float32),
                "contact_candidate_mask": _hand_vertex_field(
                    raw["contact_candidate_mask"], len(wrench),
                    "contact_candidate_mask").astype(bool),
                "contact_object_owner": _hand_vertex_field(
                    raw["hand_contact_object"], len(wrench),
                    "contact_object_owner").astype(np.int16),
                "force_candidate_object_owner": _hand_vertex_field(
                    raw["contact_candidate_object"], len(wrench),
                    "force_candidate_object_owner").astype(np.int16),
                "contact_geometry_method": np.array(str(_scalar(
                    raw, "contact_geometry_method",
                    "legacy_unspecified"))),
            }
        elif Path(dataset_root).name == "hoi4d_lerobot":
            if stage1_path is None:
                raise ValueError("HOI4D articulated canonical labels require Stage-1 geometry")
            wrench = raw["original_generalized_contact_wrench"]
            fitted = raw["fitted_generalized_wrench"]
            force = raw["vertex_force_world"].reshape(len(wrench), 2, 778, 3)
            valid = raw["force_label_valid"]
            environment = raw["environment_force_world"]
            environment_wrench = raw["environment_generalized_wrench"]
            support_n = np.linalg.norm(environment, axis=1)
            active = raw["environment_contact_active"]
            position = raw["system_com_world"]
            partition_metadata = {
                key: raw[key] for key in ("frame_index", "failures_json", "window_size", "window_overlap")
                if key in raw.files
            }
            # Stage-2 owns force allocation; immutable surface/contact fields
            # remain in the Stage-1 geometry sidecar.
            with np.load(stage1_path, allow_pickle=False) as stage1:
                geometry_metadata = {
                    "hand_contact_mask": _hand_vertex_field(
                        stage1["contact_mask"], len(wrench), "contact_mask").astype(bool),
                    "signed_distance_vertices_m": _hand_vertex_field(
                        stage1["surface_distance_m"], len(wrench),
                        "surface_distance_m").astype(np.float32),
                    "proximity_vertices": _hand_vertex_field(
                        stage1["contact_priority"], len(wrench),
                        "contact_priority").astype(np.float32),
                    "contact_candidate_mask": _hand_vertex_field(
                        stage1["contact_candidate"], len(wrench),
                        "contact_candidate").astype(bool),
                    "contact_object_owner": _hand_vertex_field(
                        stage1["tactile_contact_part"], len(wrench),
                        "tactile_contact_part").astype(np.int16),
                    "force_candidate_object_owner": _hand_vertex_field(
                        stage1["contact_part"], len(wrench),
                        "contact_part").astype(np.int16),
                    "contact_geometry_method": np.array(str(_scalar(
                        stage1, "contact_geometry_method", "legacy_unspecified"))),
                }
        else:
            if stage1_path is None:
                raise ValueError("articulated canonical labels require the Stage-1 sidecar")
            with np.load(stage1_path) as stage1:
                wrench = stage1["generalized_contact_wrench"]
                position = stage1["system_com_world"]
                start, stop = int(stage1["start"]), int(stage1["stop"])
                active = (raw["environment_contact_active"]
                          if "environment_contact_active" in raw.files
                          else stage1["environment_contact_active"])
                geometry_metadata = {
                    "hand_contact_mask": np.stack([
                        stage1[f"{side}_contact_mask"] for side in ("left", "right")
                    ], axis=1).astype(bool),
                    "signed_distance_vertices_m": np.stack([
                        stage1[f"{side}_surface_distance_m"] for side in ("left", "right")
                    ], axis=1).astype(np.float32),
                    "proximity_vertices": np.stack([
                        stage1[f"{side}_contact_priority"] for side in ("left", "right")
                    ], axis=1).astype(np.float32),
                    "contact_candidate_mask": np.stack([
                        stage1[f"{side}_contact_candidate"] for side in ("left", "right")
                    ], axis=1).astype(bool),
                    "contact_object_owner": np.stack([
                        np.where(stage1[f"{side}_contact_mask"],
                                 stage1[f"{side}_surface_owner"], -1)
                        for side in ("left", "right")
                    ], axis=1).astype(np.int16),
                    "force_candidate_object_owner": np.stack([
                        stage1[f"{side}_contact_part"] for side in ("left", "right")
                    ], axis=1).astype(np.int16),
                    "contact_geometry_method": np.array(str(_scalar(
                        stage1, "contact_geometry_method",
                        "legacy_unspecified"))),
                }
            force = raw["vertex_force_world"]
            fitted = raw["fitted_generalized_wrench"]
            valid = raw["force_label_valid"]
            environment = raw["environment_force_world"]
            environment_wrench = (raw["environment_generalized_wrench"]
                                  if "environment_generalized_wrench" in raw.files else None)
            support_n = raw["environment_support_n"]
            partition_metadata = {}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_keys = (
            "solver_fps", "source_fps", "solver_timestamp_s", "source_timestamp_s",
            "solver_transition_index", "solver_transition_timestamp_s",
            "solver_next_timestamp_s", "solver_dt_s", "source_frame_left",
            "source_frame_right", "source_interpolation_alpha",
            "next_source_frame_left", "next_source_frame_right",
            "next_source_interpolation_alpha", "solver_timeline_policy",
        )
        timeline_metadata = {key: raw[key] for key in timeline_keys if key in raw.files}
        stage2_window_size = int(_scalar(
            raw, "refine_window_size", _scalar(raw, "window_size", 0)))
        stage2_overlap = int(_scalar(
            raw, "refine_overlap",
            _scalar(raw, "window_overlap", _scalar(raw, "overlap", 0))))
        solver_metadata = {
            key: raw[key] for key in (
                "solver_eps_abs", "solver_eps_rel", "solver_polishing")
            if key in raw.files
        }
        if static_independent:
            temporal_mode = "independent_frames"
            frame_index = (raw["frame_index"] if "frame_index" in raw.files
                           else np.arange(start, start + len(wrench), dtype=np.int64))
            timeline_metadata = {"sample_frame_index": frame_index}
        else:
            temporal_mode = "trajectory"
            if len(timeline_metadata) != len(timeline_keys):
                # Articulated backends operate on true consecutive source
                # frames.  Preserve their source rate instead of pretending
                # that they use the rigid resampled 30 Hz timeline.
                fps = 30.0
                if stage1_path is not None:
                    with np.load(stage1_path) as stage1:
                        if "fps" in stage1.files:
                            fps = float(stage1["fps"])
                        elif "normalization_json" in stage1.files:
                            fps = float(json.loads(str(stage1["normalization_json"])).get("fps", fps))
                indices = np.arange(len(wrench), dtype=np.int64)
                timestamps = indices.astype(np.float64) / fps
                timeline_metadata = {
                    "solver_fps": np.float64(fps),
                    "solver_transition_index": indices,
                    "solver_transition_timestamp_s": timestamps,
                    "solver_next_timestamp_s": timestamps + 1.0 / fps,
                    "solver_dt_s": np.full(len(wrench), 1.0 / fps),
                    "solver_timeline_policy": np.array("consecutive_source_frames"),
                }
        _atomic_savez_compressed(
            output_path, canonical_format_version=np.int64(4),
            dataset=np.array(Path(dataset_root).name), topology=np.array(topology),
            temporal_mode=np.array(temporal_mode),
            episode_index=np.int64(episode), start=np.int64(start), stop=np.int64(stop),
            normalization_json=np.array(json.dumps(MASS_NORMALIZATION, sort_keys=True)),
            generalized_contact_wrench=wrench, generalized_fitted_wrench=fitted,
            system_com_world=position, hand_vertex_force_world=force,
            environment_force_world=environment, environment_support_n=support_n,
            **({"environment_wrench_world": environment_wrench}
               if environment_wrench is not None else {}),
            environment_contact_active=np.asarray(active, bool),
            force_label_valid=np.asarray(valid, bool), source_result_npz=np.array(str(raw_path)),
            force_allocation_policy=np.array(str(_scalar(raw, "force_allocation_policy", ""))),
            environment_contact_policy=np.array(str(_scalar(
                raw, "environment_contact_policy",
                _scalar(raw, "environment_detector", "")))),
            gaussian_sigma_s=np.float64(_scalar(raw, "gaussian_sigma_s", np.nan)),
            gaussian_sigma_pos_s=np.float64(_scalar(
                raw, "gaussian_sigma_pos_s",
                _scalar(raw, "gaussian_sigma_s", np.nan))),
            gaussian_sigma_rot_s=np.float64(_scalar(
                raw, "gaussian_sigma_rot_s",
                _scalar(raw, "gaussian_sigma_s", np.nan))),
            stage2_window_size=np.int64(stage2_window_size),
            stage2_overlap=np.int64(stage2_overlap),
            **geometry_metadata, **timeline_metadata, **partition_metadata,
            **solver_metadata,
            source_stage1_npz=np.array("" if stage1_path is None else str(stage1_path)),
        )
