#!/usr/bin/env python3
"""Write the final labels-v5 tree for one dataset directly from a solver run.

The formal generation chain is two commands (spec §2,
docs/labels_v5.md):

  1. solve    scripts/labels/recipes/a800_8gpu_solve.sh
              -> canonical NPZ run tree (audit/provenance endpoint, unchanged)
  2. publish  this script
              -> the exact labels-v5 tree that drops into _processed

No labels-v4 input, no oracle replay, no separate validation passes.  The
streams, index tables, statistics, and info declaration are materialized in
one pass through core.tactile_labels.materialize (the single assembly
implementation); the content hashes in label_index are part of the format
(§4.3), not a check stage.

Inputs per episode: <run>/<dataset>/sequence_*/labels/episode_*.npz
(canonical solver output) plus the sibling backend/episode_*.npz (per-object
payload); object physics and the role=target hint come from the _processed
root's objects/ tables.  Per-run target objects follow the §5.5 rule
(metadata first, otherwise loaded-object wrench energy).

--install validates copied payloads before activating a versioned manifest
for a prepared dataset. The original staging tree remains available for retry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from core.tactile_labels.materialize import (  # noqa: E402
    build_articulated_episode_labels, build_episode_labels,
    build_static_episode_labels, LabelStatsAccumulator, write_label_streams,
)
from core.tactile_labels.schema import TOPOLOGY_RIGID  # noqa: E402
from label_generation.adapters.prepared import target_instance_id, target_physics

ACCEPTED_STATUS = {"ok", "skipped", "partial"}
INSTALL_ENTRIES = ("tactile_data", "force_data", "timeline_data")
INSTALL_META_FILES = ("meta/label_index.parquet", "meta/label_stats.json", "meta/info.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-run", type=Path, required=True)
    parser.add_argument("--dataset", required=True,
                        help="dataset name (dexycb or dexycb_lerobot)")
    processed_root = os.environ.get("PROCESSED_ROOT")
    parser.add_argument("--processed-root", type=Path, default=processed_root,
                        required=not processed_root)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="staging tree (default <solver-run>/labels_v5/"
                             "<dataset>_lerobot)")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--install", action="store_true",
                        help="publish staged labels and activate the new prepared manifest")
    parser.add_argument('--reuse-staged', action='store_true',
                        help='Reuse complete staging with matching solver, geometry, episode coverage and unchanged inputs')
    return parser.parse_args()


def load_run_results(solver_run: Path, dataset: str) -> list[dict]:
    """Per-dataset results files first (the static batch driver writes
    results_<dataset>.json), then the merged trajectory-runner file."""
    for candidate in (solver_run / f"results_{dataset}.json",
                      solver_run / "results.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return load_results(solver_run)


def load_results(solver_run: Path) -> list[dict]:
    path = solver_run / "results.json"
    if path.is_file():
        return json.loads(path.read_text())
    return [json.loads(line) for line
            in (solver_run / "results.jsonl").read_text().splitlines()
            if line.strip()]


def target_of(processed_root: Path, episode_index: int, object_ids: list[str]):
    """§5.5 per-run target rule: metadata first, else loaded-object energy."""
    def select(rows: list[dict], span: tuple[int, int]):
        try:
            wanted = target_instance_id(processed_root, episode_index)
            return wanted, object_ids.index(wanted)
        except Exception:
            spanned = rows[span[0]:span[1]]
            energy = np.zeros(len(object_ids))
            loaded = np.zeros(len(object_ids), bool)
            for row in spanned:
                wrench = np.frombuffer(
                    row["object_contact_wrench"], np.float64
                ).reshape(len(object_ids), 6)
                energy += np.linalg.norm(wrench, axis=1)
                per_object = np.frombuffer(
                    row["object_hand_vertex_force_world_n_per_kg"], np.float32
                ).reshape(len(object_ids), -1)
                loaded |= np.abs(per_object).max(axis=1) > 0
            if bool(loaded.any()):
                energy = np.where(loaded, energy, -1.0)
            index = int(energy.argmax())
            return object_ids[index], index
    return select


def load_hoi4d_articulated_labels(record: dict, dataset_root: Path) -> dict:
    """Recover HOI4D state using the exact solver loader and smoother.

    HOI4D saves physics_json/points_world, unlike ARCTIC's part_N fields.
    Contact and force supervision always come from the saved canonical NPZ.
    """
    from label_generation.adapters.arctic_lerobot import NormalizedPartPhysics
    from core.physics.articulated_gaussian_wrench import estimate_articulated_discrete_gaussian_wrench
    from label_generation.adapters.hoi4d_articulated_lerobot import load_hoi4d_articulated_episode

    data = load_hoi4d_articulated_episode(dataset_root, int(record['episode_index']),
                                        view_id=str(record['selected_view']))
    with np.load(record['label'], allow_pickle=False) as canonical:
        with np.load(str(canonical['source_stage1_npz']), allow_pickle=False) as stage:
            physics = json.loads(str(stage['physics_json']))
            bodies = [NormalizedPartPhysics(
                part_index=index, name='base' if index == 0 else 'child',
                mass=float(item['mass']), volume_m3=1.0,
                com_part=np.asarray(item['com_body'], np.float64),
                inertia_com_part=np.asarray(item['inertia_body'], np.float64),
                joint_type=item['joint_type'],
                joint_axis_parent=np.asarray(item['joint_axis_parent'], np.float64),
                joint_origin_parent=np.asarray(item['joint_origin_parent'], np.float64),
            ) for index, item in enumerate(physics)]
            if len(bodies) != 2 or bodies[1].joint_type != 'revolute':
                raise ValueError('HOI4D publication requires the solver two-link revolute contract')
            if not np.array_equal(data['frame_index'], stage['frame_index']):
                raise ValueError('HOI4D source frames changed after solve')
            state = estimate_articulated_discrete_gaussian_wrench(
                data['part_to_world'][:, 0], data['joint_values'][:, 0], bodies,
                data['part_to_root'][0, 1], np.arange(len(data['frame_index']) + 1),
                fps=float(stage['fps']), sigma_s=float(stage['gaussian_sigma_pos_s']),
                sigma_rot_s=float(stage['gaussian_sigma_rot_s']),
                radius=int(stage['gaussian_radius_pos_frames']),
                radius_rot=int(stage['gaussian_radius_rot_frames']),
                gravity_world=tuple(data['gravity_world'][0]), device='cpu')
            if not np.allclose(state['generalized_contact_wrench'],
                               stage['original_generalized_contact_wrench'], rtol=1e-5, atol=1e-5):
                raise ValueError('HOI4D recovered state disagrees with saved solver dynamics')
            rows = len(canonical['solver_transition_index'])
            points = np.asarray(stage['points_world'], np.float64)[:rows]
            present = (np.isfinite(points).all(axis=(2, 3))
                       & (np.linalg.norm(np.ptp(points, axis=2), axis=2) >= 0.01))
            mass = sum(body.mass for body in bodies)
            local = data['part_to_root'][0]
            coms = np.stack([local[j, :3, :3] @ body.com_part + local[j, :3, 3]
                             for j, body in enumerate(bodies)])
            com = sum(body.mass * coms[j] for j, body in enumerate(bodies)) / mass
            inertia = np.zeros((3, 3))
            for j, body in enumerate(bodies):
                delta = coms[j] - com
                rotation = local[j, :3, :3]
                inertia += (rotation @ body.inertia_com_part @ rotation.T
                            + body.mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)))
            return build_articulated_episode_labels(
                episode_index=int(record['episode_index']), canonical_view=str(record['selected_view']),
                solver_fps=float(canonical['solver_fps']), source_fps=float(stage['fps']),
                dt_s=float(np.asarray(canonical['solver_dt_s']).reshape(-1)[0]),
                source_frame_left=np.asarray(stage['frame_index'], np.int64)[:rows],
                hand_present=present, contact=np.asarray(canonical['hand_contact_mask'], bool),
                signed_distance=np.asarray(canonical['signed_distance_vertices_m'], np.float32),
                force_valid=np.asarray(canonical['force_label_valid'], bool),
                vertex_force=np.asarray(canonical['hand_vertex_force_world'], np.float32),
                generalized_contact_wrench=np.asarray(canonical['generalized_contact_wrench'], np.float64),
                generalized_environment_wrench=np.asarray(canonical['environment_wrench_world'], np.float64),
                joint_angle=state['smoothed_joint_angle'][:rows],
                joint_velocity=state['generalized_velocity'][:rows, 6],
                base_position=state['smoothed_base_position_world'][:rows],
                base_rotation=state['smoothed_base_rotation_world'][:rows],
                base_velocity=state['generalized_velocity'][:rows, :6],
                gravity_world=data['gravity_world'][0], object_name=str(stage['object_instance_id']),
                physics_body=_ArticulatedPhysics(mass=mass, com_body=com, inertia_body=inertia))


def load_articulated_episode(record: dict) -> dict:
    """Decode one floating_base_revolute7d episode (arctic / hoi4d subsets).

    Source layout differs from rigid: no backend npz — presence and root-link
    state live in stage1, the QP payload in the canonical labels npz (7-dim
    generalized wrench).  The solver timeline maps to source frames as
    ``start + row`` (the v4 mapping fallback; arctic is 25 fps on both the
    solver and the video side).
    """
    episode_index = int(record["episode_index"])
    label_path = Path(str(record["label"]))
    with np.load(label_path, allow_pickle=False) as canonical:
        stage1_path = Path(str(canonical["source_stage1_npz"]))
        with np.load(stage1_path, allow_pickle=False) as stage1:
            rows = len(canonical["solver_transition_index"])
            present = np.zeros((rows, 2), bool)
            for side, key in ((0, "left_vertices_world"),
                              (1, "right_vertices_world")):
                if key in stage1.files:
                    vertices = np.asarray(stage1[key], np.float64)[:rows]
                    extent = np.ptp(vertices, axis=1)
                    present[:, side] = (
                        np.isfinite(vertices).all(axis=(1, 2))
                        & (np.linalg.norm(extent, axis=1) >= 0.01))
            masses = np.asarray([
                float(stage1[f"part_{part}_mass"])
                for part in range(2)])
            coms = np.stack([
                np.asarray(stage1[f"part_{part}_com"], np.float64)
                for part in range(2)])
            part_mass_total = float(masses.sum())
            com_body = (masses[:, None] * coms).sum(axis=0) / part_mass_total
            physics_body = _ArticulatedPhysics(
                mass=part_mass_total, com_body=com_body,
                inertia_body=np.zeros((3, 3)))
            return build_articulated_episode_labels(
                episode_index=episode_index,
                canonical_view=str(record.get("selected_view")),
                solver_fps=float(canonical["solver_fps"]),
                source_fps=float(canonical["solver_fps"]),
                dt_s=float(np.asarray(
                    canonical["solver_dt_s"]).reshape(-1)[0]),
                source_frame_left=int(canonical["start"])
                + np.arange(rows, dtype=np.int64),
                hand_present=present,
                contact=np.asarray(canonical["hand_contact_mask"], bool),
                signed_distance=np.asarray(
                    canonical["signed_distance_vertices_m"], np.float32),
                force_valid=np.asarray(
                    canonical["force_label_valid"], bool),
                vertex_force=np.asarray(
                    canonical["hand_vertex_force_world"], np.float32),
                generalized_contact_wrench=np.asarray(
                    canonical["generalized_contact_wrench"], np.float64),
                generalized_environment_wrench=np.asarray(
                    canonical["environment_wrench_world"], np.float64),
                joint_angle=np.asarray(stage1["joint_angle"], np.float64),
                joint_velocity=np.asarray(
                    stage1["generalized_velocity"], np.float64)[:, 6],
                base_position=np.asarray(
                    stage1["smoothed_base_position_world"],
                    np.float64)[:rows],
                base_rotation=np.asarray(
                    stage1["smoothed_base_rotation_world"],
                    np.float64)[:rows],
                base_velocity=np.asarray(
                    stage1["generalized_velocity"], np.float64)[:, :6],
                gravity_world=np.asarray((0.0, 0.0, -9.80665)),
                object_name=str(stage1["object_name"])
                if "object_name" in stage1.files else "articulated_object",
                physics_body=physics_body,
            )


class _ArticulatedPhysics:
    """Mass/COM summary of the articulated body (parts share 1 kg total)."""

    def __init__(self, mass: float, com_body: np.ndarray,
                 inertia_body: np.ndarray):
        self.mass = float(mass)
        self.com_body = tuple(np.asarray(com_body, np.float64))
        self.inertia_body = np.asarray(inertia_body, np.float64)


def _static_object_rotations(dataset_root: Path, episode_index: int,
                             view_id: str, source_frames: np.ndarray,
                             per_row_ids: list) -> np.ndarray:
    """World-frame orientation per static row from the objects table.

    The static gravity QP never stores orientation (only the COM solve
    reference), but the published ``object_rotation`` drives the review
    overlay and any state consumer — identity there floats the mesh off
    the real object.  Rows whose (frame, instance) pose is absent keep the
    identity placeholder: those are exactly the label-less rows.
    """
    import pyarrow.parquet as pq

    rows = len(per_row_ids)
    rotation = np.zeros((rows, 3, 3), np.float64)
    rotation[:, 0, 0] = rotation[:, 1, 1] = rotation[:, 2, 2] = 1.0
    observation = next(
        (r for r in pq.read_table(
            dataset_root / "meta/observations.parquet",
            filters=[("episode_index", "=", episode_index)]).to_pylist()
         if str(r["view_id"]) == view_id), None)
    if observation is None or not observation.get("objects_available"):
        return rotation
    objects = pq.ParquetFile(
        dataset_root / str(observation["objects_path"])).read_row_group(
        int(observation["objects_row_group_index"])).slice(
        int(observation["objects_row_offset"]),
        int(observation["objects_num_rows"])).to_pylist()
    c2w_of = {
        int(r["frame_index"]): np.asarray(
            r["camera.c2w"], np.float64).reshape(4, 4)
        for r in pq.ParquetFile(
            dataset_root / str(observation["hmr_path"])).read_row_group(
            int(observation["hmr_row_group_index"])).slice(
            int(observation["hmr_row_offset"]),
            int(observation["hmr_num_rows"])).to_pylist()}
    pose_of = {(int(r["frame_index"]), str(r["object_instance_id"])): r
               for r in objects}
    for local, (frame, instance) in enumerate(
            zip(np.asarray(source_frames, np.int64).tolist(), per_row_ids)):
        row = pose_of.get((int(frame), str(instance)))
        c2w = c2w_of.get(int(frame))
        if row is None or c2w is None or not instance:
            continue
        pose_cam = np.asarray(row["pose_cam"], np.float64).reshape(4, 4)
        block = pose_cam[:3, :3]
        # Some static roots store normalized meshes and carry the geometry
        # scale inside the pose rotation block (obman: R = s * Q, s per
        # category; mow was denormalized 2026-09-13).  The label contract
        # wants a rotation: strip the uniform scale.
        scale = float(np.linalg.norm(block, axis=0).mean())
        if abs(scale - 1.0) > 1e-3:
            block = block / scale
        rotation[local] = c2w[:3, :3] @ block
    return rotation


def load_static_episode(record: dict, dataset_root: Path) -> dict:
    """Decode one independent-frame static episode (obman / mow).

    Source: the per-frame static gravity QP backend's canonical NPZ.  Every
    row is a separate physical problem; obman switches the visible object
    per row, mow has one scene object per episode.  Object COM is the
    per-row solve reference; gravity is recovered from the stored static
    wrench (wrench[:, :3] = -g_dir * 9.80665).
    """
    import pyarrow.parquet as pq

    episode_index = int(record["episode_index"])
    with np.load(Path(str(record["label"])), allow_pickle=False) as static:
        rows = len(np.asarray(static["force_label_valid"]))
        distance = np.asarray(static["signed_distance_vertices_m"],
                              np.float32).reshape(rows, 2, 778)
        contact = np.asarray(static["hand_contact_mask"], bool).reshape(
            rows, 2, 778)
        # The v4-era -1.0 m sentinel marks absent hands and is finite; real
        # geometry never reaches -0.9 m at these scales.
        real = np.isfinite(distance) & (distance > -0.9)
        present = real.any(axis=2) | contact.any(axis=2)
        distance = np.where(real, distance, np.float32(0.25))
        wrench = np.asarray(static["contact_wrench_world"], np.float64)
        # The static solver uses unit reference mass. OBMan's independent
        # samples have different gravity directions, including within one
        # episode; preserve every row instead of broadcasting the first.
        gravity = -wrench[:, :3]
        if "object_instance_id" in static.files:
            per_row_ids = [str(v) for v in static["object_instance_id"]]
        else:
            instances = pq.read_table(
                dataset_root / "objects/instances.parquet",
                filters=[("episode_index", "=", episode_index)]).to_pylist()
            if len(instances) != 1:
                raise SystemExit(
                    f"episode {episode_index}: expected one scene instance "
                    f"for a static episode, got {len(instances)}")
            per_row_ids = [str(instances[0]["object_instance_id"])] * rows
        # Rows without a visible object carry the backend's "" id: they have
        # no instance metadata, so they stay outside the object table and
        # point at slot -1 instead of a phantom physics entry.
        object_ids = [value for value in dict.fromkeys(per_row_ids) if value]
        slot = np.full(rows, -1, np.int32)
        for local, value in enumerate(per_row_ids):
            if value:
                slot[local] = object_ids.index(value)
        fps = float(json.loads(
            (dataset_root / "meta/info.json").read_text()).get("fps", 30.0))
        if "frame_index" in static.files:
            source_frames = np.asarray(static["frame_index"], np.int64)
        else:
            source_frames = int(static["start"]) + np.arange(rows,
                                                             dtype=np.int64)
        return build_static_episode_labels(
            episode_index=episode_index,
            canonical_view=str(static["view_id"]),
            fps=fps,
            source_frame_left=source_frames,
            hand_present=present,
            contact=contact,
            signed_distance=distance,
            force_valid=np.asarray(static["force_label_valid"], bool),
            vertex_force=np.asarray(static["vertex_force_world"], np.float32),
            contact_wrench=wrench,
            object_com=np.asarray(static["x_com_world"], np.float64),
            object_ids=object_ids,
            object_index_per_row=slot,
            gravity_world=gravity,
            physics_of=lambda oid: target_physics(
                dataset_root, episode_index, oid, gravity[0]),
            object_rotation=_static_object_rotations(
                dataset_root, episode_index, str(static["view_id"]),
                source_frames, per_row_ids),
        )


def rigid_wrench_identity_guard(
    raw, rows: int,
    force_tolerance_n: float = 0.05,
    torque_tolerance_nm: float = 0.05,
) -> np.ndarray:
    """Rows where the QP's fitted wrench actually reproduced the target.

    The stored contact/environment wrench columns carry the inverse-dynamics
    target unconditionally.  Frames inside the Gaussian dynamics mask's
    exclusion zone (pose-gap neighborhoods) never run the hand QP — forces
    and environment stay zero there while ``force_label_valid`` remains True
    ("neutral" semantics) — so the stored vertex forces cannot reproduce
    contact − environment and the two supervision heads would contradict.
    The fitted-vs-target residual flags exactly that class: verified QP rows
    sit at ~1e-9, direct-environment rows fit by construction, broken rows
    miss by the full gravity-scale target.  The residual is a wrench, so its
    force part is reference-independent; the torque tolerance is wide enough
    to absorb any transport-reference offset at these residuals.
    """
    if "fitted_contact_wrench_world" not in raw.files:
        return np.ones(rows, bool)
    target = np.asarray(raw["contact_wrench_world"], np.float64)[:rows]
    fitted = np.asarray(raw["fitted_contact_wrench_world"], np.float64)[:rows]
    ok = (np.linalg.norm(fitted[:, :3] - target[:, :3], axis=1)
          <= force_tolerance_n)
    ok &= (np.linalg.norm(fitted[:, 3:] - target[:, 3:], axis=1)
           <= torque_tolerance_nm)
    return ok


def load_episode(record: dict, dataset_root: Path) -> dict:
    episode_index = int(record["episode_index"])
    label_path = Path(str(record["label"]))
    with np.load(label_path, allow_pickle=False) as canonical:
        topology = str(canonical["topology"]) if "topology" in canonical.files \
            else str(record.get("topology", ""))
        if topology == "floating_base_revolute7d":
            canonical.close()
            if dataset_root.name == 'hoi4d_lerobot':
                return load_hoi4d_articulated_labels(record, dataset_root)
            return load_articulated_episode(record)
        if topology == "static":
            canonical.close()
            return load_static_episode(record, dataset_root)
        backend_path = (Path(str(record["output_dir"])) / "backend"
                        / label_path.name)
        with np.load(backend_path, allow_pickle=False) as raw:
            if topology != "rigid6d":
                raise SystemExit(
                    f"episode {episode_index}: topology {topology!r} is not "
                    "supported by the labels-v5 writer yet")
            rows = len(canonical["solver_transition_index"])
            object_ids = [str(value) for value in raw["object_instance_ids"]]
            force = np.asarray(canonical["hand_vertex_force_world"], np.float32)
            if len(object_ids) == 1:
                per_object = force[:, None]
            else:
                owner = np.asarray(
                    canonical["force_candidate_object_owner"], np.int16)
                per_object = np.stack([
                    np.where((owner == index)[..., None], force, 0.0)
                    for index in range(len(object_ids))], axis=1)
            normalization = json.loads(str(canonical["normalization_json"]))
            gravity = np.asarray(raw["object_gravity_world"],
                                 np.float64).reshape(-1, 3)[0]
            return build_episode_labels(
                episode_index=episode_index,
                canonical_view=str(record.get("selected_view")
                                   or raw["view_id"]),
                solver_fps=float(canonical["solver_fps"]),
                source_fps=float(canonical["source_fps"]),
                dt_s=float(np.asarray(canonical["solver_dt_s"]).reshape(-1)[0]),
                source_frame_left=np.asarray(
                    canonical["source_frame_left"], np.int64),
                source_alpha=np.asarray(
                    canonical["source_interpolation_alpha"], np.float64),
                hand_present=np.asarray(raw["hand_present"], bool)[:rows],
                contact=np.asarray(canonical["hand_contact_mask"], bool),
                signed_distance=np.asarray(
                    canonical["signed_distance_vertices_m"], np.float32),
                force_valid=np.asarray(
                    canonical["force_label_valid"], bool)[:rows] & (
                    rigid_wrench_identity_guard(raw, rows)),
                vertex_force_per_object=per_object,
                object_contact_wrench=np.asarray(
                    raw["object_contact_wrench_world"], np.float64)[:rows],
                object_environment_wrench=np.asarray(
                    raw["object_environment_wrench_world"], np.float64)[:rows],
                object_position=np.asarray(
                    raw["object_state_position_world"], np.float64)[:rows],
                object_rotation=np.asarray(
                    raw["object_state_rotation_world"], np.float64)[:rows],
                object_velocity=np.asarray(
                    raw["object_velocity_before_world"], np.float64)[:rows],
                object_gravity_world=gravity,
                object_ids=object_ids,
                reference_mass_kg=float(
                    normalization.get("reference_total_mass_kg", 1.0)),
                physics_of=lambda oid: target_physics(
                    dataset_root, episode_index, oid, gravity),
                target_of=target_of(dataset_root, episode_index, object_ids),
                topology=TOPOLOGY_RIGID,
            )


def install(staging_root: Path, live_root: Path) -> None:
    """Prepare all entries on the target filesystem before deleting live data.

    Source staging may be on another mount. Entry renames are atomic, but the
    multi-entry commit is not. Old derived labels are not retained; preserved
    staging remains the recovery source if the commit is interrupted.
    """
    from core.geometry.mano_layer import file_sha256
    from core.data.processed_compat import is_processed
    staging_root, live_root = staging_root.resolve(), live_root.resolve()
    if staging_root == live_root:
        raise ValueError('label staging must be separate from the live dataset')
    if is_processed(live_root):
        from core.tactile_labels.prepared_publish import install_prepared_labels
        print(f'[labels-v5] activated {install_prepared_labels(staging_root, live_root)}', flush=True)
        return
    for entry in INSTALL_ENTRIES:
        if not (staging_root/entry).is_dir():
            raise ValueError(f'missing staged label stream: {entry}')
    for relative in INSTALL_META_FILES:
        if not (staging_root/relative).is_file():
            raise ValueError(f'missing staged label metadata: {relative}')
    with tempfile.TemporaryDirectory(prefix='.labels-v5-install-', dir=live_root) as temporary:
        prepared = Path(temporary)
        for entry in INSTALL_ENTRIES:
            shutil.copytree(staging_root/entry, prepared/entry)
        (prepared/'meta').mkdir()
        for relative in INSTALL_META_FILES:
            shutil.copy2(staging_root/relative, prepared/relative)
        paths: list[str] = list(INSTALL_META_FILES)
        for entry in INSTALL_ENTRIES:
            paths.extend(str(path.relative_to(staging_root))
                         for path in (staging_root/entry).rglob('*') if path.is_file())
        for relative in paths:
            if file_sha256(staging_root/relative) != file_sha256(prepared/relative):
                raise ValueError(f'label installation copy mismatch: {relative}')
        # All expensive/cross-device work must finish before the first removal.
        for entry in INSTALL_ENTRIES:
            destination = live_root/entry
            if destination.exists():
                shutil.rmtree(destination)
            os.rename(prepared/entry, destination)
        for relative in INSTALL_META_FILES:
            os.replace(prepared/relative, live_root/relative)


def can_reuse_staged(output_root: Path, solver_run: Path, records: dict,
                     episodes: list[int], contract: dict | None) -> bool:
    from core.geometry.mano_layer import contract_from_info
    from core.tactile_labels.decode import load_label_index
    required = [output_root / relative for relative in (*INSTALL_META_FILES, *INSTALL_ENTRIES)]
    if not all(path.exists() for path in required):
        return False
    stats_path = output_root / 'meta/label_stats.json'
    stats = json.loads(stats_path.read_text())
    staged_info = json.loads((output_root / 'meta/info.json').read_text())
    staged_contract = contract_from_info(staged_info)
    same_geometry = (contract is None and staged_contract is None) or (
        contract is not None and staged_contract is not None
        and staged_contract['geometry_revision'] == contract['geometry_revision'])
    input_paths = [Path(records[ep]['label']) for ep in episodes]
    for ep in episodes:
        directory = Path(records[ep]['output_dir'])
        for subdir in ('backend', 'stage1', 'stage2'):
            input_paths.extend((directory / subdir).glob('*.npz'))
    newest_input = max(path.stat().st_mtime_ns for path in input_paths)
    staged_rows = load_label_index(output_root)
    # Old static staging lost per-row gravity/object identity. Rebuild it
    # rather than treating matching solver timestamps as a format upgrade.
    import pyarrow as pa
    import pyarrow.parquet as pq
    force_schemas = {}
    optional_lists = ("generalized_wrench", "joint_state", "static_gravity_world")
    for row in staged_rows:
        for group in row["force_groups"]:
            path = group["path"]
            if path not in force_schemas:
                if not (output_root / path).is_file():
                    return False
                schema = pq.ParquetFile(output_root / path).schema_arrow
                if any(name in schema.names and pa.types.is_fixed_size_list(schema.field(name).type)
                       for name in optional_lists):
                    return False
                force_schemas[path] = schema
            if (row.get("topology") == "static"
                    and not {"static_gravity_world", "static_object_index"}.issubset(force_schemas[path].names)):
                return False
    complete_streams = all((output_root / group['path']).is_file()
                           for row in staged_rows for key in ('tactile_groups', 'force_groups')
                           for group in row[key])
    fresh_geometry = contract is None or staged_info.get('mano_dependent_labels', {}).get('state') == 'fresh'
    return (same_geometry and stats.get('solver_run_id') == solver_run.name
            and complete_streams and fresh_geometry
            and int(stats.get('episodes', -1)) == len(episodes)
            and sum(int(row['n_rows']) for row in staged_rows) == int(stats['n_rows'])
            and all((output_root / 'timeline_data' / f"episode_{ep:06d}.parquet").is_file()
                    for ep in episodes)
            and sorted(int(row['episode_index']) for row in staged_rows) == episodes
            and newest_input <= stats_path.stat().st_mtime_ns)


def main() -> None:
    args = parse_args()
    solver_run = args.solver_run.resolve()
    root_name = (args.dataset if args.dataset.endswith("_lerobot")
                 else f"{args.dataset}_lerobot")
    processed_root = args.processed_root.resolve()
    live_root = processed_root / root_name
    output_root = (args.output_root.resolve() if args.output_root
                   else solver_run / "labels_v5" / root_name)

    records = {int(row["episode_index"]): row
               for row in load_run_results(solver_run, args.dataset)
               if row.get("status") in ACCEPTED_STATUS
               and Path(str(row.get("root", ""))).name == root_name
               and Path(str(row.get("label", ""))).is_file()}
    if args.episodes is not None:
        wanted = set(map(int, args.episodes))
        records = {index: row for index, row in records.items()
                   if index in wanted}
    episodes = sorted(records)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    if not episodes:
        raise SystemExit(f"no accepted episodes for {root_name} in {solver_run}")

    accumulator = LabelStatsAccumulator()
    def materialized():
        for position, episode_index in enumerate(episodes):
            episode = load_episode(records[episode_index], live_root)
            accumulator.add(episode)
            if (position + 1) % 25 == 0 or position + 1 == len(episodes):
                print(f"[labels-v5] {root_name}: {position + 1}/{len(episodes)} episodes", flush=True)
            yield episode

    from core.data import processed_compat
    source_info = processed_compat.read_info(live_root)
    from core.geometry.mano_layer import contract_from_info, file_sha256
    contract = contract_from_info(source_info)
    if contract is not None:
        proof_path = solver_run / "mano_geometry_contract.json"
        if not proof_path.exists():
            raise ValueError("Corrected datasets require a solver MANO contract; old runs cannot clear stale labels")
        proof = json.loads(proof_path.read_text())
        name = root_name.removesuffix("_lerobot")
        record = next((r for r in proof["datasets"] if r["dataset"] == name), None)
        if record is None or record["revision"] != contract["geometry_revision"]:
            raise ValueError("Solver geometry revision differs from installed MANO")
        if any(file_sha256(live_root / p) != h for p,h in record["hmr_sha256"].items()):
            raise ValueError("HMR parameters changed after the solver run began")
        source_info["mano_dependent_labels"] = {"state":"fresh",
            "geometry_revision":contract["geometry_revision"],"solver_run_id":solver_run.name}
    reuse = args.reuse_staged and can_reuse_staged(output_root, solver_run, records, episodes, contract)
    if reuse:
        print(f'[labels-v5] reuse complete staging: {output_root}', flush=True)
    else:
        write_label_streams(output_root, materialized(),
                            {"solver_run_id": solver_run.name}, source_info)
        stats = accumulator.finalize(solver_run.name)
        (output_root / "meta/label_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[labels-v5] wrote {output_root} ({stats['episodes']} episodes, "
              f"{stats['n_rows']} solver rows)")

    if args.install:
        # publish gate: installed hand-crop streams must stay consistent
        # with the staged labels (A1 mapping, video-frame indexing, reader
        # K, bbox centering) — no GPU, sampled episodes (AGENTS.md rule).
        from core.tactile_labels.handcrop_writer import (
            labels_handcrop_consistency, labels_handcrop_geometry_consistency)
        violations = labels_handcrop_consistency(live_root, output_root)
        if contract is not None:
            violations.extend(labels_handcrop_geometry_consistency(live_root, output_root))
        if violations:
            for message in violations:
                print(f"[labels-v5] HANDCROP GATE: {message}", flush=True)
            raise SystemExit(
                "handcrop streams inconsistent with staged labels; "
                "re-run the handcrop installer for this dataset first "
                "(or drop its segments)")
        install(output_root, live_root)
        print(f"[labels-v5] installed into {live_root}")


if __name__ == "__main__":
    main()
