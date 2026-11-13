"""Recover explicit rigid-body reference states from prepared source tracks.

No force allocation is inferred here. Recovery is accepted only if the selected
target reproduces the already published smoothed state/velocity contract.
"""
from pathlib import Path

import numpy as np
import torch

from core.data import processed_compat as contract
from core.physics.backends import Physics
from core.physics.math3d import matrix_to_quat, rotation_geodesic
from core.physics.solver_timeline import (
    make_solver_timeline, interpolate_poses, interpolate_pose_samples,
    nearest_source, gaussian_observed_transition_mask)
from core.physics.trajectory_policy import rigid_gaussian_sigma_s, rigid_gaussian_sigma_rot_s
from core.physics.trajectory_wrench import (
    estimate_rigid_discrete_gaussian_wrench, gaussian_radius_for_sigma)
from core.tactile_labels.trajectory_physics import solver_gravity_world


def recover_body_states(root: Path, labels: dict, source_positions: np.ndarray,
                        canonical_c2w: np.ndarray, *, rotation_sigma_s: float | None = None) -> dict:
    """Replay the shared source interpolation/smoothing, without re-solving QP."""
    episode = int(labels['episode_index'])
    objects = contract.object_rows(root, episode).to_pylist()
    objects = [r for r in objects if r['view_id'] == labels['canonical_view']]
    frame_count = len(canonical_c2w)
    timeline = make_solver_timeline(frame_count, float(labels['source_fps']),
                                    solver_fps=float(labels['solver_fps']))
    state_count = len(timeline.solver_timestamp_s)
    indices = np.rint(np.asarray(source_positions) * labels['solver_fps'] / labels['source_fps']).astype(int)
    if np.any(indices < 0) or np.any(indices >= state_count - 1):
        raise ValueError('published solver provenance falls outside reconstructed state grid')
    source_grid = timeline.source_left_index + timeline.interpolation_alpha * (
        timeline.source_right_index - timeline.source_left_index)
    gravity = solver_gravity_world(root, episode, source_grid, labels['gravity_world'])
    positions, rotations, velocities, contacts, poses = [], [], [], [], []
    observed_rows, dynamics_rows = [], []
    sigma_pos = rigid_gaussian_sigma_s(root.name)
    sigma_rot = (rigid_gaussian_sigma_rot_s(root.name) if rotation_sigma_s is None
                 else rotation_sigma_s)
    radius = max(gaussian_radius_for_sigma(float(labels['solver_fps']), sigma_pos),
                 gaussian_radius_for_sigma(float(labels['solver_fps']), sigma_rot))
    for slot, object_id in enumerate(labels['object_ids']):
        rows = sorted((r for r in objects if r['object_instance_id'] == object_id),
                      key=lambda r: r['frame_index'])
        if len(rows) < 2:
            raise ValueError(f'{object_id}: no reconstructable source pose track')
        # Each identity is joined as a string; category/int object labels are
        # never used to select geometry or a physical body.
        frame = np.asarray([r['frame_index'] for r in rows], int)
        unique, at = np.unique(frame, return_index=True)
        observed = nearest_source(np.isin(np.arange(frame_count), unique), timeline)
        dynamics_valid = gaussian_observed_transition_mask(observed, radius)
        object_to_camera = np.asarray([rows[i]['pose_cam'] for i in at], np.float64).reshape(-1, 4, 4)
        world = canonical_c2w[unique] @ object_to_camera
        source_pose = (world if np.array_equal(unique, np.arange(frame_count)) else
                       interpolate_pose_samples(unique, world, np.arange(frame_count)))
        pose = interpolate_poses(source_pose, timeline)
        com = np.asarray(labels['object_com_body'][slot], np.float64)
        position = pose[:, :3, 3] + np.einsum('tij,j->ti', pose[:, :3, :3], com)
        physics = Physics(float(labels['object_mass_kg'][slot]), tuple(com),
                          np.asarray(labels['object_inertia_body'][slot]).reshape(3, 3),
                          tuple(gravity[0]))
        states = estimate_rigid_discrete_gaussian_wrench(position,
            matrix_to_quat(torch.as_tensor(pose[:, :3, :3])).numpy(), physics,
            np.arange(state_count), fps=float(labels['solver_fps']),
            sigma_s=sigma_pos, sigma_rot_s=sigma_rot, gravity_world=gravity[:-1])
        incoming = np.vstack((states['generalized_velocity'], states['next_generalized_velocity'][-1]))
        positions.append(states['smoothed_position_world'][indices])
        rotations.append(states['smoothed_rotation_world'][indices])
        velocities.append(incoming[indices])
        contacts.append(states['contact_wrench_world'][indices])
        poses.append(pose[indices])
        observed_rows.append(observed[indices])
        dynamics_rows.append(dynamics_valid[indices])
    return {'position': np.stack(positions, 1), 'rotation': np.stack(rotations, 1),
            'velocity': np.stack(velocities, 1), 'required_contact': np.stack(contacts, 1),
            'raw_pose': np.stack(poses, 1), 'gravity': gravity[indices],
            'pose_observed': np.stack(observed_rows, 1),
            'dynamics_valid': np.stack(dynamics_rows, 1)}


def target_state_error(recovered: dict, target_slot: np.ndarray, position: np.ndarray,
                       rotation: np.ndarray, velocity: np.ndarray, valid: np.ndarray) -> dict:
    index = np.arange(len(target_slot))
    return {
        'position_max_m': float(np.linalg.norm(
            recovered['position'][index, target_slot] - position, axis=-1)[valid].max()),
        'rotation_max_rad': float(rotation_geodesic(
            torch.as_tensor(recovered['rotation'][index, target_slot][valid]),
            torch.as_tensor(rotation[valid], dtype=torch.float64)).max()),
        'velocity_max': float(np.abs(recovered['velocity'][index, target_slot] - velocity)[valid].max()),
        'linear_velocity_max': float(np.abs(recovered['velocity'][index, target_slot, :3]
                                             - velocity[:, :3])[valid].max()),
        'rotation_frobenius_max': float(np.linalg.norm(
            recovered['rotation'][index, target_slot] - rotation, axis=(-1, -2))[valid].max()),
    }


def recover_verified_body_states(root: Path, labels: dict, source_positions: np.ndarray,
                                 slots: np.ndarray, canonical_c2w: np.ndarray,
                                 published_force: dict) -> dict:
    """Identify a historical generation bandwidth using stored states, not loss.

    Only fixed historical generator settings are tried. Unknown transforms or
    bandwidths are rejected instead of silently fabricating reference dynamics.
    """
    valid = (published_force['flags'] & 1) > 0
    if not valid.any():
        raise ValueError('no force-valid reference states')
    candidates = list(dict.fromkeys([rigid_gaussian_sigma_rot_s(root.name), .15, .075]))
    best = None
    for sigma in candidates:
        result = recover_body_states(root, labels, source_positions, canonical_c2w,
                                      rotation_sigma_s=sigma)
        error = target_state_error(result, slots, published_force['object_position'],
            published_force['object_rotation'].reshape(-1, 3, 3),
            published_force['object_velocity'], valid)
        best = error
        if (error['position_max_m'] < 1e-4 and error['rotation_frobenius_max'] < 1e-4
                and error['velocity_max'] < 1e-4):
            result.update(rotation_sigma_s=sigma, target_state_error=error, target_slots=slots)
            return result
    raise ValueError(f'cannot reproduce published state contract: {best}')


def transported_sum(wrenches: np.ndarray, positions: np.ndarray,
                    reference: np.ndarray) -> np.ndarray:
    forces = wrenches[..., :3]
    torque = wrenches[..., 3:] + np.cross(positions - reference[:, None], forces)
    return np.concatenate((forces.sum(1), torque.sum(1)), axis=-1)


def recover_body_payload(root: Path, labels: dict, source_positions: np.ndarray,
                         canonical_c2w: np.ndarray,
                         force: dict, vertices: np.ndarray,
                         signed_distance: np.ndarray, present: np.ndarray, *, device='cpu',
                         trust_degenerate=True, progress=None,
                         target_slots: np.ndarray) -> dict:
    """Geometry-derived attribution with explicit, verified environment recovery.

    This is a reconstruction of missing auxiliary reference labels, not an
    independent force measurement. The published union forces are unchanged.
    """
    from core.geometry.object_assets import instance_geometry_ids, load_geometry
    from core.physics.rigid_multi_contact import nearest_object_owners_batched
    if (target_slots.shape != (len(source_positions),)
            or np.any(target_slots < 0) or np.any(target_slots >= len(labels['object_ids']))):
        raise ValueError('Invalid target slots for rigid reference recovery')
    result = recover_verified_body_states(root, labels, source_positions,
        target_slots, canonical_c2w, force)
    count, bodies = result['position'].shape[:2]
    index = np.arange(count)
    reference = result['position'][index, result['target_slots']]
    required_union = transported_sum(result['required_contact'], result['position'], reference)
    union_error = required_union - force['contact_wrench']
    union_ok = ((np.linalg.norm(union_error[:, :3], axis=-1) < .05)
                & (np.linalg.norm(union_error[:, 3:], axis=-1) < .05))
    dynamics = result['dynamics_valid']
    force_valid = (force['flags'] & 1) > 0
    if progress is not None:
        progress(f'[states] verified; neutral_body_rows={int((~dynamics & force_valid[:, None]).sum())}')
    field = force['vertex_force_world'].reshape(count, -1, 3).astype(np.float64)
    # Before any expensive mesh work, check the reference-independent force
    # balance. Unsupported bodies were neutral in the historical solver: they
    # contributed neither hand force nor environment force, while the stored
    # contact target still included their inverse-dynamics wrench.
    expected_environment_force = np.where(dynamics[..., None],
        result['required_contact'][..., :3], 0).sum(1) - field.sum(1)
    force_balance_error = expected_environment_force - force['environment_wrench'][:, :3]
    bad_force_balance = force_valid & (np.linalg.norm(force_balance_error, axis=-1) >= .05)
    if bad_force_balance.any():
        raise ValueError(f'{int(bad_force_balance.sum())} force-valid rows fail '
                         'observed-body force balance before ownership queries')
    mapping = instance_geometry_ids(root)
    meshes = [load_geometry(str(root.resolve()), mapping[oid]) for oid in labels['object_ids']]
    points = vertices.reshape(count, -1, 3)
    # Attribute the full predicted field on valid reference rows, including
    # zero-label vertices: do not change rollout semantics by using force
    # magnitude/contact labels to decide which predictions enter physics.
    support = (np.repeat(present, 778, axis=1)
               & (force_valid & dynamics.any(1))[:, None])
    objects = []
    for slot, mesh in enumerate(meshes):
        pose = result['raw_pose'][:, slot].copy()
        # Exactly the generator's unavailable-pose convention. Keep observed
        # poses in invalid kernel neighborhoods, but exclude missing poses.
        pose[~result['pose_observed'][:, slot], :3, 3] = 1.0e6
        objects.append({'mesh': mesh, 'object_to_world': pose,
                        'pose_observed': result['pose_observed'][:, slot],
                        'dataset': root.name,
                        'object_geometry_id': mapping[labels['object_ids'][slot]]})
    if progress is not None:
        progress(f'[ownership] device={device} points={int(support.sum())} bodies={bodies}')
    owner, query_stats = nearest_object_owners_batched(points, objects,
        query_mask=support, device=device, preferred_owner=result['target_slots'],
        trust_degenerate=trust_degenerate, return_stats=True)
    if progress is not None:
        progress(f'[ownership] completed: {query_stats}')
    hand = np.zeros((count, bodies, 6), np.float64)
    unsupported_force = np.zeros(count, bool)
    for slot in range(bodies):
        selected = field * (owner == slot)[..., None]
        unsupported_force |= (~dynamics[:, slot]
                              & (np.abs(selected).max(axis=(1, 2)) > 1e-7))
        hand[:, slot, :3] = selected.sum(1)
        hand[:, slot, 3:] = np.cross(points - result['position'][:, slot, None], selected).sum(1)
    environment = np.where(dynamics[..., None], result['required_contact'] - hand, 0.)
    environment_union = transported_sum(environment, result['position'], reference)
    environment_error = environment_union - force['environment_wrench']
    union_ok &= ((np.linalg.norm(environment_error[:, :3], axis=-1) < .05)
                 & (np.linalg.norm(environment_error[:, 3:], axis=-1) < .05))
    uncovered = ((owner < 0) & (np.linalg.norm(field, axis=-1) > 1e-7)).any(-1)
    valid = dynamics & force_valid[:, None]
    valid &= (union_ok & ~uncovered & ~unsupported_force)[:, None]
    result.update(owner=owner.reshape(count, 2, 778), environment=environment,
                  recovery_row_verified=union_ok & ~uncovered & ~unsupported_force,
                  valid=valid, hand=hand,
                  recovery_report={'rotation_sigma_s': result['rotation_sigma_s'],
                      'ownership_query': query_stats,
                      'ownership_policy': ('trust_bvh_degenerate'
                          if trust_degenerate else 'exact_degenerate'),
                      'reference_semantics': 'recovered external/residual wrench, not independently measured support',
                      'target_state_error': result['target_state_error'],
                      'union_contact_max': float(np.abs(union_error).max()),
                      'union_environment_max': float(np.abs(environment_error).max()),
                      'uncovered_force_rows': int(uncovered.sum()),
                      'unsupported_force_rows': int((unsupported_force & force_valid).sum()),
                      'force_valid_neutral_body_rows': int((~dynamics & force_valid[:, None]).sum()),
                      'valid_body_rows': int(valid.sum()), 'body_rows': int(valid.size)})
    return result


def unrecovered_force_rows(payload: dict, force_valid: np.ndarray) -> np.ndarray:
    """Missing supported-body labels or a failed scene-level consistency check."""
    force_valid = np.asarray(force_valid, bool)
    expected = force_valid[:, None] & payload['dynamics_valid']
    return ((expected & ~payload['valid']).any(axis=1)
            | (force_valid & ~payload['recovery_row_verified']))
