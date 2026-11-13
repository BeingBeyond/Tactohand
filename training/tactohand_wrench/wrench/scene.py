"""Flatten verified independent rigid bodies through the shared rollout backend."""
import torch

from core.physics.wrench import aggregate_body_forces


def scene_rollout_frame_mask(batch, device):
    """Return the body slots and transition masks in flattened body order."""
    mask = batch['rigid_body_mask'].to(device).bool()
    valid = batch['rigid_body_valid_30hz'].to(device).bool() & mask[:, None]
    valid &= (batch['real_mask_30hz'].to(device).bool()
              & batch['force_valid_30hz'].to(device).bool())[:, :, None]
    if 'metric_sample_valid' in batch:
        valid &= batch['metric_sample_valid'].to(device).bool()[:, None, None]
    return mask.reshape(-1), valid.transpose(1, 2).flatten(0, 1)


def scene_rollout_prediction_mask(batch, device):
    """Expand the sample's image-aligned predictions over its body slots."""
    mask = batch.get('force_prediction_valid_30hz')
    if mask is None:
        return None
    bodies = batch['rigid_body_mask'].shape[1]
    return mask.to(device).bool()[:, None, :].expand(-1, bodies, -1).flatten(0, 1)


def scene_rollout_inputs(force, batch):
    if 'rigid_body_position_30hz' not in batch:
        return None
    if set(batch.get('dataset_mode', [])) == {'articulated_trajectory'}:
        return None
    b = force.shape[0]
    field = force.flatten(1, 2).flatten(2, 3)
    length = field.shape[1]
    position = batch['rigid_body_position_30hz'].to(field)
    bodies = position.shape[2]
    target = batch['position_30hz'][:, :-1].to(field)
    points = batch['vertex_lever_camera_30hz'].flatten(1, 2).flatten(2, 3).to(field)
    points = points + target[:, :, None]
    owner = batch['vertex_object_index_30hz'].reshape(b, length, -1).to(field.device)
    contact = aggregate_body_forces(
        field, points, position[:, :-1], owner,
        owner_range_validated=batch.get('_rigid_body_owner_range_validated') is True)
    contact = contact + batch['rigid_body_environment_30hz'].to(contact)

    def flatten(value):
        # B,L,O,... -> B*O,L,..., preserving body-major physics ordering.
        return value.transpose(1, 2).reshape((b * bodies, value.shape[1]) + value.shape[3:])

    mask, flat_mask = scene_rollout_frame_mask(batch, field.device)
    physics = []
    for records in batch['rigid_body_physics_30hz']:
        physics.extend(list(records) + [records[0]] * (bodies - len(records)))
    flat_batch = {
        'position_30hz': flatten(position),
        'rotation_30hz': flatten(batch['rigid_body_rotation_30hz'].to(field)),
        'velocity_30hz': flatten(batch['rigid_body_velocity_30hz'].to(field)),
        'gravity_camera_30hz': flatten(batch['rigid_body_gravity_30hz'].to(field)),
        'physics_30hz': physics, 'real_mask_30hz': flat_mask,
        'force_valid_30hz': flat_mask,
        'force_supervision_valid': mask,
        'rollout_eligible': mask,
        '_rigid_body_slots': bodies,
        'dataset_mode': ['rigid_trajectory'] * (b * bodies),
    }
    prediction_mask = scene_rollout_prediction_mask(batch, field.device)
    if prediction_mask is not None:
        flat_batch['force_prediction_valid_30hz'] = prediction_mask
    return flatten(contact), flat_batch
