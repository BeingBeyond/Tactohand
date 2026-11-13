"""Per-sample tensor layout for single-body and verified multi-body references."""
import numpy as np
import torch

from core.physics.backends import Physics


BODY_AXIS = {
    'rigid_body_position_30hz': 1, 'rigid_body_rotation_30hz': 1,
    'rigid_body_velocity_30hz': 1, 'rigid_body_environment_30hz': 1,
    'rigid_body_gravity_30hz': 1, 'rigid_body_valid_30hz': 1, 'rigid_body_mask': 0,
}


def attach_single_body(result, *, verified=True):
    length = result['position_30hz'].shape[0] - 1
    physics = result['physics_30hz']
    gravity = result.get('gravity_camera_30hz')
    if gravity is None:
        gravity = torch.as_tensor(physics.gravity_world).expand(length, 3).clone()
    result.update(
        rigid_body_position_30hz=result['position_30hz'].unsqueeze(1),
        rigid_body_rotation_30hz=result['rotation_30hz'].unsqueeze(1),
        rigid_body_velocity_30hz=result['velocity_30hz'].unsqueeze(1),
        rigid_body_environment_30hz=result['environment_wrench_camera_30hz'].reshape(length, 1, 6),
        rigid_body_gravity_30hz=gravity.unsqueeze(1),
        rigid_body_valid_30hz=result['force_valid_30hz'].bool().unsqueeze(1) & verified,
        rigid_body_mask=torch.tensor([verified]),
        rigid_body_physics_30hz=[physics],
        vertex_object_index_30hz=torch.zeros(result['vertex_force_camera_30hz'].shape[:-1], dtype=torch.int16),
        rigid_body_reference_verified=torch.tensor(verified),
    )
    return result


def attach_multi_body(result, body, labels, rows, boundary_rows, rotate, reference, left, available):
    total = result['real_mask'].shape[0]
    count = len(labels['object_ids'])
    sup = np.clip(np.arange(total) - left, 0, available - 1)
    # The caller already chooses the consumer's exact 30 Hz solver rows.
    interval = np.asarray(rows).reshape(available, 3)[sup].reshape(-1)
    states = np.clip(np.arange(total * 3 + 1) - left * 3, 0, len(boundary_rows) - 1)
    state_rows = np.asarray(boundary_rows)[states]
    position = (body['position'][state_rows] - reference[:3, 3]) @ rotate
    rotation = np.einsum('ij,tnjk->tnik', rotate.T, body['rotation'][state_rows])
    velocity = body['velocity'][state_rows].copy()
    velocity[..., :3] = velocity[..., :3] @ rotate
    velocity[..., 3:] = velocity[..., 3:] @ rotate
    environment = body['environment'][interval].copy()
    environment[..., :3] = environment[..., :3] @ rotate
    environment[..., 3:] = environment[..., 3:] @ rotate
    gravity = np.broadcast_to(body['gravity'][interval, None] @ rotate,
                               (total * 3, count, 3)).copy()
    physics = [Physics(float(labels['object_mass_kg'][slot]),
        tuple(np.asarray(labels['object_com_body'][slot])),
        np.asarray(labels['object_inertia_body'][slot]).reshape(3, 3),
        tuple(gravity[0, slot]), dt=result['physics_30hz'].dt,
        substeps=result['physics_30hz'].substeps) for slot in range(count)]
    owner = body['owner'][interval].reshape(total, 3, 2, 778)
    valid = body['valid'][interval].copy()
    valid &= result['real_mask_30hz'].numpy()[:, None]
    result.update(
        rigid_body_position_30hz=torch.from_numpy(position),
        rigid_body_rotation_30hz=torch.from_numpy(rotation),
        rigid_body_velocity_30hz=torch.from_numpy(velocity),
        rigid_body_environment_30hz=torch.from_numpy(environment),
        rigid_body_gravity_30hz=torch.from_numpy(gravity),
        rigid_body_valid_30hz=torch.from_numpy(valid),
        rigid_body_mask=torch.ones(count, dtype=torch.bool),
        rigid_body_physics_30hz=physics,
        vertex_object_index_30hz=torch.from_numpy(owner.copy()),
        rigid_body_reference_verified=torch.tensor(True),
    )
    result['rollout_eligible'] = torch.tensor(bool(valid.any()))
    return result


def collate_body_field(key, values):
    axis = BODY_AXIS[key]
    count = max(value.shape[axis] for value in values)
    if all(value.shape[axis] == count for value in values):
        return torch.stack(values)
    padded = []
    for value in values:
        shape = list(value.shape)
        shape[axis] = count
        output = value.new_zeros(shape)
        if key == 'rigid_body_rotation_30hz':
            output[...] = torch.eye(3, dtype=value.dtype)
        index = [slice(None)] * value.ndim
        index[axis] = slice(0, value.shape[axis])
        output[tuple(index)] = value
        padded.append(output)
    return torch.stack(padded)
