"""Bounded GT-force audits through the model's physical decoding path."""
from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
import torch

from core.geometry.force_frames import force_world_to_local
from training.tactohand_wrench.model import SubstepForceModel
from training.tactohand_wrench.wrench.metrics import rollout_metrics


def oracle_model(cfg):
    """Build only physical decoding; never allocate network/backbone weights."""
    model = SubstepForceModel.__new__(SubstepForceModel)
    torch.nn.Module.__init__(model)
    model.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model.cfg.loss.rollout.implementation = 'deferred'
    model.legacy_substeps3 = model.raw_local_mse = False
    model.force_gate_enabled = bool(cfg.loss.contact_proximity.force_gate)
    model.vertex_force_frame = str(cfg.data.vertex_force_frame)
    model.point_force_clamp_n = cfg.loss.wrench.point_force_clamp_n
    model.rollout_horizon_10hz = 4
    model.rollout_position_scale = float(cfg.loss.rollout.position_scale_m)
    model.rollout_rotation_scale = float(cfg.loss.rollout.rotation_scale_rad)
    model.rollout_position_weight = float(cfg.loss.rollout.position_weight)
    model.rollout_rotation_weight = float(cfg.loss.rollout.rotation_weight)
    model._rollout_metric_horizons = list(cfg.metrics.get('tensorboard_rollout_horizons', [4]))
    return model


@torch.no_grad()
def audit_batch(model, batch):
    """Compare raw labels, oracle head decoding and stored wrenches.

    The oracle is defined after the learned engagement gate. Only currently
    eligible rigid/static batches receive rigid-body rollout diagnostics.
    """
    if not bool(batch['force_supervision_valid'].all()):
        return {'supervision': 'contact_only', 'rollout_status': 'masked: no force supervision'}
    raw = batch['vertex_force_camera_30hz']
    aligned = raw[:, :, 0]
    if model.vertex_force_frame == 'vertex_local':
        axes = batch['vertex_local_yz_camera_30hz'][:, :, 0]
        head = torch.from_numpy(force_world_to_local(
            aligned.numpy(), axes[..., 0, :].numpy(), axes[..., 1, :].numpy()))
    else:
        head = aligned
    force, hand, contact = model._physical_outputs(head / batch['point_force_scale'], batch)
    _, raw_hand, raw_contact = model._aggregate(raw, batch)
    real = batch['real_mask'].bool()
    valid = real & batch['force_valid_30hz'].reshape(*real.shape, -1)[:, :, 0].bool()
    present = batch['hand_present_10hz'].bool()
    conflict = ((aligned.norm(dim=-1) > .05) & ~present.unsqueeze(-1)).any((-1, -2)) & valid
    target = model._clamped_point_target(batch, force)[:, :, 0]
    difference = (force[:, :, 0] - target)[valid]
    row = {'supervision': 'force', 'dataset_mode': list(batch['dataset_mode']),
           'valid_aligned_frames': int(valid.sum()),
           'presence_conflict_frames': int(conflict.sum()),
           'oracle_vertex_max_error_n': float(difference.abs().max()) if difference.numel() else None,
           'raw_hand_wrench_max_delta': float((raw_hand - batch['hand_wrench_camera_30hz']).abs().max()),
           'oracle_hand_wrench_max_delta': float((hand - batch['hand_wrench_camera_30hz']).abs().max()),
           'clamped_vertex_vectors': int((raw.norm(dim=-1) > model.point_force_clamp_n).sum())
               if model.point_force_clamp_n is not None else 0,
           'raw_vertex_force_max_n': float(raw.norm(dim=-1).max())}
    if not bool(batch['rollout_eligible'].all()):
        row['rollout_status'] = 'masked: ineligible dynamics or non-target forces'
        return row
    if set(batch['dataset_mode']) == {'articulated_trajectory'}:
        row['rollout_status'] = 'masked: articulated dynamics'
        return row
    if not bool(model._rollout_window_mask(batch, 12).any()):
        row['rollout_status'] = 'no eligible 0.4-second window'
        return row
    rollout_batch = {'position': batch['position_30hz'], 'rotation': batch['rotation_30hz'],
                     'velocity': batch['velocity_30hz'], 'real_mask': batch['real_mask_30hz'],
                     'physics': batch['physics_30hz']}
    if 'gravity_camera_30hz' in batch:
        rollout_batch['gravity'] = batch['gravity_camera_30hz']
    row['rollout_status'] = 'checked'
    row['loss_valid_windows'] = int(model._rollout_window_mask(batch, 12).sum())
    row['raw_gt_rollout_loss'] = float(model._gt_pose_rollout_loss(
        raw_contact.reshape(1, -1, 6), batch)[0])
    row['valid_force_rollout'] = {}
    for mode, value in [('raw_gt_vertex_force', raw_contact), ('oracle_head_force', contact),
                        ('stored_hand_wrench', batch['hand_wrench_camera_30hz']
                         + batch['environment_wrench_camera_30hz'])]:
        metrics = rollout_metrics(value.reshape(1, -1, 6).double(), rollout_batch,
            [12], ['real_only'], float(model.cfg.metrics.explosion_linear_speed_m_s),
            float(model.cfg.metrics.explosion_angular_speed_rad_s),
            str(model.cfg.metrics.rollout_integrator), int(model.cfg.metrics.implicit_iterations)
        )['12']['real_only']
        row[mode] = {key: None if isinstance(value, float) and not np.isfinite(value)
                     else value for key, value in metrics.items()}
        valid_batch = dict(rollout_batch)
        valid_batch['real_mask'] = (batch['real_mask_30hz'].bool()
                                    & batch['force_valid_30hz'].bool())
        metrics = rollout_metrics(value.reshape(1, -1, 6).double(), valid_batch,
            [12], ['real_only'], float(model.cfg.metrics.explosion_linear_speed_m_s),
            float(model.cfg.metrics.explosion_angular_speed_rad_s),
            str(model.cfg.metrics.rollout_integrator), int(model.cfg.metrics.implicit_iterations)
        )['12']['real_only']
        row['valid_force_rollout'][mode] = {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in metrics.items()}
    return row
