"""Small torch-only SO(3)/quaternion helpers; quaternions use xyzw ordering."""
from __future__ import annotations

import torch


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), dim=-1)


def rotation_vector_to_quat(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Convert a world-frame rotation vector to an ``xyzw`` quaternion.

    ``torch.sinc`` keeps the scale well behaved at zero, unlike a direct
    ``sin(theta / 2) / theta`` evaluation.
    """
    angle = rotation_vector.norm(dim=-1, keepdim=True)
    xyz_scale = 0.5 * torch.sinc(angle / (2.0 * torch.pi))
    return torch.cat((rotation_vector * xyz_scale, torch.cos(0.5 * angle)), dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return torch.stack((
        1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
        2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Stable enough for trajectory input; returns xyzw."""
    # The four candidates avoid trace-only instability near 180 degrees.
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)) / 2
    qx = qx.copysign(R[..., 2, 1] - R[..., 1, 2])
    qy = qy.copysign(R[..., 0, 2] - R[..., 2, 0])
    qz = qz.copysign(R[..., 1, 0] - R[..., 0, 1])
    return normalize_quat(torch.stack((qx, qy, qz, qw), dim=-1))


def so3_log(R: torch.Tensor) -> torch.Tensor:
    q = matrix_to_quat(R)
    q = torch.where(q[..., 3:4] < 0, -q, q)
    v, w = q[..., :3], q[..., 3:4]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, w.clamp_min(1.0e-12))
    return torch.where(s > 1.0e-7, v * (angle / s), 2 * v)


def rotation_geodesic(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    return so3_log(R_a.transpose(-1, -2) @ R_b).norm(dim=-1)
