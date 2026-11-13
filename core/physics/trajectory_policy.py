"""Shared fixed smoothing policy for generation and physical label recovery."""
from .trajectory_wrench import GAUSSIAN_SIGMA_S

SHOW3D_GAUSSIAN_SIGMA_S = 0.033
GAUSSIAN_SIGMA_ROT_S = 0.15


def rigid_gaussian_sigma_s(dataset: str) -> float:
    return SHOW3D_GAUSSIAN_SIGMA_S if dataset in {'show3d', 'show3d_lerobot'} else GAUSSIAN_SIGMA_S


def rigid_gaussian_sigma_rot_s(dataset: str) -> float:
    return GAUSSIAN_SIGMA_ROT_S if dataset in {'dexycb', 'dexycb_lerobot'} else rigid_gaussian_sigma_s(dataset)
