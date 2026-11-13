"""Explicit articulated-body mass and joint parameters."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NormalizedPartPhysics:
    part_index: int
    name: str
    mass: float
    volume_m3: float
    com_part: np.ndarray
    inertia_com_part: np.ndarray
    joint_type: str
    joint_axis_parent: np.ndarray
    joint_origin_parent: np.ndarray


@dataclass(frozen=True)
class LinkPhysics:
    mass: float
    com_body: np.ndarray
    inertia_body: np.ndarray
    parent: int
    joint_type: str
    joint_axis_parent: np.ndarray
    joint_origin_parent: np.ndarray
