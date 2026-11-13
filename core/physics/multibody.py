"""Multibody inverse dynamics and generalized contact maps."""
from __future__ import annotations
import numpy as np
from .parameters import LinkPhysics
from .backends import Physics
from .math3d import matrix_to_quat
from .trajectory_wrench import estimate_rigid_discrete_gaussian_wrench

def descendants(physics: list[LinkPhysics]) -> list[list[int]]:
    result = []
    for joint in range(len(physics)):
        members = []
        for link in range(len(physics)):
            cursor = link
            while cursor >= 0:
                if cursor == joint:
                    members.append(link); break
                cursor = physics[cursor].parent
        result.append(members)
    return result


def estimate_multibody_generalized_wrench(data: dict) -> dict:
    """Project per-link Newton-Euler inverse dynamics into floating-base q."""
    poses = np.asarray(data["part_to_world"], np.float64)
    T, L = poses.shape[:2]
    link_com = np.empty((T, L, 3)); link_wrench = np.empty((T, L, 6))
    smoothed_pose = poses.copy()
    for link, physics in enumerate(data["physics"]):
        rotation = poses[:, link, :3, :3]
        com = poses[:, link, :3, 3] + np.einsum("tij,j->ti", rotation, physics.com_body)
        rigid = Physics(
            physics.mass, tuple(physics.com_body), physics.inertia_body,
            tuple(data["gravity_world"][0]), dt=1 / data["fps"], substeps=1,
        )
        result = estimate_rigid_discrete_gaussian_wrench(
            com, matrix_to_quat(__import__("torch").as_tensor(rotation)).numpy(), rigid,
            np.arange(T + 1), fps=data["fps"], gravity_world=data["gravity_world"],
        )
        link_com[:, link] = result["smoothed_position_world"][:-1]
        link_wrench[:, link] = result["contact_wrench_world"]
        smoothed_pose[:, link, :3, :3] = result["smoothed_rotation_world"][:-1]
        smoothed_pose[:, link, :3, 3] = link_com[:, link] - np.einsum(
            "tij,j->ti", smoothed_pose[:, link, :3, :3], physics.com_body)
    masses = np.asarray([x.mass for x in data["physics"]])
    system_com = np.einsum("l,tli->ti", masses, link_com) / masses.sum()
    joints = [index for index, item in enumerate(data["physics"]) if item.parent >= 0]
    D = 6 + len(joints)
    target = np.zeros((T, D)); target[:, :3] = link_wrench[:, :, :3].sum(axis=1)
    target[:, 3:6] = (
        link_wrench[:, :, 3:].sum(axis=1)
        + np.cross(link_com - system_com[:, None], link_wrench[:, :, :3]).sum(axis=1)
    )
    axis_world = np.zeros((T, len(joints), 3)); pivot_world = np.zeros_like(axis_world)
    subtree = descendants(data["physics"])
    for local, joint in enumerate(joints):
        item = data["physics"][joint]; parent_pose = smoothed_pose[:, item.parent]
        axis = np.einsum("tij,j->ti", parent_pose[:, :3, :3], item.joint_axis_parent)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        pivot = parent_pose[:, :3, 3] + np.einsum(
            "tij,j->ti", parent_pose[:, :3, :3], item.joint_origin_parent)
        axis_world[:, local] = axis; pivot_world[:, local] = pivot
        members = subtree[joint]
        if item.joint_type == "revolute":
            moment = (link_wrench[:, members, 3:].sum(axis=1)
                      + np.cross(link_com[:, members] - pivot[:, None],
                                 link_wrench[:, members, :3]).sum(axis=1))
            target[:, 6 + local] = np.einsum("ti,ti->t", axis, moment)
        elif item.joint_type == "prismatic":
            force = link_wrench[:, members, :3].sum(axis=1)
            target[:, 6 + local] = np.einsum("ti,ti->t", axis, force)
    return {
        "target_generalized_wrench": target, "system_com_world": system_com,
        "link_com_world": link_com, "link_wrench_world": link_wrench,
        "part_to_world_smooth": smoothed_pose, "joint_part_indices": np.asarray(joints),
        "joint_axis_world": axis_world, "joint_pivot_world": pivot_world,
    }


def contact_generalized_map(
    point: np.ndarray, part: int, system_com: np.ndarray,
    joint_parts: np.ndarray, physics: list[LinkPhysics],
    axes: np.ndarray, pivots: np.ndarray,
) -> np.ndarray:
    D = 6 + len(joint_parts); mapping = np.zeros((D, 3)); mapping[:3] = np.eye(3)
    r = point - system_com
    mapping[3:6] = np.array(((0, -r[2], r[1]), (r[2], 0, -r[0]), (-r[1], r[0], 0)))
    ancestors = set(); cursor = int(part)
    while cursor >= 0:
        ancestors.add(cursor); cursor = physics[cursor].parent
    for local, joint in enumerate(joint_parts):
        if int(joint) not in ancestors:
            continue
        if physics[int(joint)].joint_type == "revolute":
            mapping[6 + local] = np.cross(axes[local], point - pivots[local])
        elif physics[int(joint)].joint_type == "prismatic":
            mapping[6 + local] = axes[local]
    return mapping
