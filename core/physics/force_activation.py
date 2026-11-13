"""Stable scene-level contact ownership and force-activation patches."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def _adjacency_from_faces(face_bytes: bytes, vertex_count: int) -> tuple[tuple[int, ...], ...]:
    faces = np.frombuffer(face_bytes, dtype=np.int64).reshape(-1, 3)
    neighbors = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        for left, right in ((a, b), (b, c), (c, a)):
            neighbors[int(left)].add(int(right))
            neighbors[int(right)].add(int(left))
    return tuple(tuple(sorted(row)) for row in neighbors)


def _components(mask: np.ndarray, adjacency: tuple[tuple[int, ...], ...]) -> list[np.ndarray]:
    remaining = set(np.flatnonzero(mask).tolist())
    output = []
    while remaining:
        seed = remaining.pop(); stack = [seed]; values = [seed]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor in remaining:
                    remaining.remove(neighbor); stack.append(neighbor); values.append(neighbor)
        output.append(np.asarray(values, dtype=np.int64))
    return output


def coherent_force_activation(
    contact: np.ndarray,
    owner: np.ndarray,
    signed_distance_m: np.ndarray,
    mano_faces: np.ndarray,
    object_count: int,
    *,
    hand_vertex_count: int = 778,
    minimum_patch_vertices: int = 5,
    deep_patch_vertices: int = 3,
    deep_distance_m: float = 0.001,
    enter_frames: int = 2,
    exit_frames: int = 4,
) -> dict[str, np.ndarray]:
    """Stabilize owner per connected patch and gate force without altering contact."""
    contact = np.asarray(contact, bool)
    owner = np.asarray(owner, np.int16)
    distance = np.asarray(signed_distance_m, np.float64)
    if contact.shape != owner.shape or contact.shape != distance.shape:
        raise ValueError("contact, owner, and distance must share [T,H*V] shape")
    if contact.shape[1] != 2 * hand_vertex_count or object_count < 1:
        raise ValueError("expected two MANO hands and at least one object")
    if min(minimum_patch_vertices, deep_patch_vertices, enter_frames, exit_frames) < 1:
        raise ValueError("patch and hysteresis thresholds must be positive")
    faces = np.ascontiguousarray(mano_faces, dtype=np.int64)
    adjacency = _adjacency_from_faces(faces.tobytes(), hand_vertex_count)
    stabilized = owner.copy()
    qualified = np.zeros((len(contact), object_count, 2), dtype=bool)
    patch_size = np.zeros((len(contact), object_count, 2), dtype=np.int16)

    for frame in range(len(contact)):
        for hand in range(2):
            begin = hand * hand_vertex_count; end = begin + hand_vertex_count
            local_contact = contact[frame, begin:end]
            for component in _components(local_contact, adjacency):
                values = stabilized[frame, begin + component]
                values = values[values >= 0]
                if not len(values):
                    continue
                modal = int(np.bincount(values, minlength=object_count).argmax())
                stabilized[frame, begin + component] = modal
            for object_id in range(object_count):
                mask = local_contact & (stabilized[frame, begin:end] == object_id)
                components = _components(mask, adjacency)
                largest = max((len(value) for value in components), default=0)
                patch_size[frame, object_id, hand] = largest
                deep = any(
                    len(value) >= deep_patch_vertices
                    and np.nanmin(distance[frame, begin + value]) <= deep_distance_m
                    for value in components
                )
                qualified[frame, object_id, hand] = largest >= minimum_patch_vertices or deep

    active = np.zeros_like(qualified)
    state = np.zeros((object_count, 2), dtype=bool)
    enter_count = np.zeros((object_count, 2), dtype=np.int16)
    exit_count = np.zeros((object_count, 2), dtype=np.int16)
    for frame in range(len(contact)):
        for object_id in range(object_count):
            for hand in range(2):
                begin = hand * hand_vertex_count; end = begin + hand_vertex_count
                any_contact = bool(np.any(
                    contact[frame, begin:end]
                    & (stabilized[frame, begin:end] == object_id)))
                if qualified[frame, object_id, hand]:
                    enter_count[object_id, hand] += 1; exit_count[object_id, hand] = 0
                    if enter_count[object_id, hand] >= enter_frames:
                        state[object_id, hand] = True
                else:
                    enter_count[object_id, hand] = 0; exit_count[object_id, hand] += 1
                    if exit_count[object_id, hand] >= exit_frames:
                        state[object_id, hand] = False
                # Contact remains a necessary condition even during exit hysteresis.
                active[frame, object_id, hand] = state[object_id, hand] and any_contact
    return {"stabilized_owner": stabilized, "qualified_patch": qualified,
            "force_active": active, "largest_patch_vertices": patch_size}
