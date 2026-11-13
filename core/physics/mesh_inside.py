"""Fast, repair-free inside queries on original triangle meshes."""
from __future__ import annotations

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None


RAY_DIRECTIONS = np.asarray([
    [1.0, 0.371, 0.173], [-0.293, 1.0, 0.419], [0.227, -0.331, 1.0],
    [-1.0, -0.217, 0.463], [0.389, -1.0, -0.257], [-0.179, 0.443, -1.0],
    [0.577, 0.707, -0.408],
], dtype=np.float32)
RAY_DIRECTIONS /= np.linalg.norm(RAY_DIRECTIONS, axis=1, keepdims=True)
_WARP_DIRECTION_CACHE: dict[str, object] = {}


if wp is not None:
    @wp.kernel
    def _ray_parity_votes(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        directions: wp.array(dtype=wp.vec3),
        ray_epsilon: wp.float32,
        max_hits: wp.int32,
        votes: wp.array(dtype=wp.int32),
    ):
        point_index, direction_index = wp.tid()
        direction = directions[direction_index]
        point = points[point_index]
        seed = wp.float32((point_index * 17 + direction_index * 13) % 31 - 15)
        jitter = ray_epsilon * seed / wp.float32(31.0)
        origin = point + wp.vec3(jitter, -jitter * wp.float32(0.73),
                                 jitter * wp.float32(0.41))
        hit_count = wp.int32(0)
        for hit_index in range(64):
            if hit_index >= max_hits:
                break
            query = wp.mesh_query_ray(mesh_id, origin, direction, wp.float32(1.0e20))
            if not query.result:
                break
            hit_count += 1
            origin = origin + direction * (query.t + ray_epsilon)
        votes[point_index * directions.shape[0] + direction_index] = hit_count & 1


def _inside_cpu(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Chunked NumPy fallback matching the GPU ray-vote contract."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    triangles = np.asarray(vertices, np.float64)[np.asarray(faces, np.int64)]
    if len(points) == 0 or len(triangles) == 0:
        return np.zeros(len(points), dtype=bool)
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    extent = np.ptp(np.asarray(vertices, np.float64), axis=0)
    epsilon = max(float(np.linalg.norm(extent)) * 1.0e-7, 1.0e-9)
    votes = np.zeros(len(points), np.int16)
    point_chunk, face_chunk = 256, 2048
    for direction_index, direction in enumerate(RAY_DIRECTIONS.astype(np.float64)):
        hit_count = np.zeros(len(points), np.int32)
        for point_start in range(0, len(points), point_chunk):
            point_stop = min(point_start + point_chunk, len(points))
            query = points[point_start:point_stop].copy()
            seeds = ((np.arange(point_start, point_stop) * 17
                      + direction_index * 13) % 31 - 15).astype(np.float64)
            jitter = epsilon * seeds / 31.0
            query += np.stack([jitter, -0.73 * jitter, 0.41 * jitter], axis=1)
            count = np.zeros(len(query), np.int32)
            for face_start in range(0, len(triangles), face_chunk):
                face_stop = min(face_start + face_chunk, len(triangles))
                v0 = triangles[face_start:face_stop, 0]
                e1 = edge1[face_start:face_stop]
                e2 = edge2[face_start:face_stop]
                h = np.cross(direction[None], e2)
                determinant = np.einsum("ij,ij->i", e1, h)
                parallel = np.abs(determinant) < 1.0e-10
                inverse = np.zeros_like(determinant)
                inverse[~parallel] = 1.0 / determinant[~parallel]
                delta = query[:, None] - v0[None]
                u = np.einsum("qfi,fi->qf", delta, h) * inverse[None]
                cross = np.cross(delta, e1[None])
                v = np.einsum("qfi,i->qf", cross, direction) * inverse[None]
                distance = np.einsum("qfi,fi->qf", cross, e2) * inverse[None]
                hit = ((~parallel)[None] & (u >= 0.0) & (v >= 0.0)
                       & (u + v <= 1.0) & (distance > epsilon))
                count += hit.sum(axis=1, dtype=np.int32)
            hit_count[point_start:point_stop] = count
        votes += hit_count & 1
    return votes > (len(RAY_DIRECTIONS) // 2)


def raycast_inside(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    device: str = "cuda:0",
    warp_mesh: object | None = None,
    warp_points: object | None = None,
) -> np.ndarray:
    """Classify points by majority parity over the original mesh surface."""
    shape = np.asarray(points).shape[:-1]
    flat = np.asarray(points, np.float32).reshape(-1, 3)
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32).reshape(-1, 3)
    if len(flat) == 0 or len(faces) == 0:
        return np.zeros(shape, dtype=bool)
    use_warp = (wp is not None and warp_mesh is not None
                and str(device).startswith("cuda") and wp.is_cuda_available())
    if not use_warp:
        return _inside_cpu(flat, vertices, faces).reshape(shape)

    extent = np.ptp(vertices, axis=0)
    epsilon = max(float(np.linalg.norm(extent)) * 1.0e-6, 1.0e-8)
    points_wp = (warp_points if warp_points is not None else
                 wp.array(flat, dtype=wp.vec3, device=device))
    directions_wp = _WARP_DIRECTION_CACHE.get(str(device))
    if directions_wp is None:
        directions_wp = wp.array(RAY_DIRECTIONS, dtype=wp.vec3, device=device)
        _WARP_DIRECTION_CACHE[str(device)] = directions_wp
    votes_wp = wp.zeros(len(flat) * len(RAY_DIRECTIONS), dtype=wp.int32, device=device)
    wp.launch(
        _ray_parity_votes, dim=(len(flat), len(RAY_DIRECTIONS)),
        inputs=[warp_mesh.id, points_wp, directions_wp, epsilon, 64],
        outputs=[votes_wp], device=device,
    )
    votes = votes_wp.numpy().reshape(len(flat), len(RAY_DIRECTIONS)).sum(axis=1)
    return (votes > (len(RAY_DIRECTIONS) // 2)).reshape(shape)
