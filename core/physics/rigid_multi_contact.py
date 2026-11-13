"""Geometry-derived contact support for scenes with multiple rigid objects."""
from __future__ import annotations

import numpy as np
import trimesh
import time

try:
    import warp as wp
except ImportError:  # pragma: no cover - CPU-only fallback
    wp = None

from .proximity import contact_gated_force_candidates
from .mesh_inside import raycast_inside


# These caches become effective in persistent batch workers.  Keep the Warp
# input arrays alive together with the mesh because wp.Mesh owns device-side
# BVH state backed by them.
_WARP_MESH_CACHE: dict[tuple[str, str], tuple[object, object, object]] = {}
_TRIMESH_CACHE: dict[str, trimesh.Trimesh] = {}
_OWNER_BOUNDS_CACHE: dict[str, tuple] = {}
_DEGENERATE_BOUNDS_CACHE: dict[str, tuple | None] = {}
_EXACT_BOUNDS_CACHE: dict[str, tuple] = {}


def object_bottom_support_geometry(object_to_world, mesh_vertices,
                                    gravity_direction_world, *, support_band_m=.002):
    """Reference table support geometry shared by generation and recovery."""
    poses, vertices = np.asarray(object_to_world), np.asarray(mesh_vertices)
    gravity = np.asarray(gravity_direction_world, np.float64)
    if gravity.shape == (3,):
        gravity = np.broadcast_to(gravity, (len(poses), 3))
    if (poses.shape != (len(poses), 4, 4) or vertices.ndim != 2
            or vertices.shape[-1] != 3 or gravity.shape != (len(poses), 3)
            or not np.isfinite(gravity).all() or np.any(np.linalg.norm(gravity, axis=-1) <= 0)
            or support_band_m <= 0):
        raise ValueError('invalid object support geometry')
    up = -gravity / np.linalg.norm(gravity, axis=-1, keepdims=True)
    world = np.einsum('vj,tij->tvi', vertices, poses[:, :3, :3]) + poses[:, None, :3, 3]
    height = np.einsum('tvi,ti->tv', world, up)
    bottom = height.min(axis=1)
    points = np.stack([world[t, height[t] <= bottom[t] + support_band_m].mean(0)
                       for t in range(len(poses))])
    return {'support_point_world': points, 'bottom_height_along_up_m': bottom,
            'up_direction_world': up}


def cached_object_trimesh(item: dict) -> trimesh.Trimesh:
    """Return an immutable object mesh reused by persistent workers."""
    vertices, faces = item["mesh"]
    geometry_id = str(item.get("object_geometry_id", ""))
    mesh_key = f"{item.get('dataset', '')}:{geometry_id}" if geometry_id else ""
    mesh = _TRIMESH_CACHE.get(mesh_key) if mesh_key else None
    if mesh is None:
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, np.float64),
                               faces=np.asarray(faces, np.int64), process=False)
        if mesh_key:
            _TRIMESH_CACHE[mesh_key] = mesh
            if len(_TRIMESH_CACHE) > 16:
                _TRIMESH_CACHE.pop(next(iter(_TRIMESH_CACHE)))
    return mesh


def nearest_object_owner(points_world: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Exact ownership using proven distance bounds before triangle queries.

    Nearest-vertex distance is an upper bound on surface distance. Subtracting
    the longest mesh edge gives a lower bound; AABB distance is another lower
    bound. A body is accepted only if its upper bound beats every other lower
    bound. Ambiguous points use the same triangle query as label generation.
    """
    from scipy.spatial import cKDTree
    points = np.asarray(points_world, np.float64)
    if points.ndim != 3 or points.shape[0] != 1:
        raise ValueError('bounded attribution expects one reference frame')
    upper, lower = [], []
    for item in objects:
        mesh = cached_object_trimesh(item)
        key = f"{item.get('dataset', '')}:{item.get('object_geometry_id', id(mesh))}"
        cached = _OWNER_BOUNDS_CACHE.get(key)
        if cached is None:
            triangles = mesh.vertices[mesh.faces]
            edge = max(float(np.linalg.norm(triangles[:, i] - triangles[:, (i + 1) % 3], axis=-1).max())
                       for i in range(3))
            cached = (cKDTree(mesh.vertices), edge, mesh.vertices.min(0), mesh.vertices.max(0))
            _OWNER_BOUNDS_CACHE[key] = cached
            if len(_OWNER_BOUNDS_CACHE) > 16:
                _OWNER_BOUNDS_CACHE.pop(next(iter(_OWNER_BOUNDS_CACHE)))
        tree, edge, lo, hi = cached
        pose = item['object_to_world'][0]
        local = (points[0] - pose[:3, 3]) @ pose[:3, :3]
        distance = tree.query(local, workers=1)[0]
        bbox = np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0), axis=-1)
        upper.append(distance)
        lower.append(np.maximum(bbox, np.maximum(distance - edge, 0)))
    upper, lower = np.stack(upper, 1), np.stack(lower, 1)
    winner = upper.argmin(1)
    other_lower = lower.copy()
    other_lower[np.arange(len(winner)), winner] = np.inf
    proven = upper[np.arange(len(winner)), winner] < other_lower.min(1) - 1e-10
    if not proven.all():
        _, exact = nearest_object_surface(points[:, ~proven], objects, device='cpu', signed=False)
        winner[~proven] = exact[0]
    return winner[None].astype(np.int16)


def contact_query_mesh(item: dict) -> tuple[trimesh.Trimesh, str]:
    """Return the original mesh used by surface and multi-ray queries."""
    return cached_object_trimesh(item), "original_mesh_multiray"


if wp is not None:
    @wp.kernel
    def _warp_surface_query(
        mesh_id: wp.uint64,
        query_points: wp.array(dtype=wp.vec3),
        distance_out: wp.array(dtype=wp.float32),
        closest_out: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        point = query_points[tid]
        face = wp.int32(-1)
        u = wp.float32(0.0)
        v = wp.float32(0.0)
        wp.mesh_query_point_no_sign(mesh_id, point, 1.0e38, face, u, v)
        if face >= 0:
            closest = wp.mesh_eval_position(mesh_id, face, u, v)
            distance_out[tid] = wp.length(closest - point)
            closest_out[tid] = closest
        else:
            distance_out[tid] = 1.0e20
            closest_out[tid] = wp.vec3(0.0, 0.0, 0.0)


def _warp_surface_distance(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, device: str,
    *, cache_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, object, object]:
    """Return unsigned distance, closest point, and the cached Warp mesh."""
    if wp is None:
        raise ImportError("warp is unavailable")
    wp.init()
    key = None if cache_key is None else (str(device), str(cache_key))
    cached = None if key is None else _WARP_MESH_CACHE.get(key)
    if cached is None:
        vertices_wp = wp.array(np.asarray(vertices, np.float32), dtype=wp.vec3, device=device)
        faces_wp = wp.array(np.asarray(faces, np.int32).reshape(-1), dtype=wp.int32, device=device)
        mesh = wp.Mesh(points=vertices_wp, indices=faces_wp, bvh_constructor="sah")
        if key is not None:
            _WARP_MESH_CACHE[key] = (vertices_wp, faces_wp, mesh)
            if len(_WARP_MESH_CACHE) > 16:
                _WARP_MESH_CACHE.pop(next(iter(_WARP_MESH_CACHE)))
    else:
        vertices_wp, faces_wp, mesh = cached
    flat = np.asarray(points, np.float32).reshape(-1, 3)
    points_wp = wp.array(flat, dtype=wp.vec3, device=device)
    distance_wp = wp.zeros(len(flat), dtype=wp.float32, device=device)
    closest_wp = wp.zeros(len(flat), dtype=wp.vec3, device=device)
    wp.launch(
        _warp_surface_query, dim=len(flat),
        inputs=[mesh.id, points_wp],
        outputs=[distance_wp, closest_wp], device=device,
    )
    if str(device).startswith('cuda'):
        wp.synchronize_device(str(device))
    distance = distance_wp.numpy().reshape(points.shape[:-1])
    closest = closest_wp.numpy().reshape(points.shape).astype(np.float64)
    return distance.astype(np.float64), closest, mesh, points_wp


def _exact_owners_batched(points, frames, objects, *, chunk_size=256):
    """Float64 triangle fallback grouped across frames, with proven AABB pruning."""
    from scipy.spatial import cKDTree
    result = np.zeros(len(points), np.int16)
    for begin in range(0, len(points), chunk_size):
        stop = min(begin + chunk_size, len(points))
        selected, frame = points[begin:stop], frames[begin:stop]
        local_points, bounds, meshes = [], [], []
        upper = np.full(len(selected), np.inf)
        for item in objects:
            mesh = cached_object_trimesh(item)
            key = f"{item.get('dataset', '')}:{item.get('object_geometry_id', id(mesh))}"
            cached = _EXACT_BOUNDS_CACHE.get(key)
            if cached is None:
                cached = (cKDTree(mesh.vertices), mesh.vertices.min(0), mesh.vertices.max(0))
                _EXACT_BOUNDS_CACHE[key] = cached
                if len(_EXACT_BOUNDS_CACHE) > 16:
                    _EXACT_BOUNDS_CACHE.pop(next(iter(_EXACT_BOUNDS_CACHE)))
            tree, lo, hi = cached
            poses = np.asarray(item['object_to_world'])[frame]
            local = np.einsum('pi,pij->pj', selected - poses[:, :3, 3], poses[:, :3, :3])
            upper = np.minimum(upper, tree.query(local, workers=1)[0])
            local_points.append(local)
            bounds.append(np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0), axis=-1))
            meshes.append(mesh)
        nearest = np.full(len(selected), np.inf)
        owner = np.zeros(len(selected), np.int16)
        for slot, (mesh, local, lower) in enumerate(zip(meshes, local_points, bounds)):
            # A surface cannot be closer than its AABB. A nearest mesh vertex
            # supplies an upper bound, so this excludes only proven losers.
            at = np.flatnonzero(lower <= upper + 1e-12)
            if not len(at):
                continue
            _, distance, _ = trimesh.proximity.closest_point(mesh, local[at])
            better = distance < nearest[at]
            nearest[at[better]] = distance[better]
            owner[at[better]] = slot
        if not np.isfinite(nearest).all():
            raise ValueError('No finite exact surface owner')
        result[begin:stop] = owner
    return result


def _deterministic_tie_owners(distance, tolerance, preferred=None):
    """Resolve near-ties without a float64 triangle query.

    Ownership within ``tolerance`` is physically indistinguishable (0.1 mm by
    default). Prefer the published target body when it is a candidate, then the
    lowest slot. This keeps the label deterministic without paying for exact
    triangle distances at every object boundary.
    """
    best = distance.min(1, keepdims=True)
    eligible = distance <= best + tolerance
    winner = eligible.argmax(1).astype(np.int16)
    if preferred is not None:
        preferred = np.asarray(preferred, np.int64)
        valid = (preferred >= 0) & (preferred < distance.shape[1])
        if valid.any():
            rows = np.flatnonzero(valid)
            choose = eligible[rows, preferred[rows]]
            winner[rows[choose]] = preferred[rows][choose]
    return winner


def nearest_object_owners_batched(points_world, objects, *, query_mask=None,
                                  device='cpu', tie_tolerance_m=1e-4,
                                  preferred_owner=None, tie_break='slot',
                                  trust_degenerate=False,
                                  chunk_size=65536, return_stats=False):
    """Offline BVH attribution over a complete frame chunk, without ray tests.

    Warp builds one cached BVH per immutable mesh. All points are transformed
    into each mesh's local frame in batches. Near body-distance ties are
    resolved deterministically (target slot, then lowest slot); failed queries
    always use the exact float64 triangle implementation, while degenerate
    meshes do so only when ``trust_degenerate`` is false. The tolerance controls
    numerical ambiguity, not contact/support labels.
    """
    if tie_break not in ('slot', 'exact'):
        raise ValueError("tie_break must be 'slot' or 'exact'")
    started = time.perf_counter()
    points = np.asarray(points_world, np.float64)
    if points.ndim != 3 or points.shape[-1] != 3 or not objects:
        raise ValueError('points must be [frames,vertices,3] with at least one object')
    if chunk_size < 1 or tie_tolerance_m < 0:
        raise ValueError('invalid ownership chunk/tolerance')
    for item in objects:
        if np.shape(item['object_to_world']) != (points.shape[0], 4, 4):
            raise ValueError('object poses must cover the full query timeline')
    mask = np.ones(points.shape[:2], bool) if query_mask is None else np.asarray(query_mask, bool)
    if mask.shape != points.shape[:2] or not np.isfinite(points[mask]).all():
        raise ValueError('invalid ownership query mask or points')
    if preferred_owner is not None:
        preferred_owner = np.asarray(preferred_owner, np.int64)
        if preferred_owner.shape != (points.shape[0],):
            raise ValueError('preferred_owner must have one slot per frame')
    if all('pose_observed' in item for item in objects):
        available_frames = np.stack([np.asarray(item['pose_observed'], bool) for item in objects])
        if available_frames.shape != (len(objects), points.shape[0]):
            raise ValueError('pose availability must cover the full query timeline')
        mask = mask & available_frames.any(0)[:, None]
    owner = np.full(points.shape[:2], -1, np.int16)
    frames, vertices = np.nonzero(mask)
    stats = {'queries': len(frames), 'objects': len(objects), 'exact_fallback_points': 0,
             'tie_points': 0, 'tie_break_points': 0, 'query_failure_points': 0,
             'degenerate_points': 0, 'bvh_seconds': 0., 'fallback_seconds': 0.,
             'device': str(device), 'tie_tolerance_m': tie_tolerance_m,
             'tie_break': tie_break, 'trust_degenerate': bool(trust_degenerate)}
    if len(objects) == 1:
        owner[mask] = 0
    else:
        from scipy.spatial import cKDTree
        for begin in range(0, len(frames), chunk_size):
            f, v = frames[begin:begin + chunk_size], vertices[begin:begin + chunk_size]
            distance = np.empty((len(f), len(objects)), np.float64)
            skipped_lower = np.full(len(f), np.inf)
            query_started = time.perf_counter()
            for slot, item in enumerate(objects):
                available = (np.asarray(item['pose_observed'], bool)[f]
                             if 'pose_observed' in item else np.ones(len(f), bool))
                distance[:, slot] = 1e18
                if not available.any():
                    continue
                pose = np.asarray(item['object_to_world'])[f]
                local = np.einsum('pi,pij->pj', points[f, v] - pose[:, :3, 3], pose[:, :3, :3])
                mesh = cached_object_trimesh(item)
                key = (f"{item.get('dataset', '')}:{item['object_geometry_id']}"
                       if item.get('object_geometry_id') else None)
                bounds_key = key or f'uncached:{id(mesh)}'
                if bounds_key not in _DEGENERATE_BOUNDS_CACHE:
                    triangles = mesh.vertices[mesh.faces]
                    edge0, edge1 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                    edge2 = triangles[:, 2] - triangles[:, 1]
                    denominator = (edge0**2 + edge1**2 + edge2**2).sum(-1)
                    thin = (np.linalg.norm(np.cross(edge0, edge1), axis=-1)
                            <= 2e-6 * denominator)
                    if thin.any():
                        center = triangles[thin].mean(1)
                        radius = float(np.linalg.norm(triangles[thin] - center[:, None], axis=-1).max())
                        _DEGENERATE_BOUNDS_CACHE[bounds_key] = (cKDTree(center), radius)
                    else:
                        _DEGENERATE_BOUNDS_CACHE[bounds_key] = None
                    if len(_DEGENERATE_BOUNDS_CACHE) > 16:
                        _DEGENERATE_BOUNDS_CACHE.pop(next(iter(_DEGENERATE_BOUNDS_CACHE)))
                degenerate = _DEGENERATE_BOUNDS_CACHE[bounds_key]
                if degenerate is not None:
                    tree, radius = degenerate
                    skipped_lower = np.minimum(skipped_lower,
                        np.where(available, np.maximum(tree.query(local, workers=1)[0] - radius, 0), np.inf))
                d, _, _, _ = _warp_surface_distance(local[available], mesh.vertices, mesh.faces,
                                                   str(device), cache_key=key)
                distance[available, slot] = d
            stats['bvh_seconds'] += time.perf_counter() - query_started
            winner = distance.argmin(1)
            nearest = np.partition(distance, 1, axis=1)[:, :2]
            tie = nearest[:, 1] - nearest[:, 0] <= tie_tolerance_m
            failure = (~np.isfinite(distance) | (distance > 1e19)).any(1)
            degenerate = skipped_lower <= distance.min(1) + tie_tolerance_m
            exact = failure | (degenerate & ~bool(trust_degenerate))
            stats['tie_points'] += int(tie.sum())
            stats['query_failure_points'] += int(failure.sum())
            stats['degenerate_points'] += int(degenerate.sum())
            if tie_break == 'exact':
                exact |= tie
            elif tie.any():
                preferred = None if preferred_owner is None else preferred_owner[f]
                winner[tie] = _deterministic_tie_owners(
                    distance[tie], tie_tolerance_m,
                    None if preferred is None else preferred[tie])
                stats['tie_break_points'] += int((tie & ~exact).sum())
            if exact.any():
                fallback_started = time.perf_counter()
                winner[exact] = _exact_owners_batched(
                    points[f[exact], v[exact]], f[exact], objects)
                stats['fallback_seconds'] += time.perf_counter() - fallback_started
                stats['exact_fallback_points'] += int(exact.sum())
            owner[f, v] = winner
    stats['seconds'] = time.perf_counter() - started
    return (owner, stats) if return_stats else owner


def nearest_object_surface(
    points_world: np.ndarray, objects: list[dict], *, device: str = "cuda:0",
    return_closest: bool = False,
    signed: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true nearest triangle-surface distance and object ownership.

    A legacy tactile table may have been authored against only one selected
    object.  Multi-object force allocation must instead evaluate the union of
    all object surfaces on the canonical timeline.
    """
    points = np.asarray(points_world, np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points_world must have shape [T,V,3]")
    if not objects:
        raise ValueError("at least one object is required")
    nearest = np.full(points.shape[:-1], np.inf, np.float64)
    inside_union = np.zeros(points.shape[:-1], dtype=bool)
    owner = np.full(points.shape[:-1], -1, np.int16)
    nearest_closest = (np.full(points.shape, np.nan, np.float64)
                       if return_closest else None)
    meshes = []
    for item in objects:
        vertices, faces = item["mesh"]
        vertices = np.asarray(vertices, np.float64)
        faces = np.asarray(faces, np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("every object mesh must contain triangle faces")
        query_mesh, _status = contact_query_mesh(item)
        meshes.append(query_mesh)
    use_warp = wp is not None and str(device).startswith("cuda") and wp.is_cuda_available()
    frame_chunk = 256 if use_warp else 16
    for object_index, (item, mesh) in enumerate(zip(objects, meshes)):
        poses = np.asarray(item["object_to_world"], np.float64)
        if poses.shape != (points.shape[0], 4, 4):
            raise ValueError("every object pose track must cover points_world")
        for begin in range(0, points.shape[0], frame_chunk):
            end = min(begin + frame_chunk, points.shape[0])
            local = np.einsum(
                "tvi,tij->tvj",
                points[begin:end] - poses[begin:end, None, :3, 3],
                poses[begin:end, :3, :3],
            )
            if use_warp:
                local_distance, local_closest, warp_mesh, warp_points = _warp_surface_distance(
                    local, mesh.vertices, mesh.faces, str(device),
                    cache_key=(f"{item.get('dataset', '')}:{item.get('object_geometry_id')}"
                               if item.get("object_geometry_id") else None))
                inside = (raycast_inside(
                    local, mesh.vertices, mesh.faces, device=str(device),
                    warp_mesh=warp_mesh, warp_points=warp_points) if signed
                    else np.zeros(local.shape[:-1], bool))
            else:
                local_closest, local_distance, _ = trimesh.proximity.closest_point(
                    mesh, local.reshape(-1, 3),
                )
                inside = (raycast_inside(local, mesh.vertices, mesh.faces, device="cpu").reshape(-1)
                          if signed else np.zeros(local.size // 3, bool))
                local_distance = local_distance.reshape(end - begin, points.shape[1])
                local_closest = local_closest.reshape(end - begin, points.shape[1], 3)
                inside = inside.reshape(end - begin, points.shape[1])
            inside_union[begin:end] |= inside
            better = local_distance < nearest[begin:end]
            nearest[begin:end][better] = local_distance[better]
            owner[begin:end][better] = object_index
            if nearest_closest is not None and np.any(better):
                closest_world = np.einsum(
                    "tvj,tij->tvi", local_closest, poses[begin:end, :3, :3],
                ) + poses[begin:end, None, :3, 3]
                nearest_closest[begin:end][better] = closest_world[better]
    nearest = np.where(inside_union, -nearest, nearest)
    if nearest_closest is None:
        return nearest, owner
    return nearest, owner, nearest_closest


def normal_consistent_surface_mask(
    points_world: np.ndarray,
    normals_world: np.ndarray,
    surface_distance_m: np.ndarray,
    closest_surface_world: np.ndarray,
    *,
    threshold_m: float = 0.010,
    safe_distance_m: float = 0.003,
    angle_deg: float = 90.0,
) -> np.ndarray:
    """Threshold exact surface distance and apply the standard hand-normal filter.

    Vertices inside any object, or within ``safe_distance_m`` of its
    surface, are retained without a direction test.  For the remaining
    vertices the hand normal must point towards the closest object surface.
    The closest points come from the same BVH query as proximity, so this adds
    no second mesh query.
    """
    points = np.asarray(points_world, np.float64)
    normals = np.asarray(normals_world, np.float64)
    distance = np.asarray(surface_distance_m, np.float64)
    closest = np.asarray(closest_surface_world, np.float64)
    if points.shape != normals.shape or points.shape != closest.shape:
        raise ValueError("points, normals, and closest surface points must match")
    if distance.shape != points.shape[:-1]:
        raise ValueError("surface distance must match the point leading dimensions")

    point_valid = np.isfinite(points).all(axis=-1)
    valid = point_valid & np.isfinite(normals).all(axis=-1)
    normal_norm = np.linalg.norm(normals, axis=-1)
    valid &= normal_norm > 1.0e-8
    inside = point_valid & np.isfinite(distance) & (distance < 0.0)
    contact = inside | (valid & np.isfinite(distance)
                        & (distance <= float(threshold_m)))
    directional = contact & ~inside & (distance > float(safe_distance_m))
    if np.any(directional):
        direction = closest[directional] - points[directional]
        direction_norm = np.linalg.norm(direction, axis=-1)
        well_defined = direction_norm > 1.0e-12
        dot = np.full(len(direction), -np.inf, np.float64)
        dot[well_defined] = np.einsum(
            "ij,ij->i",
            normals[directional][well_defined] /
            normal_norm[directional][well_defined, None],
            direction[well_defined] / direction_norm[well_defined, None],
        )
        keep = well_defined & (dot >= float(np.cos(np.deg2rad(angle_deg))))
        contact[directional] = keep
    return contact


def generate_surface_contact_geometry(
    points_world: np.ndarray,
    normals_world: np.ndarray,
    objects: list[dict],
    *,
    device: str = "cuda:0",
    contact_threshold_m: float = 0.010,
    normal_safe_distance_m: float = 0.003,
    normal_angle_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate exact union proximity, normal-filtered contact, and ownership."""
    distance, owner, closest = nearest_object_surface(
        points_world, objects, device=device, return_closest=True)
    contact = normal_consistent_surface_mask(
        points_world, normals_world, distance, closest,
        threshold_m=contact_threshold_m,
        safe_distance_m=normal_safe_distance_m,
        angle_deg=normal_angle_deg,
    )
    contact_owner = np.where(contact, owner, -1).astype(np.int16)
    return distance.astype(np.float32), contact, owner, contact_owner


def gate_force_candidates_by_contact(
    candidate: np.ndarray, contact: np.ndarray, *, hand_vertex_count: int = 778,
) -> np.ndarray:
    """Enable the relaxed force neighborhood only for contacting hands.

    Contact is the activation condition.  The larger candidate threshold is a
    support set used after activation, never an alternative contact detector.
    Inputs are for one object, so gating is per frame, hand, and object.
    """
    return contact_gated_force_candidates(
        candidate, contact, hand_vertex_count=hand_vertex_count,
    )
