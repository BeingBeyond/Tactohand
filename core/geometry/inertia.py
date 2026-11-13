"""Mesh mass properties, independent of dataset normalization policies."""
from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull, QhullError

def _inertia_from_mesh(verts: np.ndarray, faces: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute solid unit-density mass properties, repairing invalid topology.

    Some OakInk scan meshes contain every triangle twice with opposite winding.
    Their contact surface is still useful, but signed tetrahedron integrals
    cancel to zero.  For mass properties only, use the convex hull whenever
    duplicate faces or a non-physical integral is detected.  The caller keeps
    the original mesh for contact/proximity calculations.
    """
    vertices = np.asarray(verts, np.float64)
    triangles = np.asarray(faces, np.int64)

    def integrate(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v0, v1, v2 = (vertices[surface[:, index]] for index in range(3))
        vol6 = np.einsum("ti,ti->t", v0, np.cross(v1, v2))
        signed = float(vol6.sum())
        if abs(signed) < 1e-12:
            raise ValueError("mesh has zero signed volume")
        # A consistently inward-wound surface is physically equivalent.
        vol6 *= 1.0 if signed > 0 else -1.0
        signed = abs(signed)
        com = ((v0 + v1 + v2) / 4 * vol6[:, None]).sum(0) / signed
        a, b, c = v0 - com, v1 - com, v2 - com
        second = sum(np.einsum("ti,tj->tij", u, u) for u in (a, b, c))
        total = a + b + c
        second += np.einsum("ti,tj->tij", total, total)
        second = (second * (vol6 / 60)[:, None, None]).sum(0) * (mass / (signed / 6))
        inertia = np.eye(3) * np.trace(second) - second
        inertia = (inertia + inertia.T) / 2
        eigenvalues = np.linalg.eigvalsh(inertia)
        if not np.isfinite(com).all() or not np.isfinite(inertia).all() or eigenvalues.min() <= 0:
            raise ValueError("mesh integral produced non-SPD inertia")
        return com.astype(np.float64), inertia.astype(np.float64)

    unoriented = np.sort(triangles, axis=1)
    duplicate_faces = len(np.unique(unoriented, axis=0)) < len(triangles)
    if not duplicate_faces:
        try:
            return integrate(triangles)
        except ValueError:
            pass
    try:
        hull = ConvexHull(vertices)
    except QhullError as error:
        raise ValueError("mesh has no three-dimensional convex hull") from error
    hull_faces = np.asarray(hull.simplices, np.int64).copy()
    center = vertices[hull.vertices].mean(axis=0)
    p0, p1, p2 = (vertices[hull_faces[:, index]] for index in range(3))
    inward = np.einsum("ti,ti->t", np.cross(p1 - p0, p2 - p0),
                       (p0 + p1 + p2) / 3 - center) < 0
    hull_faces[inward] = hull_faces[inward][:, [0, 2, 1]]
    return integrate(hull_faces)


def _volume_com_inertia(vertices: np.ndarray, faces: np.ndarray, mass: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed triangular mesh volume, COM and inertia about COM."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    vol6 = np.einsum("ti,ti->t", v0, np.cross(v1, v2))
    signed6 = float(vol6.sum())
    if abs(signed6) < 1e-12:
        raise ValueError("part mesh has zero signed volume")
    if signed6 < 0:
        vol6 = -vol6; signed6 = -signed6
    volume = signed6 / 6.0
    com = ((v0 + v1 + v2) * vol6[:, None]).sum(0) / (4.0 * signed6)
    a, b, c = v0 - com, v1 - com, v2 - com
    second = sum(np.einsum("ti,tj->tij", u, u) for u in (a, b, c))
    total = a + b + c
    second += np.einsum("ti,tj->tij", total, total)
    second = (second * (vol6 / 60.0)[:, None, None]).sum(0) * (mass / volume)
    inertia = np.eye(3) * np.trace(second) - second
    inertia = (inertia + inertia.T) * 0.5
    if np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("geometry-normalized part inertia is not positive definite")
    return volume, com.astype(np.float64), inertia.astype(np.float64)
