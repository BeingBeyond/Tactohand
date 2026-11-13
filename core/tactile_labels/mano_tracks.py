"""Shared MANO vertex tracks on the labels solver timeline (spec §6.1).

The single underlying geometry implementation is
``core.data.mano_geometry.load_mano_world_geometry`` (source-frame world
vertices, zeroed on absent frames).  This module is the one entry point every
label consumer (training reader, converter validation, staging deep checks)
must use to obtain vertices aligned to solver rows; re-implementing the
completion/interpolation elsewhere is forbidden by the spec.

Lever semantics (spec decision #7): vertices are linearly interpolated from
source frames onto solver rows — the same construction the solver used — so
lever arms are exactly co-timed with force rows.
"""
from __future__ import annotations
import core.data.processed_compat as processed_contract

from pathlib import Path
import json

import numpy as np


def _interpolate_rows(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Linearly sample ``values`` (S,...) at fractional source positions."""
    left = np.floor(positions).astype(np.int64)
    alpha = (positions - left)[(...,) + (None,) * (values.ndim - 1)]
    right = np.minimum(left + 1, len(values) - 1)
    left = np.clip(left, 0, len(values) - 1)
    return ((1.0 - alpha) * values[left]
            + alpha * values[right]).astype(values.dtype, copy=False)


def solver_vertex_tracks(
    root: Path,
    episode_index: int,
    view_id: str,
    *,
    source_positions: np.ndarray,
    rows: np.ndarray | None = None,
    sides: tuple[str, ...] = ("left", "right"),
    device: str = "cpu",
    with_faces: bool = False,
    observation: dict | None = None,
    hmr_frame=None,
):
    """Return ``{side: (vertices[n_rows,778,3] f32, present[n_rows] bool)}``.

    ``source_positions[r]`` is the exact fractional source-frame position of
    solver row ``r`` (the labels-v5 ``source_position`` column, written from
    the solver's interpolation bracket).  Presence follows the nearest source
    frame; absent rows are exactly zero.  ``with_faces=True`` additionally
    returns the MANO face topology per side (identical across frames).

    ``rows`` (optional) scopes the output AND the MANO forward to the given
    solver-row indices: only the source frames bracketing those rows'
    positions run through the loader — bit-identical to the episode-scoped
    call (row-local interpolation, loader slices clamp at the episode tail),
    at O(window) instead of O(episode) cost.  The episode-wide default stays
    for the training reader, whose sequential access amortizes it through
    the episode cache.
    """
    from core.data.mano_geometry import load_mano_world_geometry

    positions_all = np.asarray(source_positions, np.float64)
    frame_start = frame_stop = None
    if rows is not None:
        positions = positions_all[np.asarray(rows, np.int64)]
        frame_start = max(0, int(np.floor(positions.min())))
        # +2 exclusive: the right interpolation bracket of the last row.
        frame_stop = int(np.floor(positions.max())) + 2
    else:
        positions = positions_all
    # Stored-vertices roots (hi4d family): the loader branch yields zero-pose
    # hands at the origin — skip it and read the stored vertices directly.
    # Flagged per process on the first detected mismatch (see below); callers
    # that also need MANO faces keep the loader path.
    info = processed_contract.read_info(Path(root))
    hmr_cap = info.get('beingh_human', {}).get('capabilities', {}).get('hmr', {})
    native_vertices = hmr_cap.get('mano') is False and hmr_cap.get('mesh_vertices') is True
    if native_vertices or (str(root.name) in _STORED_VERTICES_ROOTS and not with_faces):
        stored = _read_stored_vertices(root, episode_index, view_id, sides,
                                       positions)
        if stored is not None:
            if with_faces:
                if hmr_cap.get('mesh_topology') != 'mano_778':
                    raise ValueError('Native hand vertices require an explicit supported topology')
                from core.geometry.mano_layer import create_mano_layer
                stored['faces'] = {
                    side: create_mano_layer(side=side).th_faces.cpu().numpy().astype(np.int32)
                    for side in sides
                }
            return stored
        if native_vertices:
            raise ValueError(f'{root}: declared native hand vertices could not be read')
    result: dict = {} if not with_faces else {"faces": {}}
    # Interpolation runs slice-relative; the stored-vertices paths below
    # index the FULL hmr table and need absolute positions.
    local_positions = (positions - frame_start
                       if frame_start is not None else positions)
    for side in sides:
        source_vertices, _normals, faces = load_mano_world_geometry(
            Path(root), int(episode_index), str(view_id), side=side,
            device=device, frame_start=frame_start, frame_stop=frame_stop,
            observation=observation, compute_normals=False, hmr_frame=hmr_frame)
        source_present = (
            np.linalg.norm(source_vertices, axis=-1).max(axis=1) > 0)
        vertices = _interpolate_rows(source_vertices, local_positions)
        # Resolve half-frame ties in the original coordinate system, so a
        # different slice start cannot change which source hand is present.
        nearest = np.rint(positions).astype(np.int64)
        if frame_start is not None:
            nearest -= frame_start
        nearest = np.minimum(nearest, len(source_present) - 1)
        present = source_present[nearest]
        vertices[~present] = 0.0
        result[side] = (vertices.astype(np.float32, copy=False), present)
        if with_faces:
            result["faces"][side] = np.asarray(faces, np.int32)
    result = _stored_vertices_fallback(root, episode_index, view_id, sides,
                                       positions, result)
    return result


_FALLBACK_WARNED_ROOTS: set[str] = set()
# Roots whose loader output was once detected degenerate against the stored
# vertices_camera; every later episode of such a root skips the loader.
_STORED_VERTICES_ROOTS: set[str] = set()


def _read_stored_vertices(root: Path, episode_index: int, view_id: str,
                          sides, positions: np.ndarray):
    """Read stored ``{side}.vertices_camera`` mapped to world, or ``None``.

    ``None`` means this episode/view has no usable stored vertices (schema
    without the column, empty rows, or any read failure) — the caller then
    falls back to the plain loader branch.
    """
    try:
        import pyarrow.parquet as pq

        observations = processed_contract.observations(Path(root), int(episode_index))
        row = next(r for r in observations
                   if str(r["view_id"]) == str(view_id))
        table = processed_contract.observation_table(Path(root), row)
        if "left.vertices_camera" not in table.column_names:
            return None
        hmr = table.to_pylist()
        if not hmr or not hmr[0].get("left.vertices_camera"):
            return None
        picked = np.clip(np.rint(np.asarray(positions, np.float64)),
                         0, len(hmr) - 1).astype(np.int64)
        result: dict = {}
        for side in sides:
            world = []
            for i in picked:
                entry = hmr[int(i)]
                verts = np.frombuffer(
                    entry[f"{side}.vertices_camera"], np.float32
                ).reshape(778, 3)
                c2w = np.asarray(
                    entry["camera.c2w"], np.float64).reshape(4, 4)
                world.append(verts @ c2w[:3, :3].T + c2w[:3, 3])
            world = np.stack(world).astype(np.float32)
            present = np.linalg.norm(world, axis=-1).max(axis=1) > 0
            world[~present] = 0.0
            result[side] = (world, present)
        return result
    except Exception:  # noqa: BLE001 — stored path is best-effort
        return None


def _stored_vertices_fallback(root: Path, episode_index: int, view_id: str,
                              sides, positions: np.ndarray,
                              result: dict) -> dict:
    """Replace loader output with stored ``vertices_camera`` on mismatch.

    Some conversions (hi4d family) store the real per-frame fits only as
    ``{side}.vertices_camera`` with identity ``camera.c2w`` and empty MANO
    parameters; the plain loader branch then yields zero-pose hands at the
    origin.  Detect that against the stored vertices and fall back to them,
    mapped to world through each row's own ``camera.c2w``.  Best-effort:
    any failure keeps the loader output.  (This backstop used to live only
    in the review visualizer; training must see the same hands.)  Once a
    root is detected, ``solver_vertex_tracks`` skips the wasted loader call
    for its remaining episodes (``_STORED_VERTICES_ROOTS``).
    """
    stored = _read_stored_vertices(root, episode_index, view_id, sides,
                                   positions)
    if stored is None:
        return result
    replaced = False
    for side in sides:
        loaded = result[side][0]
        if float(np.abs(loaded[0] - stored[side][0][0]).max()) > 0.02:
            result[side] = stored[side]
            replaced = True
    if replaced:
        _STORED_VERTICES_ROOTS.add(str(root.name))
        if str(root.name) not in _FALLBACK_WARNED_ROOTS:
            # Degenerate-loader rescue (hi4d family): report once per
            # dataset root — training decodes thousands of episodes and a
            # per-episode line drowned the logs.
            _FALLBACK_WARNED_ROOTS.add(str(root.name))
            import sys
            print(f"[mano_tracks] {root.name}: stored vertices_camera "
                  f"fallback active (loader output degenerate; first ep "
                  f"{episode_index})", file=sys.stderr, flush=True)
    return result
