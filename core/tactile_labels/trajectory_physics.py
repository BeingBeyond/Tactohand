"""Solver-timeline physics inputs recovered from the published source contract."""
from pathlib import Path

import numpy as np

from core.data import processed_compat as contract


def solver_gravity_world(root: Path, episode: int, source_positions: np.ndarray,
                         fallback: np.ndarray) -> np.ndarray:
    """Match generation: normalize source directions, then interpolate vectors.

    Interpolation uses exact fractional source frames, not token/video indices.
    Independent static samples use their separate per-row physics contract.
    """
    positions = np.asarray(source_positions, np.float64)
    reader = contract.reader_for(root)
    if not reader.has_modality('gravity'):
        return np.broadcast_to(np.asarray(fallback, np.float64), positions.shape + (3,)).copy()
    table = reader.read_episode('gravity', int(episode), 'shared',
                                columns=['frame_index', 'gravity_dir_world'])
    if not table.num_rows:
        raise ValueError(f'{root.name}/{episode}: declared gravity stream is empty')
    frame = table['frame_index'].to_numpy().astype(np.float64)
    directions = np.asarray(table['gravity_dir_world'].to_pylist(), np.float64)
    order = np.argsort(frame)
    frame, directions = frame[order], directions[order]
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    if (not np.isfinite(directions).all() or not np.isfinite(positions).all()
            or np.any(norms <= 0) or np.any(np.diff(frame) <= 0)):
        raise ValueError(f'{root.name}/{episode}: invalid gravity track')
    directions /= norms
    return np.stack([np.interp(positions, frame, directions[:, axis])
                     for axis in range(3)], axis=-1) * 9.80665
