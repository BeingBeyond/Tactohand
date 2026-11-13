"""Solver-row index and supervision spans for tactile and force streams."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.tactile_labels.schema import (
    FLAG_CONTACT_VALID, FLAG_FORCE_VALID, FLAG_PROXIMITY_VALID, TOPOLOGY_STATIC,
)


def read_native_timeline(path: Path, labels: dict, revision: str) -> pa.Table:
    return validate_native_timeline(pq.read_table(path), labels, revision)


def validate_native_timeline(timeline: pa.Table, labels: dict, revision: str) -> pa.Table:
    metadata = timeline.schema.metadata or {}
    expected = {'source_revision': revision}
    if 'episode_index' in timeline.column_names:
        episodes_match = bool(np.all(timeline['episode_index'].to_numpy() == labels['episode_index']))
    else:
        episodes_match = True
        expected.update(episode_index=labels['episode_index'],
                        tactile_sha1=labels['tactile_sha1'], force_sha1=labels['force_sha1'])
    rows = timeline['solver_row'].to_numpy()
    if (not episodes_match
            or any(metadata.get(key.encode()) != str(value).encode() for key, value in expected.items())
            or len(rows) != int(labels['n_rows'])
            or not np.array_equal(rows, np.arange(len(rows)))):
        raise ValueError(f"Timeline does not match episode {labels['episode_index']} at revision {revision}")
    return timeline


def target_spans(timeline: pa.Table):
    """Keep each target contiguous and leave per-row supervision to its masks."""
    source = timeline['source_position'].to_numpy()
    targets = timeline['target_object'].fill_null(-1).to_numpy()
    changes = np.flatnonzero(targets[1:] != targets[:-1]) + 1
    boundaries = np.r_[0, changes, len(targets)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        positions = source[begin:end]
        finite = positions[np.isfinite(positions)]
        if len(finite):
            yield (int(begin), int(end), int(targets[begin]),
                   int(np.ceil(finite.min() - 1e-4)),
                   int(np.floor(finite.max() + 1e-4)) + 1)


def supervision_spans(timeline: pa.Table) -> pa.Table:
    """Compress adjacent rows with the same target and supervision flags."""
    flags = timeline['tactile_flags'].to_numpy()
    contact = (flags & FLAG_CONTACT_VALID) != 0
    proximity = (flags & FLAG_PROXIMITY_VALID) != 0
    force = timeline['force_valid'].to_numpy()
    closing = timeline['closing_solver_row'].is_valid().to_numpy()
    target = timeline['target_object'].fill_null(-1).to_numpy()
    values = np.stack((target, contact, proximity, force, force & closing), axis=1)
    changes = np.flatnonzero(np.any(values[1:] != values[:-1], axis=1)) + 1
    starts = np.r_[0, changes].astype(np.int32) if len(values) else changes.astype(np.int32)
    stops = np.r_[changes, len(values)].astype(np.int32) if len(values) else starts
    return pa.table({
        'start_solver_row': starts,
        'stop_solver_row': stops,
        'target_object': pa.array(target[starts], mask=target[starts] < 0),
        'contact_valid': contact[starts],
        'proximity_valid': proximity[starts],
        'force_valid': force[starts],
        'force_with_closing_state': (force & closing)[starts],
    }).replace_schema_metadata(timeline.schema.metadata)


def native_timeline_from_rows(labels, source, tactile_flags, force_valid, target) -> pa.Table:
    """Build the index from explicit solver-row target assignments."""
    rows = np.arange(int(labels['n_rows']), dtype=np.int32)
    static = labels['topology'] == TOPOLOGY_STATIC
    target = np.asarray(target, dtype=np.int32)
    closing = np.full(len(rows), -1, dtype=np.int32)
    if not static:
        # The next state is only in the same coordinate/COM reference when
        # both stored rows resolve to the same physical target.
        closing[:-1] = np.where((target[:-1] >= 0) & (target[:-1] == target[1:]), rows[1:], -1)
    if static:
        solver_time = pa.nulls(len(rows), pa.float64())
    else:
        fps = float(labels['solver_fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Episode {labels['episode_index']} has invalid solver FPS")
        solver_time = pa.array(rows / fps)
    table = pa.table({
        'solver_row': rows,
        'solver_time_s': solver_time,
        'source_position': pa.array(source, mask=~np.isfinite(source)),
        'tactile_flags': tactile_flags,
        'force_valid': force_valid,
        'target_object': pa.array(target, mask=target < 0),
        'closing_solver_row': pa.array(closing, mask=closing < 0),
    })
    return table.replace_schema_metadata({
        'episode_index': str(labels['episode_index']),
        'topology': str(labels['topology']),
        'source_fps': str(labels['source_fps']),
        'tactile_sha1': str(labels['tactile_sha1']),
        'force_sha1': str(labels['force_sha1']),
    })
