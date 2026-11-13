"""Explicit offline auxiliary-label store; training never reconstructs geometry."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.data import processed_compat as contract

FORMAT = 'rigid_reference_v1'
RECOVERY_REVISION = 'dense_unsigned_bvh_carried_target_v3'
FIELDS = ('position', 'rotation', 'velocity', 'environment', 'gravity', 'owner', 'valid')


def source_key(root: Path, labels: dict, *, manifest_sha256: str | None = None) -> str:
    if manifest_sha256 is None:
        reader = contract.reader_for(root)
        manifest_sha256 = hashlib.sha256(reader.path(reader.manifest_path).read_bytes()).hexdigest()
    source = {key: labels[key] for key in ('episode_index', 'tactile_sha1', 'force_sha1',
        'object_ids', 'object_mass_kg', 'object_com_body', 'object_inertia_body', 'solver_fps', 'source_fps')}
    source['manifest_sha256'] = manifest_sha256
    source['format'] = FORMAT
    source['recovery_revision'] = RECOVERY_REVISION
    # Only the interpolated-pose datasets had unsupported dynamics bodies.
    # Keep valid v3 products for complete-track datasets (e.g. OakInk/TACO).
    if root.name in {'hot3d_lerobot', 'show3d_lerobot'}:
        source['recovery_revision'] += '_observed_dynamics_v4'
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def reference_manifest_path(reference_root, root: Path, episode: int) -> Path:
    return Path(reference_root) / root.name / f'episode_{episode:06d}' / 'manifest.json'


def reference_status(reference_root, root: Path, labels: dict):
    if not reference_root:
        return None
    path = reference_manifest_path(reference_root, root, int(labels['episode_index']))
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('format') != FORMAT or value['source_key'] != source_key(root, labels):
        raise ValueError(f'stale/incompatible rigid references: {path}')
    return value


def write_reference(reference_root, root: Path, labels: dict, payload=None, error=None):
    path = reference_manifest_path(reference_root, root, int(labels['episode_index']))
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {'format': FORMAT, 'source_key': source_key(root, labels),
             'episode': int(labels['episode_index']), 'object_ids': labels['object_ids'],
             'status': 'rejected' if error else 'ready', 'error': error,
             'force_labels_unchanged': True}
    if payload is not None:
        count = len(payload['owner'])
        columns = {'solver_transition': pa.array(np.arange(count), type=pa.int32())}
        shapes = {}
        for key in FIELDS:
            array = np.asarray(payload[key])
            shapes[key] = list(array.shape[1:])
            dtype = pa.bool_() if key == 'valid' else pa.int16() if key == 'owner' else pa.float64()
            columns[key] = pa.FixedSizeListArray.from_arrays(
                pa.array(array.reshape(-1), type=dtype), int(np.prod(array.shape[1:])))
        temporary = path.parent / 'bodies.pending.parquet'
        destination = path.parent / 'bodies.parquet'
        pq.write_table(pa.table(columns), temporary, compression='zstd', row_group_size=256)
        temporary.replace(destination)
        value.update(rows=count, shapes=shapes, payload='bodies.parquet',
                     payload_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                     report=payload['recovery_report'],
                     state_layout='solver-row start states with addressable closing rows')
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
    return value


def read_reference(reference_root, root: Path, labels: dict):
    value = reference_status(reference_root, root, labels)
    if value is None or value['status'] != 'ready':
        return None, value
    path = reference_manifest_path(reference_root, root, int(labels['episode_index']))
    payload = path.parent / value['payload']
    if hashlib.sha256(payload.read_bytes()).hexdigest() != value['payload_sha256']:
        raise ValueError(f'corrupt rigid reference payload: {payload}')
    table = pq.read_table(payload, columns=list(FIELDS))
    arrays = {}
    for key in FIELDS:
        column = table[key].combine_chunks()
        arrays[key] = column.values.to_numpy(zero_copy_only=False).reshape(
            (value['rows'],) + tuple(value['shapes'][key]))
    return arrays, value
