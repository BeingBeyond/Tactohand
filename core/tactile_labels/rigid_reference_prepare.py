"""Offline recovery of missing rigid-scene references; never solve forces."""
from pathlib import Path
import time

import numpy as np

from core.data import processed_compat as contract
from core.data.camera import hmr_c2w
from core.tactile_labels.decode import read_episode_columns
from core.tactile_labels.mano_tracks import solver_vertex_tracks
from core.tactile_labels.native_timeline import validate_native_timeline
from core.tactile_labels.rigid_bodies import recover_body_payload, unrecovered_force_rows
from core.tactile_labels.rigid_reference_store import write_reference
from core.tactile_labels.schema import FLAG_FORCE_VALID, FLAG_LEFT_PRESENT, FLAG_RIGHT_PRESENT


def prepare_episode(root: Path, labels: dict, output_root: Path, *, mano_chunk=128,
                    query_device='cuda:0', trust_degenerate=True):
    """Read only prepared geometry/labels, recover every body, commit atomically."""
    started = time.monotonic()
    print('[labels] reading prepared force/tactile columns', flush=True)
    reader = contract.reader_for(root)
    timeline = validate_native_timeline(
        reader.read_episode('tactile.timeline', int(labels['episode_index']), 'shared'),
        labels, reader.revision)
    target_slots = timeline['target_object'].fill_null(-1).to_numpy()
    tactile = read_episode_columns(root, labels['tactile_groups'], ['source_position', 'flags'])
    force = read_episode_columns(root, labels['force_groups'], [
        'flags', 'vertex_force_world', 'contact_wrench', 'environment_wrench',
        'object_position', 'object_rotation', 'object_velocity'])
    count = len(force['flags'])
    if count != len(tactile['flags']) or not count:
        raise ValueError('Mismatched or empty tactile/force timelines')
    valid = (force['flags'] & FLAG_FORCE_VALID) > 0
    if not valid.any():
        raise ValueError('No valid force rows to verify historical state recovery')
    observation = next(v for v in contract.observations(root, int(labels['episode_index']))
                       if str(v['view_id']) == str(labels['canonical_view']))
    c2w = hmr_c2w(root, observation)
    # Share the same immutable table across hand sides and MANO chunks instead
    # of repeatedly decoding the whole episode from shared storage.
    hmr_frame = contract.observation_table(root, observation).to_pandas().sort_values('frame_index')
    present = np.stack([(tactile['flags'] & bit) > 0
                        for bit in (FLAG_LEFT_PRESENT, FLAG_RIGHT_PRESENT)], axis=1)
    vertices = np.zeros((count, 2, 778, 3), np.float32)
    # Only rows carrying force supervision need dense attribution. Scoped MANO
    # forwards keep per-worker intermediate allocations bounded.
    picked = np.flatnonzero(valid)
    geometry_started = time.monotonic()
    print(f'[geometry] valid_rows={len(picked)} bodies={len(labels["object_ids"])} '
          f'mano_chunk={mano_chunk}', flush=True)
    for start in range(0, len(picked), mano_chunk):
        rows = picked[start:start + mano_chunk]
        tracks = solver_vertex_tracks(root, int(labels['episode_index']),
            labels['canonical_view'], source_positions=tactile['source_position'],
            rows=rows, device='cpu', observation=observation, hmr_frame=hmr_frame)
        for side, name in enumerate(('left', 'right')):
            vertices[rows, side], available = tracks[name]
            if np.any(present[rows, side] & ~available):
                raise ValueError(f'{name}: published present hand has no source geometry')
        if start % (mano_chunk * 8) == 0 or start + mano_chunk >= len(picked):
            print(f'[geometry] {min(start + mano_chunk, len(picked))}/{len(picked)} '
                  f'elapsed={time.monotonic() - geometry_started:.1f}s', flush=True)
    geometry_seconds = time.monotonic() - geometry_started
    print(f'[recover] verifying historical states; ownership_device={query_device}', flush=True)
    payload = recover_body_payload(root, labels, tactile['source_position'],
        c2w, force, vertices, None, present, device=query_device,
        target_slots=target_slots,
        trust_degenerate=trust_degenerate,
        progress=lambda message: print(message, flush=True))
    # Historical neutral bodies are not failed reconstructions. Require every
    # force-valid, source-supported body; keep unsupported bodies masked.
    missing = unrecovered_force_rows(payload, valid)
    payload['recovery_report'].update(
        force_valid_rows=int(valid.sum()), unrecovered_force_rows=int(missing.sum()),
        geometry_seconds=geometry_seconds, preparation_seconds=time.monotonic() - started)
    if missing.any():
        raise ValueError(f'{int(missing.sum())}/{int(valid.sum())} force-valid rows '
                         f'failed recovery checks: {payload["recovery_report"]}')
    print(f'[write] report={payload["recovery_report"]}', flush=True)
    return write_reference(output_root, root, labels, payload=payload)
