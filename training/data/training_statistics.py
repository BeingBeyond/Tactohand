"""Train-only loss priors and force normalization from DexYCB/ARCTIC.

Only this fixed training subset fits the priors; other mixture sources do not
fit these statistics. Episode summaries allow interrupted preparation to resume.
No images or MANO geometry are decoded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from core.data.processed_compat import reader_for
from core.tactile_labels.decode import load_label_index, read_episode_columns
from core.tactile_labels.native_timeline import read_native_timeline, validate_native_timeline, timeline_revision
from core.tactile_labels.schema import (
    FLAG_LEFT_PRESENT, FLAG_RIGHT_PRESENT, FLAG_CONTACT_VALID, FLAG_FORCE_VALID,
)
from core.physics.scales import robust_wrench_scale
from training.data.split_manifest import select_manifest_episodes, manifest_sha256


def native_statistics_rows(timeline_path: Path, label: dict, revision: str):
    return aligned_statistics_rows(read_native_timeline(timeline_path, label, revision))


def aligned_statistics_rows(timeline):
    """Select each exactly aligned source frame once from the native index."""
    positions = timeline['source_position'].to_numpy()
    source_frames = np.rint(positions)
    aligned = np.flatnonzero(np.isfinite(positions) & (np.abs(positions - source_frames) <= 1e-4))
    _, first = np.unique(source_frames[aligned], return_index=True)
    selected = aligned[np.sort(first)]
    rows = timeline['solver_row'].to_numpy()[selected]
    targets = timeline['target_object'].fill_null(-1).to_numpy()[selected]
    return rows, targets


def prepare(root: Path, manifest: Path, output: Path, timeline_root: Path | None = None) -> None:
    digest = manifest_sha256(manifest)
    readers = {name: reader_for(root / f'{name}_lerobot') for name in ('dexycb', 'arctic')}
    prepared_manifests = {
        name: hashlib.sha256(reader.path(reader.manifest_path).read_bytes()).hexdigest()
        for name, reader in readers.items()
    }
    installed_native = [reader.has_modality('tactile.timeline') for reader in readers.values()]
    if any(installed_native) and not all(installed_native):
        raise ValueError('Statistics require the same timeline contract for DexYCB and ARCTIC')
    if timeline_root is None and not all(installed_native):
        raise ValueError('Train statistics require native solver timelines')
    sampling = 'native_rows'
    scales_path = output / 'scales.json'
    if scales_path.exists():
        existing = json.loads(scales_path.read_text())
        if existing.get('statistics_sampling', 'published_tokens') != sampling:
            raise ValueError(f'Statistics output already uses a different sampling protocol: {output}')
    cache = output / 'episodes'
    cache.mkdir(parents=True, exist_ok=True)
    total_counts = np.zeros((778, 2), np.int64)
    nonzero = np.zeros(778, np.int64)
    squared = slots = components = 0
    wrenches = []
    episodes = 0
    for name, reader in readers.items():
        data_root = root / f'{name}_lerobot'
        ids = select_manifest_episodes(data_root, str(manifest), 'train')
        if ids is None:
            raise ValueError(f'{name}: training statistics require a complete split manifest')
        revision = reader.revision
        for label in load_label_index(data_root, sorted(ids)):
            signature_parts = [digest, label]
            if reader.has_modality('tactile.timeline'):
                signature_parts += [sampling, prepared_manifests[name]]
            elif timeline_root is not None:
                timeline_path = (timeline_root / data_root.name / revision /
                                 f"episode_{label['episode_index']:06d}.parquet")
                if not timeline_path.is_file():
                    raise FileNotFoundError(timeline_path)
                signature_parts += [sampling, hashlib.sha256(timeline_path.read_bytes()).hexdigest()]
            signature = hashlib.sha256(json.dumps(
                signature_parts, sort_keys=True).encode()).hexdigest()[:16]
            path = cache / f'{name}_{label["episode_index"]}_{signature}.npz'
            if not path.exists():
                tactile = read_episode_columns(data_root, label['tactile_groups'], ['flags', 'contact'])
                if reader.has_modality('tactile.timeline'):
                    timeline = validate_native_timeline(reader.read_episode(
                        'tactile.timeline', int(label['episode_index']), 'shared'), label, timeline_revision(reader))
                    indices, targets = aligned_statistics_rows(timeline)
                else:
                    indices, targets = native_statistics_rows(timeline_path, label, revision)
                flags = tactile['flags'][indices]
                present = np.stack([(flags & FLAG_LEFT_PRESENT) > 0,
                                    (flags & FLAG_RIGHT_PRESENT) > 0], axis=1)
                contact = tactile['contact'].reshape(-1, 2, 778)[indices] > 0
                valid = present & ((flags & FLAG_CONTACT_VALID) > 0)[:, None]
                engaged = valid & contact.any(-1)
                active_contact = contact[engaged]
                counts = np.stack([(~active_contact).sum(0), active_contact.sum(0)], axis=-1)
                nz = np.zeros(778, np.int64)
                ss = nc = ns = 0
                wrench = np.zeros((0, 6), np.float64)
                if label['has_force'] and label['force_groups']:
                    force = read_episode_columns(data_root, label['force_groups'],
                                                 ['flags', 'vertex_force_world', 'contact_wrench'])
                    fvalid = valid & ((force['flags'][indices] & FLAG_FORCE_VALID) > 0)[:, None]
                    masses = np.ones(len(indices), np.float64)
                    assigned = targets >= 0
                    if np.any(fvalid[~assigned]):
                        raise ValueError(f'{name}: a supervised force row has no target slot')
                    masses[assigned] = np.asarray(label['object_mass_kg'])[targets[assigned]]
                    values = force['vertex_force_world'].reshape(-1, 2, 778, 3)[indices].astype(np.float64)
                    values *= masses[:, None, None, None]
                    values = values[fvalid]
                    if not np.isfinite(values).all():
                        raise ValueError(f'{name}: non-finite valid training force')
                    magnitude = np.linalg.norm(values, axis=-1)
                    values *= np.minimum(1, 20 / np.maximum(magnitude, 1e-12))[..., None]
                    nz = (magnitude > .05).sum(0)
                    ss, nc, ns = float(np.square(values).sum()), values.size, len(values)
                    wrench = force['contact_wrench'][indices][fvalid.any(-1)]
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, counts=counts, nonzero=nz,
                                    squared=ss, components=nc, slots=ns, wrench=wrench)
                temporary.replace(path)
            with np.load(path, allow_pickle=False) as part:
                total_counts += part['counts']
                nonzero += part['nonzero']
                squared += float(part['squared'])
                components += int(part['components'])
                slots += int(part['slots'])
                wrenches.append(part['wrench'])
            episodes += 1
            if episodes % 100 == 0:
                print(f'train-statistics: {episodes} episodes', flush=True)
    if slots <= 0 or total_counts.sum() <= 0:
        raise ValueError('No valid train-only statistics')
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'contact.npz', schema_version=1, class_counts=total_counts)
    np.savez(output / 'force.npz', schema_version=1, nonzero_counts=nonzero, supervised_slots=slots)
    stats = {'split_sha256': digest, 'source': 'dexycb+arctic train only',
             'prepared_manifest_sha256': prepared_manifests,
             'statistics_sampling': sampling,
             'episodes': episodes, 'n_rows': slots,
             'point_force_rms_n': max(float(np.sqrt(squared / components)), 1e-6),
             'wrench_scale': robust_wrench_scale(np.concatenate(wrenches)).tolist()}
    (output / 'scales.json').write_text(json.dumps(stats, indent=2) + '\n')
    (output / 'statistics.done').write_text(digest + '\n')
    print(f'PASS train-only statistics: {episodes} episodes, {slots} valid hand observations', flush=True)
