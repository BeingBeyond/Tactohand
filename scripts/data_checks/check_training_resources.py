#!/usr/bin/env python3
"""Verify a dataset release's training statistics and rigid references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.data.processed_compat import reader_for
from core.tactile_labels.rigid_reference_store import source_key
from core.tactile_labels.native_timeline import timeline_revision, validate_native_timeline
from training.data.split_manifest import manifest_sha256


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(processed_root, resources, split, metadata_only=False):
    split_hash = manifest_sha256(split)
    readers = {}
    for name in ['scales.json', 'output_scales.json']:
        stats = json.loads((resources / 'statistics' / name).read_text())
        if stats['split_sha256'] != split_hash:
            raise ValueError(f'{name}: statistics belong to another split')
        for dataset, expected in stats['prepared_manifest_sha256'].items():
            reader = reader_for(processed_root / f'{dataset}_lerobot')
            if digest(reader.path(reader.manifest_path)) != expected:
                raise ValueError(f'{dataset}: statistics belong to another dataset manifest')
            readers[dataset] = reader
    count = 0
    for dataset in sorted((resources / 'rigid_references').glob('*_lerobot')):
        root = processed_root / dataset.name
        reader = reader_for(root)
        readers[dataset.name.removesuffix('_lerobot')] = reader
        labels = {row['episode_index']: row for row in reader.embedded_table('tactile_context').to_pylist()}
        for path in sorted(dataset.glob('episode_*/manifest.json')):
            value = json.loads(path.read_text())
            if value['source_key'] != source_key(root, labels[value['episode']]):
                raise ValueError(f'Stale rigid reference: {path}')
            validate_native_timeline(
                reader.read_episode('tactile.timeline', value['episode'], 'shared'),
                labels[value['episode']], timeline_revision(reader))
            if value['status'] == 'ready':
                payload = path.parent / value['payload']
                if not payload.resolve().is_relative_to(resources.resolve()):
                    raise ValueError(f'Unsafe payload path: {payload}')
                if not payload.is_file():
                    raise FileNotFoundError(payload)
                if not metadata_only and digest(payload) != value['payload_sha256']:
                    raise ValueError(f'Corrupt rigid reference: {payload}')
            count += 1
    return dict(status='passed', datasets=len(readers), references=count,
                payload_hashes_verified=not metadata_only)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--processed-root', required=True, type=Path)
    parser.add_argument('--resources-root', required=True, type=Path)
    parser.add_argument('--split', default='core/resources/tactohand/split_v1.json')
    parser.add_argument('--metadata-only', action='store_true',
                        help='Verify all bindings and metadata; skip dense Parquet checksums')
    args = parser.parse_args()
    report = check(args.processed_root, args.resources_root, args.split, args.metadata_only)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
