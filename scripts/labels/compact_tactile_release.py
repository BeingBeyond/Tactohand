#!/usr/bin/env python3
"""Stage compact prepared tactile labels for one dataset without changing its live pointer."""
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.processed_index import ProcessedIndex
from core.tactile_labels.prepared_publish import _component, _write_json, write_payload
from core.tactile_labels.schema import FORCE_TIMELINE_COLUMNS, SAMPLE_TIMELINE_COLUMNS


def component(reader, meta, name, ranges):
    spec = reader.spec(f'tactile.{name}')
    updated = _component(reader.root, meta, name, spec['index_unit'], ranges)
    for key in ('semantics', 'source_frame_mapping'):
        if key in spec:
            updated[key] = spec[key]
    return updated


def project_shard(source, destination, mapping, reuse, scratch):
    if reuse and destination.is_file():
        with pq.ParquetFile(source) as original, pq.ParquetFile(destination) as result:
            fields = [field for field in original.schema_arrow if field.name not in mapping]
            rows = [original.metadata.row_group(i).num_rows for i in range(original.num_row_groups)]
            if result.schema_arrow.equals(pa.schema(fields)) and rows == [
                result.metadata.row_group(i).num_rows for i in range(result.num_row_groups)
            ]:
                return
    if scratch is None:
        write_payload(source, destination, mapping)
        return
    with TemporaryDirectory(dir=scratch, prefix='tactile-project-') as temporary:
        local = Path(temporary)
        local_source = local / 'source.parquet'
        local_destination = local / 'projected.parquet'
        shutil.copyfile(source, local_source)
        write_payload(local_source, local_destination, mapping)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix('.pending')
        shutil.copyfile(local_destination, pending)
        pending.replace(destination)


def stage(reader, release, solver_index_column, workers, reuse, scratch):
    root = reader.root
    manifest = deepcopy(reader.manifest)
    revision = f'{reader.revision}_{release}'
    meta = f'meta/index/{release}'
    data = f'data/tactile/releases/{release}'
    (root / meta).mkdir(parents=True, exist_ok=True)
    destination = f'meta/index/manifest.{release}.json'
    timeline_ranges = reader.range_table('tactile.timeline').to_pylist()
    if not timeline_ranges:
        raise ValueError(f'{root}: empty tactile timeline')
    timeline_by_episode = defaultdict(list)
    for row in timeline_ranges:
        timeline_by_episode[row['episode']].append(row)
    for rows in timeline_by_episode.values():
        rows.sort(key=lambda row: row['start'])

    @lru_cache(maxsize=8)
    def metadata(path):
        return pq.read_table(reader.path(path))

    def matching_metadata(row):
        start = row['start']
        stop = start + row['count']
        pieces = []
        cursor = start
        for span in timeline_by_episode[row['episode']]:
            begin = max(cursor, span['start'])
            end = min(stop, span['start'] + span['count'])
            if begin >= end:
                continue
            if begin != cursor:
                raise ValueError(f'Gap in timeline for {row}')
            pieces.append(metadata(span['path']).slice(span['offset'] + begin - span['start'], end - begin))
            cursor = end
        if cursor != stop:
            raise ValueError(f'Incomplete timeline for {row}')
        return pa.concat_tables(pieces)

    components = manifest['modalities']['tactile']['components']
    components.pop('supervision', None)
    summary = {}
    for name, mapping in (('samples', SAMPLE_TIMELINE_COLUMNS), ('transitions', FORCE_TIMELINE_COLUMNS)):
        if name not in components:
            continue
        mapping = {column: solver_index_column if field == 'solver_row' else field
                   for column, field in mapping.items()}
        by_path = defaultdict(list)
        for row in reader.range_table(f'tactile.{name}').to_pylist():
            by_path[row['path']].append(row)
        migrated = []
        total_rows = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = []
            for ordinal, (path, rows) in enumerate(sorted(by_path.items())):
                source = reader.path(path)
                physical = pq.read_table(source, columns=list(mapping))
                for row in rows:
                    timeline = matching_metadata(row)
                    payload = physical.slice(row['offset'], row['count'])
                    for old, new in mapping.items():
                        if not np.array_equal(payload[old].to_numpy(), timeline[new].to_numpy(), equal_nan=True):
                            raise ValueError(f'{root.name} {name} episode {row["episode"]}: {old} differs from timeline')
                relative = f'{data}/{name}/file-{ordinal:05d}.parquet'
                jobs.append(pool.submit(project_shard, source, root / relative, mapping, reuse, scratch))
                if len(jobs) >= 2 * workers:
                    jobs.pop(0).result()
                migrated.extend(dict(row, path=relative) for row in rows)
                total_rows += sum(row['count'] for row in rows)
            for job in jobs:
                job.result()
        components[name] = component(reader, meta, name, migrated)
        components[name]['row_metadata'] = 'tactile.timeline'
        if solver_index_column != 'solver_row':
            components[name]['solver_index_column'] = solver_index_column
        summary[name] = dict(files=len(by_path), rows=total_rows)
        print(root.name, name, summary[name], flush=True)

    timeline_paths = {}
    migrated = []
    for row in timeline_ranges:
        path = row['path']
        if path not in timeline_paths:
            relative = f'{data}/timeline/file-{len(timeline_paths):05d}.parquet'
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            table = metadata(path)
            pending = target.with_suffix('.pending')
            pq.write_table(table.replace_schema_metadata(
                (table.schema.metadata or {}) | {b'source_revision': revision.encode()}),
                pending, compression='zstd')
            pending.replace(target)
            timeline_paths[path] = relative
        migrated.append(dict(row, path=timeline_paths[path]))
    components['timeline'] = component(reader, meta, 'timeline', migrated)
    manifest['revision'] = revision
    provenance = json.loads(reader.path(manifest['provenance']).read_text())
    manifest['provenance'] = f'{meta}/provenance.json'
    _write_json(root / manifest['provenance'], provenance)
    _write_json(root / destination, manifest)
    published = ProcessedIndex(root, manifest_path=destination)
    if not published.embedded_table('tactile_context').equals(reader.embedded_table('tactile_context')):
        raise ValueError(f'{root}: episode label context changed')
    return dict(source=reader.manifest_path, manifest=destination, revision=revision,
                manifest_sha256=hashlib.sha256((root / destination).read_bytes()).hexdigest(),
                episodes=len(timeline_by_episode), **summary)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--source-release', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--solver-index-column', choices=('solver_row', 'source_position'),
                        default='solver_row')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume-shards', action='store_true',
                        help='Reuse completed atomic shard writes from the same source and release')
    parser.add_argument('--scratch-dir', type=Path, help='Local directory for shard rewriting')
    args = parser.parse_args()
    if not all(re.fullmatch('[A-Za-z0-9_-]+', name) for name in (args.release, args.source_release)):
        parser.error('invalid release name')
    if args.release == args.source_release:
        parser.error('the source and destination releases must differ')
    if args.workers < 1:
        parser.error('--workers must be positive')
    active = ProcessedIndex(args.dataset_root)
    target = f'meta/index/manifest.{args.release}.json'
    if active.manifest_path == target:
        parser.error('refusing to overwrite an active release')
    if args.receipt.exists():
        receipt = json.loads(args.receipt.read_text())
        if receipt['manifest'] != target or receipt['manifest_sha256'] != hashlib.sha256(
                (args.dataset_root / target).read_bytes()).hexdigest():
            raise ValueError('Existing receipt does not match the staged manifest')
        print(f'{args.dataset_root.name}: verified existing receipt')
        return
    reader = ProcessedIndex(args.dataset_root, manifest_path=f'meta/index/manifest.{args.source_release}.json')
    receipt = stage(reader, args.release, args.solver_index_column, args.workers,
                    args.resume_shards, args.scratch_dir)
    receipt['active_manifest'] = active.manifest_path
    receipt['active_sha256'] = hashlib.sha256(active.path(active.manifest_path).read_bytes()).hexdigest()
    _write_json(args.receipt, receipt)


if __name__ == '__main__':
    main()
