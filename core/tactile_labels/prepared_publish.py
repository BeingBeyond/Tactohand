"""Install staged solver-row labels as a versioned prepared manifest."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.data.processed_index import ProcessedIndex, RANGE_SCHEMA
from core.tactile_labels.native_timeline import read_native_timeline
from core.tactile_labels.schema import FORCE_TIMELINE_COLUMNS, SAMPLE_TIMELINE_COLUMNS


def _write_json(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def _index_spec(path, table, column):
    ids = table[column].to_numpy()
    return dict(path=path, rows=len(ids), episode_min=int(ids.min()), episode_max=int(ids.max()))


def _component(root, meta, name, index_unit, ranges):
    path = f'{meta}/{name}.parquet'
    table = pa.Table.from_pylist(ranges, schema=RANGE_SCHEMA).sort_by('episode')
    pq.write_table(table, root / path, compression='zstd')
    example = ranges[0]['path']
    schema = pq.ParquetFile(root / example).schema_arrow
    return dict(storage='parquet', index_unit=index_unit,
                streams={'shared': {'view_id': None, 'target_id': None}},
                ranges=[_index_spec(path, table, 'episode')], schema_example_path=example,
                schema=dict(fields=schema.names,
                    arrow_ipc_base64=base64.b64encode(schema.remove_metadata().serialize()).decode(),
                    column_types={field.name: str(field.type) for field in schema}),
                nullable_fields=[field.name for field in schema if field.nullable])


def write_payload(source: Path, destination: Path, row_metadata: dict[str, str]) -> np.ndarray:
    """Project out timeline columns, preserving physical row-group offsets."""
    with pq.ParquetFile(source) as parquet:
        schema = parquet.schema_arrow
        payload = [name for name in schema.names if name not in row_metadata]
        projected = pa.schema([schema.field(name) for name in payload], metadata=schema.metadata)
        pending = destination.with_suffix('.pending')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(pending, projected, compression='zstd') as writer:
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=payload)
                writer.write_table(table, row_group_size=max(1, table.num_rows))
        offsets = np.cumsum([0, *(parquet.metadata.row_group(i).num_rows
                                  for i in range(parquet.num_row_groups))])
    with pq.ParquetFile(pending) as result:
        if [result.metadata.row_group(i).num_rows for i in range(result.num_row_groups)] != list(np.diff(offsets)):
            raise ValueError(f'Changed physical row-group boundaries: {source}')
    pending.replace(destination)
    return offsets


def _sample_ranges(staging, root, data, labels, group_key, row_metadata):
    files = {}
    ranges = []
    for row in labels:
        start = 0
        for group in row[group_key]:
            source = staging / group['path']
            if source not in files:
                relative = f"{data}/{group['path']}"
                destination = root / relative
                offsets = write_payload(source, destination, row_metadata)
                files[source] = relative, offsets
            relative, offsets = files[source]
            i = group['row_group']
            count = int(offsets[i + 1] - offsets[i])
            ranges.append(dict(episode=row['episode_index'], stream='shared', start=start,
                               count=count, path=relative, offset=int(offsets[i])))
            start += count
        if start != row['n_rows'] and (group_key == 'tactile_groups' or row['has_force']):
            raise ValueError(f"Label row count differs for episode {row['episode_index']}")
    return ranges


def _packed_timeline(staging, root, data, labels, declared, revision):
    ranges = []
    for begin in range(0, len(labels), 256):
        path = f'{data}/timeline/file-{begin // 256:05d}.parquet'
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        tables = []
        for row in labels[begin:begin + 256]:
            episode = row['episode_index']
            source = staging / f'timeline_data/episode_{episode:06d}.parquet'
            table = read_native_timeline(source, row, declared['timeline_revision'])
            table = table.append_column(
                'episode_index', pa.array(np.full(table.num_rows, episode, dtype=np.int64)))
            tables.append(table.replace_schema_metadata({'source_revision': revision}))
            ranges.append(dict(episode=episode, stream='shared', start=0, count=table.num_rows,
                               path=path, offset=offset))
            offset += table.num_rows
        pq.write_table(pa.concat_tables(tables), root / path, compression='zstd')
    return ranges


def install_prepared_labels(staging: Path, root: Path) -> str:
    with (root.parent / '.native_timeline_activation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _install(staging, root)


def _install(staging, root):
    info_path = root / 'meta/info.json'
    original = info_path.read_bytes()
    reader = ProcessedIndex(root)
    if not reader.has_modality('tactile.timeline'):
        raise ValueError('Install into a prepared dataset with a native solver timeline')
    info = json.loads(original)
    staged_info = json.loads((staging / 'meta/info.json').read_text())
    declared = staged_info['beingh']['labels']
    if declared.get('timeline') != 'solver_rows_v1':
        raise ValueError('Staged labels must use the native solver-row contract')
    labels = pq.read_table(staging / 'meta/label_index.parquet').sort_by('episode_index').to_pylist()
    by_episode = {row['episode_index']: row for row in labels}
    if not labels or len(labels) != len(by_episode):
        raise ValueError('Staged labels must contain unique episode IDs')
    known = set(reader.metadata('episodes', columns=['episode_index'])['episode_index'].to_pylist())
    if not by_episode.keys() <= known:
        raise ValueError('Staged labels contain episode IDs absent from the prepared dataset')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    release = f'labels_{stamp}'
    revision = f'{reader.revision}_{release}'
    data, meta = f'data/tactile/releases/{release}', f'meta/index/{release}'
    (root / meta).mkdir()
    components = {}
    for name, index_unit, key, row_metadata in (
        ('samples', 'solver_sample', 'tactile_groups', SAMPLE_TIMELINE_COLUMNS),
        ('transitions', 'solver_transition', 'force_groups', FORCE_TIMELINE_COLUMNS),
    ):
        ranges = _sample_ranges(staging, root, data, labels, key, row_metadata)
        if ranges:
            components[name] = _component(root, meta, name, index_unit, ranges)
            components[name]['row_metadata'] = 'tactile.timeline'
    ranges = _packed_timeline(staging, root, data, labels, declared, revision)
    components['timeline'] = _component(root, meta, 'timeline', 'solver_sample', ranges)
    for name, component in components.items():
        if reader.has_modality(f'tactile.{name}'):
            source = reader.spec(f'tactile.{name}')
            for key in ('semantics', 'source_frame_mapping'):
                if key in source:
                    component[key] = source[key]

    manifest = reader.manifest
    indices = []
    for i, shard in enumerate(manifest['indices']['episodes']):
        table = pq.read_table(reader.path(shard['path']))
        field = table.schema.field('tactile')
        values = [by_episode.get(episode) for episode in table['episode_index'].to_pylist()]
        context = pa.array(values, type=field.type)
        table = table.set_column(table.schema.get_field_index('tactile'), field, context)
        path = f'{meta}/episodes-{i:05d}.parquet'
        pq.write_table(table, root / path, compression='zstd')
        indices.append(_index_spec(path, table, 'episode_index'))
    manifest['indices']['episodes'] = indices
    manifest['modalities']['tactile']['components'] = components
    manifest['contracts']['beingh']['labels'] = {
        key: value for key, value in declared.items() if key != 'timeline_revision'}
    if 'mano_dependent_labels' in staged_info:
        manifest['contracts']['mano_dependent_labels'] = staged_info['mano_dependent_labels']

    statistics = json.loads((staging / 'meta/label_stats.json').read_text())
    previous_stats = pq.read_table(reader.path(manifest['stats']))
    rows = [row for row in previous_stats.to_pylist()
            if row['modality'] not in {'tactile', 'tactile.samples'}]
    for name in ('point_force_rms_n', 'wrench_scale', 'n_rows', 'episodes'):
        value = np.asarray(statistics[name], dtype=np.float64)
        rows.append(dict(revision=revision, modality='tactile.samples', field=name,
                         scope='all', metric='source_statistic', value=value.reshape(-1).tolist(),
                         shape=list(value.shape), recipe_id=statistics['solver_run_id']))
    manifest['stats'] = f'{meta}/stats.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=previous_stats.schema), root / manifest['stats'])
    provenance = json.loads(reader.path(manifest['provenance']).read_text())
    provenance.setdefault('generators', {})['label_statistics'] = statistics
    provenance['contracts'] = manifest['contracts']
    manifest['provenance'] = f'{meta}/provenance.json'
    _write_json(root / manifest['provenance'], provenance)
    manifest['revision'] = revision
    relative_manifest = f'meta/index/manifest.{release}.json'
    _write_json(root / relative_manifest, manifest)
    published = ProcessedIndex(root, manifest_path=relative_manifest)
    if len(published.range_table('tactile.timeline')) != len(labels):
        raise ValueError('Published timeline episode count differs from staged labels')
    if info_path.read_bytes() != original:
        raise ValueError('Prepared dataset changed during installation')
    info.update(manifest=relative_manifest, revision=revision)
    _write_json(info_path, info)
    return relative_manifest
