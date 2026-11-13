"""Small compatibility views derived from the new manifest, never old indices.

The same portable modules are distributed with video2contact so both training
stacks read the published contract without depending on a private repo path.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from core.data.processed_index import ProcessedIndex


def is_processed(root: Path) -> bool:
    return json.loads((Path(root)/'meta/info.json').read_text()).get('format_version')=='processed_dataset_v1'


@lru_cache(maxsize=32)
def _reader(path: str, manifest_path: str, revision: str,
            manifest_mtime_ns: int = 0) -> ProcessedIndex:
    value=ProcessedIndex(path, manifest_path=manifest_path)
    if value.revision!=revision:raise ValueError('Manifest changed while opening compatibility view')
    return value


def reader_for(root: Path) -> ProcessedIndex:
    root=Path(root).resolve()
    info=json.loads((root/'meta/info.json').read_text())
    manifest_path = info.get('manifest', 'meta/index/manifest.json')
    revision = info['revision']
    release = os.environ.get('TACTOHAND_PREPARED_RELEASE')
    if release:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', release):
            raise ValueError(f'Invalid prepared release name: {release!r}')
        manifest_path = f'meta/index/manifest.{release}.json'
        revision = json.loads((root / manifest_path).read_text())['revision']
    manifest = root / manifest_path
    return _reader(str(root), manifest_path, revision, manifest.stat().st_mtime_ns)


def read_info(root: Path) -> dict:
    root=Path(root)
    info=json.loads((root/'meta/info.json').read_text())
    if info.get('format_version')!='processed_dataset_v1':return info
    reader=reader_for(root)
    provenance=json.loads(reader.path(reader.manifest['provenance']).read_text())
    result=dict(info,revision=reader.revision,manifest=reader.manifest_path,
                fps=reader.manifest['fps'],dataset_name=info['name'],
                total_episodes=reader.manifest['episode_total'],total_frames=reader.manifest['frame_total'])
    result.update(reader.manifest.get('contracts',{}))
    human=dict(provenance.get('source_contract',{}))
    human['capabilities']={name:dict(value) for name,value in reader.manifest.get('unavailable_modalities',{}).items()}
    for modality,spec in reader.component_specs():
        human['capabilities'][modality]=dict(spec.get('source_capabilities',{}),available=bool(spec['ranges']))
    family = reader.manifest['modalities'].get('tactile', {})
    if family.get('storage') == 'family':
        human['capabilities']['tactile'] = dict(available=reader.has_modality('tactile.samples'),
                                               layout='tactile_family_v1')
    human['camera_table']=reader.manifest['tables'].get('cameras')
    human['objects']=reader.manifest.get('object_scene',{})
    human['format_version']='processed_dataset_v1'
    result['beingh_human']=human
    result['_processed_manifest']=reader.manifest
    result['_source_info_sha256']=reader.manifest.get('source_info_sha256')
    native=reader.manifest.get('native_legacy')
    if native:
        result.update(native['info'])
        result['paths']=dict(data_pattern=native['data_pattern'],video_pattern=native['video_pattern'])
        result['mesh_store']=native.get('source_mesh_store',{})
        result['video_files']=native['source_video_files']
        paths=[];seen=set()
        for shard in reader.manifest['modalities']['native_records']['ranges']:
            for path in reader._index_table(shard['path'])['path'].unique().to_pylist():
                if path not in seen:paths.append(path);seen.add(path)
        result['data_files']=[dict(chunk=i//1000,file=i%1000,rows=pq.ParquetFile(reader.path(path)).metadata.num_rows)
                              for i,path in enumerate(paths)]
    return result


def native_sequences(root: Path) -> pa.Table:
    reader=reader_for(root)
    return pa.Table.from_pylist([dict(seq_id=r['source_seq_id'],seq_name=r['sequence_id'],split=r['split'],
        num_frames=r['num_frames'],video_start_frame=r['source_video_start_frame'],
        video_end_frame=r['source_video_end_frame_inclusive']) for r in reader.metadata('episodes').to_pylist()])


def native_row_index(root: Path) -> pa.Table:
    reader=reader_for(root);paths=[];rows=[]
    for shard in reader.manifest['modalities']['native_records']['ranges']:
        for row in reader._index_table(shard['path']).to_pylist():
            if row['path'] not in paths:paths.append(row['path'])
            number=paths.index(row['path'])
            rows.append(dict(seq_id=row['episode'],data_chunk=number//1000,data_file=number%1000,
                             row_start=row['offset'],row_end=row['offset']+row['count']))
    return pa.Table.from_pylist(rows)


def object_path(root: Path, key: str, legacy: str) -> Path:
    root=Path(root)
    if not is_processed(root):return root/legacy
    declaration=reader_for(root).manifest.get('object_scene',{})
    path=declaration.get(key) or declaration.get('articulation',{}).get(key)
    if not path:raise FileNotFoundError(f'No declared object resource: {key}')
    return reader_for(root).path(path)


def split_table(root: Path, columns: list[str] | None = None) -> pa.Table | None:
    """Read metadata split rules, with support for the previous data layout."""
    reader = reader_for(root)
    if 'splits' in reader.manifest.get('tables', {}):
        return reader.table('splits', columns)
    if 'splits' in reader.manifest['modalities']:
        return payload_table(root, 'splits', columns)
    return None


def payload_table(root: Path, modality: str, columns: list[str] | None = None) -> pa.Table:
    reader=reader_for(root)
    if modality == 'splits' and 'splits' in reader.manifest.get('tables', {}):
        return reader.table('splits', columns)
    if (reader.manifest.get('contracts', {}).get('unified_data')
            and modality in {'object_instances', 'tactile_context'}):
        if modality == 'tactile_context':
            return reader.embedded_table(modality, columns=columns)
        table = reader.embedded_table(modality)
        return table.select(columns) if columns is not None else table
    paths=[];seen=set()
    for shard in reader.spec(modality)['ranges']:
        for path in reader._index_table(shard['path'])['path'].unique().to_pylist():
            if path not in seen:paths.append(path);seen.add(path)
    return pa.concat_tables([pq.read_table(reader.path(path),columns=columns) for path in paths])


def _physical_group(reader: ProcessedIndex, path: str, offset: int, count: int) -> dict:
    file,offsets=reader._file(path)
    group=int(np.searchsorted(offsets,offset,side='right')-1)
    if offset+count>offsets[group+1]:
        raise ValueError('Legacy single-row-group API cannot represent this range; use ProcessedIndex')
    return dict(path=path,row_group_index=group,row_offset=int(offset-offsets[group]),num_rows=count)


def legacy_locator(reader: ProcessedIndex, modality: str, episode: int, stream: str) -> dict | None:
    if not reader.has_modality(modality):return None
    rows=reader.ranges(modality,episode,stream)
    if not rows:return None
    path=rows[0]['path'];offset=rows[0]['offset'];count=sum(r['count'] for r in rows)
    cursor=offset
    for row in rows:
        if row['path']!=path or row['offset']!=cursor:
            raise ValueError(f'Legacy locator cannot represent a fragmented {modality} payload')
        cursor+=row['count']
    return _physical_group(reader,path,offset,count)


def episode_rows(root: Path, episode_indices: list[int] | None = None) -> list[dict]:
    root=Path(root)
    if not is_processed(root):
        return [r for p in sorted((root/'meta/episodes').glob('chunk-*/*.parquet')) for r in pq.read_table(p).to_pylist()]
    reader=reader_for(root)
    tasks={r['task_index']:r.get('name',r.get('task')) for r in reader.table('tasks').to_pylist()} if 'tasks' in reader.manifest['tables'] else {}
    out=[]
    episodes = reader.metadata('episodes')
    if episode_indices is not None:
        episodes = episodes.filter(pc.is_in(episodes['episode_index'],
            value_set=pa.array(episode_indices, type=pa.int64())))
    for value in episodes.to_pylist():
        row=dict(value,length=value['num_frames'],dataset_from_index=value['global_frame_start'],
                 dataset_to_index=value['global_frame_start']+value['num_frames'],
                 tasks=[tasks[t] for t in value.get('task_ids',[]) if t in tasks])
        row['data/file_index']=value['episode_index'];row['data/chunk_index']=0
        for view in value.get('views',[]):
            if 'streams' in view:
                view = dict(view.get('extensions') or {}, **(view['streams'].get('processed') or {}),
                            view_id=view['view_id'])
            stream=view['view_id'];key=view.get('video_key',stream)
            ranges=reader.ranges('video',value['episode_index'],stream) if 'video' in reader.manifest['modalities'] else []
            if ranges:
                first=ranges[0];prefix=f'videos/{key}'
                row[prefix+'/path']=first['path']
                row[prefix+'/chunk_index']=0;row[prefix+'/file_index']=value['episode_index']
                row[prefix+'/from_timestamp']=first['offset']/reader.manifest['fps']
                row[prefix+'/to_timestamp']=(first['offset']+sum(r['count'] for r in ranges))/reader.manifest['fps']
        for modality,prefix in [('gravity','gravity_data'),('language','language_annotations')]:
            try:loc=legacy_locator(reader,modality,value['episode_index'],'shared')
            except ValueError:loc=None  # New readers handle cross-row-group ranges directly.
            if loc:
                for k,v in loc.items():row[prefix+'/'+k]=v
        out.append(row)
    return out


def observations(root: Path, episode: int | None = None) -> list[dict]:
    root=Path(root)
    if not is_processed(root):
        table=pq.read_table(root/'meta/observations.parquet',filters=[('episode_index','=',episode)] if episode is not None else None)
        return table.to_pylist()
    reader=reader_for(root)
    out=[]
    for ep in reader.metadata('episodes',episode).to_pylist():
        eid=ep['episode_index']
        for view in ep['views']:
            if 'streams' in view:
                availability = view.get('availability') or {}
                view = dict(view.get('extensions') or {}, **(view['streams'].get('processed') or {}),
                            view_id=view['view_id'], **availability)
            stream=view['view_id'];row=dict(view,episode_index=eid,num_frames=ep['num_frames'])
            if 'observation_context' in reader.manifest['modalities']:
                context=reader.read_episode('observation_context',eid,stream).to_pylist()
                if context:row.update(context[0])
            video=reader.ranges('video',eid,stream) if 'video' in reader.manifest['modalities'] else []
            row['video_num_frames']=sum(r['count'] for r in video)
            for modality in ('hmr','camera','objects','body_tracking','hand_instances','umetrack','calibration','segmentation'):
                ranges = reader.ranges(modality,eid,stream) if reader.has_modality(modality) else []
                try:
                    loc=legacy_locator(reader,modality,eid,stream)
                except ValueError:
                    loc=None  # Fragmented payloads are read through observation_table.
                row[modality+'_available']=bool(ranges)
                row[modality+'_num_rows']=sum(r['count'] for r in ranges)
                if loc:
                    for k,v in loc.items():row[modality+'_'+k]=v
            row['hmr_segment_count']=int(row.get('hmr_available',False))
            row['camera_dynamic']=bool(row.get('camera_available'))
            tactile=reader.spec('tactile') if reader.has_modality('tactile') else {}
            row['tactile_available']=False
            if tactile.get('index_unit')=='episode_frame':
                loc=legacy_locator(reader,'tactile',eid,'shared')
                row['tactile_available']=bool(loc)
                if loc:
                    for k,v in loc.items():row['tactile_'+k]=v
            out.append(row)
    return out


def observation_table(root: Path, observation: dict, modality: str = 'hmr',
                      columns: list[str] | None = None) -> pa.Table:
    """Read an observation across any published files and row groups."""
    if is_processed(root):
        return reader_for(root).read_episode(modality, int(observation['episode_index']),
                                              str(observation['view_id']), columns)
    return pq.ParquetFile(Path(root) / observation[modality+'_path']).read_row_group(
        int(observation[modality+'_row_group_index']), columns=columns).slice(
        int(observation[modality+'_row_offset']), int(observation[modality+'_num_rows']))


def cameras(root: Path) -> pa.Table:
    return reader_for(root).table('cameras') if is_processed(root) else pq.read_table(Path(root)/'meta/cameras.parquet')


class TactileGroups(list[dict]):
    """Payload locators with the pinned source for projected row metadata."""

    def __init__(self, locators: list[dict], reader: ProcessedIndex, episode: int, modality: str):
        super().__init__(locators)
        self.timeline_source = reader, episode, modality


def groups(reader: ProcessedIndex, modality: str, episode: int, stream='shared') -> list[dict]:
    result=[]
    if not reader.has_modality(modality):return result
    for row in reader.ranges(modality,episode,stream):
        _,offsets=reader._file(row['path']);cursor=row['offset'];end=cursor+row['count']
        while cursor<end:
            group=int(np.searchsorted(offsets,cursor,side='right')-1)
            count=min(end,int(offsets[group+1]))-cursor
            result.append(dict(path=row['path'],row_group=group,row_offset=int(cursor-offsets[group]),num_rows=count))
            cursor+=count
    if (result and modality in ('tactile', 'tactile_force')
            and reader.spec(modality).get('row_metadata') == 'tactile.timeline'):
        return TactileGroups(result, reader, episode, modality)
    return result


class EpisodeLabelRow(dict):
    """Training metadata with physical Parquet locators resolved on first use.

    No persistent cache and no payload/footer access during window enumeration.
    The root is pinned by the owning reader's manifest revision.
    """

    def __init__(self, row: dict, reader: ProcessedIndex):
        super().__init__(row)
        self.reader = reader

    def __missing__(self, key):
        modalities = {'tactile_groups': 'tactile', 'force_groups': 'tactile_force'}
        if key not in modalities:
            raise KeyError(key)
        return groups(self.reader, modalities[key], int(self['episode_index']))

    def get(self, key, default=None):
        if key in {'tactile_groups', 'force_groups'}:
            return self[key]
        return super().get(key, default)


def label_index(root: Path, episode_indices: list[int] | None = None,
                *, lazy_groups: bool = False) -> list[dict]:
    if not is_processed(root):return pq.read_table(Path(root)/'meta/label_index.parquet').to_pylist()
    reader=reader_for(root)
    if (not reader.manifest.get('contracts', {}).get('unified_data')
            and 'tactile_context' not in reader.manifest['modalities']):return []
    if reader.manifest.get('contracts', {}).get('unified_data') and not reader.has_modality('tactile'):
        return []
    context=payload_table(root,'tactile_context')
    if episode_indices is not None:
        context=context.filter(pc.is_in(context['episode_index'],
            value_set=pa.array(episode_indices, type=pa.int64())))
    values=context.to_pylist()
    sample_ranges=reader.range_table('tactile')
    sample_ranges=sample_ranges.filter(pc.equal(sample_ranges['stream'],'shared'))
    counts=sample_ranges.group_by('episode').aggregate([('count','sum')])
    row_counts=dict(zip(counts['episode'].to_pylist(),counts['count_sum'].to_pylist()))
    for row in values:
        episode = int(row['episode_index'])
        row['n_rows'] = int(row_counts.get(episode, 0))
        if not lazy_groups:
            row['tactile_groups'] = groups(reader, 'tactile', episode)
            row['force_groups'] = groups(reader, 'tactile_force', episode)
    return [EpisodeLabelRow(row, reader) for row in values] if lazy_groups else values


def object_instances(root: Path, episode: int | None = None) -> pa.Table:
    root = Path(root)
    if is_processed(root):
        return reader_for(root).object_instances(episode)
    return pq.read_table(root / 'objects/instances.parquet',
                         filters=[('episode_index', '=', episode)] if episode is not None else None)


def target_object_instance(root: Path, episode: int, instance_id: str | None = None) -> dict:
    """Select an explicitly marked target or the sole instance; never guess."""
    rows = object_instances(root, episode).to_pylist()
    if instance_id is not None:
        candidates = [r for r in rows if r['object_instance_id'] == instance_id]
    else:
        candidates = [r for r in rows if r.get('role') == 'target']
        if not candidates and len(rows) == 1:
            candidates = rows
    if len(candidates) != 1:
        raise ValueError(f'Episode {episode} needs an explicit object_instance_id; target is not unique')
    return candidates[0]


def object_rows(root: Path, episode: int, view: str | None = None,
                columns: list[str] | None = None, object_index: int | None = None) -> pa.Table:
    """Read episode-local object records through their authoritative ranges."""
    import pyarrow.compute as pc
    requested = list(columns) if columns is not None else None
    selected = list(dict.fromkeys(requested + ['object_index'])) if requested is not None and object_index is not None else requested
    if is_processed(root):
        reader = reader_for(root)
        streams = [view] if view is not None else list(reader.spec('objects')['streams'])
        parts = [reader.read_episode('objects', episode, stream, selected) for stream in streams
                 if reader.ranges('objects', episode, stream)]
        if not parts:
            raise KeyError(f'No object observations for {episode}/{view}')
        table = pa.concat_tables(parts)
    else:
        contract = read_info(root).get('beingh_human', {}).get('objects', {})
        path = contract.get('frames_path', 'objects/frames/chunk-000/file-000.parquet')
        filters = [('episode_index', '=', episode)]
        if view is not None:
            filters.append(('view_id', '=', view))
        table = pq.read_table(Path(root) / path, columns=selected, filters=filters)
    if object_index is not None:
        table = table.filter(pc.equal(table['object_index'], object_index))
    return table.select(requested) if requested is not None else table


def label_stats(root: Path) -> dict:
    if not is_processed(root):return json.loads((Path(root)/'meta/label_stats.json').read_text())
    reader=reader_for(root);table=pq.read_table(reader.path(reader.manifest['stats']))
    provenance=json.loads(reader.path(reader.manifest['provenance']).read_text())
    result=dict(provenance.get('generators',{}).get('label_statistics',{}))
    for row in table.to_pylist():
        if row['modality'] in {'tactile', 'tactile.samples'}:
            value=np.asarray(row['value']).reshape(row['shape'])
            result[row['field']]=value.item() if not row['shape'] else value.tolist()
    return result
