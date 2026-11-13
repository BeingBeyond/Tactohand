"""Projected training metadata from published ranges; no derived files."""
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from core.data import processed_compat as contract


def episode_metadata(root: Path) -> list[dict]:
    reader = contract.reader_for(root)
    path = reader.manifest['indices']['episodes'][0]['path']
    available = pq.read_schema(reader.path(path)).names
    columns = [key for key in ('episode_index', 'split', 'subject_id',
        'num_frames', 'global_frame_start') if key in available]
    return reader.metadata('episodes', columns=columns).to_pylist()


def view_metadata(root: Path, episodes: list[dict], views=None) -> list[dict]:
    """Attach video addresses and observation counts without payload footers."""
    reader = contract.reader_for(root)
    wanted = pa.array([ep['episode_index'] for ep in episodes], type=pa.int64())
    requested = set(map(str, views)) if views is not None else None
    lookups = {}
    for modality in ('video', 'hmr', 'hand_instances'):
        lookup = {}
        if reader.has_modality(modality):
            table = reader.range_table(modality)
            table = table.filter(pc.is_in(table['episode'], value_set=wanted))
            # first uses the first logical range, exactly as episode_rows.
            aggregated = table.group_by(['episode', 'stream'], use_threads=False).aggregate(
                [('count', 'sum'), ('path', 'first'), ('offset', 'first')])
            for row in aggregated.to_pylist():
                lookup[(row['episode'], row['stream'])] = (
                    row['count_sum'], row['path_first'], row['offset_first'])
        lookups[modality] = lookup
    metadata = reader.metadata('episodes', columns=['episode_index', 'views'])
    metadata = metadata.filter(pc.is_in(metadata['episode_index'], value_set=wanted))
    parent = pc.list_parent_indices(metadata['views'])
    flattened = pc.list_flatten(metadata['views'])
    episode_ids = pc.take(metadata['episode_index'], parent).to_pylist()
    view_ids = pc.struct_field(flattened, 'view_id').to_pylist()
    fields = {field.name for field in flattened.type}
    if 'streams' in fields:
        processed = pc.struct_field(flattened, ['streams', 'processed']).to_pylist()
        extras = pc.struct_field(flattened, 'extensions').to_pylist() if 'extensions' in fields else [None] * len(view_ids)
        availability = pc.struct_field(flattened, 'availability').to_pylist() if 'availability' in fields else [None] * len(view_ids)
        values = (dict(extra or {}, **(stream or {}), view_id=view, **(available or {}))
                  for view, stream, extra, available in zip(view_ids, processed, extras, availability))
    else:
        values = iter(flattened.to_pylist())
    by_episode = {int(ep['episode_index']): ep for ep in episodes}
    observations = []
    for eid, value in zip(episode_ids, values):
        ep = by_episode[eid]
        view = str(value['view_id'])
        if requested is not None and view not in requested:
            continue
        key = (eid, view)
        hmr = lookups['hmr'].get(key)
        if hmr is None:
            continue
        video = lookups['video'].get(key)
        row = dict(value, episode_index=eid, num_frames=ep['num_frames'],
            hmr_available=True, hmr_segment_count=1, hmr_num_rows=hmr[0],
            video_num_frames=video[0] if video else 0,
            hand_instances_available=key in lookups['hand_instances'],
            hand_instances_num_rows=lookups['hand_instances'].get(key, (0,))[0])
        if video:
            prefix = 'videos/' + str(row.get('video_key', view))
            ep[prefix + '/path'] = video[1]
            ep[prefix + '/from_timestamp'] = video[2] / reader.manifest['fps']
        observations.append(row)
    return observations
