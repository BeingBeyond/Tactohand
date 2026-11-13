"""Integer range reader for the processed dataset manifest contract.

Payload locations come exclusively from the published ranges. A reader pins
one manifest revision; missing indices never fall through to neighbouring data.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


RANGE_SCHEMA = pa.schema([
    pa.field("episode", pa.int64(), nullable=False),
    pa.field("stream", pa.string(), nullable=False),
    pa.field("start", pa.int64(), nullable=False),
    pa.field("count", pa.int64(), nullable=False),
    pa.field("path", pa.string(), nullable=False),
    pa.field("offset", pa.int64(), nullable=False),
])


class ProcessedIndex:
    """Read a pinned manifest with bounded per-instance metadata/data caches."""

    def __init__(
        self, root: str | Path, manifest_path: str | None = None,
        cache_entries: int = 16, row_group_cache_entries: int = 2,
    ) -> None:
        self.root = Path(root).resolve()
        if manifest_path is None:
            info = json.loads((self.root / "meta/info.json").read_text())
            manifest_path = info["manifest"]
        self.manifest_path = manifest_path
        self.manifest = json.loads(self.path(manifest_path).read_text())
        if self.manifest.get("format_version") not in {"processed_index_v1", "processed_index_v2"}:
            raise ValueError("Unsupported processed index manifest")
        if (self.manifest['format_version'] == 'processed_index_v2'
                and self.manifest.get('contracts', {}).get('unified_data') != 'unified_data_v1'):
            raise ValueError('Unsupported unified data contract')
        self.revision = self.manifest["revision"]
        self.cache_entries = max(1, int(cache_entries))
        self.row_group_cache_entries = max(0, int(row_group_cache_entries))
        self._indices: OrderedDict[str, pa.Table] = OrderedDict()
        self._files: OrderedDict[str, tuple[pq.ParquetFile, np.ndarray]] = OrderedDict()
        self._groups: OrderedDict[tuple, pa.Table] = OrderedDict()
        self._range_cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self._paths: OrderedDict[str, Path] = OrderedDict()
        self._pid = os.getpid()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        for key in ("_indices", "_files", "_groups", "_range_cache", "_paths"):
            state[key] = OrderedDict()
        return state

    def _ensure_process(self) -> None:
        if getattr(self, "_pid", None) != os.getpid():
            for key in ("_indices", "_files", "_groups", "_range_cache", "_paths"):
                setattr(self, key, OrderedDict())
            self._pid = os.getpid()

    def path(self, relative: str) -> Path:
        if relative in getattr(self, '_paths', {}):
            return self._paths[relative]
        value = Path(relative)
        if value.is_absolute() or ".." in value.parts:
            raise ValueError(f"Expected dataset-relative path: {relative!r}")
        result = (self.root / value).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError(f"Path escapes dataset root: {relative!r}")
        if hasattr(self, '_paths'):
            self._remember(self._paths, relative, result, self.cache_entries * 8)
        return result

    def spec(self, modality: str) -> dict:
        """Resolve a family component; old tactile names are in-memory aliases."""
        aliases = {'tactile': 'tactile.samples', 'tactile_force': 'tactile.transitions'}
        family = self.manifest['modalities'].get('tactile', {})
        if family.get('storage') == 'family':
            name = aliases.get(modality, modality)
            if name.startswith('tactile.'):
                return family['components'][name.split('.', 1)[1]]
        if modality.startswith('tactile.'):
            reverse = {'samples': 'tactile', 'transitions': 'tactile_force'}
            modality = reverse.get(modality.split('.', 1)[1], modality)
        return self.manifest['modalities'][modality]

    def has_modality(self, modality: str) -> bool:
        try:
            self.spec(modality)
        except KeyError:
            return False
        return True

    def component_specs(self):
        for name, spec in self.manifest['modalities'].items():
            if spec.get('storage') == 'family':
                for component, value in spec['components'].items():
                    yield name + '.' + component, value
            else:
                yield name, spec

    def embedded_table(self, name: str, episode: int | None = None,
                       columns: list[str] | None = None) -> pa.Table:
        """Materialize small compatibility tables from typed episode metadata."""
        field_name = 'objects' if name == 'object_instances' else 'tactile'
        table = self.metadata('episodes', episode, ['episode_index', field_name])
        if name == 'object_instances':
            field = table.schema.field('objects').type.field('instances')
            schema = pa.schema(list(field.type.value_type))
            rows = [dict(item, episode_index=row['episode_index'])
                    for row in table.to_pylist() for item in (row.get('objects') or {}).get('instances', [])]
            return pa.Table.from_pylist(rows, schema=schema)
        if name == 'tactile_context':
            schema = pa.schema([pa.field('episode_index', pa.int64())] + list(table.schema.field('tactile').type))
            table = table.filter(pc.is_valid(table['tactile']))
            names = columns if columns is not None else schema.names
            return pa.Table.from_arrays([
                table['episode_index'] if key == 'episode_index'
                else pc.struct_field(table['tactile'], key) for key in names],
                schema=pa.schema([schema.field(key) for key in names]))
        raise KeyError(name)

    def object_instances(self, episode: int | None = None) -> pa.Table:
        if self.manifest.get('contracts', {}).get('unified_data'):
            return self.embedded_table('object_instances', episode)
        if episode is not None:
            return self.read_episode('object_instances', episode, 'shared')
        paths = {row['path'] for shard in self.spec('object_instances')['ranges']
                 for row in self._index_table(shard['path']).to_pylist()}
        return pa.concat_tables([pq.ParquetFile(self.path(path)).read() for path in sorted(paths)])

    def tactile(self, episode: int, components=('samples',), columns: dict | None = None) -> dict:
        """Read only requested tactile components, retaining their native domains."""
        result = {'metadata': self.embedded_table('tactile_context', episode).to_pylist()
                  if self.manifest.get('contracts', {}).get('unified_data') else []}
        for component in components:
            name = 'tactile.' + component
            if not self.has_modality(name):
                result[component] = None
                continue
            result[component] = self.read_episode(name, episode, 'shared', (columns or {}).get(component))
        return result

    @staticmethod
    def _remember(cache: OrderedDict, key: Any, value: Any, limit: int) -> Any:
        if limit:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > limit:
                cache.popitem(last=False)
        return value

    def _index_table(self, path: str, columns: list[str] | None = None) -> pa.Table:
        self._ensure_process()
        key = path if columns is None else (path, tuple(columns))
        if path in self._indices:
            self._indices.move_to_end(path)
            table = self._indices[path]
            return table.select(columns) if columns is not None else table
        if key in self._indices:
            self._indices.move_to_end(key)
            return self._indices[key]
        return self._remember(
            # Episode/range lookups use searchsorted. One contiguous Arrow
            # buffer avoids copying the entire ID column on every lookup.
            self._indices, key, pq.ParquetFile(self.path(path)).read(columns=columns).combine_chunks(),
            self.cache_entries,
        )

    def _selected(self, shards: list[dict], episode: int | None) -> Iterable[dict]:
        for shard in shards:
            if episode is None or shard["episode_min"] <= episode <= shard["episode_max"]:
                yield shard

    def metadata(self, kind: str, episode: int | None = None,
                 columns: list[str] | None = None) -> pa.Table:
        if kind not in {"episodes", "frames"}:
            raise ValueError(kind)
        if kind=='frames' and not self.manifest['indices']['frames'] and self.manifest.get('frame_generation'):
            rule=self.manifest['frame_generation'];parts=[]
            for row in self.metadata('episodes',episode).to_pylist():
                n=int(row['num_frames']);tasks=row.get('task_ids') or [-1]
                parts.append(pa.table({'episode_index':np.full(n,row['episode_index'],np.int64),
                    'frame_index':np.arange(n,dtype=np.int64),
                    'timestamp':np.arange(n,dtype=np.float64)/float(rule['fps']),
                    'task_index':np.full(n,tasks[0],np.int64)}))
            result = pa.concat_tables(parts)
            return result.select(columns) if columns is not None else result
        tables = []
        for shard in self._selected(self.manifest["indices"][kind], episode):
            selected_columns = (list(dict.fromkeys(columns + ['episode_index']))
                                if columns is not None and episode is not None else columns)
            table = (self._index_table(shard["path"], selected_columns) if episode is None
                     else self._metadata_episode(shard['path'], episode, selected_columns))
            if episode is not None:
                table = self._episode_slice(table, "episode_index", episode)
            if columns is not None:
                table = table.select(columns)
            tables.append(table)
        if not tables:
            raise KeyError((kind, episode))
        return pa.concat_tables(tables)

    def _metadata_episode(self, path: str, episode: int, columns) -> pa.Table:
        """First sample reads only matching metadata row groups, not the split."""
        self._ensure_process()
        if path in self._indices:
            table = self._indices[path]
            return table.select(columns) if columns is not None else table
        key = (path, tuple(columns)) if columns is not None else path
        if key in self._indices:
            return self._indices[key]
        key = (path, int(episode), tuple(columns) if columns is not None else None)
        if key in self._indices:
            self._indices.move_to_end(key)
            return self._indices[key]
        file, _ = self._file(path)
        field = file.schema.names.index('episode_index')
        groups = []
        for i in range(file.num_row_groups):
            stats = file.metadata.row_group(i).column(field).statistics
            if stats is None or not stats.has_min_max or stats.min <= episode <= stats.max:
                groups.append(i)
        table = file.read_row_groups(groups, columns=columns).combine_chunks()
        return self._remember(self._indices, key, table, self.cache_entries)

    def iter_metadata(self, kind: str, batch_rows: int = 1024):
        if kind not in {'episodes','frames'}:raise ValueError(kind)
        if kind=='frames' and not self.manifest['indices']['frames']:
            yield from self.metadata('frames').to_pylist()
            return
        for shard in self.manifest['indices'][kind]:
            table=self._index_table(shard['path'])
            for batch in table.to_batches(max_chunksize=batch_rows):
                yield from batch.to_pylist()

    @staticmethod
    def _episode_slice(table: pa.Table, column: str, episode: int) -> pa.Table:
        values = table[column].to_numpy(zero_copy_only=False)
        lo = int(np.searchsorted(values, episode, side="left"))
        hi = int(np.searchsorted(values, episode, side="right"))
        return table.slice(lo, hi-lo)

    def table(self, name: str, columns: list[str] | None = None) -> pa.Table:
        declaration = self.manifest["tables"][name]
        paths = [declaration] if isinstance(declaration, str) else [s["path"] for s in declaration["shards"]]
        return pa.concat_tables([pq.ParquetFile(self.path(p)).read(columns=columns) for p in paths])

    def range_table(self, modality: str) -> pa.Table:
        """Published integer ranges, without opening any payload footer."""
        parts = [self._index_table(shard['path']) for shard in self.spec(modality)['ranges']]
        if not parts:
            return pa.Table.from_batches([], schema=RANGE_SCHEMA)
        table = pa.concat_tables(parts).sort_by([('episode', 'ascending'),
            ('stream', 'ascending'), ('start', 'ascending')]).combine_chunks()
        start = table['start'].to_numpy()
        count = table['count'].to_numpy()
        offset = table['offset'].to_numpy()
        if np.any(count <= 0) or np.any(start < 0) or np.any(offset < 0):
            raise ValueError(f'Invalid ranges: {modality}')
        if len(start) > 1:
            same = pc.and_(pc.equal(table['episode'].slice(1), table['episode'].slice(0, len(start)-1)),
                           pc.equal(table['stream'].slice(1), table['stream'].slice(0, len(start)-1))).to_numpy()
            if np.any(same & (start[1:] < start[:-1] + count[:-1])):
                raise ValueError(f'Overlapping ranges: {modality}')
        return table

    def ranges(self, modality: str, episode: int, stream: str) -> list[dict]:
        self._ensure_process()
        key = (modality, episode, stream)
        if key in self._range_cache:
            self._range_cache.move_to_end(key)
            return self._range_cache[key]
        spec = self.spec(modality)
        result = []
        import pyarrow.compute as pc
        for shard in self._selected(spec["ranges"], episode):
            table = self._episode_slice(self._index_table(shard["path"]), "episode", episode)
            result.extend(table.filter(pc.equal(table["stream"], stream)).to_pylist())
        result.sort(key=lambda row: row["start"])
        end = -1
        for row in result:
            if row["count"] <= 0 or min(row["start"], row["offset"]) < 0 or row["start"] < end:
                raise ValueError(f"Invalid/overlapping ranges: {(modality, episode, stream)}")
            end = row["start"] + row["count"]
        return self._remember(self._range_cache, key, result, self.cache_entries)

    def locate(
        self, modality: str, episode: int, stream: str, indices: Iterable[int],
    ) -> list[tuple[str, int] | None]:
        requested = np.asarray(list(indices))
        if requested.size == 0:
            return []
        if requested.ndim != 1 or requested.dtype.kind not in "iu" or np.any(requested < 0):
            raise ValueError("Indices must be nonnegative integers")
        requested = requested.astype(np.int64)
        ranges = self.ranges(modality, episode, stream)
        if not ranges:
            return [None] * len(requested)
        starts = np.asarray([r["start"] for r in ranges], dtype=np.int64)
        hits = np.searchsorted(starts, requested, side="right") - 1
        result = []
        for index, hit in zip(requested, hits):
            if hit < 0 or index >= ranges[hit]["start"] + ranges[hit]["count"]:
                result.append(None)
            else:
                row = ranges[hit]
                result.append((row["path"], int(row["offset"] + index - row["start"])))
        return result

    def _file(self, path: str) -> tuple[pq.ParquetFile, np.ndarray]:
        self._ensure_process()
        if path in self._files:
            self._files.move_to_end(path)
            return self._files[path]
        file = pq.ParquetFile(self.path(path))
        offsets = np.cumsum([0] + [
            file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)
        ], dtype=np.int64)
        return self._remember(self._files, path, (file, offsets), self.cache_entries)

    def _payload_group(self, file: pq.ParquetFile, path: str, group: int,
                       columns: list[str] | None) -> pa.Table:
        key = (path, int(group), tuple(columns) if columns is not None else None)
        if key in self._groups:
            self._groups.move_to_end(key)
            return self._groups[key]
        return self._remember(self._groups, key,
                              file.read_row_group(int(group), columns=columns),
                              self.row_group_cache_entries)

    def read_physical_rows(
        self, path: str, indices: Iterable[int], columns: list[str] | None = None,
    ) -> pa.Table:
        values = np.asarray(list(indices), dtype=np.int64)
        file, offsets = self._file(path)
        schema = file.schema_arrow
        if columns is not None:
            schema = pa.schema([schema.field(name) for name in columns])
        if len(values) == 0:
            return pa.Table.from_batches([], schema=schema)
        if values.min() < 0 or values.max() >= offsets[-1]:
            raise IndexError((path, int(values.min()), int(values.max()), int(offsets[-1])))
        groups = np.searchsorted(offsets, values, side="right") - 1
        pieces, positions = [], []
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            table = self._payload_group(file, path, int(group), columns)
            pieces.append(table.take(pa.array(values[selected] - offsets[group])))
            positions.extend(selected.tolist())
        joined = pa.concat_tables(pieces)
        return joined.take(pa.array(np.argsort(positions)))

    def read(
        self, modality: str, episode: int, stream: str, indices: Iterable[int],
        columns: list[str] | None = None,
    ) -> tuple[pa.Table, np.ndarray]:
        """Return present rows in request order and a request-length presence mask.

        Payload validity flags are retained in their own columns. The mask only
        reports existence; callers must not conflate the two.
        """
        if self.spec(modality)["storage"] != "parquet":
            raise ValueError(f"{modality} is not a Parquet modality; use locate for video")
        locations = self.locate(modality, episode, stream, indices)
        present = np.asarray([loc is not None for loc in locations], dtype=bool)
        by_file: dict[str, list[tuple[int, int]]] = {}
        for pos, location in enumerate(locations):
            if location is not None:
                by_file.setdefault(location[0], []).append((pos, location[1]))
        pieces, positions = [], []
        for path, pairs in by_file.items():
            pieces.append(self.read_physical_rows(path, [p[1] for p in pairs], columns))
            positions.extend(p[0] for p in pairs)
        if not pieces:
            spec = self.spec(modality)
            example = spec.get("schema_example_path")
            if example is None:
                raise ValueError(f"No schema source for {modality}")
            return self.read_physical_rows(example, [], columns), present
        joined = pa.concat_tables(pieces)
        return joined.take(pa.array(np.argsort(positions))), present

    def read_episode(
        self, modality: str, episode: int, stream: str, columns: list[str] | None = None,
    ) -> pa.Table:
        spec = self.spec(modality)
        if spec["storage"] != "parquet":
            raise ValueError(f"{modality} is not a Parquet modality; use locate for video")
        ranges = self.ranges(modality, episode, stream)
        if not ranges:
            return self.read(modality, episode, stream, [], columns)[0]
        pieces = []
        for row in ranges:
            path = row["path"]
            file, offsets = self._file(path)
            if columns is not None:
                for name in columns:
                    file.schema_arrow.field(name)
            cursor = int(row["offset"])
            end = cursor + int(row["count"])
            if end > offsets[-1]:
                raise IndexError((path, cursor, end - 1))
            group = int(np.searchsorted(offsets, cursor, side="right") - 1)
            while cursor < end:
                stop = min(end, int(offsets[group + 1]))
                table = self._payload_group(file, path, group, columns)
                pieces.append(table.slice(cursor - int(offsets[group]), stop - cursor))
                cursor = stop
                group += 1
        return pa.concat_tables(pieces)
