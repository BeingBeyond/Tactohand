"""Versioned, lossless static object registries (JSON) and legacy Parquet reads."""
from __future__ import annotations

import base64
from functools import lru_cache
import json
import math
import numpy as np
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = 'object_assets_v2'
LEGACY_FORMAT = 'object_assets_v1'
FLOAT_TAG = '__object_assets_float__'
SOURCE_ATTRIBUTES = pa.struct([
    ('category', pa.string()), ('label', pa.int64()), ('label_semantics', pa.string()),
    ('attributes_json', pa.string()), ('extensions_json', pa.string()),
])
ASSET_SCHEMA = pa.schema([
    pa.field('object_asset_id', pa.string(), nullable=False),
    pa.field('object_geometry_id', pa.string(), nullable=False),
    pa.field('name', pa.string(), nullable=False),
    pa.field('topology_type', pa.string(), nullable=False),
    pa.field('source_dataset', pa.string(), nullable=False),
    pa.field('source_id', pa.string(), nullable=False),
    ('semantic_class_id', pa.int64()), ('semantic_class_namespace', pa.string()),
    ('visual_asset_path', pa.string()), ('visual_asset_format', pa.string()),
    ('visual_asset_sha256', pa.string()),
    pa.field('source_attributes', SOURCE_ATTRIBUTES, nullable=False),
])


def canonical_assets(table: pa.Table, dataset: str) -> pa.Table:
    """Separate source labels from stable asset identity without renumbering."""
    if 'source_attributes' in table.column_names:
        return table.cast(ASSET_SCHEMA)
    known = {'object_asset_id', 'object_geometry_id', 'name', 'topology_type', 'source_id',
             'category', 'label', 'attributes_json', 'visual_asset_path',
             'visual_asset_format', 'visual_asset_sha256'}
    rows = []
    for original in table.to_pylist():
        label = original.get('label')
        semantic = dataset == 'mow' and label is not None and label >= 0
        rows.append(dict(object_asset_id=original['object_asset_id'],
            object_geometry_id=original['object_geometry_id'], name=original['name'],
            topology_type=original.get('topology_type') or 'rigid', source_dataset=dataset,
            source_id=original['source_id'], semantic_class_id=label if semantic else None,
            semantic_class_namespace='mow:object_names' if semantic else None,
            visual_asset_path=original.get('visual_asset_path'),
            visual_asset_format=original.get('visual_asset_format'),
            visual_asset_sha256=original.get('visual_asset_sha256'),
            source_attributes=dict(category=original.get('category'), label=label,
                label_semantics='semantic_class_id' if semantic else 'source_defined',
                attributes_json=original.get('attributes_json'),
                extensions_json=json.dumps(encode_values({k: v for k, v in original.items() if k not in known}),
                                           ensure_ascii=False, allow_nan=False))))
    return pa.Table.from_pylist(rows, schema=ASSET_SCHEMA)


class AssetIndex:
    """Exact string asset-to-geometry resolution shared by all consumers."""

    def __init__(self, table: pa.Table):
        self.assets = {}
        for row in table.to_pylist():
            key = row['object_asset_id']
            if not isinstance(key, str) or not key or key in self.assets:
                raise ValueError(f'Duplicate/invalid asset identity: {key!r}')
            self.assets[key] = row

    def asset(self, object_asset_id: str) -> dict:
        if not isinstance(object_asset_id, str):
            raise TypeError('Asset lookup requires an exact object_asset_id string')
        return self.assets[object_asset_id]

    def geometry_id(self, object_asset_id: str) -> str:
        return self.asset(object_asset_id)['object_geometry_id']


def encode_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {FLOAT_TAG: 'nan' if math.isnan(value) else ('inf' if value > 0 else '-inf')}
    if isinstance(value, dict):
        if FLOAT_TAG in value:
            raise ValueError('Reserved object asset encoding key')
        return {key: encode_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_values(item) for item in value]
    return value


def _decode(value):
    if set(value) == {FLOAT_TAG}:
        return float(value[FLOAT_TAG])
    return value


def registry(tables: dict[str, pa.Table], source_dataset: str | None = None) -> dict:
    """Keep rows human-readable, with source schemas for exact typed round trips."""
    tables = dict(tables)
    original_schema = None
    if 'assets' in tables and source_dataset is not None:
        original_schema = base64.b64encode(tables['assets'].schema.serialize().to_pybytes()).decode()
        tables['assets'] = canonical_assets(tables['assets'], source_dataset)
    version = FORMAT if 'assets' in tables and 'source_attributes' in tables['assets'].column_names else LEGACY_FORMAT
    return dict(format_version=version, mesh_store='meshes.bin',
                source_asset_schema=original_schema,
                identity_contract=dict(asset_key='object_asset_id', geometry_key='object_geometry_id',
                    semantic_key=['semantic_class_namespace', 'semantic_class_id'],
                    source_labels='provenance only; never mesh lookup keys'),
                binary_layout=dict(vertices_dtype='<f4', faces_dtype='<i4', offset_unit='byte'),
                tables={name: encode_values(table.to_pylist()) for name, table in tables.items()},
                arrow_schemas={name: base64.b64encode(table.schema.serialize().to_pybytes()).decode()
                               for name, table in tables.items()})


@lru_cache(maxsize=32)
def _load(path: str, mtime_ns: int, size: int) -> dict:
    value = json.loads(Path(path).read_text(), object_hook=_decode)
    if value.get('format_version') not in {FORMAT, LEGACY_FORMAT}:
        raise ValueError(f'Unknown object asset format: {path}')
    return value


def read_asset_table(path: Path, section: str, columns: list[str] | None = None) -> pa.Table:
    path = Path(path)
    if path.suffix != '.json':
        return pq.read_table(path, columns=columns)
    stat = path.stat()
    value = _load(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value['arrow_schemas'][section])))
    table = pa.Table.from_pylist(value['tables'][section], schema=schema)
    return table.select(columns) if columns is not None else table


def read_asset_schema(path: Path, section: str) -> pa.Schema:
    return read_asset_table(path, section).schema


def instance_geometry_ids(root: Path) -> dict[str, str]:
    """Authoritative string instance -> asset -> geometry join for all readers."""
    from core.data import processed_compat as contract
    instances = contract.object_instances(root).select(
        ['object_instance_id', 'object_asset_id']).to_pylist()
    assets = AssetIndex(read_asset_table(contract.object_path(
        root, 'assets_path', 'objects/assets.parquet'), 'assets',
        columns=['object_asset_id', 'object_geometry_id']))
    return {str(row['object_instance_id']): assets.geometry_id(str(row['object_asset_id']))
            for row in instances}


@lru_cache(maxsize=16)
def load_geometry(root_string: str, geometry_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one exact geometry key, without eagerly materializing a dataset."""
    from core.data import processed_compat as contract
    root = Path(root_string)
    table = read_asset_table(contract.object_path(root, 'geometries_path',
        'objects/geometries.parquet'), 'geometries', columns=[
        'object_geometry_id', 'verts_offset', 'verts_count', 'faces_offset', 'faces_count'])
    rows = [row for row in table.to_pylist() if str(row['object_geometry_id']) == geometry_id]
    if len(rows) != 1:
        raise ValueError(f'non-unique/missing geometry key {geometry_id!r}')
    row = rows[0]
    with contract.object_path(root, 'mesh_store_path', 'objects/mesh_store.bin').open('rb') as stream:
        stream.seek(int(row['verts_offset']))
        vertices = np.frombuffer(stream.read(int(row['verts_count']) * 12), '<f4').reshape(-1, 3).copy()
        stream.seek(int(row['faces_offset']))
        faces = np.frombuffer(stream.read(int(row['faces_count']) * 12), '<i4').reshape(-1, 3).copy()
    return vertices, faces
