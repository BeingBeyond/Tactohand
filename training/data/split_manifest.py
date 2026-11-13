"""Versioned sequence-level split overlays for prepared training data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.data import processed_compat as contract


def load_manifest(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text())
    if document.get('version') != 1:
        raise ValueError('Unsupported split manifest version')
    for name, mapping in document['datasets'].items():
        if not mapping or set(mapping.values()) - {'train', 'val', 'test'}:
            raise ValueError(f'Invalid split mapping for {name}')
    return document


def select_manifest_episodes(root: Path, path: str, split: str) -> set[int] | None:
    """Validate complete sequence coverage before overriding installed splits."""
    document = load_manifest(path)
    name = root.name.removesuffix('_lerobot')
    mapping = document['datasets'].get(name)
    if mapping is None:
        return None
    rows = contract.reader_for(root).metadata(
        'episodes', columns=['episode_index', 'sequence_id']).to_pylist()
    names = [str(row['sequence_id']) for row in rows]
    if len(set(names)) != len(names) or set(names) != set(mapping):
        raise ValueError(f'{name}: installed sequences differ from split manifest')
    if split not in {'train', 'val', 'test', 'all'}:
        raise ValueError(f'Unsupported manifest split {split}')
    return {int(row['episode_index']) for row in rows
            if split == 'all' or mapping[str(row['sequence_id'])] == split}


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
