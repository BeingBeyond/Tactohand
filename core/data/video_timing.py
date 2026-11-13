"""Verified physical-CFR metadata for TorchCodec's cached timestamp reader."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

from core.data.processed_compat import reader_for


def published_cfr_grid(path: str | Path) -> tuple[Fraction, int] | None:
    """Return the file's FPS/count only for published full-video CFR streams.

    Crop/legacy/unpublished streams retain exact ordinal decoding. A stream
    claiming the CFR contract but violating its header invariants is an error.
    No frame scan, decoded-frame cache or on-disk derived index is created.
    """
    path = Path(path).resolve()
    root = next((parent for parent in path.parents
                 if (parent / 'meta/info.json').is_file()), None)
    if root is None:
        return None
    relative = path.relative_to(root).parts
    # Unified processed payloads use data/video; historical layouts use
    # videos. Crop streams (data/handcrop_video) have a separate contract.
    if relative[:2] != ('data', 'video') and relative[0] != 'videos':
        return None
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if info.get('format_version') != 'processed_dataset_v1':
        return None
    semantics = reader_for(root).spec('video').get('semantics', {})
    if semantics.get('physical_timestamp_layout', {}).get('version') != 'physical_video_cfr_v1':
        return None
    if semantics.get('video_encoding', {}).get('b_frames') != 0:
        raise ValueError(f'CFR publication must declare zero B frames: {path}')

    from core.io.torchcodec_cfr import verified_cfr_grid
    return verified_cfr_grid(path)
