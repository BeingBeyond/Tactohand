"""Typed handcrop geometry v2 and decoding shared with legacy readers."""
from __future__ import annotations

import pyarrow as pa


CAM_INTR_TYPE = pa.struct([pa.field(name, pa.float64()) for name in ('fx', 'fy', 'cx', 'cy')])
GEOMETRY_SCHEMA_V2 = pa.schema([
    pa.field('episode_index', pa.int64(), nullable=False),
    pa.field('view_id', pa.string(), nullable=False),
    pa.field('target_id', pa.string(), nullable=False),
    pa.field('frame_index', pa.int64(), nullable=False),
    *[pa.field(name, pa.list_(pa.field('element', pa.float64()), 4), nullable=False) for name in (
        'crop_box_xyxy_orig', 'hand_bbox_xywh_orig', 'hand_bbox_xywh')],
    pa.field('frame_valid', pa.bool_(), nullable=False),
    pa.field('crop_valid', pa.bool_(), nullable=False),
    pa.field('cam_intr', CAM_INTR_TYPE),
], metadata={b'beingh.handcrop.geometry.version': b'v2',
             b'beingh.handcrop.full_frame_coordinates': b'normalized_observation',
             b'beingh.handcrop.crop_coordinates': b'crop_pixels'})

# Canonical geometry stores positions/masks only. Camera intrinsics remain in
# the observation camera table and are transformed by frame_intrinsics.
GEOMETRY_SCHEMA_V3 = GEOMETRY_SCHEMA_V2.remove(
    GEOMETRY_SCHEMA_V2.get_field_index('cam_intr')).with_metadata({
        b'beingh.handcrop.geometry.version': b'v3',
        b'beingh.handcrop.full_frame_coordinates': b'normalized_observation',
        b'beingh.handcrop.crop_coordinates': b'crop_pixels',
        b'beingh.handcrop.intrinsics': b'derived_from_observation_camera_and_crop_box'})


def decode_cam_intr(value):
    """Expose the established focal/princpt API for binary v1 and typed v2."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        import msgpack
        value = msgpack.loads(value, raw=False) if value else None
    if value is None:
        return None
    if all(name in value for name in ('fx', 'fy', 'cx', 'cy')):
        return {'focal': [value['fx'], value['fy']], 'princpt': [value['cx'], value['cy']]}
    if 'focal' not in value or 'princpt' not in value:
        raise ValueError('Unknown handcrop camera intrinsics encoding')
    return value


def typed_cam_intr(value):
    decoded = decode_cam_intr(value)
    if decoded is None:
        return None
    if set(decoded) != {'focal', 'princpt'}:
        raise ValueError('Additional camera intrinsics fields require explicit migration')
    fx, fy = decoded['focal']
    cx, cy = decoded['princpt']
    return dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def geometry_path(view: str, hand: str, file_number: int) -> str:
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', view) or hand not in ('left', 'right') or file_number < 0:
        raise ValueError('Invalid handcrop geometry stream identity')
    return (f'crop_data/hand/observation.images.{view}.{hand}/'
            f'chunk-{file_number // 1000:03d}/file-{file_number % 1000:03d}.parquet')


def reserve_geometry_path(root, view: str, hand: str) -> str:
    """Reserve a fresh stream file without overwriting a packed geometry shard."""
    import fcntl
    from pathlib import Path
    import re
    root = Path(root)
    directory = root / 'crop_data/hand' / f'observation.images.{view}.{hand}'
    with (root / 'meta/cropped_video_observations.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        numbers = []
        for path in directory.glob('chunk-*/file-*.parquet'):
            match = re.fullmatch(r'chunk-(\d+)/file-(\d+)\.parquet', path.relative_to(directory).as_posix())
            if match:
                numbers.append(int(match[1]) * 1000 + int(match[2]))
        relative = geometry_path(view, hand, max(numbers, default=-1) + 1)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.open('xb').close()
        return relative
