"""Shared utilities, loaded on demand so optional workflows stay optional."""
from importlib import import_module


_EXPORTS = {
    "create_video_ffmpeg": "io.img2video",
    "batch_encode_videos_gpu": "io.img2video",
    "ensure_mesh_store": "geometry.mesh_store",
    "load_mesh": "geometry.mesh_store",
    "write_extra_parquet": "io.utils_extra_parquet",
    "StreamingParquetWriter": "io.utils_parquet_streaming",
    "build_seq_row_ranges": "io.utils_seq_index",
    "build_seq_video_ranges": "io.utils_seq_index",
    "write_seq_row_index": "io.utils_seq_index",
    "write_seq_video_ranges": "io.utils_seq_index",
    "build_video_index": "io.utils_video_index",
    "compute_global_frame_idx": "io.utils_video_index",
    "locate_video_frame": "io.utils_video_index",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
