"""Shared cached video reading and GPU decoding, independent of dataset schemas."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

try:
    from torchcodec.decoders import VideoDecoder as _TorchCodecDecoder
    _HAS_TORCHCODEC = True
except Exception:
    _HAS_TORCHCODEC = False

try:
    import decord
    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

_VIDEO_BACKENDS = []
if _HAS_TORCHCODEC:
    _VIDEO_BACKENDS.append("torchcodec")
if _HAS_DECORD:
    _VIDEO_BACKENDS.append("decord")
_VIDEO_BACKENDS.append("cv2")


def _resolve_video_backend(requested: Optional[str]) -> str:
    """Pick a backend: honour explicit request, else auto-detect."""
    if requested is not None:
        req = requested.lower()
        if req == "torchcodec" and not _HAS_TORCHCODEC:
            raise ImportError("torchcodec requested but not available")
        if req == "decord" and not _HAS_DECORD:
            raise ImportError("decord requested but not available")
        return req
    return _VIDEO_BACKENDS[0]


class VideoReader:
    """Lazy video decoder with LRU cache of open handles.

    Parameters
    ----------
    video_paths : dict mapping video_id -> Path
    cache_size : int
        Max open video handles kept in LRU cache.
    image_mode : str
        ``"rgb"`` or ``"bgr"``.
    backend : str or None
        ``"torchcodec"``, ``"decord"``, ``"cv2"``, or ``None`` (auto).
    cfr_grid_provider : callable or None
        Optional published timing lookup; absent metadata uses exact decoding.
    """

    def __init__(self, video_paths: Dict[int, Path],
                 cache_size: int = 4, image_mode: str = "rgb",
                 backend: Optional[str] = None,
                 device: Optional[str] = None,
                 torchcodec_seek_mode: str = "exact",
                 torchcodec_num_ffmpeg_threads: int = 1,
                 cfr_grid_provider: Optional[Callable[[str], tuple | None]] = None):
        self._cache_size = cache_size
        self._image_mode = image_mode
        self._backend = _resolve_video_backend(backend)
        self._device = device
        self._torchcodec_seek_mode = str(torchcodec_seek_mode)
        if self._torchcodec_seek_mode not in {"exact", "approximate", "cfr_timestamp"}:
            raise ValueError(f"Unsupported TorchCodec seek mode: {torchcodec_seek_mode}")
        self._torchcodec_num_ffmpeg_threads = max(1, int(torchcodec_num_ffmpeg_threads))
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._timestamp_grids: dict[int, tuple] = {}
        self._cfr_grid_provider = cfr_grid_provider
        self._paths = video_paths
        # When torchcodec+cuda, keep frames as GPU tensors
        self._returns_tensor = (
            self._backend == "torchcodec" and device is not None and "cuda" in device
        )

    @property
    def returns_tensor(self) -> bool:
        """True if read_frame/read_frames return torch.Tensor (on GPU)."""
        return self._returns_tensor

    def _open(self, video_id: int) -> Any:
        if video_id in self._cache:
            self._cache.move_to_end(video_id)
            return self._cache[video_id]
        path = str(self._paths[video_id])
        if self._backend == "torchcodec":
            grid = None
            mode = self._torchcodec_seek_mode
            if mode == "cfr_timestamp":
                grid = (self._cfr_grid_provider(path)
                        if self._cfr_grid_provider is not None else None)
                mode = "approximate" if grid is not None else "exact"
            tc_kwargs = {
                "dimension_order": "NHWC",
                "seek_mode": mode,
                "num_ffmpeg_threads": self._torchcodec_num_ffmpeg_threads,
            }
            if self._device is not None:
                tc_kwargs["device"] = self._device
            handle = _TorchCodecDecoder(path, **tc_kwargs)
            if grid is not None:
                self._timestamp_grids[video_id] = grid
        elif self._backend == "decord":
            handle = decord.VideoReader(path, ctx=decord.cpu(0))
        else:
            handle = cv2.VideoCapture(path)
            if not handle.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
        self._cache[video_id] = handle
        self._cache.move_to_end(video_id)
        while len(self._cache) > self._cache_size:
            old_id, old = self._cache.popitem(last=False)
            self._timestamp_grids.pop(old_id, None)
            if self._backend == "cv2" and isinstance(old, cv2.VideoCapture):
                old.release()
        return handle

    def _read_timestamp_frames(self, video_id: int, handle, indices):
        rate, count = self._timestamp_grids[video_id]
        if any(index < 0 or index >= count for index in indices):
            raise IndexError(f"CFR frame index outside [0, {count}): {indices}")
        # Frame centers avoid float rounding onto the previous frame boundary.
        seconds = [(int(index) + 0.5) / float(rate) for index in indices]
        frames = handle.get_frames_played_at(seconds)
        ordinals = frames.pts_seconds.cpu().numpy() * float(rate)
        if not np.allclose(ordinals, indices, rtol=0, atol=1e-5):
            raise ValueError(f"CFR timestamp lookup returned incorrect frame PTS: {self._paths[video_id]}")
        return frames.data

    def read_frame(self, video_id: int, frame_idx: int) -> Union[np.ndarray, torch.Tensor]:
        handle = self._open(video_id)
        if self._backend == "torchcodec":
            frame_t = (self._read_timestamp_frames(video_id, handle, [int(frame_idx)])[0]
                       if video_id in self._timestamp_grids else handle.get_frame_at(frame_idx).data)
            if self._returns_tensor:
                if self._image_mode == "bgr":
                    frame_t = frame_t.flip(-1).contiguous()
                return frame_t
            frame = frame_t.cpu().numpy()
            if self._image_mode == "bgr":
                frame = frame[..., ::-1].copy()
        elif self._backend == "decord":
            frame = handle[frame_idx].asnumpy()  # RGB
            if self._image_mode == "bgr":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            handle.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = handle.read()
            if not ret:
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} from video {video_id}")
            if self._image_mode == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def read_frames(self, video_id: int, frame_indices: List[int]) -> Union[np.ndarray, torch.Tensor]:
        """Read multiple frames. Returns (N, H, W, 3) ndarray or tensor.

        Indices are clamped to the decoded stream's frame range: label
        timelines occasionally reference one or two frames past a packed
        video tail (rate-conversion rounding), and the static edge-hold
        semantics make the last frame the correct substitute.
        """
        handle = self._open(video_id)
        frame_indices = [int(index) for index in frame_indices]
        grid = self._timestamp_grids.get(video_id)
        total = grid[1] if grid else getattr(getattr(handle, "metadata", None), "num_frames", None)
        if isinstance(total, int) and total > 0:
            frame_indices = [min(max(index, 0), total - 1) for index in frame_indices]
        if self._backend == "torchcodec":
            # Decode ascending unique positions, then restore the exact
            # request order (including held/padded duplicates).
            ordered, inverse = np.unique(frame_indices, return_inverse=True)
            positions = ordered.tolist()
            batch_t = (self._read_timestamp_frames(video_id, handle, positions)
                       if grid else handle.get_frames_at(positions).data)
            batch_t = batch_t[torch.as_tensor(inverse, device=batch_t.device)]
            if self._returns_tensor:
                if self._image_mode == "bgr":
                    batch_t = batch_t.flip(-1).contiguous()
                return batch_t
            batch = batch_t.cpu().numpy()
            if self._image_mode == "bgr":
                batch = batch[..., ::-1].copy()
            return batch
        if self._backend == "decord" and isinstance(handle, decord.VideoReader):
            frames = handle.get_batch(frame_indices).asnumpy()
            if self._image_mode == "bgr":
                frames = frames[..., ::-1].copy()
            return frames
        # cv2 fallback
        out = []
        for fi in frame_indices:
            out.append(self.read_frame(video_id, fi))
        return np.stack(out, axis=0)

    def close(self):
        for handle in self._cache.values():
            if self._backend == "cv2" and isinstance(handle, cv2.VideoCapture):
                handle.release()
        self._cache.clear()
        self._timestamp_grids.clear()


class GPUVideoDecoder:
    """Decodes video frames on GPU using torchcodec.

    Used in deferred mode: DataLoader workers load annotations and return
    frame locations, then this decoder batch-decodes on GPU in the main
    process.  Achieves 2-3x throughput vs CPU-in-workers.

    Parameters
    ----------
    video_paths : dict mapping video_id -> Path
    cache_size : int
        Max open VideoDecoder handles kept in LRU cache.
    device : str
        CUDA device, e.g. ``"cuda"`` or ``"cuda:0"``.
    """

    def __init__(self, video_paths: Dict[int, Path],
                 cache_size: int = 16, device: str = "cuda",
                 torchcodec_seek_mode: str = "exact",
                 torchcodec_num_ffmpeg_threads: int = 1):
        if not _HAS_TORCHCODEC:
            raise ImportError("GPUVideoDecoder requires torchcodec")
        self._paths = video_paths
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._cache_size = cache_size
        self._device = device
        self._torchcodec_seek_mode = str(torchcodec_seek_mode)
        self._torchcodec_num_ffmpeg_threads = max(1, int(torchcodec_num_ffmpeg_threads))

    def _evict_one(self) -> None:
        """Evict one cached decoder handle (LRU)."""
        if not self._cache:
            return
        _, dec = self._cache.popitem(last=False)
        close_fn = getattr(dec, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    def _open(self, video_id: int) -> Any:
        if video_id in self._cache:
            self._cache.move_to_end(video_id)
            return self._cache[video_id]

        # Open with retry: if CUDA HW decoder resources are temporarily
        # exhausted, evict stale decoder handles and retry.
        while True:
            try:
                dec = _TorchCodecDecoder(
                    str(self._paths[video_id]),
                    device=self._device,
                    dimension_order="NHWC",
                    seek_mode=self._torchcodec_seek_mode,
                    num_ffmpeg_threads=self._torchcodec_num_ffmpeg_threads,
                )
                self._cache[video_id] = dec
                self._cache.move_to_end(video_id)
                while len(self._cache) > self._cache_size:
                    self._evict_one()
                return dec
            except RuntimeError as e:
                msg = str(e)
                is_hw_open_error = (
                    "Failed to create specified HW device" in msg
                    or "Unknown error occurred" in msg
                )
                if is_hw_open_error and len(self._cache) > 0:
                    self._evict_one()
                    continue
                raise

    @staticmethod
    def _should_decode_dense_range(indices: List[int]) -> bool:
        """Prefer sequential range decode for locally dense frame requests.

        TorchCodec's sparse ``get_frames_at`` is very slow for exact random
        access on our sequential clip batches, while decoding a short dense
        range and gathering on GPU is much faster.
        """
        if not indices:
            return False
        span = indices[-1] - indices[0] + 1
        return span <= 2048 and span <= max(256, len(indices) * 8)

    def _get_frames_for_indices(self, dec: Any, indices: List[int]) -> torch.Tensor:
        """Return frames aligned with sorted unique ``indices``."""
        if self._should_decode_dense_range(indices):
            start = indices[0]
            stop = indices[-1] + 1
            dense = dec.get_frames_in_range(start, stop, 1).data
            local = torch.tensor(
                [idx - start for idx in indices],
                device=dense.device,
                dtype=torch.long,
            )
            return dense.index_select(0, local)
        return dec.get_frames_at(indices).data

    def decode_locations(self, locations: List[Tuple[int, int]]) -> torch.Tensor:
        """Decode frame locations into a (N, H, W, 3) GPU tensor.

        Parameters
        ----------
        locations : list of (video_id, frame_idx) tuples
        """
        # Group by video shard for batch decode
        groups: Dict[int, List[Tuple[int, int]]] = {}
        for pos, (vid_id, frame_idx) in enumerate(locations):
            groups.setdefault(vid_id, []).append((pos, frame_idx))

        frames: list = [None] * len(locations)
        for vid_id, items in groups.items():
            dec = self._open(vid_id)
            frame_to_positions: Dict[int, List[int]] = {}
            for pos, frame_idx in items:
                frame_to_positions.setdefault(frame_idx, []).append(pos)
            indices = sorted(frame_to_positions)
            batch = self._get_frames_for_indices(dec, indices)
            for i, frame_idx in enumerate(indices):
                frame = batch[i]
                for pos in frame_to_positions[frame_idx]:
                    frames[pos] = frame
        return torch.stack(frames)  # (N, H, W, 3) on GPU

    def decode_locations_batch(
        self, locations_batch: List[List[Tuple[int, int]]]
    ) -> List[torch.Tensor]:
        """Decode a batch of clips while grouping frame requests by video.

        This reduces Python and TorchCodec overhead versus decoding each clip
        separately, especially when many samples in a batch come from the same
        video shard.
        """
        frames: List[List[Any]] = [
            [None] * len(locations) for locations in locations_batch
        ]
        groups: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
        for sample_idx, locations in enumerate(locations_batch):
            for frame_pos, (vid_id, frame_idx) in enumerate(locations):
                groups.setdefault(vid_id, {}).setdefault(frame_idx, []).append(
                    (sample_idx, frame_pos)
                )

        for vid_id, frame_to_positions in groups.items():
            dec = self._open(vid_id)
            indices = sorted(frame_to_positions)
            batch = self._get_frames_for_indices(dec, indices)
            for i, frame_idx in enumerate(indices):
                frame = batch[i]
                for sample_idx, frame_pos in frame_to_positions[frame_idx]:
                    frames[sample_idx][frame_pos] = frame

        return [torch.stack(sample_frames) for sample_frames in frames]

    def decode_batch(self, locations_batch: List[List[Tuple[int, int]]],
                     ) -> torch.Tensor:
        """Decode a batch of frame location lists.

        Parameters
        ----------
        locations_batch : list of B location lists, each with T entries

        Returns
        -------
        torch.Tensor of shape (B, T, H, W, 3) on GPU, uint8 RGB
        """
        clips = self.decode_locations_batch(locations_batch)
        return torch.stack(clips)  # (B, T, H, W, 3)

    def close(self):
        self._cache.clear()
