"""Cross-process GPU Stage-1 slots for mixed GPU/CPU annotation pipelines."""
from __future__ import annotations

import atexit
import fcntl
import os
from pathlib import Path
import time


class GpuStageLease:
    def __init__(self, stream: object | None, path: Path | None) -> None:
        self._stream = stream
        self.path = path
        if stream is not None:
            atexit.register(self.release)

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    def release(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "GpuStageLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def acquire_gpu_stage1_lease(*, poll_interval_s: float = 0.05) -> GpuStageLease:
    """Acquire one configured slot for the worker's physical GPU.

    The feature is disabled unless both ``STAGE1_GPU_LOCK_ROOT`` and a positive
    ``STAGE1_WORKERS_PER_GPU`` are exported. ``BEINGH_PHYSICAL_GPU`` is set by
    the unified scheduler before CUDA devices are remapped for child isolation.
    """
    root_value = os.environ.get("STAGE1_GPU_LOCK_ROOT", "")
    slots = int(os.environ.get("STAGE1_WORKERS_PER_GPU", "0"))
    if not root_value or slots <= 0:
        return GpuStageLease(None, None)
    physical_gpu = os.environ.get("BEINGH_PHYSICAL_GPU", "0")
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"gpu_{physical_gpu}_slot_{slot:02d}.lock" for slot in range(slots)]
    while True:
        for path in paths:
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                stream.seek(0)
                stream.truncate()
                stream.write(f"pid={os.getpid()}\n")
                stream.flush()
                return GpuStageLease(stream, path)
            except BlockingIOError:
                stream.close()
        time.sleep(poll_interval_s)


def acquire_cpu_stage2_lease(*, poll_interval_s: float = 0.05) -> GpuStageLease:
    """Acquire one global CPU Stage-2 slot when pipeline limiting is enabled."""
    root_value = os.environ.get("STAGE2_CPU_LOCK_ROOT", "")
    slots = int(os.environ.get("STAGE2_CPU_SLOTS", "0"))
    if not root_value or slots <= 0:
        return GpuStageLease(None, None)
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"stage2_slot_{slot:03d}.lock" for slot in range(slots)]
    while True:
        for path in paths:
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                stream.seek(0)
                stream.truncate()
                stream.write(f"pid={os.getpid()}\n")
                stream.flush()
                return GpuStageLease(stream, path)
            except BlockingIOError:
                stream.close()
        time.sleep(poll_interval_s)
