"""H.264 video encoding via ffmpeg pipes (design §1, §2 L3)."""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np


def encoder(path: Path, width: int, height: int, *, fps: float,
            preset: str = "fast", crf: int = 18) -> subprocess.Popen:
    """Open an H.264 rawvideo pipe; caller writes BGR frames then waits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", preset,
         "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags",
         "+faststart", str(path)], stdin=subprocess.PIPE)


def encode_frames(path: Path, frames: np.ndarray, *, fps: float,
                  preset: str = "fast", crf: int = 18) -> None:
    """Encode a (T, H, W, 3) BGR stack straight to an H.264 mp4."""
    frames = np.asarray(frames, np.uint8)
    process = encoder(path, frames.shape[2], frames.shape[1], fps=fps,
                      preset=preset, crf=crf)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode,
                                            process.args)


def transcode_h264(video_path: Path, *, crf: int = 20) -> None:
    """Re-encode any video file to H.264 in place (yuv420p, faststart)."""
    video_path = Path(video_path)
    temporary = video_path.with_suffix(".pre_h264.mp4")
    video_path.rename(temporary)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary),
             "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(video_path)], check=True)
    finally:
        temporary.unlink(missing_ok=True)
