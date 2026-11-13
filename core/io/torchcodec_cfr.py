"""Validate a file header against an explicit physical-CFR contract."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path


def verified_cfr_grid(path: str | Path) -> tuple[Fraction, int] | None:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        if stream.codec_context.has_b_frames or stream.start_time not in (0, None):
            raise ValueError(f'CFR stream must have zero-origin PTS and no B frames: {path}')
        if rate is None or rate <= 0 or not stream.frames:
            raise ValueError(f'CFR stream requires rational FPS and frame count: {path}')
        rate = Fraction(rate)
        if (Fraction(1, 1) / rate / stream.time_base).denominator != 1:
            raise ValueError(f'CFR stream requires an integral PTS grid: {path}')
        return rate, int(stream.frames)
