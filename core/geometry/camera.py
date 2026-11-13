"""Camera value types and crop projection math."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class CameraIntrinsics:
    """OpenCV pinhole intrinsics; attribute access matches the legacy
    ``camera`` rows consumed by ``project_camera``."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_row(cls, row: Any) -> "CameraIntrinsics":
        return cls(float(row.fx), float(row.fy), float(row.cx),
                   float(row.cy))


def crop_intrinsics(intrinsics: CameraIntrinsics,
                    box_xyxy: tuple[float, float, float, float],
                    target_size: float) -> CameraIntrinsics:
    """Intrinsics for a resized axis-aligned crop (handcrop-style panels).

    ``box_xyxy`` is the crop in source pixels; the crop is resized to
    ``(target_size, target_size)``.
    """
    x1, y1, x2, y2 = box_xyxy
    scale_x = target_size / max(1.0, x2 - x1)
    scale_y = target_size / max(1.0, y2 - y1)
    return CameraIntrinsics(intrinsics.fx * scale_x, intrinsics.fy * scale_y,
                            (intrinsics.cx - x1) * scale_x,
                            (intrinsics.cy - y1) * scale_y)
