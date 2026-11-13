"""Shared color maps for labels/contact/proximity/force (design §2 L0)."""
from __future__ import annotations

import cv2
import numpy as np

FORCE_COLOR = (0, 255, 255)
TORQUE_COLOR = (255, 80, 255)
SUPPORT_COLOR = (255, 255, 0)

_TURBO_LUT = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
# Target/object resultant force uses a warm map so it cannot be confused with
# the per-MANO-vertex Turbo arrows.
_TARGET_FORCE_LUT = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_HOT)[:, 0]
_PROXIMITY_STOPS = np.array(
    [[0, 0, 255], [0, 96, 255], [128, 200, 0], [255, 160, 0], [255, 80, 0]],
    np.float32)  # 0 / 4cm / 12cm / 25cm clip, BGR
_FORCE_STOPS = np.array(
    [[70, 70, 70], [0, 200, 200], [0, 255, 255], [0, 0, 255]],
    np.float32)  # 0 → scale, gray→cyan→yellow→red, BGR


def _lerp_stops(stops: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    low = np.floor(x).astype(int)
    frac = (x - low)[..., None]
    return (stops[low] * (1 - frac)
            + stops[np.minimum(low + 1, len(stops) - 1)] * frac
            ).astype(np.uint8)


def turbo_color(ratio: float) -> tuple[int, int, int]:
    """Turbo BGR color without one OpenCV kernel launch per arrow."""
    color = _TURBO_LUT[int(np.clip(round(ratio * 255), 0, 255))]
    return int(color[0]), int(color[1]), int(color[2])


def target_force_color(ratio: float) -> tuple[int, int, int]:
    color = _TARGET_FORCE_LUT[int(np.clip(round(ratio * 255), 0, 255))]
    return int(color[0]), int(color[1]), int(color[2])


def proximity_color(distance_m: np.ndarray) -> np.ndarray:
    """Near-contact red → 10 mm orange → ~12 cm green → far blue (BGR, uint8)."""
    return _lerp_stops(_PROXIMITY_STOPS,
                       np.asarray(distance_m, np.float64) / 0.25)


def force_color(magnitude: np.ndarray, scale: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Force-magnitude map; returns ``(colors, radii)`` for vertex clouds."""
    t = np.clip(np.asarray(magnitude, np.float64)
                / max(float(scale), 1e-9), 0.0, 1.0)
    return (_lerp_stops(_FORCE_STOPS, t),
            1 + (t * 2.2).astype(int))
