"""Shared full-frame image preprocessing for TactoHand datasets."""
from __future__ import annotations

import cv2
import numpy as np


def letterbox_square(image: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize centered on a black square canvas."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError(f"invalid letterbox geometry: image={image.shape}, size={size}")
    scale = min(float(size) / width, float(size) / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def edge_hold(array: np.ndarray, total: int, left: int, right: int):
    """Extend axis 0 to ``total`` by repeating the edge rows (the exact
    padding semantics the reader applies to the batch label tensors)."""
    if left:
        array = np.concatenate(
            [np.repeat(array[:1], left, axis=0), array], axis=0)
    if right:
        array = np.concatenate(
            [array, np.repeat(array[-1:], right, axis=0)], axis=0)
    return array
