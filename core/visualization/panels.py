"""Panel composition and contact sheets (design §2 L3)."""
from __future__ import annotations

import cv2
import numpy as np

from core.visualization.drawing import put_text


def compose_row(panels: list[np.ndarray], *, gap: int = 6,
                header: str = "", header_scale: float = 0.5,
                footer: str = "",
                footer_color: tuple[int, int, int] = (200, 200, 255)
                ) -> np.ndarray:
    """Stack same-height BGR panels horizontally with header/footer bands."""
    height = max(panel.shape[0] for panel in panels)
    header_h = 26 if header else 0
    footer_h = 20 if footer else 0
    width = sum(panel.shape[1] for panel in panels) + gap * (len(panels) + 1)
    row = np.full((header_h + height + footer_h, width, 3), 16, np.uint8)
    x = gap
    for panel in panels:
        row[header_h:header_h + height, x:x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    if header:
        put_text(row, header, (gap + 2, 18), header_scale)
    if footer:
        put_text(row, footer, (gap + 2, header_h + height + 14), 0.38,
                 footer_color)
    return row


def contact_sheet(rows: list[np.ndarray], legend: str, *, scale: float = 0.62
                  ) -> np.ndarray:
    """Legend band + downscaled frame rows stacked into one reviewable PNG."""
    if not rows:
        raise ValueError("contact_sheet needs at least one row")
    resized = [cv2.resize(row, (int(row.shape[1] * scale),
                                int(row.shape[0] * scale)))
               for row in rows]
    width = max(row.shape[1] for row in resized)
    band = np.full((46, width, 3), 12, np.uint8)
    put_text(band, legend, (6, 18), 0.42, (0, 255, 255))
    return np.concatenate([band] + [self_pad(row, width) for row in resized])


def self_pad(row: np.ndarray, width: int) -> np.ndarray:
    if row.shape[1] >= width:
        return row
    pad = np.full((row.shape[0], width - row.shape[1], 3), 12, np.uint8)
    return np.concatenate([row, pad], axis=1)
