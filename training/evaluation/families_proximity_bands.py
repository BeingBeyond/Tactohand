"""Proximity distance-band family: metrics restricted to GT distance ranges.

Answers "what is the error within [a, b) metres of true contact distance?"
in ONE pass: the family accumulates per-band sums while every window
streams through, so sweeping band edges costs nothing extra at eval time.

Bands are defined on the DECODED GT distance (``BatchView.distance_mm_gt``,
which saturates at the label clip bound ``PROXIMITY_CLIP_M``), so the last
band is open-ended and absorbs everything beyond the clip.  An entry enters
exactly one band: the first whose upper bound exceeds its GT distance.
Bands whose upper bound is below the clip are half-open ``[a, b)``; the
final band is ``[a, inf)``.

Reported per band: ADE (mm), bias (pred-gt, mm, sign shows over/under-
estimation), RMSE (mm), and the share of supervised slots in the band.
Select via ``--metrics proximity_bands``; custom edges via
``--band-edges 0 0.005 0.01 0.025 0.05 0.25`` (metres, ascending; default:
contact / 5 mm / 1 cm / 2.5 cm / 5 cm / clip).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from training.evaluation.registry import MetricFamily, Sums, register
from training.evaluation.view import BatchView, MetricSpec

DEFAULT_BAND_EDGES = (0.0, 0.005, 0.010, 0.025, 0.050, float("inf"))


def normalize_band_edges(edges: Optional[Sequence[float]]) -> Tuple[float, ...]:
    """Validate CLI band edges (metres, ascending) and append +inf."""
    values = tuple(float(e) for e in (edges or DEFAULT_BAND_EDGES))
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"band edges must be strictly ascending: {values}")
    if values[0] < 0.0:
        raise ValueError(f"band edges must be non-negative: {values}")
    if math.isinf(values[-1]):
        return values
    return values + (float("inf"),)


def band_label(edges: Sequence[float], index: int) -> str:
    low, high = edges[index], edges[index + 1]
    high_text = "inf" if math.isinf(high) else f"{high * 1000:.0f}"
    return f"{low * 1000:.0f}-{high_text}"


@register("proximity_bands")
class ProximityBandsFamily(MetricFamily):
    """Per-band ADE / bias / RMSE over GT distance ranges (one pass)."""

    def __init__(self, edges: Optional[Sequence[float]] = None) -> None:
        self._sums = Sums()
        self.edges = normalize_band_edges(edges)

    @staticmethod
    def specs() -> List[MetricSpec]:
        # Band keys are dynamic (depend on the edges); specs() only carries
        # the shared note — _report tolerates missing specs with a blank one.
        return [
            MetricSpec("proximity_bands/note", "",
                       note="per GT-distance-band errors; bands are "
                            "[a,b) mm, last is open-ended"),
        ]

    def update(self, view: BatchView) -> None:
        # Full-window tensors [T, 2, 778]; valid = supervised & finite GT.
        valid = (view.proximity_mask.bool().expand_as(view.distance_mm_gt)
                 & torch.isfinite(view.distance_mm_gt))
        gt = view.distance_mm_gt
        err = (view.distance_mm_hat - gt).abs()
        signed = view.distance_mm_hat - gt
        # Edges are metres (CLI unit), the view tensors are millimetres.
        edges_mm = [e * 1000.0 for e in self.edges]
        for b in range(len(edges_mm) - 1):
            mask = valid & (gt >= edges_mm[b]) & (gt < edges_mm[b + 1])
            self._sums.add(f"band/{b}/abs", (err * mask).sum(), mask.sum())
            self._sums.add(f"band/{b}/signed", (signed * mask).sum(),
                           mask.sum())
            self._sums.add(f"band/{b}/sq", (err.square() * mask).sum(),
                           mask.sum())
            self._sums.add(f"band/{b}/share", mask.sum(), 1.0)
        self._sums.add("band/total", valid.sum(), 1.0)

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        total = self._sums.sum("band/total")
        for b in range(len(self.edges) - 1):
            label = band_label(self.edges, b)
            count = self._sums.sum(f"band/{b}/share")
            out[f"proximity_bands/{label}/ade_mm"] = \
                self._sums.mean(f"band/{b}/abs")
            out[f"proximity_bands/{label}/bias_mm"] = \
                self._sums.mean(f"band/{b}/signed")
            out[f"proximity_bands/{label}/rmse_mm"] = math.sqrt(
                self._sums.mean(f"band/{b}/sq"))
            out[f"proximity_bands/{label}/share"] = (
                count / max(total, 1.0))
        return out
