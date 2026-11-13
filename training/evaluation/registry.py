"""Metric-family registry for the TactoHand standalone evaluator.

A metric family is a class with ``specs() -> list[MetricSpec]``,
``update(view: BatchView)`` and ``compute() -> dict[str, float]``; it
registers itself under a selector name via :func:`register`.  The harness
selects families by name on the CLI (``--metrics contact,proximity``), so
adding a metric never touches the runner — write the family, register it,
use it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

from training.evaluation.view import BatchView, MetricSpec

FamilyClass = Type["MetricFamily"]


class MetricFamily:
    """Base class: accumulate across batches, emit one float per spec key."""

    def update(self, view: BatchView) -> None:
        raise NotImplementedError

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError

    @staticmethod
    def specs() -> List[MetricSpec]:
        raise NotImplementedError


_REGISTRY: Dict[str, FamilyClass] = {}


def register(name: str) -> Callable[[FamilyClass], FamilyClass]:
    def wrap(cls: FamilyClass) -> FamilyClass:
        if name in _REGISTRY:
            raise KeyError(f"duplicate metric family selector: {name}")
        _REGISTRY[name] = cls
        return cls
    return wrap


def available_families() -> List[str]:
    return sorted(_REGISTRY)


def build_families(selectors: List[str]) -> List[MetricFamily]:
    unknown = [s for s in selectors if s not in _REGISTRY]
    if unknown:
        raise KeyError(
            f"unknown metric family selector(s) {unknown}; "
            f"available: {available_families()}")
    return [_REGISTRY[s]() for s in selectors]


class Sums:
    """Accumulator bag keyed by spec key.

    Values/counts may be Python floats or 0-dim GPU tensors: metric
    families pass tensor sums straight through so accumulation never
    forces a device sync; :meth:`mean`/:meth:`sum` sync once each when
    the report is built.
    """

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._counts: Dict[str, Any] = {}

    def add(self, key: str, value, count=1.0) -> None:
        self._values[key] = self._values.get(key, 0.0) + value
        self._counts[key] = self._counts.get(key, 0.0) + count

    def mean(self, key: str) -> float:
        return float(self._values[key]) / max(float(self._counts[key]), 1.0)

    def sum(self, key: str) -> float:
        return float(self._values[key])

    def keys(self) -> List[str]:
        return sorted(self._values)


# Import families for their registration side effect.
from training.evaluation import families_contact
from training.evaluation import families_proximity
from training.evaluation import families_proximity_bands
from training.evaluation import families_force
from training.evaluation import families_rollout
