"""Standalone metric evaluation for TactoHand wrench runs.

Selection interface (see ``training/eval_tactohand.py``)::

    from training.evaluation.harness import EvalHarness

Metric families register via ``registry.register``; the harness consumes
whatever is registered, so new metrics need no runner changes.
"""
from training.evaluation.harness import EvalHarness, format_console, resolve_checkpoint, write_report
from training.evaluation.registry import available_families, build_families
from training.evaluation.view import BatchView, MetricSpec

__all__ = [
    "EvalHarness",
    "available_families",
    "build_families",
    "BatchView",
    "MetricSpec",
    "resolve_checkpoint",
    "write_report",
    "format_console",
]
