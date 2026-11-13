"""Shared dynamics; importing small numerical helpers does not load backends."""
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends import NewtonBackend, Physics, RigidState, TorchBackend
    from .articulated_newton import ArticulatedNewtonBackend
    from .articulated_torch import ArticulatedState, ArticulatedTorchBackend
    from .backends import replay_sequence, summarize_replay

_EXPORTS = {
    "Physics": "backends",
    "RigidState": "backends",
    "TorchBackend": "backends",
    "NewtonBackend": "backends",
    "ArticulatedState": "articulated_torch",
    "ArticulatedTorchBackend": "articulated_torch",
    "ArticulatedNewtonBackend": "articulated_newton",
    "replay_sequence": "backends",
    "summarize_replay": "backends",
}
__all__ = [
    "Physics", "RigidState", "TorchBackend", "NewtonBackend",
    "ArticulatedState", "ArticulatedTorchBackend", "ArticulatedNewtonBackend",
    "replay_sequence", "summarize_replay",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
