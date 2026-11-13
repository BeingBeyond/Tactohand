"""Recover each independent static sample's gravity and exact object slot."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.data import processed_compat


def static_row_physics(root: Path, labels: dict, source_positions: np.ndarray,
                       force_valid: np.ndarray, contact_wrench: np.ndarray,
                       *, object_index: np.ndarray | None = None,
                       gravity_world: np.ndarray | None = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Read explicit row metadata, or recover old static-gravity labels.

    Legacy static QP targets encode F_contact = -m*g, independently of the
    allocated vertex forces. Never derive gravity from a model prediction.
    Object identity comes from the source observation's string instance key,
    not an asset label or the first object in the episode.
    """
    valid = np.asarray(force_valid, bool)
    count = len(valid)
    ids = list(map(str, labels["object_ids"]))
    if not ids:
        if valid.any():
            raise ValueError("A force-valid static episode has no object physics")
        return np.full(count, -1, np.int32), np.zeros((count, 3), np.float64)
    if len(set(ids)) != len(ids):
        raise ValueError("Static physics requires unique object instance ids")
    if object_index is None:
        frames = np.rint(source_positions).astype(np.int64)
        if not np.allclose(frames, source_positions, atol=1e-5, rtol=0):
            raise ValueError("Static samples require integral source-frame positions")
        if len(ids) == 1:
            slots = np.zeros(count, np.int32)
        else:
            observations = processed_compat.object_rows(
                root, int(labels["episode_index"]), str(labels["canonical_view"]),
                columns=["frame_index", "object_instance_id"])
            slot_of = {key: index for index, key in enumerate(ids)}
            by_frame: dict[int, set[int]] = {}
            for frame, instance in zip(observations["frame_index"].to_pylist(),
                                       observations["object_instance_id"].to_pylist()):
                if str(instance) in slot_of:
                    by_frame.setdefault(int(frame), set()).add(slot_of[str(instance)])
            slots = np.full(count, -1, np.int32)
            for row, frame in enumerate(frames):
                candidates = by_frame.get(int(frame), set())
                if len(candidates) == 1:
                    slots[row] = next(iter(candidates))
                elif valid[row]:
                    raise ValueError(
                        f"Static episode {labels['episode_index']} frame {frame}: "
                        f"expected one target instance, got {len(candidates)}")
    else:
        values = np.asarray(object_index)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError("Malformed static object-index column")
        slots = values.astype(np.int32)
        if not np.array_equal(values, slots):
            raise ValueError("Static object indices must be integers")
    in_range = (slots >= 0) & (slots < len(ids))
    if not in_range[valid].all():
        raise ValueError("A force-valid static row has no matching object physics")
    masses = np.asarray(labels["object_mass_kg"], np.float64)
    if len(masses) != len(ids) or not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("Invalid static object masses")
    # Invalid/absent rows keep a finite placeholder and never enter rollout.
    row_mass = masses[np.where(in_range, slots, 0)]
    expected = -np.asarray(contact_wrench, np.float64)[:, :3] / row_mass[:, None]
    gravity = expected if gravity_world is None else np.asarray(gravity_world, np.float64)
    if gravity.shape != (count, 3) or not np.isfinite(gravity[valid]).all():
        raise ValueError("Malformed static per-row gravity")
    if not np.allclose(gravity[valid], expected[valid], rtol=1e-5, atol=1e-5):
        raise ValueError("Static gravity disagrees with the stored equilibrium target")
    gravity = np.where(valid[:, None], gravity, 0.0)
    return slots, gravity
