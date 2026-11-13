"""Contact priority and candidate support from outside-positive signed distance."""
from __future__ import annotations

import numpy as np


STANDARD_CONTACT_THRESHOLD_M = 0.010
STANDARD_FORCE_CANDIDATE_THRESHOLD_M = 0.020
# Missing hands have no queried surface. Match the labels-v5 absent-hand fill.
ABSENT_HAND_DISTANCE_M = 0.25


def signed_distance_contact_priority(
    signed_distance_m: np.ndarray,
    *,
    distance_scale_m: float = 0.004,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``q=2/(1+exp(s/ell))`` with stable evaluation.

    Negative signed distance (penetration) has ``q>1``, contact has ``q=1``,
    and positive separation has ``q<1``. Invalid frames receive zero priority.
    """
    signed = np.asarray(signed_distance_m, dtype=np.float64)
    if signed.ndim < 1 or not np.isfinite(signed).all():
        raise ValueError("signed_distance_m must be a finite array")
    if distance_scale_m <= 0:
        raise ValueError("distance_scale_m must be positive")
    scaled = signed / distance_scale_m
    exp_negative = np.exp(-np.maximum(scaled, 0.0))
    exp_positive = np.exp(np.minimum(scaled, 0.0))
    priority = np.where(
        scaled >= 0.0,
        2.0 * exp_negative / (1.0 + exp_negative),
        2.0 / (1.0 + exp_positive),
    )
    if valid is not None:
        frame_valid = np.asarray(valid, dtype=bool)
        if frame_valid.shape != signed.shape[:-1]:
            raise ValueError("valid must have signed_distance_m.shape[:-1]")
        priority = np.where(frame_valid[..., None], priority, 0.0)
    return priority


def signed_distance_candidate_mask(
    signed_distance_m: np.ndarray,
    *,
    d_off_m: float = 0.020,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Return the strict candidate support ``s < d_off``."""
    signed = np.asarray(signed_distance_m, dtype=np.float64)
    if signed.ndim < 1 or not np.isfinite(signed).all():
        raise ValueError("signed_distance_m must be a finite array")
    if d_off_m <= 0:
        raise ValueError("d_off_m must be positive")
    candidate = signed < d_off_m
    if valid is not None:
        frame_valid = np.asarray(valid, dtype=bool)
        if frame_valid.shape != signed.shape[:-1]:
            raise ValueError("valid must have signed_distance_m.shape[:-1]")
        candidate &= frame_valid[..., None]
    return candidate


def signed_distance_contact_mask(
    signed_distance_m: np.ndarray,
    *,
    contact_threshold_m: float = STANDARD_CONTACT_THRESHOLD_M,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild the solver contact mask under the canonical 10 mm contract.

    Dataset contact binaries may intentionally use other task presets (for
    example MOW uses 35 mm).  Force allocation must therefore use preserved
    signed distance rather than inheriting those dataset-specific binaries.
    Inside-mesh vertices from contact generation have non-positive/zero signed
    distance and remain contact under this definition.
    """
    signed = np.asarray(signed_distance_m, dtype=np.float64)
    if signed.ndim < 1 or not np.isfinite(signed).all():
        raise ValueError("signed_distance_m must be a finite array")
    if contact_threshold_m <= 0:
        raise ValueError("contact_threshold_m must be positive")
    contact = signed <= contact_threshold_m
    if valid is not None:
        frame_valid = np.asarray(valid, dtype=bool)
        if frame_valid.shape != signed.shape[:-1]:
            raise ValueError("valid must have signed_distance_m.shape[:-1]")
        contact &= frame_valid[..., None]
    return contact


def resolve_standard_contact_mask(
    signed_distance_m: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    stored_contact_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Prefer a stored contact mask when it already obeys the 10 mm contract.

    Checking equivalence is an inexpensive array operation over the preserved
    signed distance.  It does not recompute point-to-mesh geometry.  Dataset
    presets using another threshold automatically fall back to rebuilding the
    mask from signed distance.
    """
    canonical = signed_distance_contact_mask(signed_distance_m, valid=valid)
    if stored_contact_mask is None:
        return canonical, "signed_distance_10mm"
    stored = np.asarray(stored_contact_mask, dtype=bool)
    if stored.shape != canonical.shape:
        raise ValueError("stored_contact_mask must match signed_distance_m")
    if valid is not None:
        stored = stored & np.asarray(valid, dtype=bool)[..., None]
    if np.array_equal(stored, canonical):
        return stored, "stored_contact_mask_verified_10mm"
    return canonical, "signed_distance_10mm_preset_override"


def contact_gated_force_candidates(
    candidate: np.ndarray,
    contact: np.ndarray,
    *,
    hand_vertex_count: int = 778,
) -> np.ndarray:
    """Gate a relaxed force neighborhood by real contact for each hand.

    ``candidate`` may use a looser distance such as 20 mm, but it is enabled
    only when the same frame/hand/object block contains at least one contact
    vertex (the standard contact threshold is 10 mm).  This function is the
    shared solver boundary; callers must pass one object's contact mask when a
    scene contains multiple objects.
    """
    support = np.asarray(candidate, dtype=bool)
    onset = np.asarray(contact, dtype=bool)
    if support.shape != onset.shape or support.ndim < 2:
        raise ValueError("candidate and contact must have equal shape [...,V]")
    if hand_vertex_count <= 0 or support.shape[-1] % hand_vertex_count:
        raise ValueError("last dimension must contain complete hand blocks")
    gated = support.copy()
    for begin in range(0, support.shape[-1], hand_vertex_count):
        end = begin + hand_vertex_count
        active = onset[..., begin:end].any(axis=-1)
        gated[..., begin:end] &= active[..., None]
    return gated
