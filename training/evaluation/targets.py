"""Native evaluation targets, independent of input crop overrides."""
from __future__ import annotations
import torch

def _native_contact_target(batch, device):
    """OFFICIAL scoring targets (2026-09-22, both view builders): when the
    reader supplies the crop-override-free native targets, score against
    THEM.  The handcrop pipeline zeroes the off-crop hand's contact, which
    would otherwise hand a handcrop method free empty-GT credit on frames
    where the unseen hand is actually in contact (arctic ego: GT-contact
    frame-hand share collapsed 85% -> 43% under the crop override).
    Validity stays per-pipeline: a frame a method's own protocol cannot
    score stays out of ITS denominators; targets are the shared scene
    truth.  Training losses keep reading contact_target_10hz."""
    if "contact_target_native_10hz" in batch:
        return batch["contact_target_native_10hz"].to(device).bool()
    return batch["contact_target_10hz"].to(device).bool()


def _native_contact_valid(batch, device):
    """Fixed scene-level evaluation universe, independent of crop policy."""
    valid = batch.get("contact_valid_native_10hz",
                      batch["contact_valid_10hz"]).to(device).bool()
    # Reader-native validity is stored vertex-broadcast-ready as
    # [B,T,H,1], while the view builder combines hand validity with
    # real_mask in [B,T,1] before adding the vertex singleton itself.
    if valid.ndim == 4 and valid.shape[-1] == 1:
        valid = valid.squeeze(-1)
    present = batch.get("hand_present_native_10hz")
    if present is not None:
        valid = valid & present.to(device).bool()
    return valid
