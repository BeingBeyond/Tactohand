"""Dataset-specific trajectory smoothing and resume-validation policy."""
from __future__ import annotations
import os
from core.physics.trajectory_wrench import GAUSSIAN_SIGMA_S
from core.physics.trajectory_policy import (
    rigid_gaussian_sigma_s, rigid_gaussian_sigma_rot_s)

SHOW3D_GAUSSIAN_SIGMA_S = 0.033
# Rotation annotations in DexYCB carry roughly 2.9 mrad/frame of high-frequency
# jitter (~40% of the median per-frame increment), which the second derivative
# amplifies into the dominant torque-label noise.  A wider rotation-only kernel
# (measured on raw DexYCB tracks: 10 Hz torque persistence skill 0.57 -> 0.80)
# removes it while the translation kernel stays untouched so force labels are
# unchanged.  Datasets without a measured rotation-noise profile keep the
# single-sigma behavior until they are calibrated the same way.
GAUSSIAN_SIGMA_ROT_S = 0.15
RIGID_SPLIT_ROT_SIGMA_S = {
    "dexycb_lerobot": GAUSSIAN_SIGMA_ROT_S,
}
GAUSSIAN_SIGMA_POS_ENV = "GAUSSIAN_SIGMA_POS_S"
GAUSSIAN_SIGMA_ROT_ENV = "GAUSSIAN_SIGMA_ROT_S"


def _env_float(environ: dict[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def resolve_rigid_gaussian_sigmas(
    dataset: str,
    *,
    overrides: tuple[float | None, float | None] = (None, None),
    environ: dict[str, str] | None = None,
) -> tuple[float, float]:
    """Resolve (sigma_pos_s, sigma_rot_s) for one rigid dataset root name.

    Precedence per channel: explicit override, then environment
    (GAUSSIAN_SIGMA_POS_S / GAUSSIAN_SIGMA_ROT_S), then the per-dataset
    constants.  The solver and the resume-validation share this function so a
    re-solve with different smoothing always invalidates stale labels.
    """
    env = os.environ if environ is None else environ
    pos_override, rot_override = overrides
    pos = (float(pos_override) if pos_override is not None else
           _env_float(env, GAUSSIAN_SIGMA_POS_ENV, rigid_gaussian_sigma_s(dataset)))
    rot = (float(rot_override) if rot_override is not None else
           _env_float(env, GAUSSIAN_SIGMA_ROT_ENV, rigid_gaussian_sigma_rot_s(dataset)))
    if pos <= 0.0 or rot <= 0.0:
        raise ValueError(
            f"gaussian sigmas must be positive; got pos={pos}, rot={rot}")
    return pos, rot
