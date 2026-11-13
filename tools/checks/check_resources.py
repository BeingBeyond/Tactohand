"""Check bundled content, consumer weights and path overrides.

Run with the training Python environment. No datasets, MANO assets or GPU
are required. The test does not scan or regenerate labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.data.resources import RESOURCE_ROOT, resolve_resource_path
from training.losses.haco_losses import VCBLoss
from training.tactohand_wrench.model import SubstepForceModel


def main() -> None:
    manifest = json.loads((RESOURCE_ROOT / "manifest.json").read_text())
    for name, spec in manifest["resources"].items():
        payload = (RESOURCE_ROOT / name).read_bytes()
        assert len(payload) == spec["bytes"]
        # When the old local artifact is available, verify exact compatibility.
        previous = ROOT / spec["previous_path"] if spec.get("previous_path") else None
        if previous is not None and previous.is_file():
            assert previous.read_bytes() == payload

    with tempfile.TemporaryDirectory(prefix="tactohand-resources-") as temporary:
        base = Path(temporary)
        old_cwd = Path.cwd()
        try:
            os.chdir(base)
            assert resolve_resource_path("core/resources/tactohand/contact_statistics_v5.npz").is_file()
            assert resolve_resource_path("external.npz") == Path("external.npz")
            assert resolve_resource_path(base / "custom.npz") == base / "custom.npz"
            assert resolve_resource_path("artifacts/tactohand/contact_statistics_v5.npz") == Path("artifacts/tactohand/contact_statistics_v5.npz")
            for invalid in ("core/resources/tactohand/../private.npz",
                            "core/resources/tactohand//etc/passwd"):
                try:
                    resolve_resource_path(invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid bundled path accepted")

            # Exercise VCBLoss's real statistics consumer with synthetic mesh
            # regressors; mesh mode leaves them unused, so no MANO is needed.
            regressor = base / "regressor.npy"
            np.save(regressor, np.zeros((1, 778), np.float32))
            with np.load(RESOURCE_ROOT / "contact_statistics_v5.npz") as stats:
                counts = stats["class_counts"].copy()
            custom = base / "custom.npz"
            np.savez(custom, schema_version=np.int64(1), class_counts=counts)
            kwargs = dict(contact_data_path="unused.npy",
                          v_regressor_336_path=str(regressor),
                          v_regressor_84_path=str(regressor),
                          j_regressor=torch.zeros(21, 778), total_epochs=3)
            a = VCBLoss(**kwargs, contact_statistics_path="core/resources/tactohand/contact_statistics_v5.npz")
            b = VCBLoss(**kwargs, contact_statistics_path=str(custom))
            for epoch in range(3):
                assert torch.equal(a.samples_per_cls[epoch], b.samples_per_cls[epoch])
            # The current mesh loss must also work without any HACO assets.
            without_haco = VCBLoss(
                contact_data_path=None, v_regressor_336_path=None,
                v_regressor_84_path=None, j_regressor=torch.zeros(21, 778),
                total_epochs=3,
                contact_statistics_path="core/resources/tactohand/contact_statistics_v5.npz")
            for epoch in range(3):
                assert torch.equal(a.samples_per_cls[epoch], without_haco.samples_per_cls[epoch])
            for mode, count in (("mesh_336", 336), ("mesh_84", 84)):
                reduced = base / f"{mode}.npy"
                matrix = np.full((count, 778), 1.0 / 778, np.float32)
                np.save(reduced, matrix)
                paths: dict[str, str | None] = dict(v_regressor_336_path=None, v_regressor_84_path=None)
                paths[f"v_regressor_{count}_path"] = str(reduced)
                loss = VCBLoss(
                    contact_data_path=None, **paths,
                    j_regressor=torch.zeros(21, 778), total_epochs=3, v_type=mode,
                    contact_statistics_path="core/resources/tactohand/contact_statistics_v5.npz")
                expected = torch.from_numpy(matrix) @ a.samples_per_cls[0]
                assert torch.equal(expected, loss.samples_per_cls[0])
            changed = np.tile(np.array([[1000, 1]], np.int64), (778, 1))
            np.savez(custom, schema_version=np.int64(1), class_counts=changed)
            alternate = VCBLoss(**kwargs, contact_statistics_path=str(custom))
            assert not torch.equal(a.samples_per_cls[0], alternate.samples_per_cls[0])

            a_force, b_force = torch.nn.Module(), torch.nn.Module()
            source = RESOURCE_ROOT / "force_statistics_v5.npz"
            custom.write_bytes(source.read_bytes())
            for module, path in ((a_force, "core/resources/tactohand/force_statistics_v5.npz"),
                                 (b_force, str(custom))):
                SubstepForceModel._load_force_vertex_weights(module, path, beta=0.9999, count_floor=100.0)
            assert torch.equal(a_force.force_vertex_weights, b_force.force_vertex_weights)
            with np.load(source) as stats:
                changed_force = {key: stats[key].copy() for key in stats.files}
            changed_force["nonzero_counts"] = np.zeros(778, np.int64)
            np.savez(custom, **changed_force)
            alternate_force = torch.nn.Module()
            SubstepForceModel._load_force_vertex_weights(alternate_force, str(custom), beta=0.9999, count_floor=100.0)
            assert not torch.equal(a_force.force_vertex_weights, alternate_force.force_vertex_weights)

        finally:
            os.chdir(old_cwd)
    print("PASS: resource content, cwd independence, overrides, identical consumer weights")


if __name__ == "__main__":
    main()
