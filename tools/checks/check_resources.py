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

import numpy as np
import torch

from core.data.resources import RESOURCE_ROOT, resolve_resource_path
from training.losses.haco_losses import VCBLoss
from training.tactohand_wrench.model import SubstepForceModel


def main() -> None:
    from types import SimpleNamespace
    from core.tactile_labels.native_timeline import (
        native_timeline_from_rows, timeline_revision, validate_native_timeline,
    )
    labels = dict(episode_index=106, n_rows=2, topology='rigid', source_fps=30,
                  solver_fps=30, tactile_sha1='contact-content', force_sha1='force-content')
    timeline = native_timeline_from_rows(labels, np.array([0., 1.]),
        np.array([15, 15], np.uint8), np.array([True, True]), np.array([0, 0]))
    metadata = dict(timeline.schema.metadata or {})
    metadata[b'source_revision'] = b'payload_v1'
    timeline = timeline.replace_schema_metadata(metadata)
    canonical = SimpleNamespace(revision='payload_v1', manifest={}, spec=lambda _: {})
    uploaded = SimpleNamespace(revision='payload_v1_upload', spec=lambda _: {},
        manifest={'upload_profile': {'scope': 'upload_only', 'source_revision': 'payload_v1'}})
    for reader in (canonical, uploaded):
        validate_native_timeline(timeline, labels, timeline_revision(reader))
    foreign_metadata = dict(metadata)
    foreign_metadata[b'source_revision'] = b'other_payload'
    foreign = timeline.replace_schema_metadata(foreign_metadata)
    try:
        validate_native_timeline(foreign, labels, timeline_revision(uploaded))
    except ValueError:
        pass
    else:
        raise AssertionError('A foreign timeline revision was accepted')
    uploaded.spec = lambda _: {'source_revision': 'new_payload'}
    assert timeline_revision(uploaded) == 'new_payload'

    from training.data.split_manifest import load_manifest
    for name in ('split_v1.json', 'h2o_official_split_v1.json'):
        assert load_manifest('core/resources/tactohand/' + name)['version'] == 1
    allowlist = json.loads((RESOURCE_ROOT / 'gigahands_tracking_good_objects.allowlist.json').read_text())
    assert allowlist['gigahands'] and all(isinstance(name, str) for name in allowlist['gigahands'])

    with tempfile.TemporaryDirectory(prefix="tactohand-resources-") as temporary:
        base = Path(temporary)
        old_cwd = Path.cwd()
        try:
            os.chdir(base)
            assert resolve_resource_path("core/resources/tactohand/split_v1.json").is_file()
            assert resolve_resource_path("external.npz") == Path("external.npz")
            assert resolve_resource_path(base / "custom.npz") == base / "custom.npz"
            assert resolve_resource_path("training_resources/tactohand_v1/statistics/contact.npz") == Path("training_resources/tactohand_v1/statistics/contact.npz")
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
            counts = np.tile(np.array([[900, 100]], np.int64), (778, 1))
            source_contact = base / "contact.npz"
            np.savez(source_contact, schema_version=1, class_counts=counts)
            custom = base / "custom.npz"
            np.savez(custom, schema_version=np.int64(1), class_counts=counts)
            kwargs = dict(contact_data_path="unused.npy",
                          v_regressor_336_path=str(regressor),
                          v_regressor_84_path=str(regressor),
                          j_regressor=torch.zeros(21, 778), total_epochs=3)
            a = VCBLoss(**kwargs, contact_statistics_path=str(source_contact))
            b = VCBLoss(**kwargs, contact_statistics_path=str(custom))
            for epoch in range(3):
                assert torch.equal(a.samples_per_cls[epoch], b.samples_per_cls[epoch])
            # The current mesh loss must also work without any HACO assets.
            without_haco = VCBLoss(
                contact_data_path=None, v_regressor_336_path=None,
                v_regressor_84_path=None, j_regressor=torch.zeros(21, 778),
                total_epochs=3,
                contact_statistics_path=str(source_contact))
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
                    contact_statistics_path=str(source_contact))
                expected = torch.from_numpy(matrix) @ a.samples_per_cls[0]
                assert torch.equal(expected, loss.samples_per_cls[0])
            changed = np.tile(np.array([[1000, 1]], np.int64), (778, 1))
            np.savez(custom, schema_version=np.int64(1), class_counts=changed)
            alternate = VCBLoss(**kwargs, contact_statistics_path=str(custom))
            assert not torch.equal(a.samples_per_cls[0], alternate.samples_per_cls[0])

            a_force, b_force = torch.nn.Module(), torch.nn.Module()
            source = base / "force.npz"
            np.savez(source, schema_version=1, nonzero_counts=np.full(778, 500), supervised_slots=1000)
            custom.write_bytes(source.read_bytes())
            for module, path in ((a_force, str(source)),
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
    print("PASS: bundled protocols and external-statistics consumers, overrides, cwd independence")


if __name__ == "__main__":
    main()
