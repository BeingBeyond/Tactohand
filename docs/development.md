# Development

[English](development.md) | [简体中文](development.zh-CN.md)

## Architecture

| Directory | Responsibility |
|---|---|
| `core/data`, `core/io` | Prepared manifests, tables, indexes and video/file IO |
| `core/geometry` | MANO, meshes, cameras, projections and mass properties |
| `core/physics` | Numerical dynamics, contact solving and wrench aggregation |
| `core/tactile_labels` | Label storage, readers/writers and handcrop access |
| `core/visualization` | Shared scene rendering and colormaps |
| `training/` | Models, learning data, training, inference, evaluation and result display |
| `label_generation/` | Offline solver adaptation, scheduling, audits and publication |
| `scripts/` | Dataset/model operations and hardware recipes |
| `tools/` | Development checks, packaging and workspace maintenance |

Core does not import training, label_generation, scripts or tools. Training and
label generation consume core and exchange label files without importing one
another. Numerical physics receives explicit states and parameters; readers do
not depend on rendering. Shared implementation belongs in libraries, not CLIs.

Learning windows, augmentation and supervision eligibility belong in
`training/data/`. Model-specific networks and runtime adapters live in
`training/tactohand_wrench/`; HOPE is a separate task. Retain stable model paths,
checkpoint keys and saved Hydra targets when moving code.

## Operations and checks

Run commands from the repository root. Put prepared-data checks in
`scripts/data_checks/`, developer regressions in `tools/checks/`, and presets
under `scripts/<workflow>/recipes/`. Recipes set parameters and call reusable
operations. Platform launchers must wait for the real task and return its exit
code. Long jobs need separate logs, incremental outputs and completion markers.

```bash
tools/run_checks.sh --quick
python3 -S tools/checks/test_inference_dashboard.py
python tools/checks/test_inference_visibility.py
```

`--quick` checks shell syntax, tracked-file hygiene and dependencies. `--full`
adds launcher contracts and pyright. Prediction/resource/rollout checks run
separately; select the checks relevant to the change. Use tiny subsets for
data-dependent checks and verify actual consumers after geometry/crop changes.

## Packaging

`packaging/public_files.toml` selects wheel and source contents. The source
package includes both documentation languages; local data, checkpoints and
third-party model payloads remain external.

```bash
python tools/packaging/build_public_source.py --output /tmp/tactohand-source.zip
python tools/checks/check_distribution.py \
  --output artifacts/release_checks/distribution.json \
  --log-dir logs/release_checks/distribution
```

The distribution check builds source/wheel/sdist, compares rebuilt library
contents and exercises an installed package outside the checkout with synthetic
CPU inputs. It reuses existing dependencies; fresh-environment installation
and real-weight evaluation are separate checks. Reports are incremental and
`.done` marks success.

Package selection does not change Git visibility or remove research history.
Public updates must contain selected source files rather than research history.
Review code, statistics, dependencies and model assets for redistribution rights;
keep provenance metadata even when payloads are not distributed.

## Contributions

Use focused commits with an imperative scope-prefixed subject, such as
`training: fix force masking`. Include purpose, affected datasets and commands
run in the PR. Update [label format](label_format.md) when schemas or data
contracts change. Keep English and Chinese guides aligned.

History retains retired recipes and investigation notes. Current workflow
instructions belong in [usage](usage.md), data conventions in
[datasets](datasets.md).
