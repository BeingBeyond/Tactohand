# Repository development and release tools

Use `tools/` to maintain or verify code and the repository. Dataset/checkpoint
operations, training, evaluation, rendering and performance measurements belong
in `scripts/`, including per-business `recipes/`.
Shared implementation belongs in `core/`, `training/` or `label_generation/`.

| Tool | Purpose |
|---|---|
| `checks/` | On-demand regression checks for prediction, resources and model rollout |
| `checks/check_distribution.py` | Build ZIP/wheel/sdist, check file scope and docs, then exercise an installed package outside the checkout |
| `run_checks.sh` | `--quick`: shell syntax, dependency boundaries and tracked-file hygiene; `--full`: adds launcher command checks and pyright |
| `checks/check_workspace_hygiene.py` | Inspect tracked caches, credential filenames, large files and malformed root paths; report-only unless `--fail-on-findings` |
| `checks/test_launcher_dry_run.sh` | Check five current launcher command contracts without training; filter with `--only NAME` |
| `packaging/public_source_scope.py` | Shared wheel/source file selection used by setup.py |
| `packaging/build_public_source.py` | Build a local source ZIP from the inclusion policy |
| `packaging/build_opensource_inventory.py` | Optional file-level release review inventory |
| `workspace/cleanup_generated.py` | Optional registered-output retention tool; dry-run by default |

```bash
tools/run_checks.sh --quick
tools/run_checks.sh --full
python tools/packaging/build_public_source.py --output /tmp/tactohand-source.zip
python tools/checks/check_distribution.py --output artifacts/release_checks/distribution.json --log-dir logs/release_checks/distribution
```

Quick checks do not scan datasets, model payloads or nested third-party repos.
Ignored local caches and private files are not findings. Failed checks return a
nonzero status; no historical runbook baseline is silently accepted.

`checks/check_prediction.py`, `checks/check_resources.py`, and
`checks/check_rollout.py` verify code behavior; they are separate from the
`--quick`/`--full` repository checks and run only when explicitly invoked.
Model regression checks may require the training environment. The rollout CLI
uses shared validation code in `training/tactohand_wrench/validation/rollout.py`.

The distribution check uses the current interpreter's dependencies. It runs a
synthetic CPU prediction, never starts training and does not certify a fresh
dependency environment or real trained-checkpoint accuracy. Step logs and the
incremental JSON report remain available on failure; `.done` means success.

Neither training nor operational scripts import tools CLI implementations.

Model-specific regression CLIs are grouped under `checks/tactohand/`; model batch
assembly and rollout-validation helpers are shared from `training/`, not copied
into command wrappers.

`python tools/checks/tactohand/test_articulated_supervision.py` checks that
articulated samples retain contact/proximity/force gradients and skip rollout
loss and metrics, including mixed rigid/articulated batches. Eligible rigid
rollout remains enabled. This CPU check uses synthetic inputs and no MANO assets.

`checks/check_architecture.py` checks library dependency direction, retired roots,
internal import resolution and saved Hydra compatibility targets without importing
heavy runtime dependencies. It is included in `run_checks.sh --quick`.

`checks/check_assets.py` verifies the local asset manifest, streaming SHA-256 and
relative links without unpickling. `--include-archive` also verifies relocated
installers/retired resources; `--output PATH` writes a JSON report. It is optional
and does not make private model assets a requirement for repository quick checks.
