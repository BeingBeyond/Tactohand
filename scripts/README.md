# Experiment and data operations

Use `scripts/` to run work on datasets, checkpoints or experiment results.
Named experiment/hardware presets live in each business directory’s `recipes/`. Repository development,
regression and release checks belong in `tools/`.
Run commands from the repository root.

| Directory | Purpose |
|---|---|
| `training/` | Reusable training invocation, environment, logging and training recipes |
| `evaluation/` | Evaluation recipes, result merging and paid API cache operations |
| `labels/` | Final label solve and publish entrypoints |
| `data_checks/` | Check prepared-label readiness and handcrop consumer geometry |
| `visualization/` | Inspect training inputs and compare predictions with GT |
| `profiling/` | Training/data-loading measurements and profiling recipes |

`data_checks/` checks actual dataset inputs. Developer regression checks for
inference, resources and rollout implementations live in `tools/checks/`.
Profiling measures cost/throughput; it consumes shared rollout-validation code
from `training/tactohand_wrench/validation/rollout.py`, not a tools CLI.

Shared implementation belongs in `core/`, `training/` or `label_generation/`.
Prepared scenes live in `core/visualization/prepared.py`; learning-batch and
prediction rendering lives in `training/visualization/`. Scripts are consumers.
API cache storage remains `artifacts/api_cache/contactprompt/`.

Presets select parameters; scripts expose operations; libraries implement shared behavior.
Example: `scripts/training/recipes/tactohand/train_384.sh` → `scripts/training/run_experiment.sh` → `training/train.py`.

`profiling/profile_tactohand_loader.py --help` describes bounded real-data loader
profiling. Its `--matrix` compares the current standard (`standard_*`) and fast
(`optimized_*`) data paths, with independent logs and resumable case outputs.
The historical `--legacy-augment` oracle and `original_*` cases were retired;
use a new output directory for the current matrix. This comparison does not
reproduce the retired photometric implementation.

Recipe locations are indexed in `scripts/recipes.md`. Shell helpers stay with
their consuming recipe family; recipes select parameters and call reusable CLIs.
