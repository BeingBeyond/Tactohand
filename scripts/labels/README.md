# Final label operations

Portable solving and publishing entrypoints, run from the repository root:

```bash
python scripts/labels/solve.py --help
python scripts/labels/publish.py --solver-run <run> --dataset <name> --install
```

`solve.py` forwards CLI options to the final scheduler. Set data/output paths,
workers and devices through its arguments. It does not require an A800 profile.

For the existing eight-A800 production recipe, use:

```bash
bash scripts/labels/recipes/a800_8gpu_solve.sh
```

The recipe preserves the established parameters and preflight/audit stages.
Shared workers live in `label_generation/`, numerical code in `core/physics/`.
No dataset generation or installation is performed by repository reorganization.
