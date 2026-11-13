# Offline label generation

`label_generation/` owns dataset solver adapters, scheduling, stage coordination,
preflight, audit and publishing. It consumes `core/` and does not import training.

- `scheduler.py` dispatches complete episodes with existing retry/resume behavior;
  `dispatch.py` selects rigid, static or articulated workers.
- `adapters/` defines solver input, reference mass, completion and smoothing policy.
  Table/mesh reads come from `core/data`; mass properties from `core/geometry`;
  numerical dynamics and contact allocation from `core/physics`.
- `stage_pipeline.py` controls worker stage slots. `io.py` atomically saves results.
- `publish.py` materializes and installs a selected run through the core label writer.

Use `scripts/labels/solve.py` or `scripts/labels/publish.py`; production parameters
are in `scripts/labels/recipes/a800_8gpu_solve.sh`. The package is included in the
selected source release and remains outside the default inference wheel.
