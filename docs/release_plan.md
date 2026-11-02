# Release plan

Status: early source preview. This roadmap describes planned work, not completed
validation or a guarantee of release dates. Items are checked only after their
acceptance criteria are met.

## Stage 0 — Small public preview

- [x] Publish a self-contained force-to-wrench utility from the implementation.
- [x] Include a runnable synthetic example and explain units and limitations.
- [x] Document the staged release scope.
- [ ] Finalize the software license and contributor/resource attribution.

The full video prediction model, training pipeline and checkpoints are not part
of this stage.

## Stage 1 — Model and inference

- [ ] Release the current model implementation and matching configuration.
- [ ] Publish a checkpoint with its provenance and usage conditions.
- [ ] Provide tensor/video inference with documented preprocessing.
- [ ] Document contact/proximity outputs and force coordinates, normalization and units.
- [ ] Run a real-checkpoint example in a clean environment outside the development repository.
- [ ] Provide a redistributable example input and expected output description.

Acceptance: a new user can install the documented dependencies, obtain the
required resources and run the example without private paths or development data.

## Stage 2 — Evaluation and training

- [ ] Release evaluation entrypoints with metric definitions and coverage reporting.
- [ ] Document labels-v5 readers, dataset preparation and handcrop conventions.
- [ ] Provide a portable training configuration and launcher.
- [ ] Validate a small real-data training/evaluation run and document hardware requirements.
- [ ] Document MANO/backend/backbone installation and third-party notices.
- [ ] Add automated packaging, import and command-line checks.

Acceptance: the documented data subset and configuration reproduce the stated
checks; unvalidated datasets and workflows are explicitly identified.

## Stage 3 — Optional research workflows

- [ ] Review label-generation dependencies and resource redistribution conditions.
- [ ] Document label-generation acceptance checks before releasing that workflow.
- [ ] Release baseline adapters only with their separate setup and validation notes.

## Distribution boundaries

Raw datasets, restricted MANO assets, private credentials, machine-specific
configuration, experiment logs and internal research history are excluded.
Each resource will be distributed only under its applicable terms. Features and
artifacts will be announced here as they become available; this roadmap does not
claim that results or reproduction have already been verified.
