# Model development and runtime

`training/` owns training, inference, evaluation and model-result visualization.
It consumes `core/`; it does not import operational scripts, developer tools or
label-generation workflows. Supported training tasks are `tactohand_wrench` and
`hope_baseline`. The default is `experiment=tactohand_384 data=tactohand`.

| Module | Responsibility |
|---|---|
| `train.py`, `eval_tactohand.py`, `infer_tactohand.py`, `cli.py` | Stable entrypoints |
| `conf/`, `configs/` | Experiment configuration and task/data schemas |
| `data/` | Shared label datasets, windows, padding, sampling, augmentation and DataModule |
| `runtime/`, `callbacks/`, `losses/` | Shared schedules, lifecycle callbacks and reusable losses |
| `evaluation/` | Standard metric views, families, reports and predictor assembly |
| `visualization/` | Training batch and prediction adapters to shared scene rendering |
| `tactohand_wrench/` | TactoHand network, Lightning model, targets, checkpoint/physical-output adapters |
| `hope_baseline/` | HOPE VertexFormer network and Lightning model |

Model `model.py` locations, module attributes and checkpoint parameter names remain
unchanged. Old `training.tactohand_wrench.datamodule` and
`training.tactohand_wrench.data.datamodule` imports bridge saved Hydra targets to
`training.data.datamodule`; new configurations use the shared path.

The evaluator selects model-specific loading and decoding lazily through
`evaluation/tasks.py`. Native checkpoint evaluation currently supports TactoHand;
this reorganization does not add HOPE checkpoint evaluation. Official ARCTIC
FieldSF/FieldLSTM remains in `evaluation/predictors_interfield.py` and loads external
checkpoints. Research adapters remain optional in the public distribution.

Articulated episodes keep contact, proximity and force supervision, with
`rollout_eligible=false`. `force_no_rollout` admits them without enabling unsupported
rollout. The Show3D dataset remains available as TactoHand data; competition-specific
Show3D training is retired.

Regression CLIs belong in `tools/checks/`. Shared batch assembly lives in
`data/sample_batch.py`, model rollout validation in `tactohand_wrench/validation/`.
See [architecture](../docs/architecture.md).
