# TactoHand

Video-based hand–object contact, proximity and per-vertex force prediction.

**Early source preview — full model and pretrained checkpoints are not released yet.**

TactoHand is being prepared for a staged public release. This repository currently
contains a small utility from the implementation, a runnable synthetic example,
and the [release plan](docs/release_plan.md).

## Available now

- Differentiable aggregation of per-vertex contact forces into a 6D wrench.
- A synthetic example that requires no dataset, model weights or MANO assets.
- A public roadmap for the model, inference and reproduction releases.

This preview does not predict contact or force from video. The example demonstrates
force aggregation only; it is not an accuracy or physical-reconstruction benchmark.

## Run the example

Use Python 3.11 or newer and an appropriate PyTorch installation for your machine.
From this repository:

```bash
python -m pip install -e .
python -m examples.aggregate_forces
```

Expected output, ordered as `[Fx, Fy, Fz, Tx, Ty, Tz]`:

```text
Wrench [N, N, N, N*m, N*m, N*m]: [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
```

Forces are expressed in newtons and lever arms in meters, in the same coordinate
frame. Torque is measured about the origin used to define those lever arms. Gravity
and environment forces are separate from this hand-contact wrench.

## Planned releases

See the [release plan](docs/release_plan.md) for scope and acceptance criteria.
No release date is committed yet. Star or watch this repository to follow updates;
questions about intended usage are welcome through GitHub Issues.

## License and resources

Licensing terms for this preview are pending; no software license is granted yet.
A license will be added before the supported software release. Datasets, MANO
assets, backbone weights and trained checkpoints are not included in this preview;
their access and redistribution conditions will be documented separately.
