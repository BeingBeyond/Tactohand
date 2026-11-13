# Numerical physics library

Import shared numerical models through `core.physics`.

- `backends`, `articulated_torch`, `articulated_newton`, `rollout_fast`: rigid
  and articulated forward dynamics and differentiable rollout.
- `math3d`, `discrete_kinematics`, `trajectory_wrench`: coordinate and temporal math.
- `parameters`, `multibody`: explicit articulated-body parameters and generalized maps.
- `wrench`: differentiable vertex-force aggregation.
- Contact, proximity and QP modules: numerical force allocation.

Functions accept explicit states, arrays, geometry and physical parameters.
Prepared-data reads live in `core/data`; dataset normalization/smoothing policy,
worker stage slots and orchestration live in `label_generation/`.
Use `scripts/labels/solve.py` and `scripts/labels/publish.py` for operations.
