"""Shared model-rollout correctness checks used by profiling and development."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def validate(device="cpu", implementation="deferred"):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from core.physics.backends import BatchedTorchBackend, Physics, RigidState
    from core.physics.rollout_fast import midpoint_ex, compiled_midpoint
    from training.tactohand_wrench.model import SubstepForceModel

    graph_requested = implementation == "graphed_fused3"
    if graph_requested:
        if torch.device(device).type != "cuda":
            raise ValueError("CUDA Graph validation requires a CUDA device")
        implementation = "compiled_fused3"
    torch.set_num_threads(1)
    dtype = torch.float64
    generator = torch.Generator().manual_seed(1729)
    reports = []
    for name, diagonal in [("ordinary", [0.01, 0.02, 0.025]),
                           ("ill_conditioned", [1e-8, 0.5, 1.0]),
                           ("zero_wrench", [0.01, 0.02, 0.025])]:
        b, t = 2, 18
        physics = [Physics(mass=1.0 + i, com_body=(0., 0., 0.),
                           inertia_body=np.diag(diagonal), gravity_world=(0., 0., 0.),
                           dt=1 / 120, substeps=4) for i in range(b)]
        force = torch.randn(b, t, 6, generator=generator, dtype=dtype) * 1e-3
        # Keep high-condition-number cases stable without hiding their
        # ill-conditioned inertia; exercise nonzero angular response.
        force[..., 3:] *= torch.tensor(diagonal, dtype=dtype)
        if name == "zero_wrench":
            force.zero_()
        force = force.to(device)
        batch = {
            "dataset_mode": ["rigid_trajectory"] * b,
            "real_mask_30hz": torch.ones(b, t, dtype=torch.bool, device=device),
            "position_30hz": torch.zeros(b, t + 1, 3, dtype=dtype, device=device),
            "rotation_30hz": torch.eye(3, dtype=dtype, device=device).expand(b, t + 1, 3, 3),
            "velocity_30hz": torch.zeros(b, t + 1, 6, dtype=dtype, device=device),
            "physics_30hz": physics,
        }
        batch["real_mask_30hz"][1, -2:] = False
        outputs = []
        for mode in ("reference", implementation):
            trajectory = []

            class TracedBackend(BatchedTorchBackend):
                def rollout_interval(self, state, wrench):
                    result = super().rollout_interval(state, wrench)
                    trajectory.append(torch.cat((result.x_com, result.quat_xyzw,
                                                 result.v_world, result.omega_world), -1).detach())
                    return result

            model = SimpleNamespace(
                cfg=OmegaConf.create({"metrics": {"rollout_integrator": "implicit_midpoint",
                                                 "implicit_iterations": 6},
                                      "loss": {"rollout": {"implementation": mode}}}),
                rollout_horizon_10hz=4, rollout_position_scale=0.01,
                rollout_rotation_scale=0.1, rollout_position_weight=1., rollout_rotation_weight=1.)
            predicted = force.clone().requires_grad_(True)
            aligned = (torch.arange(t, device=device) % 3 == 0)[None, :, None]
            hybrid = torch.where(aligned, predicted, force)
            with patch("training.tactohand_wrench.model.BatchedTorchBackend", TracedBackend):
                losses = SubstepForceModel._gt_pose_rollout_loss(model, hybrid, batch)
            gradient, = torch.autograd.grad(losses[0], predicted)
            assert bool(torch.isfinite(gradient).all()), (name, mode)
            assert torch.count_nonzero(gradient[:, 1::3]) == 0
            assert torch.count_nonzero(gradient[:, 2::3]) == 0
            outputs.append((torch.stack(losses).detach(), gradient.detach(), torch.stack(trajectory)))
        for actual, expected in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9)
        reports.append({"case": name,
                        "loss_max_abs": float((outputs[1][0] - outputs[0][0]).abs().max()),
                        "gradient_max_abs": float((outputs[1][1] - outputs[0][1]).abs().max()),
                        "trajectory_max_abs": float((outputs[1][2] - outputs[0][2]).abs().max())})

    # Exercise a genuine singular solve and verify its failed graph is rejected
    # at the boundary rather than zero-masked into backward.
    reference = torch.zeros(1, 1, 6, dtype=dtype, device=device, requires_grad=True)
    backend = BatchedTorchBackend([physics[0]], reference, midpoint_implementation="deferred")
    zero = reference[..., :3] * 0
    quat = torch.zeros(1, 1, 4, dtype=dtype, device=device)
    quat[..., 3] = 1
    real_solve = torch.linalg.solve_ex

    def singular_solve(a, rhs, **kwargs):
        return real_solve(torch.zeros_like(a), rhs, **kwargs)

    with patch("torch.linalg.solve_ex", side_effect=singular_solve):
        backend.rollout_interval(RigidState(zero, quat, zero, zero), reference)
    try:
        backend.check_solve_errors()
    except torch._C._LinAlgError:
        pass
    else:
        raise AssertionError("Singular solve was not rejected")
    # Same graph-connected fallback as the training caller; no failed solver
    # graph is involved in backward.
    fallback = reference.sum() * 0
    fallback.backward()
    assert bool(torch.isfinite(reference.grad).all())
    assert torch.count_nonzero(reference.grad) == 0
    # Deliberately inconsistent inverse makes the first Newton Jacobian
    # exactly [[1,0,1],[0,1,0],[1,0,1]]. Check actual eager/compiled kernel info
    # propagation, without patching code inside a compiled graph.
    omega = torch.zeros(2, 7, 3, dtype=dtype, device=device)
    omega[..., 1] = 2
    eye = torch.eye(3, dtype=dtype, device=device)
    inertia = torch.diag(torch.tensor([1., 2., 3.], dtype=dtype, device=device)).expand(2, 1, 3, 3)
    solver = compiled_midpoint() if implementation in {"compiled", "compiled_substep"} else midpoint_ex
    _, failed = solver(omega, torch.zeros_like(omega), inertia,
                       eye.expand(2, 1, 3, 3), torch.ones(2, 1, 1, dtype=dtype, device=device),
                       eye, 6)
    if implementation in {"compiled_substep", "compiled_solve3", "compiled_fused3"}:
        from core.physics.rollout_fast import compiled_substep
        zeros = torch.zeros_like(omega)
        quat = torch.zeros(2, 7, 4, dtype=dtype, device=device)
        quat[..., 3] = 1
        ones = torch.ones(2, 1, 1, dtype=dtype, device=device)
        *_, failed = compiled_substep()(zeros, quat, zeros, omega, zeros, zeros,
                                       ones, inertia, eye.expand(2, 1, 3, 3),
                                       ones, eye, 6,
                                       specialize_solve=implementation != "compiled_substep",
                                       specialize_products=implementation == "compiled_fused3")
    assert bool(failed), "Kernel discarded singular solve info"
    if implementation in {"compiled_solve3", "compiled_fused3"}:
        from itertools import permutations
        from core.physics.rollout_fast import _Solve3

        # Small-motion rollouts can leave every pivot on the diagonal.
        # Force both row swaps and check the transpose solve in backward.
        rows = torch.tensor(list(permutations(range(3))), device=device)
        base = torch.tensor([[4., 1., 2.], [.5, 1., 3.], [1., 3., 1.]], dtype=dtype, device=device)
        matrix = base[rows].requires_grad_()
        rhs = torch.arange(18, dtype=dtype, device=device).reshape(6, 3).requires_grad_()
        actual, singular = _Solve3.apply(matrix, rhs)  # pyright: ignore[reportGeneralTypeIssues]
        expected = torch.linalg.solve(matrix, rhs)
        assert not bool(singular.any())
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        seed = torch.arange(1, 19, dtype=dtype, device=device).reshape(6, 3)
        for computed, reference in zip(torch.autograd.grad(actual, (matrix, rhs), seed),
                                       torch.autograd.grad(expected, (matrix, rhs), seed)):
            torch.testing.assert_close(computed, reference, rtol=1e-12, atol=1e-12)
    result = {"implementation": implementation, "device": device, "comparisons": reports,
              "singular_graph_rejected": True, "kernel_singular_info_preserved": True}
    if graph_requested:
        from core.physics.rollout_fast import RigidRolloutGraph

        # Capture an ordinary solve, then change only the inverse so replay
        # encounters the same singular Jacobian tested above.
        graph = RigidRolloutGraph()
        backend = SimpleNamespace(
            mass=torch.ones(2, 1, 1, dtype=dtype, device=device), inertia=inertia,
            inertia_inv=torch.linalg.inv(inertia).contiguous(),
            dt=torch.ones(2, 1, 1, dtype=dtype, device=device), _identity=eye,
            substeps=1, implicit_iterations=6)
        net = torch.zeros(2, 7, 6, dtype=dtype, device=device, requires_grad=True)
        position = torch.zeros(2, 8, 3, dtype=dtype, device=device)
        rotation = eye.expand(2, 8, 3, 3)
        velocity = torch.zeros(2, 8, 6, dtype=dtype, device=device)
        velocity[..., 4] = 2
        mask = torch.ones(2, 7, dtype=torch.bool, device=device)
        graph_args = (backend, net, position, rotation, velocity, mask)
        scales = dict(horizon=1, position_scale=0.01, rotation_scale=0.1)
        good = graph.run(*graph_args, **scales)
        assert good is not None and not bool(good[2])
        torch.autograd.grad(good[0] + good[1], net)
        backend.inertia_inv = eye.expand(2, 1, 3, 3).contiguous()
        bad = graph.run(*graph_args, **scales)
        assert bad is not None and bool(bad[2]), "CUDA replay lost the singular-solve flag"
        result.update(implementation="graphed_fused3", replay_singular_info_preserved=True)
    return result


def validate_real(dataset, device, implementation):
    import torch
    from omegaconf import OmegaConf
    from training.data.sample_batch import make_batch
    from training.tactohand_wrench.model import SubstepForceModel

    batch = make_batch(dataset, 1, True, max_episodes=8)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    if set(batch["dataset_mode"]) != {"rigid_trajectory"}:
        raise ValueError("Real-data gate requires a rigid trajectory")
    gt = (batch["hand_wrench_camera_30hz"].double()
          + batch["environment_wrench_camera_30hz"].double()).reshape(1, -1, 6)
    outputs = []
    perturbations = (1.01, 0.98) if implementation == "graphed_fused3" else (1.01,)
    for mode in ("reference", implementation):
        model = SimpleNamespace(
            cfg=OmegaConf.create({"metrics": {"rollout_integrator": "implicit_midpoint",
                                             "implicit_iterations": 6},
                                  "loss": {"rollout": {"implementation": mode}}}),
            rollout_horizon_10hz=4, rollout_position_scale=0.01,
            rollout_rotation_scale=0.1, rollout_position_weight=1., rollout_rotation_weight=1.)
        aligned = (torch.arange(gt.shape[1], device=device) % 3 == 0)[None, :, None]
        observations = []
        for scale in perturbations:
            predicted = (gt * scale).detach().requires_grad_(True)
            loss = SubstepForceModel._gt_pose_rollout_loss(model, torch.where(aligned, predicted, gt), batch)
            gradient, = torch.autograd.grad(loss[0], predicted)
            assert bool(torch.isfinite(gradient).all())
            observations.append((torch.stack(loss).detach(), gradient.detach()))
        if mode == "graphed_fused3":
            assert getattr(model, "_rigid_rollout_graph", None) is not None
        outputs.append(observations)
    loss_errors, grad_errors = [], []
    for (actual_loss, actual_grad), (expected_loss, expected_grad) in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-9)
        loss_errors.append(float((actual_loss - expected_loss).abs().max()))
        grad_errors.append(float((actual_grad - expected_grad).abs().max()))
    return {"dataset": dataset, "loss_max_abs": max(loss_errors), "gradient_max_abs": max(grad_errors),
            "perturbations": len(perturbations)}
