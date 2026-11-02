"""Synthetic unit check; no learned prediction or dataset required."""
import torch
from tactohand_preview import aggregate_vertex_forces


def main():
    force = torch.tensor([[0., 0., 1.], [0., 0., 1.]], requires_grad=True)
    lever = torch.tensor([[0.1, 0., 0.], [-0.1, 0., 0.]])
    wrench = aggregate_vertex_forces(force, lever)
    torch.testing.assert_close(wrench, torch.tensor([0., 0., 2., 0., 0., 0.]))
    wrench.square().sum().backward()
    assert force.grad is not None and torch.isfinite(force.grad).all()
    print("Wrench [N, N, N, N*m, N*m, N*m]:", wrench.detach().tolist())


if __name__ == "__main__":
    main()
