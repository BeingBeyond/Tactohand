"""HACO loss functions: VCBLoss, RegLoss, SmoothRegLoss.

Ported from HACO_RELEASE/lib/core/loss.py.
All external dependencies (lib.core.config, lib.utils.human_models) removed.
Paths are read from OmegaConf config instead.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data.resources import resolve_resource_path


class VCBLoss(nn.Module):
    """Vertex-level Class-Balanced BCE Loss.

    Parameters
    ----------
    contact_data_path : str
        Path to ``contact_data_all.npy`` (N_samples, 778) binary.
    v_regressor_336_path, v_regressor_84_path : str
        Paths to vertex regressors.
    j_regressor : Tensor (21, 778)
        MANO joint regressor.
    total_epochs : int
        Used to pre-compute per-epoch CB weights.
    beta : float
        CB loss beta (default 0.9999).
    v_type : str
        One of ``mesh``, ``mesh_336``, ``mesh_84``, ``joint``.
    """

    def __init__(
        self,
        contact_data_path: str | None,
        v_regressor_336_path: str | None,
        v_regressor_84_path: str | None,
        j_regressor: torch.Tensor,
        total_epochs: int = 50,
        beta: float = 0.9999,
        v_type: str = "mesh",
        contact_statistics_path: str | None = None,
    ):
        super().__init__()
        self.beta = beta
        self.v_type = v_type

        # Full MANO mesh supervision does not use reduced-mesh regressors.
        # Load only the selected representation's asset; legacy reduced-mesh
        # and joint modes retain their existing calculations.
        self.v_regressor_336 = self._load_regressor(v_regressor_336_path, "336") if v_type == "mesh_336" else None
        self.v_regressor_84 = self._load_regressor(v_regressor_84_path, "84") if v_type == "mesh_84" else None
        self.mano_regressor = j_regressor.float()

        if contact_statistics_path:
            artifact = np.load(resolve_resource_path(contact_statistics_path), allow_pickle=False)
            version = int(np.asarray(artifact["schema_version"]).item())
            if version != 1:
                raise ValueError(
                    f"unsupported HACO contact statistics schema {version}"
                )
            counts = torch.from_numpy(np.asarray(
                artifact["class_counts"], dtype=np.int64
            )).float()
            if tuple(counts.shape) != (778, 2):
                raise ValueError("HACO class_counts must have shape [778, 2]")
            self.samples_per_cls = self._get_samples_per_cls_from_counts(
                counts, total_epochs
            )
        else:
            if not contact_data_path:
                raise ValueError("Set loss.haco.contact_statistics_path or loss.haco.contact_data_path")
            all_contact_samples = torch.from_numpy(
                np.load(contact_data_path)).to(torch.int32)
            self.samples_per_cls = self._get_samples_per_cls(
                all_contact_samples, total_epochs)

    @staticmethod
    def _load_regressor(path: str | None, size: str) -> torch.Tensor:
        if not path or not Path(path).expanduser().is_file():
            raise FileNotFoundError(
                f"mesh_{size} loss requires loss.haco.v_regressor_{size}_path; "
                "provide the matching external HACO regressor asset.")
        return torch.tensor(np.load(Path(path).expanduser()), dtype=torch.float32)

    def _get_samples_per_cls(
        self, all_contact_samples: torch.Tensor, total_epoch: int,
        total_sample_num: int = 1000,
    ) -> dict[int, torch.Tensor]:
        count_v_0 = (all_contact_samples == 0).sum(dim=0)
        count_v_1 = (all_contact_samples == 1).sum(dim=0)
        all_v_contact_counts = torch.stack([count_v_0, count_v_1], dim=1).float()

        return self._get_samples_per_cls_from_counts(
            all_v_contact_counts, total_epoch, total_sample_num
        )

    def _get_samples_per_cls_from_counts(
        self, all_v_contact_counts: torch.Tensor, total_epoch: int,
        total_sample_num: int = 1000,
    ) -> dict[int, torch.Tensor]:

        all_contact_counts = all_v_contact_counts.sum(dim=0)

        denom_v = all_v_contact_counts.sum(dim=1, keepdim=True).clamp(min=1e-6)
        samples_per_cls_v = all_v_contact_counts / denom_v * total_sample_num

        denom = all_contact_counts.sum().clamp(min=1e-6)
        samples_per_cls = all_contact_counts / denom * total_sample_num

        result: dict[int, torch.Tensor] = {}
        v_weight_max = 0.3
        for epoch in range(total_epoch):
            v_weight = v_weight_max * (epoch / max(total_epoch - 1, 1))
            all_weight = 1.0 - v_weight
            blended = (v_weight * samples_per_cls_v.clone()
                       + all_weight * samples_per_cls.clone())
            blended = blended.clamp(min=20.0)
            blended = blended / blended.sum(dim=1, keepdim=True) * total_sample_num
            result[epoch] = blended

        # Apply vertex/joint regressors
        if self.v_type == "mesh":
            pass
        elif self.v_type == "mesh_336":
            assert self.v_regressor_336 is not None
            for k in result:
                result[k] = torch.matmul(self.v_regressor_336, result[k])
        elif self.v_type == "mesh_84":
            assert self.v_regressor_84 is not None
            for k in result:
                result[k] = torch.matmul(self.v_regressor_84, result[k])
        elif self.v_type == "joint":
            for k in result:
                result[k] = torch.matmul(self.mano_regressor, result[k])
        else:
            raise ValueError(f"Unknown v_type: {self.v_type}")

        return result

    def _compute_class_weight(self, samples_per_cls: torch.Tensor) -> torch.Tensor:
        effective_num = 1.0 - torch.pow(self.beta, samples_per_cls)
        class_weights = (1.0 - self.beta) / effective_num
        class_weights = class_weights / class_weights.sum(dim=1, keepdim=True)
        return class_weights

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, epoch: int,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gt = gt.to(pred.device).float()
        if valid is not None:
            if not valid.any().item():
                # Graph-connected zero (not a bare constant): contact-only
                # batches must keep the contact-head parameters in the
                # autograd graph, or DDP ranks with different data present
                # different unused-parameter sets and desynchronize the
                # gradient-bucket collective order.
                return pred.sum() * 0.0
            pred, gt = pred[valid], gt[valid]

        if epoch not in self.samples_per_cls:
            epoch = max(self.samples_per_cls.keys())
        class_weights = self._compute_class_weight(self.samples_per_cls[epoch])

        # Expand per-vertex class weights to batch
        w = class_weights[None, :, :].to(pred.device)
        g = gt.long()[..., None]
        cw = torch.gather(w.expand(gt.size(0), -1, -1), 2, g)[..., 0]

        vcb_loss = F.binary_cross_entropy_with_logits(
            pred, gt, reduction="none"
        ) * cw
        pred_prob = pred.sigmoid()
        fp_reg_loss = ((gt == 0).float() * pred_prob).mean()
        return vcb_loss.mean() + 0.5 * fp_reg_loss


class RegLoss(nn.Module):
    """L1 regularization toward contact means."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss(reduction="mean")

    def forward(
        self, pred: torch.Tensor, gt_means: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gt_means = gt_means.to(pred.device).float()
        if valid is not None:
            if not valid.any().item():
                return torch.tensor(0.0, device=pred.device)
            pred, gt_means = pred[valid], gt_means[valid]
        return self.criterion(pred.sigmoid(), gt_means)


class SmoothRegLoss(nn.Module):
    """Mesh-topology smoothness regularization using MANO faces."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _build_adjacency(num_verts: int, faces: torch.Tensor) -> torch.Tensor:
        device = faces.device
        row_idx = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2],
                             faces[:, 1], faces[:, 2], faces[:, 0]])
        col_idx = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0],
                             faces[:, 0], faces[:, 1], faces[:, 2]])
        indices = torch.stack([row_idx, col_idx], dim=0)
        values = torch.ones(indices.shape[1], device=device)
        adj = torch.sparse_coo_tensor(indices, values,
                                      (num_verts, num_verts), device=device)
        self_loops = torch.eye(num_verts, device=device).to_sparse()
        return adj + self_loops

    def forward(self, pred: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        batch_size, num_verts = pred.shape
        pred_contact = pred.sigmoid()
        pred_non_contact = 1.0 - pred_contact
        faces = faces.to(pred.device).long()

        adj = self._build_adjacency(num_verts, faces)

        prop_c = torch.stack([
            torch.sparse.mm(adj, pred_contact[b].unsqueeze(-1)).squeeze(-1)
            for b in range(batch_size)
        ], dim=0)
        prop_nc = torch.stack([
            torch.sparse.mm(adj, pred_non_contact[b].unsqueeze(-1)).squeeze(-1)
            for b in range(batch_size)
        ], dim=0)

        norm_c = prop_c / (prop_c.max(dim=1, keepdim=True)[0] + 1e-6)
        norm_nc = prop_nc / (prop_nc.max(dim=1, keepdim=True)[0] + 1e-6)

        iso_c = torch.abs(pred_contact - norm_c)
        iso_nc = torch.abs(pred_non_contact - norm_nc)

        total_c = prop_c.sum(dim=1)
        total_nc = prop_nc.sum(dim=1)

        iso_score = iso_c.sum(dim=1) + iso_nc.sum(dim=1)
        norm_factor = total_c + total_nc + 1e-3
        return torch.log1p(iso_score / norm_factor).mean()
