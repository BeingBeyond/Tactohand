"""Compact TensorBoard Custom Scalars layout for formal TactoHand training."""


def custom_scalar_layout() -> dict:
    return {
        "Step / Optimization": {
            "Total objective": ["Multiline", ["step/objective/total"]],
            "Learning rate": ["Multiline", ["lr-AdamW/pg1", "lr-AdamW/pg2"]],
        },
        "Epoch / Objectives": {
            "Total": ["Multiline", [
                "epoch/objective/train/total",
                "epoch/objective/val/total",
            ]],
            "Point force and total wrench": ["Multiline", [
                "loss/train/point_force",
                "loss/val/point_force",
                "loss/train/wrench_force",
                "loss/val/wrench_force",
                "loss/train/wrench_torque",
                "loss/val/wrench_torque",
            ]],
            "Contact and proximity": ["Multiline", [
                "epoch/objective/train/contact",
                "epoch/objective/val/contact",
                "epoch/objective/train/proximity",
                "epoch/objective/val/proximity",
            ]],
            "Weighted rollout": ["Multiline", [
                "epoch/objective/train/rollout_weighted",
                "epoch/objective/val/rollout_weighted",
            ]],
        },
        "Epoch / Physical Quality": {
            "Point force RMSE (N)": ["Multiline", [
                "epoch/metrics/train/force/point_rmse_n",
            ]],
            "Resultant force RMSE (N)": ["Multiline", [
                "epoch/metrics/train/force/resultant_rmse_n",
            ]],
            "Resultant torque RMSE (Nm)": ["Multiline", [
                "epoch/metrics/train/force/torque_rmse_nm",
            ]],
            "Contact F1": ["Multiline", [
                "epoch/metrics/train/contact/f1",
            ]],
            "Proximity MAE": ["Multiline", [
                "epoch/metrics/train/proximity/mae",
            ]],
        },
        "Epoch / Rollout (0.4s)": {
            "Position ADE (m)": ["Multiline", [
                "epoch/rollout/train/h4_position_ade_m",
                "epoch/rollout/val/h4_position_ade_m",
            ]],
            "Rotation ADE (rad)": ["Multiline", [
                "epoch/rollout/train/h4_rotation_ade_rad",
                "epoch/rollout/val/h4_rotation_ade_rad",
            ]],
            "Explosion rate": ["Multiline", [
                "epoch/rollout/train/h4_explosion_rate",
                "epoch/rollout/val/h4_explosion_rate",
            ]],
        },
        "Epoch / Schedule": {
            "Rollout objective weight": ["Multiline", [
                "epoch/schedule/rollout_weight",
            ]],
        },
    }
