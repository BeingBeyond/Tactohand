"""Run recoverable windowed two-hand articulated QP on ARCTIC Stage-1 sidecars."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.physics.articulated_contact_qp import (  # noqa: E402
    apply_direct_environment_balance,
    prepare_direct_environment_balance,
    solve_articulated_contact_qp_windowed,
)


def atomic_savez(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream: np.savez_compressed(stream,**values)
        os.replace(temporary,path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def valid_stage2(path: Path, solver_backend: str, window_size: int, overlap: int,
                 solver_eps: float, solver_polishing: bool) -> bool:
    try:
        with np.load(path) as z:
            return (str(z["status"])=="optimal" and int(z["format_version"])==15
                    and str(z["environment_model"])=="strict_hand_qp_or_complete_environment_wrench_v2"
                    and str(z["environment_detector"])=="hand_interval_height_lifecycle_v1"
                    and str(z["stage1_method"])=="articulated_newton_compatible_inverse_dynamics_v2"
                    and str(z["solver_backend"])==solver_backend
                    and np.isclose(float(z["solver_eps_abs"]),solver_eps,rtol=0,atol=1e-15)
                    and np.isclose(float(z["solver_eps_rel"]),solver_eps,rtol=0,atol=1e-15)
                    and bool(z["solver_polishing"])==solver_polishing
                    and int(z["window_size"])==window_size
                    and int(z["overlap"])==overlap)
    except (OSError,KeyError,ValueError):return False


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path,nargs="*")
    parser.add_argument("--input-dir",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path(
        "artifacts/label_solver/arctic_normalized_forces"))
    parser.add_argument("--window-size",type=int,default=32);parser.add_argument("--overlap",type=int,default=4)
    parser.add_argument("--solver-backend",choices=("osqp","piqp"),default="osqp")
    parser.add_argument("--osqp-eps", type=float, default=float(
        os.environ.get("ARTICULATED_OSQP_EPS", "1e-8")))
    parser.add_argument("--osqp-polishing", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("ARTICULATED_OSQP_POLISHING", "1") == "1")
    parser.add_argument("--force-temporal-regularizer",type=float,default=1.0)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--num-shards",type=int,default=1);parser.add_argument("--shard-id",type=int,default=0)
    args=parser.parse_args()
    if not np.isfinite(args.osqp_eps) or args.osqp_eps <= 0:
        parser.error("--osqp-eps must be finite and positive")
    if args.num_shards<1 or not 0<=args.shard_id<args.num_shards:
        parser.error("invalid shard configuration")
    inputs=list(args.inputs)
    if args.input_dir:inputs.extend(sorted(args.input_dir.glob("episode_*.npz")))
    inputs=sorted(set(inputs));inputs=[path for index,path in enumerate(inputs) if index%args.num_shards==args.shard_id]
    if not inputs:parser.error("no input sidecars")
    for path in inputs:
        z=np.load(path);episode=int(z["episode_index"]);output=args.output_dir/f"episode_{episode:06d}.npz"
        if not args.overwrite and valid_stage2(
                output,args.solver_backend,args.window_size,args.overlap,
                args.osqp_eps,args.osqp_polishing):
            print({"status":"skipped","episode":episode,"output":str(output)},flush=True);continue
        original_target=np.asarray(z["generalized_contact_wrench"],np.float64)
        candidate=np.stack((z["left_contact_candidate"],z["right_contact_candidate"]),1)
        points=np.stack((z["left_vertices_world"],z["right_vertices_world"]),1)
        normals=np.stack((z["left_normals_world"],z["right_normals_world"]),1)
        priority=np.stack((z["left_contact_priority"],z["right_contact_priority"]),1)
        contact_part=np.stack((z["left_contact_part"],z["right_contact_part"]),1)
        # Distance alone cannot make a vertex usable when its reconstructed
        # MANO geometry is non-finite or has no articulated-part assignment.
        geometry_valid=(np.isfinite(points).all(axis=-1)
                        & np.isfinite(normals).all(axis=-1)
                        & np.isfinite(priority) & (contact_part >= 0))
        candidate &= geometry_valid
        # Contact annotations alone do not guarantee that the QP has a usable
        # force variable after geometry/normal filtering.  Match the rigid-v5
        # contract: only frames with at least one usable hand-force candidate
        # enter the QP; every other frame is balanced exactly by the abstract
        # environment, independent of the table lifecycle classification.
        hand_force_active = np.asarray(z["hand_contact_active"], bool) & candidate.any(axis=(1, 2))
        effective_target,direct_support=prepare_direct_environment_balance(
            original_target,hand_force_active)
        qp_candidate = candidate.copy()
        qp_candidate[direct_support] = False
        qp_contact_part = contact_part.copy()
        qp_contact_part[direct_support] = -1
        qp_environment_active = np.asarray(z["environment_contact_active"], bool).copy()
        qp_environment_active[direct_support] = False
        result=solve_articulated_contact_qp_windowed(
            target=effective_target,system_com=z["system_com_world"],
            joint_axis=z["joint_axis_world"],joint_pivot=z["joint_pivot_world"],
            environment_point=z["environment_contact_world"],
            environment_active=qp_environment_active,
            points=points,
            normals=normals,
            priority=priority,
            candidate=qp_candidate,
            contact_part=qp_contact_part,
            window_size=args.window_size,overlap=args.overlap,
            # Slack is used only by one-frame feasibility probes.  Frames
            # which need more than the normal strict tolerance are removed
            # from the recovered strict window and explicitly marked invalid.
            recovery_max_force_slack_n=20.0,
            recovery_max_torque_slack_nm=2.0,
            allow_wrench_slack=False,
            temporal_regularizer=args.force_temporal_regularizer,
            solver_backend=args.solver_backend,
            eps_abs=args.osqp_eps,eps_rel=args.osqp_eps,
            osqp_polishing=args.osqp_polishing)
        # A non-empty contact patch can still be physically insufficient: for
        # example, all candidates may lie on one hand and one articulated part
        # and cannot span the requested 7D wrench under strict friction cones.
        # Keep the strict model.  If per-frame recovery proves it infeasible,
        # record the fallback explicitly and let the environment own the full
        # external wrench rather than emitting a partial/zero force label.
        strict_hand_qp_infeasible = hand_force_active & ~result["quality_valid"]
        direct_balance = direct_support | strict_hand_qp_infeasible
        effective_target[strict_hand_qp_infeasible] = 0.0
        apply_direct_environment_balance(result,original_target,direct_balance)
        effective_environment_active = qp_environment_active | direct_balance
        stats=[{"start":0,"stop":len(z["generalized_contact_wrench"]),"status":"optimal",
                "windows":result.pop("window_stats")}]
        length=len(z["generalized_contact_wrench"])
        result["force_label_valid"]=result.pop("quality_valid")
        result["table_contact_excluded"]=np.zeros(length,dtype=bool)
        normalization_json = (
            np.array(str(z["normalization_json"]))
            if "normalization_json" in z.files
            else np.array(json.dumps({
                "reference_total_mass_kg": 1.0,
                "force_normalization": "per_unit_total_object_mass",
                "specific_force_unit": "N/kg",
                "specific_torque_unit": "N*m/kg",
                "mass_scaling_contract": "force_and_torque_scale_linearly_with_total_mass",
                "link_mass_distribution": "closed_mesh_volume_ratio",
            }, sort_keys=True))
        )
        result.pop("allow_wrench_slack", None)
        atomic_savez(output,format_version=np.int64(15),
                     normalization_json=normalization_json,
                     environment_model=np.array(
                         "strict_hand_qp_or_complete_environment_wrench_v2"),
                     environment_detector=np.array(str(z["environment_detector"])),
                     environment_contact_active=effective_environment_active,
                     geometric_environment_contact_active=z["environment_contact_active"],
                     allow_wrench_slack=np.bool_(False),
                     force_temporal_regularizer=np.float64(args.force_temporal_regularizer),
                     solver_backend=np.array(args.solver_backend),
                     solver_eps_abs=np.float64(args.osqp_eps),
                     solver_eps_rel=np.float64(args.osqp_eps),
                     solver_polishing=np.bool_(args.osqp_polishing),
                     stage1_method=np.array(str(z["stage1_method"])),
                     gaussian_sigma_s=z["gaussian_sigma_s"],
                     gaussian_sigma_pos_s=z["gaussian_sigma_pos_s"],
                     gaussian_sigma_rot_s=z["gaussian_sigma_rot_s"],
                     original_generalized_contact_wrench=original_target,
                     effective_generalized_contact_wrench=effective_target,
                     direct_environment_support_projection=direct_balance,
                     hand_free_direct_environment_balance=direct_support,
                     strict_hand_qp_infeasible_environment_fallback=strict_hand_qp_infeasible,
                     hand_force_active=hand_force_active,
                     hand_qp_active=hand_force_active & ~strict_hand_qp_infeasible,
                     force_allocation_policy=np.array(
                         "strict_hand_qp_infeasible_environment_balance_v6"),
                     hand_contact_active=z["hand_contact_active"],
                     tactile_contact_min_vertices=z["tactile_contact_min_vertices"],
                     direct_environment_support_projection_delta=effective_target-original_target,
                     episode_index=np.int64(episode),source_npz=np.array(str(path)),
                     segment_stats_json=np.array(json.dumps(stats,sort_keys=True)),**result)
        valid=result["force_label_valid"]
        print({"status":"ok" if valid.all() else "partial","episode":episode,"segments":len(stats),
               "valid_frames":int(valid.sum()),
               "residual_max":float(np.nanmax(np.abs(result["residual"][valid]))) if valid.any() else float("nan"),
               "closure_residual_max":float(np.nanmax(np.abs(result["closure_residual"][valid]))) if valid.any() else float("nan"),
               "force_slack_max_n":float(np.max(np.linalg.norm(result["force_slack_world"],axis=1))),
               "torque_slack_max_nm":float(np.max(np.linalg.norm(result["torque_slack_world"],axis=1))),
               "direct_environment_support_frames":int(direct_balance.sum()),
               "strict_hand_qp_infeasible_fallback_frames":int(
                   strict_hand_qp_infeasible.sum()),
               "output":str(output)},flush=True)


if __name__=="__main__":main()
