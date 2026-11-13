"""Direct OSQP allocation for two hands on a two-part articulated object."""
from __future__ import annotations

import time

import numpy as np
import osqp
from scipy import sparse

from .articulated_contact_geometry import contact_generalized_map
from .contact_constraints import support_force_world
from .contact_constraints import friction_facets, normalize_candidate_normals


def prepare_direct_environment_balance(
    target: np.ndarray,
    hand_force_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove hand-free frames from the QP and return their direct mask.

    The environment owns the complete external 6D wrench in these frames.  The
    articulated joint coordinate is balanced independently as an internal
    torque.  Both are restored after the QP, so the QP target is exactly zero
    and contributes no hand, table-support, or joint variables on direct
    frames.
    """
    target = np.asarray(target, np.float64)
    hand_force_active = np.asarray(hand_force_active, bool)
    if target.ndim != 2 or target.shape[1] != 7:
        raise ValueError("target must have shape (T,7)")
    if hand_force_active.shape != (len(target),):
        raise ValueError("hand_force_active must have shape (T,)")
    direct = ~hand_force_active
    qp_target = target.copy()
    qp_target[direct] = 0.0
    return qp_target, direct


def apply_direct_environment_balance(
    result: dict,
    original_target: np.ndarray,
    direct: np.ndarray,
) -> dict:
    """Overlay exact environment-owned labels on hand-free QP output."""
    target = np.asarray(original_target, np.float64)
    direct = np.asarray(direct, bool)
    if target.ndim != 2 or target.shape[1] != 7:
        raise ValueError("original_target must have shape (T,7)")
    if direct.shape != (len(target),):
        raise ValueError("direct must have shape (T,)")

    result["vertex_force_world"][direct] = 0.0
    result["environment_force_world"][direct] = target[direct, :3]
    # This compatibility scalar remains a non-negative force magnitude.  The
    # complete vector force and torque are retained separately below.
    result["environment_support_n"][direct] = np.linalg.norm(target[direct, :3], axis=1)
    result["joint_internal_torque"][direct] = target[direct, 6]
    result["force_slack_world"][direct] = 0.0
    result["torque_slack_world"][direct] = 0.0
    result["fitted_generalized_wrench"][direct] = target[direct]
    result["residual"][direct] = 0.0
    result["closure_residual"][direct] = 0.0
    result["quality_valid"][direct] = True

    environment_wrench = np.zeros_like(target)
    environment_wrench[:, :3] = result["environment_force_world"]
    environment_wrench[direct, :6] = target[direct, :6]
    result["environment_generalized_wrench"] = environment_wrench
    result["direct_environment_balance"] = direct.copy()
    return result


def solve_articulated_contact_qp(
    target: np.ndarray,
    system_com: np.ndarray,
    joint_axis: np.ndarray,
    joint_pivot: np.ndarray,
    environment_point: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    priority: np.ndarray,
    candidate: np.ndarray,
    contact_part: np.ndarray,
    *,
    environment_active: np.ndarray | None = None,
    mu: float = 1.2,
    friction_sides: int = 6,
    force_regularizer: float = 1.0,
    temporal_regularizer: float = 1.0,
    environment_temporal_regularizer: float = 1.0,
    environment_support_regularizer: float = 1e-4,
    joint_internal_regularizer: float = 100.0,
    joint_internal_temporal_regularizer: float = 10.0,
    allow_wrench_slack: bool = False,
    wrench_slack_regularizer: float = 100.0,
    force_slack_scale_n: float = 1.0,
    torque_slack_scale_nm: float = 0.05,
    max_force_slack_n: float = 0.1,
    max_torque_slack_nm: float = 0.005,
    torque_scale_nm: float = .1,
    joint_torque_scale_nm: float = .1,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-8,
    solver_backend: str = "osqp",
    osqp_polishing: bool = True,
    initial_vertex_force_world: np.ndarray | None = None,
    initial_environment_support_n: np.ndarray | None = None,
    initial_joint_internal_torque: np.ndarray | None = None,
) -> dict:
    """Allocate exact 7D generalized wrench to left/right MANO forces.

    Input hand axes use ``H=2`` in left/right order and ``V=778``. A smooth
    vertical environment-support scalar is included, matching the canonical
    DexYCB table-support abstraction under total normalized mass 1 kg.
    """
    begin = time.perf_counter()
    phase_time: dict[str, float] = {}
    target = np.asarray(target, np.float64); T = len(target)
    expected = (T, 2, 778)
    if target.shape != (T, 7) or points.shape != (*expected, 3) or normals.shape != (*expected, 3):
        raise ValueError("inconsistent articulated QP geometry shapes")
    if priority.shape != expected or candidate.shape != expected or contact_part.shape != expected:
        raise ValueError("inconsistent articulated QP contact shapes")
    if solver_backend not in ("auto", "osqp", "piqp"):
        raise ValueError("solver_backend must be auto, osqp or piqp")
    if environment_active is None:
        environment_active = np.ones(T,dtype=bool)
    environment_active = np.asarray(environment_active,dtype=bool)
    if environment_active.shape != (T,):
        raise ValueError("environment_active must have shape (T,)")
    if solver_backend == "auto":
        solver_backend = "osqp"
    normals = normalize_candidate_normals(normals, candidate)
    validated = time.perf_counter()
    phase_time["validate"] = validated - begin

    active: list[list[tuple[int, int, int]]] = []
    for t in range(T):
        frame = []
        for hand in range(2):
            for part in range(2):
                ids = np.flatnonzero(candidate[t, hand] & (contact_part[t, hand] == part)
                                     & np.isfinite(normals[t, hand]).all(axis=1))
                frame.extend((hand, int(vertex), part) for vertex in ids)
        active.append(frame)
    selected = time.perf_counter()
    phase_time["candidate_index"] = selected - validated
    offsets = np.zeros(T + 1, np.int64)
    for t, frame in enumerate(active): offsets[t + 1] = offsets[t] + 3 * len(frame)
    force_size = int(offsets[-1]); environment_offset = force_size
    joint_internal_offset = environment_offset + T
    slack_offset = joint_internal_offset + T
    slack_size = 6*T if allow_wrench_slack else 0
    nvar = force_size + 2*T + slack_size
    # Assemble OSQP's upper-triangular quadratic matrix directly. Repeated
    # coordinates are summed by COO -> CSC conversion; avoiding LIL scalar
    # get/set operations makes long articulated windows substantially faster.
    p_rows=[]; p_cols=[]; p_vals=[]
    def add_p(left: int, right: int, value: float) -> None:
        if value:
            if left > right: left,right=right,left
            p_rows.append(left);p_cols.append(right);p_vals.append(value)
    total_active = max(sum(map(len, active)), 1)
    lookup = []
    for t, frame in enumerate(active):
        current = {}
        for local, (hand, vertex, part) in enumerate(frame):
            identity = part * 1556 + hand * 778 + vertex
            current[identity] = local
            weight = 1.0 / (float(priority[t, hand, vertex]) ** 2 + 1e-4)
            base = int(offsets[t] + 3 * local)
            for component in range(3):
                add_p(base+component,base+component,2*force_regularizer*weight/total_active)
        lookup.append(current)

    def add_difference(terms: list[tuple[int, float]], coefficient: float) -> None:
        for left_index,(left,a) in enumerate(terms):
            for right,b in terms[left_index:]:
                add_p(left,right,2.0*coefficient*a*b)

    first_count = sum(len(set(lookup[t]) | set(lookup[t+1])) for t in range(T-1))
    if first_count:
        coefficient = temporal_regularizer / first_count
        for t in range(T-1):
            for identity in set(lookup[t]) | set(lookup[t+1]):
                for component in range(3):
                    terms=[]
                    if identity in lookup[t]: terms.append((int(offsets[t]+3*lookup[t][identity]+component),-1.0))
                    if identity in lookup[t+1]: terms.append((int(offsets[t+1]+3*lookup[t+1][identity]+component),1.0))
                    add_difference(terms, coefficient)
    for t in range(T):
        add_p(environment_offset+t,environment_offset+t,2.0*environment_support_regularizer/T)
    for t in range(T-1):
        add_difference([(environment_offset+t,-1.0),(environment_offset+t+1,1.0)],
                       environment_temporal_regularizer/max(T-1,1))
    for t in range(T):
        add_p(joint_internal_offset+t,joint_internal_offset+t,
              2.0*joint_internal_regularizer/(joint_torque_scale_nm**2*T))
    for t in range(T-1):
        add_difference([(joint_internal_offset+t,-1.0),(joint_internal_offset+t+1,1.0)],
                       joint_internal_temporal_regularizer/(joint_torque_scale_nm**2*max(T-1,1)))
    if allow_wrench_slack:
        for t in range(T):
            for component in range(3):
                add_p(slack_offset+6*t+component,slack_offset+6*t+component,
                      2.0*wrench_slack_regularizer/(force_slack_scale_n**2*T))
                add_p(slack_offset+6*t+3+component,slack_offset+6*t+3+component,
                      2.0*wrench_slack_regularizer/(torque_slack_scale_nm**2*T))
    P=sparse.coo_matrix((p_vals,(p_rows,p_cols)),shape=(nvar,nvar)).tocsc()
    P.sum_duplicates();P.eliminate_zeros()
    q = np.zeros(nvar, np.float64)
    p_assembled = time.perf_counter()
    phase_time["assemble_P"] = p_assembled - selected

    rows=[]; cols=[]; vals=[]; lower=[]; upper=[]
    def constraint(columns, values, lo, hi):
        row=len(lower); rows.extend([row]*len(columns)); cols.extend(columns); vals.extend(values)
        lower.append(lo); upper.append(hi)
    scales = np.array((1.,1.,1.,torque_scale_nm,torque_scale_nm,torque_scale_nm,joint_torque_scale_nm))
    for t, frame in enumerate(active):
        support_column=environment_offset+t
        max_support=np.inf if environment_active[t] else 0.0
        constraint([support_column],[1.0],0.0,max_support)
        if allow_wrench_slack:
            for component in range(3):
                constraint([slack_offset+6*t+component],[1.0],
                           -max_force_slack_n,max_force_slack_n)
                constraint([slack_offset+6*t+3+component],[1.0],
                           -max_torque_slack_nm,max_torque_slack_nm)
        maps=[]
        for hand, vertex, part in frame:
            maps.append(contact_generalized_map(points[t,hand,vertex],part,system_com[t],joint_axis[t],joint_pivot[t]))
        for component in range(7):
            columns=[]; values=[]
            for local,mapping in enumerate(maps):
                for force_component in range(3):
                    value=float(mapping[component,force_component]/scales[component])
                    if value: columns.append(int(offsets[t]+3*local+force_component)); values.append(value)
            # Match DexYCB: table support contributes only a vertical world
            # force, without a mesh-point-dependent torque.
            value=1.0 if component == 2 else 0.0
            if value: columns.append(support_column); values.append(value)
            if component == 6:
                columns.append(joint_internal_offset+t); values.append(1.0/joint_torque_scale_nm)
            if allow_wrench_slack and component < 6:
                columns.append(slack_offset+6*t+component)
                values.append(1.0/scales[component])
            rhs=float(target[t,component]/scales[component]); constraint(columns,values,rhs,rhs)
        for local,(hand,vertex,_) in enumerate(frame):
            normal=normals[t,hand,vertex]
            base=int(offsets[t]+3*local); columns=[base,base+1,base+2]
            constraint(columns,(-normal).tolist(),-np.inf,0.0)
            for facet in friction_facets(normal, mu, friction_sides):
                constraint(columns,facet.tolist(),-np.inf,0.0)
    a_triplets_built = time.perf_counter()
    phase_time["assemble_A_triplets"] = a_triplets_built - p_assembled
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(len(lower),nvar)).tocsc(); A.eliminate_zeros()
    assembled=time.perf_counter()
    phase_time["sparse_conversion"] = assembled - a_triplets_built
    lower_array=np.asarray(lower);upper_array=np.asarray(upper)
    warm_start_used = False
    if solver_backend == "osqp":
        solver=osqp.OSQP()
        solver.setup(P=P,q=q,A=A,l=lower_array,u=upper_array,eps_abs=eps_abs,eps_rel=eps_rel,
                     max_iter=200000,polishing=osqp_polishing,verbose=False)
        setup=time.perf_counter()
        warm_start_used = initial_vertex_force_world is not None
        if warm_start_used:
            initial_force = np.asarray(initial_vertex_force_world, np.float64)
            if initial_force.shape != points.shape:
                raise ValueError("initial_vertex_force_world must match points")
            initial_support = np.zeros(T) if initial_environment_support_n is None else np.asarray(
                initial_environment_support_n, np.float64)
            initial_joint = np.zeros(T) if initial_joint_internal_torque is None else np.asarray(
                initial_joint_internal_torque, np.float64)
            if initial_support.shape != (T,) or initial_joint.shape != (T,):
                raise ValueError("initial support and joint torque must have shape (T,)")
            initial_x = np.zeros(nvar)
            for frame_index, frame in enumerate(active):
                for local, (hand, vertex, _) in enumerate(frame):
                    base = int(offsets[frame_index] + 3 * local)
                    initial_x[base:base + 3] = initial_force[frame_index, hand, vertex]
            initial_x[environment_offset:joint_internal_offset] = initial_support
            initial_x[joint_internal_offset:slack_offset] = initial_joint
            solver.warm_start(x=initial_x)
        result=solver.solve(); solved=time.perf_counter()
        if str(result.info.status).lower() not in ("solved","solved inaccurate"):
            raise RuntimeError(f"articulated contact QP failed: {result.info.status}")
        solution=np.asarray(result.x)
        iterations=int(result.info.iter)
        objective=float(result.info.obj_val)
        backend_timing = {
            "setup": float(getattr(result.info, "setup_time", np.nan)),
            "solve": float(getattr(result.info, "solve_time", np.nan)),
            "polish": float(getattr(result.info, "polish_time", np.nan)),
            "run": float(getattr(result.info, "run_time", np.nan)),
        }
    else:
        try:
            import piqp
        except ImportError as error:
            raise RuntimeError("PIQP backend requested but piqp is not installed") from error
        equality = np.isfinite(lower_array) & (lower_array == upper_array)
        A_eq = A[equality].tocsc()
        b_eq = lower_array[equality]
        G = A[~equality].tocsc()
        h_l = lower_array[~equality]
        h_u = upper_array[~equality]
        solver=piqp.SparseSolver()
        solver.settings.eps_abs=eps_abs
        solver.settings.eps_rel=eps_rel
        solver.settings.eps_duality_gap_abs=eps_abs
        solver.settings.eps_duality_gap_rel=eps_rel
        solver.settings.compute_timings=True
        solver.settings.verbose=False
        solver.setup(P,q,A_eq,b_eq,G,h_l,h_u)
        setup=time.perf_counter(); status=solver.solve(); solved=time.perf_counter()
        result=solver.result
        if status != piqp.PIQP_SOLVED:
            raise RuntimeError(f"articulated contact QP failed: {status}")
        solution=np.asarray(result.x)
        iterations=int(result.info.iter)
        objective=float(result.info.primal_obj)
        backend_timing = {
            "setup": float(result.info.setup_time),
            "solve": float(result.info.solve_time),
            "polish": 0.0,
            "run": float(result.info.run_time),
            "kkt_factor": float(result.info.kkt_factor_time),
            "kkt_solve": float(result.info.kkt_solve_time),
        }
    forces=np.zeros((T,2,778,3),np.float64)
    for t,frame in enumerate(active):
        for local,(hand,vertex,part) in enumerate(frame):
            force=solution[offsets[t]+3*local:offsets[t]+3*local+3]
            forces[t,hand,vertex]=force
    raw_environment_support_n=solution[environment_offset:joint_internal_offset].copy()
    environment_support_n=raw_environment_support_n.copy()
    np.maximum(environment_support_n,0.0,out=environment_support_n)
    environment_support_n[~environment_active] = 0.0
    environment_force=support_force_world(
        environment_support_n, np.array((0.0,0.0,1.0)),
        contact_active=environment_active)
    # Recompute after deterministic clamping of OSQP's feasibility-tolerance
    # residue so off-table labels are exactly zero, not merely approximately.
    fitted=np.zeros((T,7),np.float64)
    for t,frame in enumerate(active):
        for local,(hand,vertex,part) in enumerate(frame):
            force=forces[t,hand,vertex]
            fitted[t]+=contact_generalized_map(points[t,hand,vertex],part,system_com[t],joint_axis[t],joint_pivot[t])@force
        fitted[t,:3]+=environment_force[t]
        fitted[t,6]+=solution[joint_internal_offset+t]
    wrench_slack=(solution[slack_offset:slack_offset+slack_size].reshape(T,6)
                  if allow_wrench_slack else np.zeros((T,6),np.float64))
    if allow_wrench_slack:
        wrench_slack[:,2]+=raw_environment_support_n-environment_support_n
    generalized_slack=np.pad(wrench_slack,((0,0),(0,1)))
    postprocessed = time.perf_counter()
    phase_time["postprocess"] = postprocessed - solved
    phase_time["python_build_total"] = assembled - begin
    phase_time["osqp_setup_wall"] = setup - assembled
    phase_time["osqp_solve_wall"] = solved - setup
    phase_time["total"] = postprocessed - begin
    active_counts = np.asarray(list(map(len, active)), dtype=np.int64)
    return {"vertex_force_world":forces,"environment_force_world":environment_force,
            "environment_support_n":environment_support_n,
            "joint_internal_torque":solution[joint_internal_offset:joint_internal_offset+T],
            "force_slack_world":wrench_slack[:,:3],"torque_slack_world":wrench_slack[:,3:],
            "allow_wrench_slack":bool(allow_wrench_slack),
            "fitted_generalized_wrench":fitted,"residual":fitted-target,
            "closure_residual":fitted+generalized_slack-target,
            "active_count":active_counts,
            "status":"optimal","objective":objective,"osqp_iterations":iterations,
            "solver_iterations":iterations,
            "solver_backend":solver_backend,
            "warm_start_used":warm_start_used,
            "timing_s":phase_time,"osqp_timing_s":backend_timing,
            "backend_timing_s":backend_timing,
            "candidate_stats":{"frames":int(T),"active_total":int(active_counts.sum()),
                "active_min":int(active_counts.min() if len(active_counts) else 0),
                "active_max":int(active_counts.max() if len(active_counts) else 0),
                "active_mean":float(active_counts.mean() if len(active_counts) else 0.0)},
            "P_nnz":int(P.nnz),"A_nnz":int(A.nnz),"variables":nvar,"constraints":A.shape[0],
            }


def solve_articulated_contact_qp_windowed(
    *, window_size: int = 96, overlap: int = 16,
    recovery_max_force_slack_n: float | None = None,
    recovery_max_torque_slack_nm: float | None = None,
    recovery_wrench_slack_regularizer: float = 1e4, **inputs,
) -> dict:
    """Solve long sequences in overlapping windows and blend exact solutions."""
    target = np.asarray(inputs["target"]); T = len(target)
    if window_size < 2 or not 0 <= overlap < window_size:
        raise ValueError("expected window_size>=2 and 0<=overlap<window_size")
    stride = window_size - overlap
    starts = list(range(0, max(T - window_size, 0) + 1, stride))
    if not starts or starts[-1] + window_size < T:
        starts.append(max(0, T - window_size))
    starts = sorted(set(starts))
    force_sum = np.zeros((T, 2, 778, 3)); environment_sum = np.zeros((T, 3))
    force_slack_sum = np.zeros((T,3)); torque_slack_sum = np.zeros((T,3))
    joint_sum = np.zeros(T); weight_sum = np.zeros(T)
    window_stats = []
    sliced_keys = {"target", "system_com", "joint_axis", "joint_pivot", "environment_point", "environment_active",
                   "points", "normals", "priority", "candidate", "contact_part"}
    for start in starts:
        stop = min(T, start + window_size); local_inputs = dict(inputs)
        for key in sliced_keys & inputs.keys():
            local_inputs[key] = np.asarray(inputs[key])[start:stop]
        used_quality_recovery=False
        local_quality_valid=np.ones(stop-start,dtype=bool)
        try:
            result = solve_articulated_contact_qp(**local_inputs)
        except RuntimeError as error:
            if recovery_max_force_slack_n is None or recovery_max_torque_slack_nm is None:
                raise RuntimeError(f"articulated window [{start}, {stop}) failed: {error}") from error
            frame_keys=sliced_keys & local_inputs.keys()
            for frame in range(stop-start):
                frame_inputs=dict(local_inputs)
                for key in frame_keys:
                    frame_inputs[key]=np.asarray(local_inputs[key])[frame:frame+1]
                # Final labels require exact zero-slack closure.  A bounded
                # slack probe can quantify infeasibility, but a small slack is
                # not evidence that the strict problem is feasible.  Probe
                # each frame strictly with the recovery numerical tolerance;
                # frames that still fail are excluded from the recovered
                # strict window below.
                frame_inputs.update(
                    allow_wrench_slack=False,
                    temporal_regularizer=0.0,
                    environment_temporal_regularizer=0.0,
                    joint_internal_temporal_regularizer=0.0,
                    eps_abs=1e-4,eps_rel=1e-4)
                try:
                    solve_articulated_contact_qp(**frame_inputs)
                    local_quality_valid[frame]=True
                except RuntimeError:
                    local_quality_valid[frame]=False
            recovered_inputs=dict(local_inputs)
            recovered_inputs.update(
                eps_abs=max(float(local_inputs.get("eps_abs",1e-5)),1e-4),
                eps_rel=max(float(local_inputs.get("eps_rel",1e-5)),1e-4),
                wrench_slack_regularizer=recovery_wrench_slack_regularizer)
            if not local_quality_valid.all():
                recovered_inputs["target"]=np.asarray(local_inputs["target"]).copy()
                recovered_inputs["candidate"]=np.asarray(local_inputs["candidate"]).copy()
                recovered_inputs["contact_part"]=np.asarray(local_inputs["contact_part"]).copy()
                recovered_inputs["environment_active"]=np.asarray(local_inputs["environment_active"]).copy()
                recovered_inputs["target"][~local_quality_valid]=0.0
                recovered_inputs["candidate"][~local_quality_valid]=False
                recovered_inputs["contact_part"][~local_quality_valid]=-1
                recovered_inputs["environment_active"][~local_quality_valid]=False
            try:
                result=solve_articulated_contact_qp(**recovered_inputs)
                used_quality_recovery=True
            except RuntimeError as recovery_error:
                raise RuntimeError(
                    f"articulated window [{start}, {stop}) failed after bounded-slack "
                    f"recovery: {recovery_error}") from recovery_error
        length = stop - start; weights = np.ones(length)
        if overlap:
            ramp = (np.arange(overlap) + 1) / (overlap + 1)
            if start > 0: weights[:min(overlap,length)] = ramp[:min(overlap,length)]
            if stop < T: weights[-min(overlap,length):] = ramp[:min(overlap,length)][::-1]
        # A frame can be covered by several overlapping windows.  Only blend
        # strict zero-slack solutions for that frame; a failed probe in one
        # window must not invalidate a strict solution from another window.
        valid_weights = weights * local_quality_valid
        force_sum[start:stop] += result["vertex_force_world"] * valid_weights[:,None,None,None]
        environment_sum[start:stop] += result["environment_force_world"] * valid_weights[:,None]
        force_slack_sum[start:stop] += result["force_slack_world"] * valid_weights[:,None]
        torque_slack_sum[start:stop] += result["torque_slack_world"] * valid_weights[:,None]
        joint_sum[start:stop] += result["joint_internal_torque"] * valid_weights
        weight_sum[start:stop] += valid_weights
        window_stats.append({"start":start,"stop":stop,"timing_s":result["timing_s"],
                             "backend_timing_s":result.get("backend_timing_s",{}),
                             "candidate_stats":result["candidate_stats"],
                             "variables":result["variables"],"constraints":result["constraints"],
                             "P_nnz":result["P_nnz"],"A_nnz":result["A_nnz"],
                             "objective":result["objective"],
                             "iterations":result.get("solver_iterations",result.get("osqp_iterations",0)),
                             "pcg_iterations":result.get("pcg_iterations",0),
                             "kkt":({key:result["kkt"][key] for key in (
                                 "primal_equality_residual_max","inequality_violation_max",
                                 "dual_residual_max","complementarity_max")}
                                 if "kkt" in result else {}),
                             "used_quality_recovery":used_quality_recovery,
                             "quality_invalid_frames":np.flatnonzero(~local_quality_valid).tolist()})
    quality_valid = weight_sum > 0.0
    safe_weight_sum = np.where(quality_valid, weight_sum, 1.0)
    forces=force_sum/safe_weight_sum[:,None,None,None]
    environment=environment_sum/safe_weight_sum[:,None]
    force_slack=force_slack_sum/safe_weight_sum[:,None]
    torque_slack=torque_slack_sum/safe_weight_sum[:,None]
    joint_internal=joint_sum/safe_weight_sum
    fitted=np.zeros_like(target)
    for t in range(T):
        ids=np.argwhere(np.linalg.norm(forces[t],axis=-1)>0)
        for hand,vertex in ids:
            part=int(inputs["contact_part"][t,hand,vertex])
            fitted[t]+=contact_generalized_map(inputs["points"][t,hand,vertex],part,
                inputs["system_com"][t],inputs["joint_axis"][t],inputs["joint_pivot"][t])@forces[t,hand,vertex]
        fitted[t,:3]+=environment[t]
        fitted[t,6]+=joint_internal[t]
    wrench_slack=np.pad(np.concatenate((force_slack,torque_slack),axis=1),((0,0),(0,1)))
    return {"vertex_force_world":forces,"environment_force_world":environment,
            "environment_support_n":environment[:,2],
            "force_slack_world":force_slack,"torque_slack_world":torque_slack,
            "quality_valid":quality_valid,
            "allow_wrench_slack":bool(inputs.get("allow_wrench_slack",False)),
            "joint_internal_torque":joint_internal,"fitted_generalized_wrench":fitted,
            "residual":fitted-target,"closure_residual":fitted+wrench_slack-target,
            "window_stats":window_stats,"window_size":window_size,"overlap":overlap,
            "status":"optimal"}
