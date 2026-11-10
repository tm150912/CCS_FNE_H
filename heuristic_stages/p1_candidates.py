"""P1：仅求解 ESM 的 E-Select，审计后提取候选设施和逐期见证。"""
from __future__ import annotations

from collections import defaultdict

import gurobipy as gp

from ESM import esm_common as ESM
from ESM import RSSforESM as rss
from .common import (
    CandidatePool, Plan, VARIABLES, _attempt,
)


def _log_p1_progress(ctx, stage, **details):
    """打印顺序步骤，同时保留原事件名和完整历史字段。"""
    number = details["candidate"] + 1
    steps = 2
    if stage == "P1_initial_start":
        message = (f"[{number}/{steps}] 求解第 {number} 个 E-Select "
                   f"(方法={details['solver_method']})")
    elif stage == "P1_initial":
        message = (f"  E-Select 结果：status={details['initial_solver_status']}, "
                   f"static_feasible={details['initial_solver_feasible']}, "
                   f"objective={details['initial_solver_objective']}")
    else:  # P1_SP：静态解与确定性逐期见证已完成审计。
        message = (f"  E-Select 审计结果：status={details['solver_status']}, "
                   f"candidate_feasible={details['candidate_feasible']}, "
                   f"period_feasible={details['period_feasible']}, "
                   f"objective={details['surrogate_objective']}")
    ctx.record(stage, message=message, **details)


def _lift_period_solution(ctx, solution):
    """首期建设并同比缩放峰值流量；当前模型没有逐期施工或跨期流量额度。

    所有 K_t <= K_max，静态容量约束因而保证各期容量可行；流平衡在
    同比缩放后保持。见证不参与静态目标，生成后仍需通过三重独立审计。
    """
    peak_target = max(ctx.c.capture_target.values())
    period = {name: {} for name in VARIABLES}
    for name in VARIABLES:
        construction = name in ("s", "r", "w", "y")
        for key, value in solution[name].items():
            if not value:
                continue
            indices = key if name in ("s", "y", "a", "x") else (key,)
            if construction:
                period[name][(*indices, ctx.t[0])] = value
            elif peak_target:
                for t in ctx.t:
                    if ctx.c.capture_target[t]:
                        period[name][(*indices, t)] = value * ctx.c.capture_target[t] / peak_target
    return period


def _audit_p1_solution(ctx, data, solution, period_solution):
    """同时审计静态解、原逐期约束以及两者的建设总量耦合。"""
    audit = ESM.audit_e_select_solution(data, solution)
    terminal_valid = audit["valid"]
    if period_solution is None:
        period_audit = dict(feasible=False, violations={"missing_period_witness": 1.0},
                            max_violation=None)
    else:
        period_audit = ctx.audit(Plan(**period_solution))
    coupling = None
    if terminal_valid and period_solution is not None:
        coupling = 0.0
        for name in ("s", "r", "w", "y"):
            for key, value in solution[name].items():
                indices = key if name in ("s", "y") else (key,)
                total = sum(period_solution[name].get((*indices, t), 0) for t in ctx.t)
                coupling = max(coupling, abs(total - value))
    audit["violations"].extend(f"period: {name}" for name in period_audit["violations"])
    if coupling is not None and coupling > ctx.options.feasibility_tolerance:
        audit["violations"].append(f"period construction totals differ from static design: {coupling}")
    valid = (terminal_valid and period_audit["feasible"] and coupling is not None
             and coupling <= ctx.options.feasibility_tolerance)
    audit.update(valid=valid, terminal_valid=terminal_valid,
                 period_feasible=period_audit["feasible"], period_audit=period_audit,
                 coupling_max_residual=coupling,
                 max_residual=max((value for value in (
                     audit["max_residual"], period_audit["max_violation"], coupling)
                     if value is not None), default=None))
    if not valid:
        audit.update(cost=None, cost_breakdown={})
    return audit


def solve_p1_model(ctx, reference, end, *, progress=None):
    """将 P1 剩余预算交给所选 E-Select 后端，再生成并审计逐期见证。"""
    def record(stage, **details):
        if progress is None:
            _log_p1_progress(ctx, stage, **details)
        else:
            progress(stage, **details)

    end = min(end, ctx.deadline(ctx.options.p1_time_limit))
    ctx.check_time(end)
    data = ESM.aggregate_e_select_data(ctx.n, ctx.c, ctx.d,
        ctx.pipeline_install_limit, target=max(ctx.c.capture_target.values()))
    peak = max(ctx.t, key=ctx.c.capture_target.get)
    initial_solution = None
    if reference is not None:
        initial_solution = {}
        for name in VARIABLES:
            warm = defaultdict(float)
            for key, value in getattr(reference, name).items():
                if name in ESM.BUILD_GROUPS or key[-1] == peak:
                    prefix = key[:-1] if name in ("s", "y", "a", "x") else key[0]
                    warm[prefix] += value
            initial_solution[name] = {key: warm.get(key, 0.0) for key in data.indices[name]}

    method = ctx.options.sp_solver_method
    initial_budget = ctx.check_time(end)
    record("P1_initial_start", candidate=0, problem="E-Select", solver_method=method,
           time_limit=initial_budget)
    initial_options = dict(time_limit=initial_budget, threads=ctx.options.threads,
        seed=ctx.options.seed, output_flag=int(ctx.options.gurobi_output_flag),
        mip_gap=ctx.options.mip_gap)
    options = rss.ESelectOptions(**initial_options,
        print_progress=bool(ctx.options.output_flag))
    solve = rss.solve_e_select
    initial_error = None
    try:
        initial = solve(data, options=options, initial_solution=initial_solution)
    except gp.GurobiError as exc:
        initial_error = str(exc)
        initial = dict(status="SOLVER_ERROR", feasible=False, e_select_objective=None)
    stage_details = dict(problem="E-Select", initial_solver_method=method,
        initial_algorithm=initial.get("algorithm", method),
        initial_solver_status=initial["status"],
        initial_solver_objective=initial["e_select_objective"],
        initial_solver_feasible=initial["feasible"])
    if initial_error is not None:
        stage_details["initial_solver_error"] = initial_error
    elif initial.get("error") is not None:
        stage_details["initial_solver_error"] = initial["error"]
    if initial.get("termination_reason") is not None:
        stage_details["initial_termination_reason"] = initial["termination_reason"]
    record("P1_initial", candidate=0, **stage_details)
    if initial["feasible"]:
        initial_solution = initial["solution"]

    # 直接求解结果必须通过静态、逐期及建设总量耦合审计。
    solution = initial_solution
    period_solution = _lift_period_solution(ctx, solution) if solution is not None else None
    audit = _audit_p1_solution(ctx, data, solution, period_solution)
    stage_details.update(period_feasibility_model=False,
        period_witness_method="static_scaling", feasibility_periods=len(ctx.t),
        initial_candidate_feasible=audit["valid"], candidate_source="direct")
    status = initial["status"]
    record("P1_SP", candidate=0, solver_method=method,
        solver_status=status, surrogate_objective=audit["cost"],
        terminal_feasible=audit["terminal_valid"], period_feasible=audit["period_feasible"],
        candidate_feasible=audit["valid"], **stage_details)
    return dict(problem="E-Select", candidate=0,
        algorithm=method, solver_method=method, status=status,
        feasible=audit["valid"], e_select_objective=audit["cost"],
        objective_scope="E-Select static proxy model only; not original CCS_FNE UB/LB",
        solution=solution if audit["valid"] else None,
        period_solution=period_solution if audit["valid"] else None,
        period_feasible=audit["period_feasible"], period_weights=dict(data.weights),
        audit=audit, data=data, stage_details=stage_details,
        cost_breakdown=audit["cost_breakdown"], max_constraint_residual=audit["max_residual"])


def _static_pool(ctx, reference, end):
    """候选设施只从通过三重审计的 E-Select 解中提取。"""
    result = solve_p1_model(ctx, reference, end)
    if not result["feasible"]:
        return None
    selected = {name: {key: int(round(value)) for key, value in result["solution"][name].items()
                       if value > 0.5} for name in ESM.BUILD_GROUPS}
    pool = CandidatePool(set(selected["s"]), set(selected["r"]), selected["w"], selected["y"])
    return pool, Plan(**result["period_solution"])


def run_p1(ctx):
    """只求一组 E-Select 候选；失败则返回 None，不引入其他来源设施。"""
    ctx.check_time()
    if ctx.options.p1_time_limit <= 0:
        ctx.record("P1_skipped", candidate=0, reason="E-Select disabled",
                   message="  P1 E-Select 候选求解已禁用")
        return None
    end = ctx.deadline(ctx.options.p1_time_limit)
    candidate = _attempt(ctx, "P1_SP", _static_pool, ctx.incumbent, end)
    if candidate is None:
        ctx.record("P1_no_solution", valid_candidates=0,
                   message="  E-Select 未获得通过审计的逐期可行解，无初始候选池")
        return None
    pool, witness = candidate
    ctx.consider(witness, "P1_incumbent")
    ctx.search_pool = pool
    ctx.record("P1_pool", problem="E-Select", units=len(pool.units), sinks=len(pool.sinks),
        pipe_types=len(pool.pipes), valid_candidates=1,
        message=f"[2/2] E-Select 设施候选集：机组 {len(pool.units)}，封存场 {len(pool.sinks)}，"
                f"井数上限合计 {sum(pool.wells.values())}，管道数上限合计 {sum(pool.pipes.values())}")
    return pool
