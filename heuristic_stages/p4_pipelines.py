"""P4：固定设施排期，恢复管道建设和全部运行流量。"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite

import gurobipy as gp

from .common import (
    COST_FIELDS, _attempt, _cumulative, _facility_signature, _flow_constraints, _objective,
    _solver_metrics, _status_name, _values, _verified, _warm_start,
)
from .pipeline_polish import _polish_and_merge
from .warm_start import optimize_with_reference


def _fixed_facility_cost(ctx, reference):
    return sum(getattr(ctx.c, COST_FIELDS[name])[key] * value
               for name in ("s", "r", "w") for key, value in getattr(reference, name).items())


def _recovery_key(pool, reference):
    return (_facility_signature(reference),
            frozenset((key, value) for key, value in pool.pipes.items() if value))


def _operating_signature(reference):
    return tuple(frozenset((key, value) for key, value in getattr(reference, name).items() if value)
                 for name in ("y", "a", "b", "x"))


def _recover_pipelines(ctx, pool, reference, *, stage_prefix="P4"):
    """P4：固定 s/r/w，重新优化 y 以及全部 a/b/x，比较真实全期成本。"""
    if not pool.covers(reference):
        raise ValueError("Pipeline recovery reference must belong to its candidate pool")
    end = ctx.deadline(ctx.options.pipe_time_limit)
    sc, wc = (_cumulative(getattr(reference, n), ctx.t) for n in ("s", "w"))
    with ctx.model("pipeline_recovery") as model:
        keys = [(*key, t) for key, count in sorted(pool.pipes.items()) if count for t in ctx.t]
        y = model.addVars(keys, lb=0, ub=ctx.pipeline_install_limit,
                          vtype=gp.GRB.INTEGER, name="y")
        x = model.addVars(keys, lb=0, name="x")
        a = model.addVars(sorted(k for k, v in sc.items() if v), lb=0, name="a")
        b = model.addVars(sorted(k for k, v in wc.items() if v), lb=0, name="b")
        for key, var in a.items():
            var.UB = ctx.n.source_capacity[key[:2]] * sc[key]
        for key, var in b.items():
            var.UB = ctx.qw[key[0]] * wc[key]
        by_arc = defaultdict(list)
        for (u, v, d), count in sorted(pool.pipes.items()):
            if not count:
                continue
            by_arc[u, v].append(d)
            installed = gp.LinExpr()
            for t in ctx.t:
                installed += y[u, v, d, t]
                model.addConstr(x[u, v, d, t] <= ctx.n.pipe_capacity[d] * installed)
            model.addConstr(installed <= count)
        for (u, v), ds in by_arc.items():
            model.addConstr(gp.quicksum(y[u, v, d, t] for d in ds for t in ctx.t)
                            <= ctx.pipe_totals[u, v], name=f"pipe_total_limit[{u},{v}]")
        _flow_constraints(model, ctx.nodes, ctx.t, ctx.c.capture_target, a, b, x)
        variables = dict(y=y, a=a, b=b, x=x)
        constant = _fixed_facility_cost(ctx, reference)
        model.setObjective(constant + gp.quicksum(getattr(ctx.c, COST_FIELDS[name])[k] * var
            for name, mapping in variables.items() for k, var in mapping.items()), gp.GRB.MINIMIZE)
        for name, mapping in variables.items():
            _warm_start(mapping, getattr(reference, name))
        target = ctx.best - ctx.options.improvement_tolerance * max(1.0, abs(ctx.best))
        stopped_bound = None

        def stop_dominated(current, where):
            nonlocal stopped_bound
            if where == gp.GRB.Callback.MIP:
                bound = current.cbGet(gp.GRB.Callback.MIP_OBJBND)
                if isfinite(bound) and bound >= target:
                    stopped_bound = bound
                    current.terminate()

        optimize_with_reference(ctx, model, end, stage=stage_prefix, callback=stop_dominated)
        bound = float(model.ObjBound) if model.IsMIP else None
        ctx._ph_p4_bound = bound if bound is not None and isfinite(bound) else None
        if stopped_bound is not None:
            ctx.record(f"{stage_prefix}_bound_stop", restricted_bound=stopped_bound,
                       improvement_target=target, scope="fixed_facilities_and_pool")
        ctx.record(f"{stage_prefix}_pipelines", solver_status=_status_name(model.Status),
                   objective=float(model.ObjVal) if model.SolCount else None,
                   start_objective=_objective(ctx.c, reference), **_solver_metrics(model))
        if not model.SolCount:
            return reference
        result = reference.copy()
        for name, mapping in variables.items():
            setattr(result, name, _values(mapping, name == "y"))
        # Fixed facilities remain fixed here; unused construction can be removed in P5.
        return result if _verified(ctx, result, f"{stage_prefix}_audit") and _objective(ctx.c, result) < _objective(ctx.c, reference) else reference


def _recover_and_polish(ctx, pool, reference):
    reference = _polish_and_merge(ctx, pool, reference)
    ctx.check_time()
    key, signature = _recovery_key(pool, reference), _operating_signature(reference)
    recent = getattr(ctx, "_ph_p4_recent", None)
    target = ctx.best - ctx.options.improvement_tolerance * max(1.0, abs(ctx.best))
    if recent is not None and recent["key"] == key:
        certified = recent["bound"] is not None and recent["bound"] >= target
        if certified or (recent["signature"] == signature and recent["skip_once"]):
            recent["skip_once"] = False
            ctx.record("P4_skipped", reason="restricted_bound" if certified else "unchanged_reference_cooldown",
                       restricted_bound=recent["bound"], scope="fixed_facilities_and_pool")
            return reference
    ctx._ph_p4_bound = None
    recovered = _attempt(ctx, "P4_pipelines", _recover_pipelines, pool, reference)
    reference_cost = _objective(ctx.c, reference)
    tolerance = ctx.options.improvement_tolerance * max(1.0, abs(reference_cost))
    no_gain = recovered is None or _objective(ctx.c, recovered) >= reference_cost - tolerance
    ctx._ph_p4_recent = dict(key=key, signature=signature, bound=ctx._ph_p4_bound,
                            skip_once=no_gain)
    if recovered is not None:
        reference = recovered
        ctx.consider(reference, "P4_incumbent")
    return _polish_and_merge(ctx, pool, reference)


def run_p4(ctx, pool, scheduled):
    """P4：选取恢复起点，优化管道与流量，并更新通过审计的最好解。"""
    # 设施相同则用真实费用更低的方案启动 P4；P3 失败时也直接精修 incumbent。
    recovery_start = scheduled or ctx.incumbent.copy()
    if (_facility_signature(recovery_start) == _facility_signature(ctx.incumbent)
            and _objective(ctx.c, recovery_start) > ctx.best):
        recovery_start = ctx.incumbent.copy()
    # All remaining original costs are nonnegative: this is a valid bound,
    # unlike the P3 proxy or the candidate's (possibly poor) actual cost.
    lower_bound = _fixed_facility_cost(ctx, recovery_start)
    target = ctx.best - ctx.options.improvement_tolerance * max(1.0, abs(ctx.best))
    if lower_bound >= target and _facility_signature(recovery_start) != _facility_signature(ctx.incumbent):
        ctx.record("P4_candidate_screen", reason="fixed_facility_cost_bound",
                   restricted_bound=lower_bound, improvement_target=target)
        recovery_start = ctx.incumbent.copy()
    _recover_and_polish(ctx, pool, recovery_start)
