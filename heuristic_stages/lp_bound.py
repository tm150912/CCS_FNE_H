"""LP：完整原问题连续松弛，仅最优目标值用于全局下界。"""
from __future__ import annotations

from time import perf_counter

import gurobipy as gp

from .common import COST_FIELDS, _BudgetExpired, _check_lower_bound, _flow_constraints, _status_name


_CONSTRUCTION = ("s", "r", "w", "y")


def _build_original_relaxation(ctx):
    """完整原问题的连续松弛：无候选筛选、无固定排期、无代理成本。

    s/r 从二元变量松弛到 [0,1]，w/y 从非负整数松弛到非负连续量。
    同弧所有管径、所有期间的建设总数受全期管道上限约束。
    a/b/x 保持非负连续；累计容量、开场联动、平衡及逐期捕集目标均保留。
    """
    model = ctx.model("CCS_FNE_original_LP")
    try:
        units = [(i, g, t) for i, g in ctx.n.ig for t in ctx.t]
        sinks = [(j, t) for j in ctx.n.sinks for t in ctx.t]
        pipes = [(u, v, d, t) for u, v in ctx.n.ij for d in ctx.d for t in ctx.t]
        s = model.addVars(units, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="s")
        r = model.addVars(sinks, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="r")
        w = model.addVars(sinks, lb=0, vtype=gp.GRB.CONTINUOUS, name="w")
        y = model.addVars(pipes, lb=0,
                          vtype=gp.GRB.CONTINUOUS, name="y")
        a = model.addVars(units, lb=0, name="a")
        b = model.addVars(sinks, lb=0, name="b")
        x = model.addVars(pipes, lb=0, name="x")
        for i, g in ctx.n.ig:
            active = gp.LinExpr()
            for t in ctx.t:
                active += s[i, g, t]
                model.addConstr(a[i, g, t] <= ctx.n.source_capacity[i, g] * active)
            model.addConstr(active <= 1)
        for j in ctx.n.sinks:
            active, wells = gp.LinExpr(), gp.LinExpr()
            for t in ctx.t:
                active += r[j, t]
                wells += w[j, t]
                model.addConstr(wells <= ctx.max_wells[j] * active)
                model.addConstr(b[j, t] <= ctx.qw[j] * wells)
            model.addConstr(active <= 1)
        for u, v in ctx.n.ij:
            for d in ctx.d:
                active = gp.LinExpr()
                for t in ctx.t:
                    active += y[u, v, d, t]
                    model.addConstr(x[u, v, d, t] <= ctx.n.pipe_capacity[d] * active)
            model.addConstr(gp.quicksum(y[u, v, d, t] for d in ctx.d for t in ctx.t)
                            <= ctx.pipe_totals[u, v], name=f"pipe_total_limit[{u},{v}]")
        _flow_constraints(model, ctx.nodes, ctx.t, ctx.c.capture_target, a, b, x)
        variables = dict(s=s, r=r, w=w, y=y, a=a, b=b, x=x)
        model.setObjective(gp.quicksum(getattr(ctx.c, COST_FIELDS[name])[key] * var
            for name, mapping in variables.items() for key, var in mapping.items()), gp.GRB.MINIMIZE)
        return model
    except BaseException:
        model.dispose()
        raise


def _build_cumulative_relaxation(ctx):
    """返回等价累计 LP 和核验用映射；s/r/w/y 为累计量，a/b/x 含义不变。"""
    model = ctx.model("CCS_FNE_cumulative_LP")
    try:
        units = [(i, g, t) for i, g in ctx.n.ig for t in ctx.t]
        sinks = [(j, t) for j in ctx.n.sinks for t in ctx.t]
        pipes = [(u, v, d, t) for u, v in ctx.n.ij for d in ctx.d for t in ctx.t]
        variables = {}
        for name, keys in (("s", units), ("r", sinks), ("w", sinks), ("y", pipes),
                           ("a", units), ("b", sinks), ("x", pipes)):
            variables[name] = model.addVars(keys, lb=0,
                ub=1 if name in ("s", "r") else gp.GRB.INFINITY,
                vtype=gp.GRB.CONTINUOUS,
                name="cum_" + name if name in _CONSTRUCTION else name)
        s, r, w, y, a, b, x = (variables[name] for name in ("s", "r", "w", "y", "a", "b", "x"))

        # _Context requires consecutive periods 1,...,T. The zero-period state
        # is the constant zero, represented by first-period variable lower bounds.
        for name in _CONSTRUCTION:
            for key, variable in variables[name].items():
                if key[-1] > 1:
                    model.addConstr(variable >= variables[name][(*key[:-1], key[-1] - 1)])
        for i, g, t in units:
            model.addConstr(a[i, g, t] <= ctx.n.source_capacity[i, g] * s[i, g, t])
        for j, t in sinks:
            model.addConstr(w[j, t] <= ctx.max_wells[j] * r[j, t])
            model.addConstr(b[j, t] <= ctx.qw[j] * w[j, t])
        for u, v, d, t in pipes:
            model.addConstr(x[u, v, d, t] <= ctx.n.pipe_capacity[d] * y[u, v, d, t])
        last = ctx.t[-1]
        for u, v in ctx.n.ij:
            model.addConstr(gp.quicksum(y[u, v, d, last] for d in ctx.d)
                            <= ctx.pipe_totals[u, v], name=f"pipe_total_limit[{u},{v}]")
        _flow_constraints(model, ctx.nodes, ctx.t, ctx.c.capture_target, a, b, x)

        objective = gp.LinExpr()
        for name, mapping in variables.items():
            costs = getattr(ctx.c, COST_FIELDS[name])
            objective += gp.quicksum(
                (costs[key] - (costs[(*key[:-1], key[-1] + 1)]
                              if name in _CONSTRUCTION and key[-1] < last else 0)) * var
                for key, var in mapping.items())
        model.setObjective(objective, gp.GRB.MINIMIZE)
        return model, variables
    except BaseException:
        model.dispose()
        raise


def run_lp(ctx, *, formulation="original"):
    """独立构建并求解完整原 LP；仅最优目标值更新全局 lower bound。"""
    if formulation not in ("original", "cumulative"):
        raise ValueError("LP formulation must be original or cumulative")
    started = perf_counter()
    end = ctx.end if ctx.options.lp_time_limit is None else ctx.deadline(ctx.options.lp_time_limit)
    ctx.lp_status = "BUILDING"
    diagnostic = {"lp_formulation": formulation} if formulation == "cumulative" else {}
    ctx.record("global_LP_start", time_limit=ctx.options.lp_time_limit, **diagnostic)
    try:
        with (_build_original_relaxation(ctx) if formulation == "original"
              else _build_cumulative_relaxation(ctx)[0]) as model:
            model.Params.Method = ctx.options.lp_method
            model.Params.Crossover = ctx.options.lp_crossover
            ctx.optimize(model, end)
            ctx.lp_status = _status_name(model.Status)
            if model.Status == gp.GRB.OPTIMAL:
                lower_bound = float(model.ObjVal)
                _check_lower_bound(ctx, lower_bound)
                ctx.lp_objective = lower_bound
                ctx.lower_bound = max(0.0, lower_bound)
                ctx.bound_origin = "full_original_LP"
            # 超时状态下的 LP 原始可行目标值不一定是下界，不予采用。
    except _BudgetExpired:
        ctx.lp_status = "TIME_LIMIT"
        raise
    except gp.GurobiError:
        ctx.lp_status = "SOLVER_ERROR"
        raise
    finally:
        ctx.lp_runtime = perf_counter() - started
        ctx.record("global_LP", solver_status=ctx.lp_status, lp_objective=ctx.lp_objective,
                   lower_bound=ctx.lower_bound, lower_bound_origin=ctx.bound_origin, lp_runtime=ctx.lp_runtime,
                   **diagnostic)
