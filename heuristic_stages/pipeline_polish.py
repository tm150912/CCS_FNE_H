"""P3/P4/P5 复用的管径与建设时序精修。"""
from __future__ import annotations

from collections import defaultdict

import gurobipy as gp

from .common import (
    CandidatePool, _BudgetExpired, _attempt, _objective, _status_name, _values, _verified,
    _warm_start,
)


def _pipe_arc_cost(ctx, y, x):
    return (sum(ctx.c.pipe_fixed[key] * value for key, value in y.items())
            + sum(ctx.c.pipe_variable[key] * value for key, value in x.items()))


def _arc_state(y, x, demand):
    return (frozenset((key, value) for key, value in y.items() if value),
            frozenset((key, value) for key, value in x.items() if value),
            frozenset((t, value) for t, value in demand.items() if value))


def _pipe_fixed_flow_split(ctx, arc, y, demand):
    """给定管道排期，按真实运行费分配固定逐期弧流量，保留微小正流量。"""
    cumulative, result = defaultdict(float), {}
    for t in ctx.t:
        for d in ctx.d:
            cumulative[d] += y.get((*arc, d, t), 0)
        remaining = demand.get(t, 0.0)
        for d in sorted(ctx.d, key=lambda item: ctx.c.pipe_variable[(*arc, item, t)]):
            value = min(remaining, ctx.n.pipe_capacity[d] * cumulative[d])
            if value > 0:
                result[(*arc, d, t)] = value
                remaining -= value
        # Do not enlarge capacity to conceal a solver's feasibility tolerance.
        if remaining > 0:
            return None
    return result


def _polish_pipe_arc(ctx, arc, y_reference, x_reference, demand, end):
    """独立原约束小 MILP：各管径共用每弧全期总量上限，允许同期及跨期追加。"""
    keys = [(*arc, d, t) for d in ctx.d for t in ctx.t]
    with ctx.model("pipeline_arc_polish") as model:
        y = model.addVars(keys, lb=0, ub=ctx.pipe_totals[arc],
                          vtype=gp.GRB.INTEGER, name="y")
        x = model.addVars(keys, lb=0, name="x")
        for d in ctx.d:
            installed = gp.LinExpr()
            for t in ctx.t:
                installed += y[(*arc, d, t)]
                model.addConstr(x[(*arc, d, t)] <= ctx.n.pipe_capacity[d] * installed)
        model.addConstr(gp.quicksum(y.values()) <= ctx.pipe_totals[arc], name="pipe_total_limit")
        for t in ctx.t:
            model.addConstr(gp.quicksum(x[(*arc, d, t)] for d in ctx.d) == demand.get(t, 0.0))
        model.setObjective(gp.quicksum(ctx.c.pipe_fixed[key] * y[key]
            + ctx.c.pipe_variable[key] * x[key] for key in keys), gp.GRB.MINIMIZE)
        _warm_start(y, y_reference)
        _warm_start(x, x_reference)
        ctx.optimize(model, end)
        status = _status_name(model.Status)
        if not model.SolCount:
            return None, None, status
        new_y = _values(y, integer=True)
        # The fixed-q split is exact for these y and avoids retaining numerical
        # balance residuals from the MILP's continuous solution.
        new_x = _pipe_fixed_flow_split(ctx, arc, new_y, demand)
        return new_y, new_x, status


def _polish_pipelines(ctx, reference, *, optimize_multibuild=True,
                      time_limit=None, full_sweep=False):
    """固定设施与逐期弧流量，精修管径和建设期；只更新经原模型审计的上界。"""
    if not ctx.options.pipe_polish_enabled or ctx.options.pipe_polish_time_limit <= 0:
        return reference
    time_limit = ctx.options.pipe_polish_time_limit if time_limit is None else time_limit
    end = ctx.deadline(time_limit)
    original_value = _objective(ctx.c, reference)
    result, trial = reference, reference.copy()
    builds, flows, demand = defaultdict(dict), defaultdict(dict), defaultdict(lambda: defaultdict(float))
    single_examined = single_changed = multi_examined = multi_changed = 0
    multi, single_scan_completed = [], False
    statuses = defaultdict(int)
    stopped = None
    single_skipped = multi_skipped = 0
    if not hasattr(ctx, "_ph_pipe_single_cache"):
        ctx._ph_pipe_single_cache, ctx._ph_pipe_multi_cache = {}, {}
    single_cache, multi_cache = ctx._ph_pipe_single_cache, ctx._ph_pipe_multi_cache

    def replace_arc(arc, new_y, new_x):
        # 同一弧可能先完成单管替换，再进入多管 MILP；清除当前方案全部管径。
        for d in ctx.d:
            for t in ctx.t:
                key = (*arc, d, t)
                trial.y.pop(key, None)
                trial.x.pop(key, None)
        trial.y.update(new_y)
        trial.x.update(new_x)

    def save(stage):
        nonlocal result, trial
        # Audit at the budget boundary before returning: a completed cheap
        # improvement must survive a later MILP exhausting the remaining time.
        if _objective(ctx.c, trial) < _objective(ctx.c, result):
            if _verified(ctx, trial, stage + "_audit"):
                ctx.consider(trial, stage)
                result = trial
            trial = result.copy()

    def improves(old, new):
        return new < old - ctx.options.improvement_tolerance * max(1.0, abs(old))

    try:
        ctx.check_time(end)
        for key, value in reference.y.items():
            builds[key[:2]][key] = value
        for key, value in reference.x.items():
            flows[key[:2]][key] = value
            demand[key[:2]][key[-1]] += value
        for arc, old_y in sorted(builds.items()):
            ctx.check_time(end)
            old_x, q = flows[arc], demand[arc]
            if any(value < 0 for value in q.values()):
                continue
            count = sum(round(value) for value in old_y.values())
            old_cost = _pipe_arc_cost(ctx, old_y, old_x)
            if count != 1:
                if count > 1:
                    multi.append((old_cost, arc))
                continue
            # 一条大管也可能由多条便宜的小管替代；单管枚举无法发现此改进。
            if ctx.pipe_totals[arc] > 1 and len(ctx.d) > 1:
                multi.append((old_cost, arc))
            if single_cache.get(arc) == _arc_state(old_y, old_x, q):
                single_skipped += 1
                continue
            single_examined += 1
            peak = max(q.values(), default=0.0)
            best_cost, best_y, best_x = old_cost, None, None
            if peak == 0:
                best_cost, best_y, best_x = 0.0, {}, {}
            else:
                first_required = min(t for t, value in q.items() if value > 0)
                for d in ctx.d:
                    ctx.check_time(end)
                    if ctx.n.pipe_capacity[d] < peak:
                        continue
                    new_x = {(*arc, d, t): value for t, value in q.items() if value}
                    variable = sum(ctx.c.pipe_variable[key] * value for key, value in new_x.items())
                    # Enumerate all feasible periods: investment need not fall
                    # monotonically and the cheapest date can precede demand.
                    for t in ctx.t:
                        if t > first_required:
                            break
                        if ctx.pipe_totals[arc] == 0:
                            continue
                        value = ctx.c.pipe_fixed[(*arc, d, t)] + variable
                        if value < best_cost:
                            best_cost, best_y, best_x = value, {(*arc, d, t): 1}, new_x
            if best_y is not None and improves(old_cost, best_cost):
                replace_arc(arc, best_y, best_x)
                single_changed += 1
                single_cache[arc] = _arc_state(best_y, best_x, q)
            else:
                single_cache[arc] = _arc_state(old_y, old_x, q)
        single_scan_completed = True
        save("P5_pipe_single")
        max_arcs = (len(multi) if full_sweep and ctx.options.pipe_polish_max_arcs > 0
                    else ctx.options.pipe_polish_max_arcs)
        if optimize_multibuild and ctx.options.pipe_arc_time_limit > 0:
            for _, arc in sorted(multi, key=lambda item: (-item[0], item[1])):
                if multi_examined >= max_arcs:
                    stopped = "arc_limit"
                    break
                ctx.check_time(end)
                keys = [(*arc, d, t) for d in ctx.d for t in ctx.t]
                current_y = {key: trial.y[key] for key in keys if trial.y.get(key, 0)}
                current_x = {key: trial.x[key] for key in keys if trial.x.get(key, 0)}
                cached = multi_cache.get(arc)
                state = _arc_state(current_y, current_x, demand[arc])
                if cached is not None and cached["state"] == state:
                    if cached["status"] == "OPTIMAL" or cached["skip_once"]:
                        cached["skip_once"] = False
                        multi_skipped += 1
                        continue
                arc_end = min(end, ctx.deadline(ctx.options.pipe_arc_time_limit))
                multi_examined += 1
                try:
                    new_y, new_x, status = _polish_pipe_arc(ctx, arc, current_y, current_x, demand[arc], arc_end)
                except _BudgetExpired:
                    statuses["TIME_LIMIT"] += 1
                    multi_cache[arc] = dict(state=state, status="TIME_LIMIT", skip_once=True)
                    ctx.check_time(end)
                    continue
                except gp.GurobiError as exc:
                    statuses["SOLVER_ERROR"] += 1
                    multi_cache[arc] = dict(state=state, status="SOLVER_ERROR", skip_once=True)
                    ctx.record("P5_pipe_arc", arc=str(arc), solver_error=str(exc))
                    continue
                statuses[status] += 1
                if new_y is None or new_x is None:
                    multi_cache[arc] = dict(state=state, status="NO_USABLE_SOLUTION", skip_once=True)
                    continue
                if improves(_pipe_arc_cost(ctx, current_y, current_x), _pipe_arc_cost(ctx, new_y, new_x)):
                    replace_arc(arc, new_y, new_x)
                    multi_changed += 1
                    save("P5_pipe_multi")
                # OPTIMAL here means solved to the configured MIP tolerance;
                # no global bound or exact-optimality claim is inferred.
                saved_y = {key: trial.y[key] for key in keys if trial.y.get(key, 0)}
                saved_x = {key: trial.x[key] for key in keys if trial.x.get(key, 0)}
                accepted = not improves(_pipe_arc_cost(ctx, saved_y, saved_x),
                                       _pipe_arc_cost(ctx, new_y, new_x))
                multi_cache[arc] = dict(state=_arc_state(saved_y, saved_x, demand[arc]),
                    status=status if accepted else "AUDIT_REJECTED", skip_once=True)
    except _BudgetExpired:
        stopped = "time_limit"
    finally:
        save("P5_pipe_partial")
        value = _objective(ctx.c, result)
        ctx.record("P5_pipe_polish", objective=value, improvement=original_value - value,
            single_examined=single_examined, single_changed=single_changed,
            multi_examined=multi_examined, multi_changed=multi_changed,
            solver_statuses=dict(statuses), stopped=stopped,
            multibuild_enabled=optimize_multibuild,
            time_limit=time_limit, full_sweep=full_sweep,
            single_scan_completed=single_scan_completed, multi_candidates=len(multi),
            single_skipped_unchanged=single_skipped, multi_skipped_unchanged=multi_skipped)
    return result


def _polish_and_merge(ctx, pool, reference, *, optimize_multibuild=True,
                      time_limit=None, full_sweep=False):
    """新管径必须同时加入候选池；精修失败时仍保留原可行方案。"""
    if ctx.options.pipe_polish_enabled and ctx.options.pipe_polish_time_limit > 0:
        polished = _attempt(ctx, "P5_pipe_polish", _polish_pipelines, reference,
                            optimize_multibuild=optimize_multibuild,
                            time_limit=time_limit, full_sweep=full_sweep)
        if polished is not None:
            reference = polished
    pool.merge(CandidatePool.from_plan(reference))
    ctx.consider(reference, "polished_incumbent")
    return reference
