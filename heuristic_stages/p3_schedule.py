"""P3：保留全期流量约束的设施排期。"""
from __future__ import annotations

from collections import defaultdict
from math import ceil

import gurobipy as gp

from .common import (
    COST_FIELDS, Plan, _attempt, _facility_constraints, _flow_constraints, _solver_metrics,
    _status_name, _trim, _values, _verified, _warm_start,
)
from .p0_feasibility import _envelope, _split_flows
from .pipeline_polish import _polish_and_merge
from .warm_start import optimize_with_reference


def _pipe_proxy(ctx, pool, reference, envelope, capacities, end):
    """容量包络只约束流量；新旧弧均按预计流量所需的建设方案估计费用。"""
    flow = defaultdict(float)
    reference_builds, envelope_builds = defaultdict(dict), defaultdict(dict)
    for (u, v, d, t), value in reference.x.items():
        flow[u, v, t] += value
    for key, value in reference.y.items():
        if value > 0:
            reference_builds[key[:2]][key[2:]] = value
    for key, value in envelope.items():
        envelope_builds[key[:2]][key[2:]] = value
    by_arc = defaultdict(list)
    for (u, v, d), count in pool.pipes.items():
        if count:
            by_arc[u, v].append(d)
    peaks = sorted(max(flow[(*edge, t)] for t in ctx.t) for edge in reference_builds)
    peaks = [value for value in peaks if value > 0]
    representative = peaks[len(peaks) // 2] if peaks else max(ctx.n.source_capacity.values(), default=0.0)
    peak_target = max(ctx.c.capture_target.values())
    result = {}
    for edge, ds in by_arc.items():
        ctx.check_time(end)
        profile = {t: flow[(*edge, t)] for t in ctx.t}
        if not any(profile.values()):
            profile = {t: min(capacities[(*edge, t)], representative * ctx.c.capture_target[t] / peak_target)
                       if peak_target else 0.0 for t in ctx.t}
        denominator = sum(ctx.c.durations[t] * profile[t] for t in ctx.t)
        first = next((t for t in ctx.t if profile[t] > 0), None)
        if first is None:
            result.update({(*edge, t): min(ctx.c.pipe_variable[(*edge, d, t)] for d in ds) for t in ctx.t})
            continue

        def estimate(builds):
            fixed = sum(ctx.c.pipe_fixed[(*edge, d, t)] * count for (d, t), count in builds.items())
            active, rates, operation = defaultdict(float), {}, 0.0
            for t in ctx.t:
                for d in ds:
                    active[d] += ctx.n.pipe_capacity[d] * builds.get((d, t), 0)
                remaining, cost = profile[t], 0.0
                for d in sorted(ds, key=lambda d: ctx.c.pipe_variable[(*edge, d, t)]):
                    amount = min(remaining, active[d])
                    cost += amount * ctx.c.pipe_variable[(*edge, d, t)]
                    remaining -= amount
                if remaining > ctx.options.feasibility_tolerance:
                    return None
                rates[t] = cost / profile[t] if profile[t] else min(ctx.c.pipe_variable[(*edge, d, t)] for d in ds)
                operation += cost
            return fixed + operation, fixed, rates

        candidates = []
        if edge in reference_builds:
            candidate = estimate(reference_builds[edge])
            if candidate is not None:
                candidates.append(candidate)
        required = max(profile.values())
        for d in ds:
            if ctx.n.pipe_capacity[d] >= required and ctx.pipe_totals[edge] > 0:
                dates = [t for t in ctx.t if t <= first]
                if dates:
                    build_t = min(dates, key=lambda t: ctx.c.pipe_fixed[(*edge, d, t)])
                    candidates.append(estimate({(d, build_t): 1}))
        # 只计覆盖预计峰值所需的包络前缀；同一期的一批管道也按所需条数截断。
        prefix, installed = {}, 0.0
        for (d, t), count in sorted(envelope_builds[edge].items(), key=lambda item: item[0][1]):
            needed = ceil(max(0.0, required - installed) / ctx.n.pipe_capacity[d])
            count = min(count, needed)
            if count:
                prefix[d, t] = count
                installed += ctx.n.pipe_capacity[d] * count
            if installed >= required:
                break
        candidate = estimate(prefix)
        if candidate is not None:
            candidates.append(candidate)
        if not candidates:
            raise RuntimeError(f"No feasible pipeline proxy profile on {edge}")
        _, fixed, rates = min(candidates, key=lambda item: item[0])
        for t in ctx.t:
            result[(*edge, t)] = rates[t] + ctx.options.pipe_cost_weight * ctx.c.durations[t] * fixed / denominator
    return result


def _schedule(ctx, pool, reference, free_nodes=None, *, stage_prefix="P3",
              recovery_reserve=5.0):
    """P3：保留多期流量与累计容量约束；管道用可构造容量包络和代理成本。"""
    if not pool.covers(reference):
        raise ValueError("Scheduling reference must belong to its candidate pool")
    # Experimental margin for NP=1 recovery at this instance size; not a hard deadline guarantee.
    remaining = ctx.check_time()
    if remaining <= recovery_reserve:
        ctx.record(f"{stage_prefix}_schedule", skipped="insufficient_recovery_budget",
                   remaining=remaining, recovery_reserve=recovery_reserve)
        return None
    end = min(ctx.deadline(ctx.options.schedule_time_limit),
              min(ctx.end, ctx.phase_end) - recovery_reserve)
    ctx.record(f"{stage_prefix}_recovery_budget", remaining=remaining,
               recovery_reserve=recovery_reserve, time_limit=ctx.options.schedule_time_limit)
    envelope, capacities = _envelope(ctx, pool)
    ctx.check_time(end)
    proxy = _pipe_proxy(ctx, pool, reference, envelope, capacities, end)
    ctx.check_time(end)
    with ctx.model("facility_schedule") as model:
        s = model.addVars([(i, g, t) for i, g in sorted(pool.units) for t in ctx.t], vtype=gp.GRB.BINARY, name="s")
        r = model.addVars([(j, t) for j in sorted(pool.sinks) for t in ctx.t], vtype=gp.GRB.BINARY, name="r")
        w = model.addVars(list(r), vtype=gp.GRB.INTEGER, lb=0, name="w")
        a = model.addVars(list(s), lb=0, name="a")
        b = model.addVars(list(r), lb=0, name="b")
        x = model.addVars(sorted(proxy), lb=0, ub={k: capacities[k] for k in proxy}, name="X")
        ctx.check_time(end)
        _facility_constraints(ctx, model, pool, s, r, w, a, b)
        _flow_constraints(model, ctx.nodes, ctx.t, ctx.c.capture_target, a, b, x)
        ctx.check_time(end)
        variables = dict(s=s, r=r, w=w, a=a, b=b)
        for name, mapping in variables.items():
            _warm_start(mapping, getattr(reference, name))
            if free_nodes is not None and name in ("s", "r", "w"):
                for key, var in mapping.items():
                    if key[0] not in free_nodes:
                        var.LB = var.UB = getattr(reference, name).get(key, 0)
        aggregate = defaultdict(float)
        for (u, v, d, t), value in reference.x.items():
            aggregate[u, v, t] += value
        _warm_start(x, aggregate)
        ctx.check_time(end)
        model.setObjective(gp.quicksum(getattr(ctx.c, COST_FIELDS[name])[key] * var
            for name, mapping in variables.items() for key, var in mapping.items())
            + gp.quicksum(proxy[key] * var for key, var in x.items()), gp.GRB.MINIMIZE)
        start = optimize_with_reference(ctx, model, end, stage=stage_prefix)
        ctx.record(f"{stage_prefix}_schedule", solver_status=_status_name(model.Status),
                   surrogate_objective=float(model.ObjVal) if model.SolCount else None,
                   free_facilities="all" if free_nodes is None else len(free_nodes), **_solver_metrics(model))
        if not model.SolCount:
            return None
        reference_objective = start.get('reference_objective')
        if (reference_objective is not None
                and model.ObjVal > reference_objective + ctx.options.improvement_tolerance * max(1.0, abs(reference_objective))):
            ctx.record(f'{stage_prefix}_reference_fallback', reason='worse_surrogate_than_reference',
                       surrogate_objective=float(model.ObjVal), reference_surrogate_objective=reference_objective)
            return reference.copy()
        result = Plan(**{name: _values(mapping, name in ("s", "r", "w")) for name, mapping in variables.items()}, y=envelope)
        _split_flows(ctx, result, _values(x))
        result = _trim(ctx, result)
        return result if _verified(ctx, result, f"{stage_prefix}_witness") else None


def _neighborhood(ctx, pool, round_number):
    """按轮次释放候选设施，保持局部排期规模并轮换覆盖全部节点。"""
    nodes = sorted({i for i, g in pool.units} | pool.sinks)
    if not nodes:
        return set()
    limit = ctx.options.neighborhood_size
    offset = (round_number * max(1, limit // 2)) % len(nodes)
    return set((nodes[offset:] + nodes[:offset])[:limit])


def run_p3(ctx, pool, reference, round_number, open_all):
    """P3：选择排期邻域并精修可行见证；全释放由入口调度器决定。"""
    free_nodes = None if open_all else _neighborhood(ctx, pool, round_number)
    if not open_all and getattr(ctx, 'ph_schedule_nodes', None):
        free_nodes = ctx.ph_schedule_nodes
    schedule_options, polish_options = {}, {}
    if (round_number == 0 and ctx.options.first_pipe_polish_time_limit is not None
            and ctx.options.pipe_polish_enabled and ctx.options.pipe_polish_time_limit > 0):
        # 短预算时最多预留可用搜索时间的一半，仍给 P3 留出尝试机会。
        polish_time = min(ctx.options.first_pipe_polish_time_limit,
                          max(0.0, ctx.check_time() - 5.0) / 2)
        schedule_options["recovery_reserve"] = 5.0 + polish_time
        polish_options = dict(time_limit=polish_time, full_sweep=True)
        ctx.record("P3_first_round_budget", time_limit=ctx.options.schedule_time_limit,
            recovery_reserve=schedule_options["recovery_reserve"],
            pipe_polish_time_limit=polish_time, full_sweep=True)
    scheduled = _attempt(ctx, "P3_schedule", _schedule, pool, reference, free_nodes,
                         **schedule_options)
    if scheduled is not None:
        ctx.consider(scheduled, "P3_witness")
        scheduled = _polish_and_merge(ctx, pool, scheduled, **polish_options)
    return scheduled
