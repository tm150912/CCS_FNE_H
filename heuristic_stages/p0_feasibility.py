"""Build the transportation template and allocate its aggregate flows."""
from __future__ import annotations

from collections import defaultdict

from .common import _cumulative


def _envelope(ctx, pool):
    """按每弧全期管道总数上限在首期优先建设大管径，生成容量包络。"""
    by_arc = defaultdict(list)
    for (u, v, d), number in pool.pipes.items():
        by_arc[u, v].append((d, min(int(number), ctx.pipe_totals[u, v])))
    y, capacities = {}, {}
    for index, (edge, choices) in enumerate(by_arc.items()):
        if index % 512 == 0:
            ctx.check_time()
        choices.sort(key=lambda item: ctx.n.pipe_capacity[item[0]], reverse=True)
        slots = ctx.pipe_totals[edge]
        capacity = 0.0
        for d, number in choices:
            count = min(slots, number)
            if count:
                y[(*edge, d, ctx.t[0])] = count
                capacity += ctx.n.pipe_capacity[d] * count
                slots -= count
            if not slots:
                break
        capacities.update({(*edge, t): capacity for t in ctx.t})
    return y, capacities


def _split_flows(ctx, plan, aggregate):
    cumulative = _cumulative(plan.y, ctx.t)
    by_arc = defaultdict(set)
    for u, v, d, _ in plan.y:
        by_arc[u, v].add(d)
    plan.x = {}
    for (u, v, t), value in aggregate.items():
        remaining = value
        for d in sorted(by_arc[u, v], key=lambda d: ctx.c.pipe_variable[u, v, d, t]):
            amount = min(remaining, ctx.n.pipe_capacity[d] * cumulative[u, v, d, t])
            if amount > 0:
                plan.x[u, v, d, t] = amount
            remaining -= amount
        if remaining > ctx.options.feasibility_tolerance:
            raise RuntimeError("Pipeline envelope cannot carry its aggregate flow")
