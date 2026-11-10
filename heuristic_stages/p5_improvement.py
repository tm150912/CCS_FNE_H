"""P5：延迟建设、走廊替代、联合邻域、候选扩充与最终精修。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil, floor, isfinite

import gurobipy as gp
import networkx as nx

from .common import (
    COST_FIELDS, CandidatePool, Plan, _BudgetExpired, _attempt, _delay_actions, _delayed_plan,
    _facility_constraints, _flow_constraints, _objective, _period_flow, _pool_signature,
    _status_name, _trim, _values, _verified, _warm_start,
)
from .p4_pipelines import _recover_and_polish
from .pipeline_polish import _polish_and_merge


def _local_delays(ctx, reference):
    """按预计固定费用节省排序，逐次重算实际变化时期并审计可行性。"""
    current = reference.copy()
    actions = _delay_actions(ctx, current)
    for action in actions[:ctx.options.local_search_moves]:
        ctx.check_time()
        changed = _delayed_plan(ctx, current, action)
        if changed is None:
            continue
        flow = _attempt(ctx, "P5_delay", _period_flow, changed, action.key[-1])
        if not flow or flow["plan"] is None:
            continue
        trial = _trim(ctx, flow["plan"])
        if not _verified(ctx, trial, "P5_audit"):
            continue
        old, new = _objective(ctx.c, current), _objective(ctx.c, trial)
        if new < old - ctx.options.improvement_tolerance * max(1.0, abs(old)):
            current = trial
            ctx.record("P5_delay", kind=action.kind, key=str(action.key), improvement=old - new)
            ctx.consider(current, "P5_incumbent")
    return current


def _expand_pool(ctx, pool, round_number=0):
    """按完整源汇通道扩充候选；同时提供替代管径和额外建设数量。"""
    if not ctx.options.expansion_batch:
        return False
    before = (frozenset(pool.units), frozenset(pool.sinks), frozenset(pool.wells.items()), frozenset(pool.pipes.items()))
    graph = nx.DiGraph()
    expansion_arcs = set()
    graph.add_nodes_from(ctx.nodes)
    for u, v in ctx.n.ij:
        if not ctx.pipe_totals[u, v]:
            continue
        weight = min(ctx.c.pipe_fixed[u, v, d, 1] / ctx.n.pipe_capacity[d]
                     + sum(ctx.c.pipe_variable[u, v, d, t] for t in ctx.t) for d in ctx.d)
        graph.add_edge(u, v, weight=weight)

    def add_path(path):
        for u, v in zip(path, path[1:]):
            expansion_arcs.add((u, v))
            for d in ctx.d:
                pool.pipes[u, v, d] = ctx.pipe_totals[u, v]

    def connect(source, sinks):
        ctx.check_time()
        distances, paths = nx.single_source_dijkstra(graph, source)
        reachable = [j for j in sinks if j in distances and j != source]
        if not reachable:
            return None
        sink = min(reachable, key=lambda j: (distances[j], str(j)))
        add_path(paths[sink])
        return sink

    usable_sinks = {j for j in ctx.n.sinks if floor(ctx.n.max_wells[j]) > 0}
    units = sorted(ctx.ig - pool.units, key=lambda ig: (
        (min(ctx.c.source_fixed[(*ig, t)] for t in ctx.t)
         + sum(ctx.c.source_variable[(*ig, t)] * ctx.n.source_capacity[ig] for t in ctx.t))
        / max(ctx.n.source_capacity[ig], 1e-12), str(ig)))
    added = 0
    for i, g in units:
        if added >= ctx.options.expansion_batch:
            break
        if ctx.n.source_capacity[i, g] <= 0:
            continue
        sink = connect(i, usable_sinks)
        if sink is not None:
            pool.units.add((i, g))
            pool.sinks.add(sink)
            added += 1
    # 从每个备选 sink 反向找一个候选源，成组引入注入设施及整条通道。
    sources = {i for i, g in pool.units}
    reverse = graph.reverse(copy=False)
    alternatives = sorted(usable_sinks - pool.sinks,
        key=lambda j: (min(ctx.c.sink_fixed[j, t] + ctx.c.well_fixed[j, t] for t in ctx.t), str(j)))
    for j in alternatives[:ctx.options.expansion_batch]:
        ctx.check_time()
        distances, paths = nx.single_source_dijkstra(reverse, j)
        reachable = [i for i in sources if i in distances]
        if reachable:
            i = min(reachable, key=lambda node: (distances[node], str(node)))
            add_path(list(reversed(paths[i])))
            pool.sinks.add(j)
    # 即使没有新节点，也允许替代管径与源汇路径；各管径候选共享每弧全期总数上限。
    for u, v in {(u, v) for u, v, d in list(pool.pipes)}:
        for d in ctx.d:
            pool.pipes[u, v, d] = ctx.pipe_totals[u, v]
    for j in pool.sinks:
        pool.wells[j] = floor(ctx.n.max_wells[j])
    ordered_sources = sorted(sources)
    offset = (round_number * ctx.options.expansion_batch) % max(1, len(ordered_sources))
    for i in (ordered_sources[offset:] + ordered_sources[:offset])[:ctx.options.expansion_batch]:
        connect(i, pool.sinks)
    after = (frozenset(pool.units), frozenset(pool.sinks), frozenset(pool.wells.items()), frozenset(pool.pipes.items()))
    changed = before != after
    # A new route may reuse old corridor edges. Preserve its complete support
    # when the entry point recycles unused candidates in the next round.
    nodes = {node for arc in expansion_arcs for node in arc}
    sinks = pool.sinks & nodes
    ctx.ph_expansion_support = CandidatePool(
        {key for key in pool.units if key[0] in nodes}, sinks,
        {j: pool.wells[j] for j in sinks},
        {key: value for key, value in pool.pipes.items() if key[:2] in expansion_arcs})
    ctx.record("P5_expand", changed=changed, units=len(pool.units), sinks=len(pool.sinks), pipe_types=len(pool.pipes))
    return changed


@dataclass
class RerouteNeighborhood:
    """一条现用走廊及完整替代通道；仅这些弧和设施允许重新施工。"""
    arcs: set[tuple]
    facilities: set[str]
    score: float = 0.0
    label: str = ""


def _flow_corridors(ctx, reference, end):
    """分解实际逐期正流量，保留共享弧流量守恒，忽略无源汇循环。"""
    aggregate, volume, expense = defaultdict(float), defaultdict(float), defaultdict(float)
    for (u, v, d, t), amount in reference.x.items():
        aggregate[u, v, t] += amount
        volume[u, v] += amount * ctx.c.durations[t]
        expense[u, v] += amount * ctx.c.pipe_variable[u, v, d, t]
    for (u, v, d, t), number in reference.y.items():
        expense[u, v] += number * ctx.c.pipe_fixed[u, v, d, t]
    unit_cost = {edge: expense[edge] / value for edge, value in volume.items() if value > 0}
    supply = defaultdict(float)
    for (i, g, t), amount in reference.a.items():
        supply[i, t] += amount
    corridors = {}
    terminal = object()
    tol = ctx.options.feasibility_tolerance

    def hop_cost(u, v, data):
        ctx.check_time(end)
        return 1.0

    for t in ctx.t:
        ctx.check_time(end)
        graph = nx.DiGraph()
        graph.add_node(terminal)
        for (u, v, period), amount in aggregate.items():
            if period == t and amount > tol:
                graph.add_edge(u, v, amount=amount)
        for (j, period), amount in reference.b.items():
            if period == t and amount > tol:
                graph.add_edge(j, terminal, amount=amount)
        for source in sorted(ctx.n.sources, key=lambda i: (-supply[i, t], str(i))):
            remaining = supply[source, t]
            while remaining > tol and source in graph:
                ctx.check_time(end)
                try:
                    path = nx.shortest_path(graph, source, terminal, weight=hop_cost)
                except nx.NetworkXNoPath:
                    break
                amount = min(remaining, *(graph[u][v]["amount"] for u, v in zip(path, path[1:])))
                route = tuple(path[:-1])
                item = corridors.setdefault(route, dict(profile=defaultdict(float), score=0.0))
                item["profile"][t] += amount
                item["score"] += amount * ctx.c.durations[t] * sum(
                    unit_cost.get(edge, 0.0) for edge in zip(route, route[1:]))
                remaining -= amount
                for u, v in zip(path, path[1:]):
                    graph[u][v]["amount"] -= amount
                    if graph[u][v]["amount"] <= tol:
                        graph.remove_edge(u, v)
    return corridors


def _route_cost_graph(ctx, profile, end):
    """以实际走廊流量估计独立新建费用，不把最大容量包络当作投资。

    各管径分别按需求构造最晚的可行单弧排期，再合计真实建设费与运行费；
    这只是通道筛选权重，共享管线和混合管径的经济性由联合 MILP 判断。
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(ctx.nodes)
    required = {d: {t: ceil(max(0.0, profile.get(t, 0.0) - ctx.options.feasibility_tolerance)
                            / ctx.n.pipe_capacity[d]) for t in ctx.t} for d in ctx.d}
    schedule_cache = {}
    for index, (u, v) in enumerate(ctx.n.ij):
        if index % 64 == 0:
            ctx.check_time(end)
        limit = ctx.pipe_totals[u, v]
        if limit not in schedule_cache:
            schedules = {}
            for d in ctx.d:
                if max(required[d].values()) > limit:
                    continue
                dates, installed = {}, 0
                for t in ctx.t:
                    needed = max(0, required[d][t] - installed)
                    if needed:
                        dates[t] = needed
                        installed += needed
                schedules[d] = dates
            schedule_cache[limit] = schedules
        choices = []
        for d, dates in schedule_cache[limit].items():
            fixed = sum(ctx.c.pipe_fixed[u, v, d, t] * count for t, count in dates.items())
            operation = sum(ctx.c.pipe_variable[u, v, d, t] * profile.get(t, 0.0) for t in ctx.t)
            choices.append((fixed + operation, fixed, operation))
        if choices:
            total, fixed, operation = min(choices)
            graph.add_edge(u, v, total=total, fixed=min(item[1] for item in choices),
                           operation=min(item[2] for item in choices))
    return graph


def _reroute_neighborhoods(ctx, pool, reference, round_number=0):
    """从真实高费用走廊定位替代路，并轮换覆盖其余现用源。"""
    if (not ctx.options.reroute_enabled or not ctx.options.reroute_batch
            or ctx.options.reroute_time_limit <= 0):
        return []
    pool.merge(CandidatePool.from_plan(reference))
    end = ctx.deadline(ctx.options.reroute_time_limit)
    before = set(pool.pipes)
    result = []
    budget_exhausted = False
    try:
        corridors = _flow_corridors(ctx, reference, end)
        by_source = defaultdict(list)
        for route, item in corridors.items():
            if len(route) > 1:
                by_source[route[0]].append((route, item))
        sources = sorted(by_source, key=lambda i: (-sum(v["score"] for _, v in by_source[i]), str(i)))
        count = min(ctx.options.reroute_batch, len(sources))
        priority_count = max(1, count // 2)
        selected = sources[:priority_count]
        rest = sources[priority_count:]
        offset = (round_number * max(1, count - priority_count)) % max(1, len(rest))
        selected += (rest[offset:] + rest[:offset])[:max(0, count - priority_count)]
        usable_sinks = [j for j in ctx.n.sinks
            if floor(ctx.n.max_wells[j]) > 0]
        facility_volume = defaultdict(float)
        for name in ("a", "b"):
            for key, amount in getattr(reference, name).items():
                facility_volume[key[0]] += amount * ctx.c.durations[key[-1]]
        facility_nodes = set(ctx.n.sources) | set(ctx.n.sinks)
        for source in selected:
            ctx.check_time(end)
            routes = sorted(by_source[source], key=lambda pair: (-pair[1]["score"], tuple(map(str, pair[0]))))
            route, item = routes[round_number % len(routes)]
            graph = _route_cost_graph(ctx, item["profile"], end)
            if source not in graph:
                continue
            neighborhood = RerouteNeighborhood(set(zip(route, route[1:])),
                {source, route[-1]}, item["score"], f"{source}->{route[-1]}")

            def weight(mode):
                def value(u, v, data):
                    ctx.check_time(end)
                    return data[mode]
                return value

            def append_path(path, sink):
                # Only complete, original directed paths may enter the candidate pool.
                edges = set(zip(path, path[1:]))
                if not edges <= ctx.arcs:
                    raise ValueError("Reroute path contains a non-original arc")
                if sink not in neighborhood.facilities and len(neighborhood.facilities) >= ctx.options.joint_neighborhood_size:
                    return
                neighborhood.arcs.update(edges)
                neighborhood.facilities.add(sink)

            distances, paths = nx.single_source_dijkstra(graph, source, weight=weight("total"))
            if route[-1] in paths:
                append_path(paths[route[-1]], route[-1])
            for mode in ("fixed", "operation")[:max(0, ctx.options.reroute_paths - 1)]:
                try:
                    append_path(nx.shortest_path(graph, source, route[-1], weight=weight(mode)), route[-1])
                except nx.NetworkXNoPath:
                    pass
            # Alternative sink cost includes its facilities and injection operation.
            peak = max(item["profile"].values(), default=0.0)
            first = min(item["profile"], default=ctx.t[0])
            def sink_cost(j):
                wells = ceil(max(0.0, peak - ctx.options.feasibility_tolerance) / ctx.qw[j])
                return (distances[j] + ctx.c.sink_fixed[j, first]
                        + wells * ctx.c.well_fixed[j, first]
                        + sum(ctx.c.sink_variable[j, t] * value for t, value in item["profile"].items()))
            alternatives = sorted((j for j in usable_sinks if j != route[-1] and j in paths),
                                  key=lambda j: (sink_cost(j), str(j)))
            for j in alternatives[:ctx.options.reroute_paths]:
                append_path(paths[j], j)
            # Preserve selected endpoints first, then release related source/sink
            # construction along either route, prioritizing actual horizon throughput.
            related = ({node for edge in neighborhood.arcs for node in edge}
                       & facility_nodes) - neighborhood.facilities
            room = max(0, ctx.options.joint_neighborhood_size - len(neighborhood.facilities))
            neighborhood.facilities.update(sorted(related,
                key=lambda node: (-facility_volume[node], str(node)))[:room])
            # Save a fully generated neighborhood atomically; interrupted paths are discarded.
            for u, v in neighborhood.arcs:
                for d in ctx.d:
                    pool.pipes[u, v, d] = ctx.pipe_totals[u, v]
            pool.units.update(ig for ig in ctx.ig if ig[0] in neighborhood.facilities)
            for j in neighborhood.facilities & set(ctx.n.sinks):
                pool.sinks.add(j)
                pool.wells[j] = floor(ctx.n.max_wells[j])
            result.append(neighborhood)
    except _BudgetExpired:
        budget_exhausted = True
    ctx.record("P5_reroute_candidates", neighborhoods=len(result),
               added_pipe_types=len(set(pool.pipes) - before), budget_exhausted=budget_exhausted,
               selected_corridors=[item.label for item in result])
    return result


def _joint_neighborhood(ctx, pool, reference, neighborhood):
    """全期联合局部 MILP：邻域外施工固定，所有连续流量重新优化。"""
    if not ctx.options.joint_lns_enabled or ctx.options.joint_lns_time_limit <= 0:
        return reference
    state = (_construction_signature(reference), _objective(ctx.c, reference))
    if state != getattr(ctx, "_ph_joint_state", None):
        ctx._ph_joint_state, ctx._ph_joint_cooldown = state, {}
    key = (frozenset(neighborhood.arcs), frozenset(neighborhood.facilities))
    cooldown = getattr(ctx, "_ph_joint_cooldown", {})
    remaining, status = cooldown.get(key, (0, None))
    if remaining:
        cooldown[key] = (remaining - 1, status)
        ctx.record("P5_joint_cooldown", label=neighborhood.label,
                   remaining_skips=remaining - 1, previous_solver_status=status,
                   skipped="no_improvement_cooldown",
                   solver_status="TIME_LIMIT" if status == "TIME_LIMIT" else "NOT_RUN")
        return reference
    ctx._ph_joint_status = None
    try:
        result = _joint_neighborhood_milp(ctx, pool, reference, neighborhood)
    except _BudgetExpired:
        ctx.record("P5_joint_reroute", solver_status="TIME_LIMIT", reason="local_budget",
                   start_objective=_objective(ctx.c, reference),
                   label=neighborhood.label, accepted=False)
        result = reference
        ctx._ph_joint_status = "TIME_LIMIT"
    old = _objective(ctx.c, reference)
    gain = old - _objective(ctx.c, result)
    if gain <= ctx.options.improvement_tolerance * max(1.0, abs(old)):
        status = ctx._ph_joint_status
        if status in ("OPTIMAL", "TIME_LIMIT"):
            # A timed-out solve gets one skipped repeat, never an optimality claim.
            cooldown[key] = (2 if status == "OPTIMAL" else 1, status)
    else:
        cooldown.pop(key, None)
    ctx._ph_joint_cooldown = cooldown
    return result


def _construction_signature(plan):
    return tuple(frozenset((key, value) for key, value in getattr(plan, name).items() if value)
                 for name in ("s", "r", "w", "y"))


def _joint_neighborhood_milp(ctx, pool, reference, neighborhood):
    """执行单个邻域；局部建模预算耗尽由调用层保留原解。"""
    if not neighborhood.arcs <= ctx.arcs:
        raise ValueError("Joint neighborhood contains a non-original arc")
    end = ctx.deadline(ctx.options.joint_lns_time_limit)
    start_objective = _objective(ctx.c, reference)
    facilities = set(neighborhood.facilities)
    if not facilities <= set(ctx.n.sources) | set(ctx.n.sinks):
        raise ValueError("Joint neighborhood facilities must be original source/sink nodes")
    pool.merge(CandidatePool.from_plan(reference))
    built_arcs = {key[:2] for key, value in reference.y.items() if value}
    arcs = sorted(built_arcs | neighborhood.arcs)
    # Omitted outside-neighborhood construction is fixed to zero in the full model.
    # Keep all original nodes in flow balance: unopened facilities can still transit flow.
    built_units = {key[:2] for key, value in reference.s.items() if value}
    built_sinks = {key[0] for key, value in reference.r.items() if value}
    built_types = {key[:3] for key, value in reference.y.items() if value}
    local_units = {ig for ig in ctx.ig if ig[0] in facilities} | built_units
    local_sinks = (set(ctx.n.sinks) & facilities) | built_sinks
    pipe_types = {(u, v): [d for d in ctx.d
                  if (u, v) in neighborhood.arcs or (u, v, d) in built_types]
                  for u, v in arcs}
    local_pool = CandidatePool(local_units, local_sinks,
        {j: floor(ctx.n.max_wells[j]) for j in local_sinks}, {})
    with ctx.model("joint_reroute_neighborhood") as model:
        ctx.check_time(end)
        units = [(i, g, t) for i, g in sorted(local_pool.units) for t in ctx.t]
        sinks = [(j, t) for j in sorted(local_pool.sinks) for t in ctx.t]
        pipes = [(u, v, d, t) for u, v in arcs for d in pipe_types[u, v] for t in ctx.t]
        fixed, construction = {}, {}
        for name, keys in (("s", units), ("r", sinks), ("w", sinks), ("y", pipes)):
            free_keys = [key for key in keys
                         if (key[:2] in neighborhood.arcs if name == "y" else key[0] in facilities)]
            free_set = set(free_keys)
            fixed[name] = {key: getattr(reference, name).get(key, 0)
                           for key in keys if key not in free_set}
            construction[name] = model.addVars(free_keys, lb=0,
                ub=1 if name in ("s", "r") else ctx.pipeline_install_limit if name == "y" else gp.GRB.INFINITY,
                vtype=gp.GRB.BINARY if name in ("s", "r") else gp.GRB.INTEGER, name=name)
        s, r, w, y = ({**fixed[name], **construction[name]} for name in ("s", "r", "w", "y"))
        a = model.addVars(units, lb=0, name="a")
        b = model.addVars(sinks, lb=0, name="b")
        x = model.addVars(pipes, lb=0, name="x")
        _facility_constraints(ctx, model, local_pool, s, r, w, a, b)
        for number, (u, v) in enumerate(arcs):
            if number % 64 == 0:
                ctx.check_time(end)
            for d in pipe_types[u, v]:
                active = gp.LinExpr()
                for t in ctx.t:
                    active += y[u, v, d, t]
                    model.addConstr(x[u, v, d, t] <= ctx.n.pipe_capacity[d] * active)
            model.addConstr(gp.quicksum(y[u, v, d, t] for d in pipe_types[u, v] for t in ctx.t)
                            <= ctx.pipeline_install_limit)
        _flow_constraints(model, ctx.nodes, ctx.t, ctx.c.capture_target, a, b, x)
        variables = dict(**construction, a=a, b=b, x=x)
        for name, mapping in variables.items():
            ctx.check_time(end)
            _warm_start(mapping, getattr(reference, name))
        constant = sum(getattr(ctx.c, COST_FIELDS[name])[key] * value
                       for name, mapping in fixed.items() for key, value in mapping.items())
        model.setObjective(constant + gp.quicksum(getattr(ctx.c, COST_FIELDS[name])[key] * var
            for name, mapping in variables.items() for key, var in mapping.items()), gp.GRB.MINIMIZE)
        from .warm_start import optimize_with_reference
        optimize_with_reference(ctx, model, end, stage="P5_joint_reroute")
        ctx._ph_joint_status = _status_name(model.Status)
        bound = float(model.ObjBound) if model.IsMIP else None
        gap = float(model.MIPGap) if model.IsMIP and model.SolCount else None
        ctx.record("P5_joint_reroute", solver_status=_status_name(model.Status),
            start_objective=start_objective, solver_runtime=float(model.Runtime),
            node_count=float(model.NodeCount),
            objective=float(model.ObjVal) if model.SolCount else None,
            restricted_bound=bound if bound is not None and isfinite(bound) else None,
            restricted_gap=gap if gap is not None and isfinite(gap) else None,
            free_arcs=len(neighborhood.arcs), free_facilities=len(facilities),
            variable_count=model.NumVars, construction_variable_count=model.NumIntVars,
            solution_count=model.SolCount, label=neighborhood.label)
        if not model.SolCount:
            return reference
        result = Plan(**{name: {**{key: value for key, value in fixed.get(name, {}).items() if value},
                               **_values(mapping, name in ("s", "r", "w", "y"))}
                         for name, mapping in variables.items()})
        # Do not trim outside construction: that would violate the stated neighborhood.
        if not _verified(ctx, result, "P5_joint_reroute_audit"):
            ctx._ph_joint_status = "AUDIT_REJECTED"
            return reference
        value, old_value = _objective(ctx.c, result), _objective(ctx.c, reference)
        threshold = ctx.options.improvement_tolerance * max(1.0, abs(old_value))
        if value < old_value - threshold:
            ctx.consider(result, "P5_joint_reroute_accept")
            pool.merge(CandidatePool.from_plan(result))
            return result
        return reference


def _reoptimize_operations(ctx, reference):
    """固定建设后重算各期运行流量，逐期保存通过原约束审计的改善。"""
    current = reference
    signature = _construction_signature(reference)
    completed = getattr(ctx, "_ph_optimal_operations", {})

    def operation(plan, period):
        return sum(getattr(ctx.c, COST_FIELDS[name])[key] * value
                   for name in ("a", "b", "x")
                   for key, value in getattr(plan, name).items() if key[-1] == period)

    for t in ctx.t:
        ctx.check_time()
        if t in completed:
            previous_signature, optimum = completed[t]
            tolerance = ctx.options.improvement_tolerance * max(1.0, abs(optimum))
            if previous_signature == signature and operation(current, t) <= optimum + tolerance:
                ctx.record("P5_operations", period=t, skipped="unchanged_optimal_operations")
                continue
        evaluated = _attempt(ctx, "P5_operations", _period_flow, current, t)
        if evaluated and evaluated["plan"] is not None:
            trial = evaluated["plan"]
            cheaper = _objective(ctx.c, trial) < _objective(ctx.c, current)
            valid = _verified(ctx, trial, "P5_operations_audit")
            if valid and evaluated["status"] == "OPTIMAL":
                completed[t] = (signature, operation(trial, t))
                ctx._ph_optimal_operations = completed
            if cheaper and valid:
                current = trial
                ctx.consider(current, "P5_operations")
    return current


def final_polish(ctx):
    """在仍有效的 Gurobi 环境和硬预算内，直接从当前最好解进行最后精修。"""
    if ctx.incumbent is None:
        return
    ctx.check_time()
    pool = getattr(ctx, "search_pool", CandidatePool.from_plan(ctx.incumbent))
    pool.merge(CandidatePool.from_plan(ctx.incumbent))
    ctx.record("final_polish_start", remaining=ctx.check_time())
    # Complete inexpensive improvements before spending the tail on another MIP.
    remaining, previous_end = ctx.check_time(), ctx.phase_end
    reserve = min(ctx.options.pipe_polish_time_limit, 0.25 * remaining)
    ctx.phase_end = ctx.deadline(remaining - reserve)
    try:
        _reoptimize_operations(ctx, ctx.incumbent.copy())
    except _BudgetExpired:
        ctx.record("P5_operations", stopped="small_polish_reserve")
    finally:
        ctx.phase_end = previous_end
    ctx.check_time()
    _polish_and_merge(ctx, pool, ctx.incumbent.copy(), optimize_multibuild=False)
    for neighborhood in getattr(ctx, "_ph_prepared_neighborhoods", []):
        if ctx.check_time() < ctx.options.joint_lns_time_limit + ctx.options.pipe_polish_time_limit:
            break
        joint = _attempt(ctx, "P5_joint", _joint_neighborhood, pool, ctx.incumbent.copy(), neighborhood)
        if joint is not None:
            ctx.consider(joint, "P5_joint_incumbent")
            _polish_and_merge(ctx, pool, joint, optimize_multibuild=False)
    if (ctx.options.max_rounds > 0 and
            ctx.check_time() >= ctx.options.pipe_time_limit + ctx.options.pipe_polish_time_limit):
        _recover_and_polish(ctx, pool, ctx.incumbent.copy())
    else:
        ctx.record("final_polish_recovery", skipped="insufficient_full_recovery_budget")
    ctx.record("final_polish_end", objective=ctx.best)


def run_p5(ctx, pool, neighborhoods):
    """P5：按固定费用节省延迟建设，再搜索联合邻域，更新最好解和候选池。"""
    reference = _local_delays(ctx, ctx.incumbent)
    pool.merge(CandidatePool.from_plan(reference))
    for neighborhood in neighborhoods:
        ctx.check_time()
        joint = _attempt(ctx, "P5_joint", _joint_neighborhood, pool, ctx.incumbent.copy(), neighborhood)
        if joint is not None:
            ctx.consider(joint, "P5_joint_incumbent")
            _polish_and_merge(ctx, pool, joint)
    ctx.check_time()


def prepare_neighborhoods(ctx, pool, reference, round_number):
    """每轮 P3 前扩充走廊，返回联合邻域及候选池是否变化。"""
    before = _pool_signature(pool)
    neighborhoods = _attempt(ctx, "P5_reroute", _reroute_neighborhoods, pool, reference, round_number) or []
    ctx._ph_prepared_neighborhoods = neighborhoods
    changed_pool = before != _pool_signature(pool)
    ctx.check_time()
    return neighborhoods, changed_pool
