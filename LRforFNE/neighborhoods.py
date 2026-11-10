"""LR-selected delays, final-count time windows, and complete-path replacement."""
from collections import defaultdict, OrderedDict
from dataclasses import replace
from math import floor, ceil, isfinite
from time import perf_counter

import gurobipy as gp
import networkx as nx

from heuristic_stages.common import (
    COST_FIELDS, CandidatePool, Plan, _BudgetExpired, _delay_actions,
    _delayed_plan, _facility_constraints, _flow_constraints, _objective,
    _period_flow, _solver_metrics, _status_name, _values, _warm_start)
from heuristic_stages.p5_improvement import (
    RerouteNeighborhood, _flow_corridors, _route_cost_graph)

from .guidance import CONSTRUCTION
from .corridor_expansion import expand_scheduled
from .local_search import solve_neighborhood as _joint_neighborhood


def _construction_signature(plan):
    return tuple(frozenset((key, value) for key, value in getattr(plan, name).items() if value)
                 for name in CONSTRUCTION)


def _search_state(ctx, plan):
    """Cache explored regions only for the current construction, retaining rotation."""
    state = getattr(ctx, '_lr_neighborhood_state', None)
    if state is None:
        state = dict(signature=None, completed=set(), n1_failed=set(), n2_done=set(),
                     cursor=0, calls=0, sweep=set(), sweep_complete=False,
                     refinable={}, refined=set(), variants={}, anchors={}, metadata={},
                     graphs=OrderedDict(), corridor_reference=None, corridors=None,
                     expansions={}, expansion_reference=None)
        ctx._lr_neighborhood_state = state
    signature = _construction_signature(plan)
    if state['signature'] != signature:
        for name in ('variants', 'anchors'):
            for item in state[name].values():
                item['failures'] //= 2
        state.update(signature=signature, completed=set(), n1_failed=set(),
                     n2_done=set(), sweep=set(), sweep_complete=False,
                     refinable={}, refined=set(), expansions={})
    if state['expansion_reference'] is not plan:
        state['expansions'] = {}
        state['expansion_reference'] = plan
    return state


def _neighborhood_signature(neighborhood):
    return frozenset(neighborhood.arcs), frozenset(neighborhood.facilities)


def _scheduled_signature(state, neighborhood, config):
    signature = _neighborhood_signature(neighborhood)
    if getattr(config, 'n3_expand_corridors', 0):
        # An unexpanded original domain may equal a completed expanded domain,
        # yet expand further itself. Only a known mapping can certify a repeat.
        return state.get('expansions', {}).get(signature)
    return signature


def n1_delays(ctx, info, max_moves):
    """Rank actual one-unit delay actions with LR signals, then audit full cost."""
    reference = ctx.incumbent
    desired = info["suggested"]
    def priority(action):
        obj = (action.kind, action.key[:-1])
        t = action.key[-1]
        mismatch = sum(v for key, v in getattr(reference, action.kind).items()
                       if key[:-1] == obj[1] and key[-1] <= t) - sum(
                           v for key, v in getattr(desired, action.kind).items()
                           if key[:-1] == obj[1] and key[-1] <= t)
        return (obj not in info["objects"], -mismatch,
                -len(info["reasons"].get(obj, ())), -action.saving, str(action.key))
    actions = sorted(_delay_actions(ctx, reference), key=priority)
    attempts = 0
    for action in actions:
        if attempts >= max_moves:
            break
        ctx.check_time()
        state = _search_state(ctx, ctx.incumbent)
        key = (action.kind, action.key)
        if key in state['n1_failed']:
            ctx.record('LR_N1_skip', kind=action.kind, key=list(action.key),
                       skipped='same_construction_proven_infeasible')
            continue
        trial = _delayed_plan(ctx, ctx.incumbent, action)
        if trial is None:
            continue
        attempts += 1
        result = _period_flow(ctx, trial, action.key[-1])
        if result['status'] == 'INFEASIBLE':
            state['n1_failed'].add(key)
        accepted = False
        if result["plan"] is not None and ctx.audit(result["plan"])["feasible"]:
            accepted = ctx.consider(result["plan"], "LR_N1_accept")
        ctx.record("LR_N1", kind=action.kind, key=list(action.key),
                   reasons=sorted(info["reasons"].get((action.kind, action.key[:-1]), ("delay_rotation",))),
                   accepted=accepted, solver_status=result["status"])
    return ctx.incumbent


def window_limits(additions, periods, tau_minus, tau_plus):
    """Return the exact cumulative window bounds, including out-of-period edges."""
    cumulative = [0.0]
    for t in periods:
        cumulative.append(cumulative[-1] + additions.get(t, 0))
    horizon = len(periods)
    return {t: (cumulative[max(0, h - tau_plus)],
                cumulative[min(horizon, h + tau_minus)])
            for h, t in enumerate(periods, 1)}


def n2_time_window(ctx, reference, released, tau_minus, tau_plus, time_limit):
    """Only nonzero final objects exist; uncreated variables are identically zero."""
    end = ctx.deadline(time_limit)
    pool = CandidatePool.from_plan(reference)
    groups = {name: {key[:-1] for key, value in getattr(reference, name).items() if value > 0}
              for name in CONSTRUCTION}
    # All built sites remain even if their final well count is zero.
    sink_keys = [(j, t) for j in sorted(pool.sinks, key=str) for t in ctx.t]
    unit_keys = [(*ig, t) for ig in sorted(pool.units, key=str) for t in ctx.t]
    pipe_keys = [(*ed, t) for ed in sorted(groups["y"], key=str) for t in ctx.t]
    with ctx.model("LR_N2_time_window") as model:
        variables = dict(
            s=model.addVars(unit_keys, vtype=gp.GRB.BINARY, name="s"),
            r=model.addVars(sink_keys, vtype=gp.GRB.BINARY, name="r"),
            w=model.addVars(sink_keys, vtype=gp.GRB.INTEGER, name="w"),
            y=model.addVars(pipe_keys, vtype=gp.GRB.INTEGER, name="y"),
            a=model.addVars(unit_keys, name="a"), b=model.addVars(sink_keys, name="b"),
            x=model.addVars(pipe_keys, name="x"))
        _facility_constraints(ctx, model, pool, variables["s"], variables["r"],
                              variables["w"], variables["a"], variables["b"])
        arc_variables = defaultdict(list)
        for ed in sorted(groups["y"], key=str):
            ctx.check_time(end)
            active = gp.LinExpr()
            for t in ctx.t:
                key = (*ed, t)
                active += variables["y"][key]
                model.addConstr(variables["x"][key] <= ctx.n.pipe_capacity[ed[2]] * active)
                arc_variables[ed[:2]].append(variables["y"][key])
        for vals in arc_variables.values():
            model.addConstr(gp.quicksum(vals) <= ctx.pipeline_install_limit)
        _flow_constraints(model, ctx.nodes, ctx.t, ctx.c.capture_target,
                          variables["a"], variables["b"], variables["x"])
        for name in CONSTRUCTION:
            prefixes = {key[:-1] for key in variables[name]}
            for prefix in prefixes:
                ctx.check_time(end)
                additions = {t: getattr(reference, name).get((*prefix, t), 0) for t in ctx.t}
                free = (name, prefix) in released
                limits = window_limits(additions, ctx.t, tau_minus if free else 0,
                                       tau_plus if free else 0)
                active = gp.LinExpr()
                for t in ctx.t:
                    active += variables[name][(*prefix, t)]
                    lo, hi = limits[t]
                    model.addConstr(active >= lo)
                    model.addConstr(active <= hi)
                model.addConstr(active == sum(additions.values()))
        for name, mapping in variables.items():
            _warm_start(mapping, getattr(reference, name))
        model.setObjective(gp.quicksum(getattr(ctx.c, COST_FIELDS[name])[key] * var
            for name, mapping in variables.items() for key, var in mapping.items()))
        ctx.optimize(model, end)
        ctx.record("LR_N2", solver_status=_status_name(model.Status),
                   tau_minus=tau_minus, tau_plus=tau_plus,
                   released=[dict(kind=n, object=list(p)) for n, p in sorted(released, key=str)],
                   **_solver_metrics(model))
        if not model.SolCount:
            return reference
        result = Plan(**{name: _values(mapping, name in CONSTRUCTION)
                         for name, mapping in variables.items()})
        if not ctx.audit(result)["feasible"]:
            return reference
        # No trimming or diameter change belongs to this neighborhood.
        for name in CONSTRUCTION:
            totals = defaultdict(float)
            for key, value in getattr(result, name).items():
                totals[key[:-1]] += value
            old = defaultdict(float)
            for key, value in getattr(reference, name).items():
                old[key[:-1]] += value
            if any(abs(totals[k] - old[k]) > ctx.options.feasibility_tolerance
                   for k in totals.keys() | old.keys()):
                raise RuntimeError("N2 changed a final construction count")
        if _objective(ctx.c, result) < _objective(ctx.c, reference) - (
                ctx.options.improvement_tolerance * max(1, abs(_objective(ctx.c, reference)))):
            ctx.consider(result, "LR_N2_accept")
            return result
        return reference


def _rank_nodes(ctx, info, nodes):
    def rank(node):
        objects = [(kind, prefix) for kind, prefix in info["objects"]
                   if kind != "y" and prefix[0] == node]
        rotation = any("rotation" in info["reasons"].get(obj, ()) for obj in objects)
        difference = sum(info["metrics"]["cumulative_difference"].get(obj, 0) for obj in objects)
        return (not rotation, -difference, -len(objects), str(node))
    return sorted(nodes, key=rank)


def _facility_price(ctx, node, profile):
    """Greedy original-cost estimate for ranking only, never a bound."""
    active = [t for t in ctx.t if profile.get(t, 0) > ctx.options.feasibility_tolerance]
    if not active:
        return 0.0
    first, peak = active[0], max(profile.values())
    if node in ctx.n.sinks:
        wells = ceil(peak / ctx.qw[node])
        if wells > ctx.max_wells[node]:
            return float('inf')
        return (ctx.c.sink_fixed[node, first] + wells * ctx.c.well_fixed[node, first]
                + sum(ctx.c.sink_variable[node, t] * profile.get(t, 0) for t in ctx.t))
    units = sorted(ctx.n.source_units[node], key=lambda g: (
        (ctx.c.source_fixed[node, g, first] + sum(
            ctx.c.source_variable[node, g, t] * min(profile.get(t, 0), ctx.n.source_capacity[node, g])
            for t in ctx.t)) / max(ctx.n.source_capacity[node, g], 1e-12), str(g)))
    selected, capacity, value = [], 0.0, 0.0
    for g in units:
        if capacity >= peak:
            break
        selected.append(g)
        capacity += ctx.n.source_capacity[node, g]
        value += ctx.c.source_fixed[node, g, first]
    if capacity < peak:
        return float('inf')
    for t in ctx.t:
        remaining = profile.get(t, 0)
        for g in sorted(selected, key=lambda g: ctx.c.source_variable[node, g, t]):
            amount = min(remaining, ctx.n.source_capacity[node, g])
            value += amount * ctx.c.source_variable[node, g, t]
            remaining -= amount
    return value


def _cached_route_graph(ctx, state, profile, end):
    # Original network/costs are constant within a context; exact profiles are keys.
    key = tuple(profile.get(t, 0) for t in ctx.t)
    if key in state['graphs']:
        state['graphs'].move_to_end(key)
        return state['graphs'][key], True
    graph = _route_cost_graph(ctx, profile, end)
    state['graphs'][key] = graph
    if len(state['graphs']) > 8:
        state['graphs'].popitem(last=False)
    return graph, False


def _released_costs(ctx, reference):
    facilities, arcs = defaultdict(float), defaultdict(float)
    facility_volume, arc_volume = defaultdict(float), defaultdict(float)
    for name in (*CONSTRUCTION, 'a', 'b', 'x'):
        target = arcs if name in ('y', 'x') else facilities
        for key, value in getattr(reference, name).items():
            target[key[:2] if name in ('y', 'x') else key[0]] += (
                getattr(ctx.c, COST_FIELDS[name])[key] * value)
            if name in ('a', 'b', 'x'):
                volume = arc_volume if name == 'x' else facility_volume
                volume[key[:2] if name == 'x' else key[0]] += value * ctx.c.durations[key[-1]]
    return facilities, arcs, facility_volume, arc_volume


def _retired_cost(ctx, routes, costs):
    # Allocate shared original expenditure to the flows being replaced. New
    # endpoints' unrelated existing expenditure is not counted as removable.
    facilities, arcs, facility_volume, arc_volume = costs
    moved_facilities, moved_arcs = defaultdict(float), defaultdict(float)
    for path, profile in routes:
        volume = sum(profile.get(t, 0) * ctx.c.durations[t] for t in ctx.t)
        for node in (path[0], path[-1]):
            moved_facilities[node] += volume
        for edge in zip(path, path[1:]):
            moved_arcs[edge] += volume
    return (sum(facilities[node] * min(1.0, value / max(facility_volume[node], 1e-12))
                for node, value in moved_facilities.items())
            + sum(arcs[edge] * min(1.0, value / max(arc_volume[edge], 1e-12))
                  for edge, value in moved_arcs.items()))


def _replacement_price(ctx, routes):
    """Price summed replacement profiles, counting shared facilities/arcs once."""
    facilities, edges = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
    for path, profile in routes:
        for node in (path[0], path[-1]):
            for t, amount in profile.items():
                facilities[node][t] += amount
        for edge in zip(path, path[1:]):
            for t, amount in profile.items():
                edges[edge][t] += amount
    price = sum(_facility_price(ctx, node, profile) for node, profile in facilities.items())
    for (u, v), profile in edges.items():
        choices = []
        for diameter in ctx.d:
            capacity, installed, cost = ctx.n.pipe_capacity[diameter], 0, 0.0
            for t in ctx.t:
                required = ceil(max(0, profile.get(t, 0) - ctx.options.feasibility_tolerance) / capacity)
                additions = max(0, required - installed)
                installed += additions
                cost += additions * ctx.c.pipe_fixed[u, v, diameter, t]
                cost += profile.get(t, 0) * ctx.c.pipe_variable[u, v, diameter, t]
            if installed <= ctx.pipe_totals[u, v]:
                choices.append(cost)
        price += min(choices, default=float('inf'))
    return price


def _search_feedback(state, meta, elapsed, improvement):
    """Persistent discounted rewards; construction changes do not erase learning."""
    for mapping, key in ((state['variants'], meta['variant']),
                         (state['anchors'], meta['anchor'])):
        item = mapping.setdefault(key, dict(gain=0.0, wall=0.0, failures=0, calls=0, last_batch=0))
        item['gain'] = 0.85 * item['gain'] + max(0, improvement)
        item['wall'] = 0.85 * item['wall'] + elapsed
        item['failures'] = 0 if improvement > 0 else item['failures'] + 1
        item['calls'] += 1
        item['last_batch'] = state['calls']


def _schedule_neighborhoods(ctx, state, neighborhoods, config):
    """Spend a bounded batch on economic candidates and one rotating exploration."""
    if not neighborhoods:
        return []
    limit = min(config.n3_candidate_limit, len(neighborhoods))
    if getattr(config, 'n3_expand_corridors', 0):
        limit = min(limit, 4)
    fresh = [n for n in neighborhoods if _scheduled_signature(state, n, config) not in state['completed']]
    ordered = sorted(fresh, key=lambda n: n.label)
    exploration = ordered[(state['calls'] - 1) % len(ordered)] if ordered else None
    best_score = max(n.score for n in neighborhoods)
    economic_scale = max(1.0, best_score - min(n.score for n in neighborhoods))
    rates = {name: {key: item['gain'] / max(item['wall'], 1e-6)
                   for key, item in state[name].items()} for name in ('variants', 'anchors')}
    rate_scales = {name: max(1.0, max(values.values(), default=0)) for name, values in rates.items()}

    def priority(neighborhood):
        meta = state['metadata'][neighborhood.label]
        learned, failures = 0.0, 0
        for name, key in (('variants', meta['variant']), ('anchors', meta['anchor'])):
            item = state[name].get(key, {})
            learned += rates[name].get(key, 0) / rate_scales[name]
            failures += item.get('failures', 0)
        economic_rank = 1.0 + (neighborhood.score - best_score) / economic_scale
        score = (economic_rank + learned) / (1 + failures / 8)
        return score, -len(neighborhood.arcs), neighborhood.label

    def cooling(candidate):
        meta = state['metadata'][candidate.label]
        history = state['anchors'].get(meta['anchor'], {})
        failures = history.get('failures', 0)
        return (failures >= 12 and
                state['calls'] - history.get('last_batch', 0) < min(32, failures))

    def refinement_priority(candidate):
        meta = state['metadata'][candidate.label]
        history = state['anchors'].get(meta['anchor'], {})
        weight = (1 - 0.85 ** history.get('calls', 0)) / 0.15
        expected = history.get('wall', 0) / max(1, weight)
        room = state['refinable'][_scheduled_signature(state, candidate, config)]
        return room / max(0.05, expected), priority(candidate)

    # An independent refinement slot must not be blocked by an approximate
    # negative replacement price. The restricted bound supplies the local room.
    pending = [n for n in neighborhoods
               if _scheduled_signature(state, n, config) in state['refinable'] and not cooling(n)]
    reserve = min(config.n3_refine_candidates, max(0, limit - int(bool(fresh))))
    chosen = sorted(pending, key=refinement_priority, reverse=True)[:reserve]
    refinements = {candidate.label for candidate in chosen}
    if exploration is not None:
        chosen.append(exploration)
    # Negative proxies are useful for relative ranking, not exclusion proofs.
    # Use a smaller batch when all proxies are negative, retaining exploration.
    batch_limit = min(limit, max(2, limit // 2)) if best_score <= 0 else limit
    batch_limit = max(batch_limit, len(chosen))
    ranked = sorted(fresh, key=priority, reverse=True)
    for candidate in ranked:
        if len(chosen) >= batch_limit:
            break
        if candidate in chosen:
            continue
        if cooling(candidate):
            continue
        chosen.append(candidate)
    chosen.sort(key=priority, reverse=True)
    for candidate in chosen:
        if len(refinements) >= config.n3_refine_candidates:
            break
        # First-time candidates also need a chance: after their coarse solve,
        # the local solver checks the actual bound before spending extra time.
        refinements.add(candidate.label)
    selected = {candidate.label for candidate in chosen}
    ctx.record('LR_N3_schedule', generated=len(neighborhoods), selected=len(chosen),
               exploration=exploration.label if exploration else None, refinement_candidates=sorted(refinements),
               candidates=[dict(label=n.label, economic_score=n.score, priority=priority(n)[0],
                                selected=n.label in selected) for n in neighborhoods])
    return [(candidate, candidate.label in refinements) for candidate in chosen]


def n3_neighborhoods(ctx, reference, info, config, iteration):
    """Rotate old corridors and build separate complete base/via/joint closures."""
    started = perf_counter()
    end = ctx.deadline(config.neighborhood_time_limit)
    state = _search_state(ctx, reference)
    state['sweep_complete'] = False
    if state['corridor_reference'] is reference:
        corridors = state['corridors']
        corridor_cache_hit = True
    else:
        corridors = _flow_corridors(ctx, reference, end)
        state['corridor_reference'], state['corridors'] = reference, corridors
        corridor_cache_hit = False
    state['metadata'] = {}
    original_costs = _released_costs(ctx, reference)
    graph_cache_hits = 0
    old_routes = sorted(corridors, key=lambda p: (-corridors[p]["score"], str(p)))
    sources = _rank_nodes(ctx, info, set(ctx.n.sources) & info["facilities"])
    sinks = _rank_nodes(ctx, info, set(ctx.n.sinks) & info["facilities"])
    # An existing corridor can still be redesigned when no alternate endpoint is selected.
    if not sources:
        sources = list(dict.fromkeys(route[0] for route in old_routes))
    if not sinks:
        sinks = list(dict.fromkeys(route[-1] for route in old_routes))
    if not sources or not sinks:
        state['sweep_complete'] = True
        return []
    count = min(config.guidance_top_k + 1, max(1, len(old_routes)))
    offset = state['cursor'] % max(1, len(old_routes))
    anchors = [old_routes[(offset+k) % len(old_routes)] if old_routes else None
               for k in range(count)]
    state['cursor'] += count
    endpoint_offset = state['calls']
    state['calls'] += 1
    selected_arcs = sorted(info["arcs"], key=lambda e: (
        -max((info["metrics"]["transport_build_gap"].get(("y", (*e, d)), 0) +
              info["metrics"]["transport_flow_gap"].get(("y", (*e, d)), 0)
              for d in ctx.d), default=0), str(e)))
    by_node = defaultdict(set)
    for index, route in enumerate(old_routes):
        for node in route:
            by_node[node].add(index)
    result, generated = [], set()

    def add(routes, replacement, variant, old, source, sink, index, companion=None):
        ctx.check_time(end)
        arcs = {edge for route in routes for edge in zip(route, route[1:])}
        facilities = {node for route in routes for node in (route[0], route[-1])}
        if not arcs <= ctx.arcs:
            raise RuntimeError("Path closure contains a non-original directed arc")
        label = f"LR:{variant}:{source}->{sink}:{index}"
        details = dict(label=label, variant=variant,
                       anchor=None if old is None else list(old))
        if len(arcs) > config.n3_max_arcs or len(facilities) > config.n3_max_facilities:
            ctx.record('LR_N3_selection', skipped='complete_path_exceeds_limit',
                       arcs=len(arcs), facilities=len(facilities), **details)
            return
        signature = frozenset(arcs), frozenset(facilities)
        actual_signature = (state['expansions'].get(signature)
                            if getattr(config, 'n3_expand_corridors', 0) else signature)
        if signature in generated or (actual_signature in state['completed'] and actual_signature not in state['refinable']):
            ctx.record('LR_N3_selection', skipped='duplicate_neighborhood', **details)
            return
        generated.add(signature)
        retired = [(old, corridors[old]['profile'])] if old else []
        if companion:
            retired.append((companion, corridors[companion]['profile']))
        released = _retired_cost(ctx, retired, original_costs)
        estimate = _replacement_price(ctx, replacement)
        score = released - estimate if isfinite(estimate) else -max(1.0, abs(ctx.best))
        neighborhood = RerouteNeighborhood(arcs, facilities, score, label)
        state['metadata'][label] = dict(variant=variant, anchor=old, economic_score=score)
        ctx.record('LR_N3_selection', facilities=sorted(facilities, key=str),
                   paths=[list(p) for p in routes], arcs=[list(e) for e in sorted(arcs, key=str)],
                   reason='complete_corridor_replacement', economic_score=score,
                   released_original_cost=released,
                   replacement_cost_estimate=estimate if isfinite(estimate) else None, **details)
        result.append(neighborhood)

    for index, old in enumerate(anchors):
        ctx.check_time(end)
        source = sources[(endpoint_offset+index) % len(sources)]
        sink = sinks[(endpoint_offset+index) % len(sinks)]
        source_cap = sum(ctx.n.source_capacity[source, g] for g in ctx.n.source_units[source])
        endpoint_cap = min(source_cap, ctx.qw[sink] * ctx.max_wells[sink])
        profile = {t: min(endpoint_cap, corridors[old]['profile'].get(t, 0)
                          if old else ctx.c.capture_target[t]) for t in ctx.t}
        if max(profile.values(), default=0) <= 0:
            state['sweep'].add(old)
            continue
        graph, graph_hit = _cached_route_graph(ctx, state, profile, end)
        graph_cache_hits += graph_hit
        def path(start, finish):
            def weight(u, v, data):
                ctx.check_time(end)
                return data['total']
            return nx.shortest_path(graph, start, finish, weight=weight)
        try:
            direct = path(source, sink)
            # One in four anchors preserves endpoint exploration. Other anchors
            # compare a bounded set of cheap sources against all guided sinks.
            if (endpoint_offset + index) % 4:
                source_prices = {node: _facility_price(ctx, node, profile) for node in sources}
                sink_prices = {node: _facility_price(ctx, node, profile) for node in sinks}
                best = (source_prices[source] + sink_prices[sink]
                        + sum(graph[u][v]['total'] for u, v in zip(direct, direct[1:])), source, sink, direct)
                for candidate_source in sorted(sources, key=lambda n: (source_prices[n], str(n)))[:3]:
                    ctx.check_time(end)
                    if not isfinite(source_prices[candidate_source]):
                        continue
                    lengths, paths = nx.single_source_dijkstra(graph, candidate_source, weight='total')
                    ctx.check_time(end)
                    for candidate_sink in sinks:
                        if candidate_sink not in lengths:
                            continue
                        cost = source_prices[candidate_source] + sink_prices[candidate_sink] + lengths[candidate_sink]
                        if cost < best[0]:
                            best = cost, candidate_source, candidate_sink, paths[candidate_sink]
                _, source, sink, direct = best
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            state['sweep'].add(old)
            continue
        base = ([list(old)] if old else []) + [direct]
        add(base, [(direct, profile)], 'base', old, source, sink, index)
        if selected_arcs:
            u, v = selected_arcs[(endpoint_offset+index) % len(selected_arcs)]
            if graph.has_edge(u, v):
                try:
                    via = path(source, u) + path(v, sink)
                    # Via supplies an alternative; do not charge both alternatives.
                    replacement = min((direct, via), key=lambda route: _replacement_price(ctx, [(route, profile)]))
                    add(base + [via], [(replacement, profile)], 'via', old, source, sink, index)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        if old:
            # Join another complete old corridor sharing a facility or transit hub.
            connected = set().union(*(by_node[node] for node in old))
            connected = sorted(i for i in connected if old_routes[i] != old)
            if connected:
                companion = old_routes[connected[endpoint_offset % len(connected)]]
                joint = base + [list(companion)]
                replacements = [(direct, profile), (list(companion), corridors[companion]['profile'])]
                try:
                    alternative = path(source, companion[-1])
                    joint.append(alternative)
                    replacements[-1] = (alternative, corridors[companion]['profile'])
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                add(joint, replacements, 'joint', old, source, sink, index, companion)
        state['sweep'].add(old)
    if not old_routes or set(old_routes) <= state['sweep']:
        state['sweep_complete'] = True
        state['sweep'].clear()
    ctx.record('LR_N3_generation', anchors=[None if p is None else list(p) for p in anchors],
               old_corridors=len(old_routes), candidates=len(result),
               corridor_cache_hit=corridor_cache_hit, route_graph_cache_hits=graph_cache_hits,
               rotation_sweep_complete=state['sweep_complete'], runtime=perf_counter()-started)
    return result


def improve(ctx, pool, info, config, iteration, *, n3_only=False):
    """Search new complete regions, preserving the latest audited incumbent."""
    start, history_start = ctx.best, len(ctx.history)
    started = perf_counter()
    attempted, generated = 0, 0
    state = _search_state(ctx, ctx.incumbent)
    state['sweep_complete'] = False
    ctx.record("LR_guidance", evaluation=info["evaluation"],
               selected=[dict(kind=n, object=list(p), reasons=sorted(info["reasons"][n, p]))
                         for n, p in sorted(info["objects"], key=str)],
               block_gaps=info["block_gaps"], gap_identity_error=info["gap_identity_error"])
    try:
        if not n3_only:
            n1_delays(ctx, info, config.n1_max_moves)
            widen = min(2, info["stagnation"] // max(1, 3 * config.neighborhood_every))
            # Unbuilt selected objects have no variables in N2, so they do not
            # distinguish two time-window neighborhoods of the same incumbent.
            built = {(name, key[:-1]) for name in CONSTRUCTION
                     for key, value in getattr(ctx.incumbent, name).items() if value}
            n2_key = (frozenset(info['objects'] & built),
                      config.n2_tau_minus+widen, config.n2_tau_plus+widen)
            state = _search_state(ctx, ctx.incumbent)
            if n2_key not in state['n2_done']:
                before = len(ctx.history)
                signature = state['signature']
                n2_time_window(ctx, ctx.incumbent, info['objects'], n2_key[1], n2_key[2],
                               config.neighborhood_time_limit)
                records = [row for row in ctx.history[before:] if row['stage'] == 'LR_N2']
                if (records and records[-1]['solver_status'] == 'OPTIMAL'
                        and _construction_signature(ctx.incumbent) == signature):
                    state['n2_done'].add(n2_key)
            else:
                ctx.record('LR_N2_skip', skipped='same_construction_and_time_window')
        neighborhoods = n3_neighborhoods(ctx, ctx.incumbent, info, config, iteration)
        generated = len(neighborhoods)
        scheduled = _schedule_neighborhoods(ctx, state, neighborhoods, config)
        expansion_enabled = getattr(config, 'n3_expand_corridors', 0) > 0
        if expansion_enabled:
            scheduled, domains = expand_scheduled(ctx, scheduled, state['corridors'] or {}, config)
            state['expansions'].update(domains)
        for neighborhood, refine in scheduled:
            ctx.check_time()
            state = _search_state(ctx, ctx.incumbent)
            signature = _neighborhood_signature(neighborhood)
            if signature in state['completed'] and not refine:
                ctx.record('LR_N3_skip', label=neighborhood.label, skipped='duplicate_neighborhood')
                continue
            pool.merge(CandidatePool.from_plan(ctx.incumbent))
            for u, v in neighborhood.arcs:
                for d in ctx.d:
                    pool.pipes[u, v, d] = max(pool.pipes.get((u, v, d), 0), ctx.pipeline_install_limit)
            pool.units.update(ig for ig in ctx.ig if ig[0] in neighborhood.facilities)
            for j in neighborhood.facilities & set(ctx.n.sinks):
                pool.sinks.add(j)
                pool.wells[j] = max(pool.wells.get(j, 0), floor(ctx.max_wells[j]))
            before, old_signature, old_objective = len(ctx.history), state['signature'], ctx.best
            meta = state['metadata'][neighborhood.label]
            solve_started = perf_counter()
            original_options = ctx.options
            try:
                if expansion_enabled:
                    ctx.options = replace(ctx.options, joint_lns_time_limit=config.n3_expanded_time_limit)
                if refine:
                    _joint_neighborhood(ctx, pool, ctx.incumbent, neighborhood, refine=True,
                        absolute_gap=config.n3_refine_absolute_gap, time_limit=config.n3_refine_time_limit)
                else:
                    _joint_neighborhood(ctx, pool, ctx.incumbent, neighborhood)
            finally:
                ctx.options = original_options
            wall = perf_counter() - solve_started
            attempted += 1
            records = [row for row in ctx.history[before:] if row['stage'] == 'P5_joint_reroute']
            status = records[-1]['solver_status'] if records else 'NOT_RECORDED'
            # A tolerance-optimal result is an explored region, not an exact
            # local-optimality certificate. Incomplete solves remain retryable.
            if (status in ('OPTIMAL', 'INFEASIBLE')
                    and _construction_signature(ctx.incumbent) == old_signature):
                state['completed'].add(signature)
                bound = records[-1].get('restricted_bound') if records else None
                if refine and records and records[-1].get('refinement_attempted'):
                    state['refined'].add(signature)
                    state['refinable'].pop(signature, None)
                elif (config.n3_refine_candidates > 0 and status == 'OPTIMAL'
                      and signature not in state['refined'] and bound is not None
                      and old_objective - bound > config.n3_refine_absolute_gap):
                    state['refinable'][signature] = old_objective - bound
                else:
                    state['refinable'].pop(signature, None)
            _search_feedback(state, meta, wall, old_objective - ctx.best)
            ctx.record('LR_N3_attempt', label=neighborhood.label, solver_status=status,
                       improvement=old_objective-ctx.best, wall_runtime=wall,
                       refinement_selected=refine, economic_score=neighborhood.score)
            if status == 'INTERRUPTED':
                raise KeyboardInterrupt
    except _BudgetExpired:
        ctx.record('LR_neighborhood_stop', stopped='time_limit')
    if ctx.best < start and getattr(config, 'pipeline_polish_time_limit', 0) > 0:
        from .pipeline_polish import polish_pipelines
        try:
            polish_pipelines(ctx, pool, config, source='neighborhood')
        except _BudgetExpired:
            ctx.record('LR_neighborhood_stop', stopped='time_limit')
    pool.merge(CandidatePool.from_plan(ctx.incumbent))
    state = _search_state(ctx, ctx.incumbent)
    return dict(improvement=start-ctx.best, events=ctx.history[history_start:],
                n3_attempts=attempted, n3_new_candidates=generated,
                n3_rotation_sweep_complete=state['sweep_complete'], runtime=perf_counter()-started)
