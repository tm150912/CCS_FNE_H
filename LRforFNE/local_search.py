"""Sparse, audited N3 MILPs with optional refinement of the same restricted model.

All bounds here concern one fixed-outside neighborhood, never the global LR bound.
"""
from collections import defaultdict
from math import floor, isfinite
from time import perf_counter

import gurobipy as gp

from heuristic_stages.common import (
    COST_FIELDS, CandidatePool, Plan, _BudgetExpired, _facility_constraints,
    _flow_constraints, _objective, _status_name)


CONSTRUCTION = ('s', 'r', 'w', 'y')


def _build(ctx, model, reference, neighborhood, end):
    """Keep all active flow nodes, including unopened facilities used in transit."""
    facilities = neighborhood.facilities
    units_by_source = getattr(ctx, '_lr_local_units_by_source', None)
    if units_by_source is None:
        units_by_source = defaultdict(set)
        for ig in ctx.ig:
            units_by_source[ig[0]].add(ig)
        ctx._lr_local_units_by_source = units_by_source
    built_types = {key[:3] for key, value in reference.y.items() if value}
    arcs = sorted({key[:2] for key in built_types} | neighborhood.arcs)
    local_units = {key[:2] for key, value in reference.s.items() if value}
    for node in facilities:
        local_units.update(units_by_source.get(node, ()))
    local_sinks = ({key[0] for key, value in reference.r.items() if value}
                   | (facilities & set(ctx.n.sinks)))
    pipe_types = {(u, v): [d for d in ctx.d
                  if (u, v) in neighborhood.arcs or (u, v, d) in built_types]
                  for u, v in arcs}
    units = [(*ig, t) for ig in sorted(local_units) for t in ctx.t]
    sinks = [(j, t) for j in sorted(local_sinks) for t in ctx.t]
    pipes = [(u, v, d, t) for u, v in arcs for d in pipe_types[u, v] for t in ctx.t]
    keys = dict(s=units, r=sinks, w=sinks, y=pipes, a=units, b=sinks, x=pipes)
    variables = {}
    for name, indices in keys.items():
        ctx.check_time(end)
        kind = gp.GRB.BINARY if name in ('s', 'r') else (
            gp.GRB.INTEGER if name in ('w', 'y') else gp.GRB.CONTINUOUS)
        variables[name] = model.addVars(indices, lb=0,
            ub=ctx.pipeline_install_limit if name == 'y' else (
                1.0 if name in ('s', 'r') else gp.GRB.INFINITY),
            obj=getattr(ctx.c, COST_FIELDS[name]), vtype=kind, name=name)
    s, r, w, y, a, b, x = (variables[name] for name in keys)
    local_pool = CandidatePool(local_units, local_sinks,
                              {j: floor(ctx.max_wells[j]) for j in local_sinks}, {})
    _facility_constraints(ctx, model, local_pool, s, r, w, a, b)
    for number, (u, v) in enumerate(arcs):
        if number % 64 == 0:
            ctx.check_time(end)
        for d in pipe_types[u, v]:
            active = gp.LinExpr()
            for t in ctx.t:
                active += y[u, v, d, t]
                model.addConstr(x[u, v, d, t] <= ctx.n.pipe_capacity[d] * active)
        model.addConstr(gp.quicksum(y[u, v, d, t] for d in pipe_types[u, v]
                                    for t in ctx.t) <= ctx.pipeline_install_limit)
    active_nodes = ({i for i, _ in local_units} | local_sinks
                    | {node for edge in arcs for node in edge})
    # Omitted nodes have no capture, injection or flow variables: their rows are 0=0.
    _flow_constraints(model, sorted(active_nodes), ctx.t, ctx.c.capture_target, a, b, x)
    rounding_bound = 0.0
    for name, mapping in variables.items():
        ctx.check_time(end)
        original = getattr(reference, name)
        model.setAttr('Start', list(mapping.values()), [original.get(key, 0) for key in mapping])
        if name in CONSTRUCTION:
            fixed = [(key, var) for key, var in mapping.items()
                     if not (key[:2] in neighborhood.arcs if name == 'y'
                             else key[0] in facilities)]
            fixed_vars = [var for _, var in fixed]
            fixed_values = [original.get(key, 0) for key, _ in fixed]
            model.setAttr('LB', fixed_vars, fixed_values)
            model.setAttr('UB', fixed_vars, fixed_values)
        # Conservative guard for integer rounding and removal of tiny positive flows.
        tolerance = model.Params.IntFeasTol if name in CONSTRUCTION else 1e-10
        rounding_bound += tolerance * sum(getattr(ctx.c, COST_FIELDS[name])[key] for key in mapping)
    model.ModelSense = gp.GRB.MINIMIZE
    return variables, len(active_nodes) * len(ctx.t), rounding_bound


def _metrics(model, phase):
    count = int(model.SolCount)
    bound = float(model.ObjBound)
    gap = float(model.MIPGap) if count else None
    return dict(phase=phase, solver_status=_status_name(model.Status),
        solver_runtime=float(model.Runtime), node_count=float(model.NodeCount),
        objective=float(model.ObjVal) if count else None,
        restricted_bound=bound if isfinite(bound) and abs(bound) < gp.GRB.INFINITY else None,
        restricted_gap=gap if gap is not None and isfinite(gap) else None,
        solution_count=count)


def _accept_candidate(ctx, pool, model, variables, rounding_bound, timings):
    """Filter clearly unchanged objectives, then audit a rounded candidate exactly once."""
    if not model.SolCount:
        return False
    threshold = ctx.options.improvement_tolerance * max(1.0, abs(ctx.best))
    if float(model.ObjVal) - rounding_bound >= ctx.best - threshold:
        timings['objective_filter_skips'] += 1
        return False
    started = perf_counter()
    values = {}
    for name, mapping in variables.items():
        raw = model.getAttr('X', list(mapping.values()))
        converted = (int(round(value)) if name in CONSTRUCTION else max(0.0, value)
                     for value in raw)
        values[name] = {key: value for key, value in zip(mapping, converted) if value > 1e-10}
    candidate = Plan(**values)
    timings['extract_wall'] += perf_counter() - started
    # The actual rounded objective provides a second cheap filter before the full audit.
    if _objective(ctx.c, candidate) >= ctx.best - threshold:
        timings['objective_filter_skips'] += 1
        return False
    started = perf_counter()
    audit = ctx.audit(candidate)
    timings['audit_wall'] += perf_counter() - started
    timings['audit_count'] += 1
    if not audit['feasible']:
        ctx.record('P5_joint_reroute_audit', rejected='original_model_audit',
                   max_violation=audit['max_violation'])
        return False
    # Exactly the acceptance semantics of _Context.consider, reusing the audit above.
    if audit['objective'] < ctx.best - threshold:
        ctx.incumbent, ctx.best = candidate.copy(), audit['objective']
        ctx.record('P5_joint_reroute_accept', accepted=True, objective=ctx.best)
        pool.merge(CandidatePool.from_plan(candidate))
        return True
    return False


def solve_neighborhood(ctx, pool, reference, neighborhood, *, refine=False,
                       absolute_gap=100.0, time_limit=None):
    """Solve one restricted N3 model, optionally spending a bounded extra effort on it.

    ``time_limit`` caps only refinement; coarse and fine passes share the original
    neighborhood/global deadline. A coarse audited improvement survives fine-pass failure.
    """
    if not ctx.options.joint_lns_enabled or ctx.options.joint_lns_time_limit <= 0:
        return reference
    if not neighborhood.arcs <= ctx.arcs:
        raise ValueError('Joint neighborhood contains a non-original arc')
    if not neighborhood.facilities <= set(ctx.n.sources) | set(ctx.n.sinks):
        raise ValueError('Joint neighborhood facilities must be original source/sink nodes')
    if not isfinite(absolute_gap) or absolute_gap < 0:
        raise ValueError('absolute_gap must be finite and nonnegative')
    if time_limit is not None and (not isfinite(time_limit) or time_limit <= 0):
        raise ValueError('time_limit must be finite and positive')
    started = perf_counter()
    start_objective = _objective(ctx.c, reference)
    end = ctx.deadline(ctx.options.joint_lns_time_limit)
    timings = dict(build_wall=0.0, solve_wall=0.0, extract_wall=0.0, audit_wall=0.0,
                   audit_count=0, objective_filter_skips=0)
    passes = []
    solve_calls = 0
    refinement_attempted = False
    refinement_reason = 'not_requested'
    metadata = dict(variable_count=0, construction_variable_count=0, balance_rows=0)
    status = 'TIME_LIMIT'
    expired = False
    try:
        with ctx.model('LR_joint_reroute_neighborhood') as model:
            variables, balances, rounding_bound = _build(ctx, model, reference, neighborhood, end)
            timings['build_wall'] = perf_counter() - started
            before = perf_counter()
            solve_calls += 1
            try:
                ctx.optimize(model, end)
            finally:
                timings['solve_wall'] += perf_counter() - before
            coarse = _metrics(model, 'coarse')
            passes.append(coarse)
            status = coarse['solver_status']
            metadata.update(variable_count=model.NumVars,
                            construction_variable_count=model.NumIntVars, balance_rows=balances)
            ctx.record('LR_N3_solve_pass', label=neighborhood.label, **coarse)
            _accept_candidate(ctx, pool, model, variables, rounding_bound, timings)
            if refine:
                threshold = ctx.options.improvement_tolerance * max(1.0, abs(ctx.best))
                room = (None if coarse['restricted_bound'] is None else
                        ctx.best - coarse['restricted_bound'])
                refinement_reason = 'insufficient_improvement_room'
                if status not in ('OPTIMAL', 'TIME_LIMIT', 'SUBOPTIMAL') or not coarse['solution_count']:
                    refinement_reason = 'incomplete_or_failed_coarse_solve'
                elif room is not None and room > max(absolute_gap, threshold):
                    refinement_reason = 'remaining_local_improvement_room'
                    fine_end = min(end, ctx.deadline(1.0 if time_limit is None else time_limit))
                    ctx.check_time(fine_end)
                    model.Params.MIPGap = 0.0
                    model.Params.MIPGapAbs = absolute_gap
                    before = perf_counter()
                    refinement_attempted = True
                    solve_calls += 1
                    try:
                        ctx.optimize(model, fine_end)
                    finally:
                        timings['solve_wall'] += perf_counter() - before
                    fine = _metrics(model, 'refine')
                    passes.append(fine)
                    status = fine['solver_status']
                    ctx.record('LR_N3_solve_pass', label=neighborhood.label, **fine)
                    _accept_candidate(ctx, pool, model, variables, rounding_bound, timings)
    except _BudgetExpired:
        expired, status = True, 'TIME_LIMIT'
        if not solve_calls:
            timings['build_wall'] = perf_counter() - started
        elif refine:
            refinement_reason = 'shared_deadline'
    except KeyboardInterrupt:
        status = 'INTERRUPTED'
        raise
    except gp.GurobiError:
        status = 'SOLVER_ERROR'
        raise
    finally:
        wall = perf_counter() - started
        timings['bookkeeping_wall'] = max(0.0, wall - sum(timings[key]
            for key in ('build_wall', 'solve_wall', 'extract_wall', 'audit_wall')))
        witnesses = [p['objective'] for p in passes if p['objective'] is not None]
        bounds = [p['restricted_bound'] for p in passes if p['restricted_bound'] is not None]
        objective = min(witnesses) if witnesses else None
        bound = max(bounds) if bounds else None
        gap = (None if objective is None or bound is None or objective == 0
               else max(0.0, objective - bound) / abs(objective))
        if objective == bound == 0:
            gap = 0.0
        ctx.record('P5_joint_reroute', solver_status=status,
            start_objective=start_objective,
            solver_runtime=sum(p['solver_runtime'] for p in passes),
            node_count=sum(p['node_count'] for p in passes), objective=objective,
            restricted_bound=bound, restricted_gap=gap,
            solution_count=passes[-1]['solution_count'] if passes else 0,
            label=neighborhood.label, free_arcs=len(neighborhood.arcs),
            free_facilities=len(neighborhood.facilities), solve_passes=solve_calls,
            pass_metrics=passes, refinement_requested=refine,
            refinement_attempted=refinement_attempted,
            refinement_status=status if refinement_attempted else None,
            refinement_reason=refinement_reason, local_budget_expired=expired,
            wall_runtime=wall, **metadata, **timings)
    return ctx.incumbent if ctx.best < start_objective else reference
