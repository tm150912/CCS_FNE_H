"""One initialization, explicit provenance, no historical-summary imports."""
from copy import deepcopy
from dataclasses import dataclass, field, asdict, replace
from hashlib import sha256
import json
from math import isfinite
from time import perf_counter

import run_PH as heuristic_api
import gurobipy as gp
from heuristic_stages.common import HeuristicOptions, Plan, VARIABLES, COST_FIELDS, _BudgetExpired, _Context, _result
from read import read_well_injection_capacities
from .multiplier_init import solve_multiplier_lp


WarmStartOptions = HeuristicOptions


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def instance_definition(network, costs, diameters, pipeline_install_limit):
    # Stream ordered coefficients instead of duplicating the nationwide cost tables.
    digest = sha256()
    for label, data in (
        ('nodes', list(network.sources) + list(network.sinks) + list(network.intersections)),
        ('sources', list(network.sources)), ('sinks', list(network.sinks)),
        ('arcs', network.ij), ('units', network.ig),
        ('diameters', list(diameters)), ('NP', pipeline_install_limit),
        ('periods', costs.periods)):
        digest.update(_encoded([label, data]).encode('utf-8'))
    mappings = [('source_capacity', network.source_capacity), ('pipe_capacity', network.pipe_capacity),
                ('well_capacity', read_well_injection_capacities(network.well_capacity, network.sinks)),
                ('max_wells', network.max_wells),
                ('durations', costs.durations), ('capture_target', costs.capture_target)]
    mappings.extend((name, getattr(costs, name)) for name in COST_FIELDS.values())
    for label, data in mappings:
        digest.update(label.encode('utf-8'))
        for key in sorted(data, key=_encoded):
            digest.update(_encoded([key, data[key]]).encode('utf-8'))
    return dict(fingerprint=digest.hexdigest(), fingerprint_version=1,
                project_years=sum(costs.durations.values()), periods=list(costs.periods),
                durations=dict(costs.durations), capture_target=dict(costs.capture_target),
                diameters=list(diameters), pipeline_install_limit=pipeline_install_limit,
                nodes=list(network.sources) + list(network.sinks) + list(network.intersections),
                source_count=len(network.sources), sink_count=len(network.sinks),
                arc_count=len(network.ij), unit_count=len(network.ig))


@dataclass
class WarmStartRecord:
    plan: object
    lower_bound: object
    lower_bound_origin: str
    instance: dict
    options: object
    runtime: float
    timings: dict
    proven_infeasible: bool
    status: str
    initial_multipliers: object
    multiplier_initialization: dict
    initialization_history: list = field(default_factory=list)
    model_parameters: dict = field(default_factory=dict)


def solve_primal_warm_start(network, costs, diameters, pipeline_install_limit, options):
    """Reuse run_PH's primal search, without its mandatory bound-only LP."""
    ctx = _Context(network, costs, diameters, replace(options, lp_time_limit=0),
                   pipeline_install_limit=pipeline_install_limit)
    ctx.configuration.update(initialization='P1_without_global_LP', time_limit_scope='primal_warm_start')
    status = 'FEASIBLE_HEURISTIC'
    try:
        with gp.Env(empty=True) as environment:
            environment.setParam('OutputFlag', int(ctx.options.gurobi_output_flag))
            environment.start()
            ctx.env = environment
            reserve = min(ctx.options.final_polish_reserve, 0.2 * ctx.check_time())
            ctx.phase_end = ctx.end - reserve
            ctx.record('search_budget', reserve=reserve, search_remaining=ctx.check_time())
            try:
                status = heuristic_api._search_heuristic(ctx)
            except _BudgetExpired:
                if perf_counter() >= ctx.end:
                    raise
                status = 'FEASIBLE_SEARCH_LIMIT' if ctx.incumbent is not None else 'SEARCH_LIMIT_NO_SOLUTION'
                ctx.record('search_stop', reason='final_polish_reserve')
            finally:
                ctx.phase_end = ctx.end
            heuristic_api.final_polish(ctx)
            if (status == 'FEASIBLE_SEARCH_LIMIT' and ctx.incumbent is not None
                    and getattr(ctx, 'search_round', 0) < ctx.options.max_rounds):
                ctx.record('search_resume', reason='unused_final_polish_budget',
                           next_round=getattr(ctx, 'search_round', 0) + 1,
                           remaining=ctx.check_time())
                status = heuristic_api._search_rounds(ctx, ctx.search_pool)
    except _BudgetExpired:
        status = 'FEASIBLE_TIME_LIMIT' if ctx.incumbent is not None else 'TIME_LIMIT_NO_SOLUTION'
        ctx.record('stop', reason=status)
    except gp.GurobiError as exc:
        status = 'FEASIBLE_SOLVER_ERROR' if ctx.incumbent is not None else 'SOLVER_ERROR_NO_SOLUTION'
        ctx.record('stop', reason=status, solver_error=str(exc))
    finally:
        ctx.env = None
    return _result(ctx, status)


def prepare_warm_start(network, costs, diameters, pipeline_install_limit,
                       heuristic_options=None, initial_result=None):
    started = perf_counter()
    instance = instance_definition(network, costs, diameters, pipeline_install_limit)
    model_parameters = deepcopy(getattr(costs, 'model_parameters', {}))
    options = replace(heuristic_options or heuristic_api.build_options(), lp_method=2, lp_crossover=0)
    internal = initial_result is None
    h_result = initial_result
    if internal:
        h_result = solve_primal_warm_start(network, costs, diameters,
                    options=options, pipeline_install_limit=pipeline_install_limit)
    plan = None
    if h_result.objective is not None:
        plan = Plan(**{name: dict(getattr(h_result, name)) for name in VARIABLES})
    nonnegative = all(isfinite(v) and v >= 0 for name in COST_FIELDS.values()
                      for v in getattr(costs, name).values())
    lower, origin = (0.0, 'nonnegative_costs') if nonnegative else (None, 'unavailable')
    multipliers, multiplier_info = None, dict(status='NOT_RUN', reason='no_initial_feasible_solution', runtime=0.0)
    if plan is not None:
        multipliers, multiplier_info = solve_multiplier_lp(network, costs, diameters,
                                                           pipeline_install_limit, options)
        lp_lower = multiplier_info.get('lower_bound')
        if lp_lower is not None and (lower is None or lp_lower > lower):
            lower, origin = lp_lower, 'full_original_LP_multiplier_initialization'
    elapsed = perf_counter() - started
    warm_options = asdict(replace(options, lp_time_limit=0))
    timings = dict(warm_start_wall=elapsed,
                   warm_start_lp=h_result.lp_runtime if internal else 0.0,
                   warm_start_lp_status=h_result.lp_status if internal else 'NOT_RUN_EXTERNAL_INITIAL_RESULT',
                   warm_start_search=max(0.0, h_result.runtime-h_result.lp_runtime) if internal else 0.0,
                   external_historical_runtime=0.0 if internal else h_result.runtime,
                   external_lower_bound_ignored=not internal,
                   multiplier_lp_wall=multiplier_info['runtime'],
                   multiplier_lp_solve=multiplier_info.get('solve_runtime', 0.0),
                   warm_start_options=warm_options)
    history = [dict(row, initialization_source='primal_warm_start' if internal
                    else 'external_initial_result') for row in h_result.history]
    history.append(dict(stage='primal_warm_start_result',
        initialization_source='primal_warm_start' if internal else 'external_initial_result',
        status=h_result.status, objective=h_result.objective, runtime=h_result.runtime))
    history.append(dict(multiplier_info, stage='multiplier_initialization'))
    return WarmStartRecord(plan, lower, origin, instance, options, elapsed, timings,
                           False, h_result.status,
                           multipliers, multiplier_info, history, model_parameters)


def primal_options(record, config):
    return replace(record.options, time_limit=config.lr_time_limit, lp_time_limit=0,
                   final_polish_reserve=0, flow_time_limit=config.flow_time_limit,
                   joint_lns_enabled=True, joint_lns_time_limit=config.neighborhood_time_limit,
                   feasibility_tolerance=config.feasibility_tolerance,
                   improvement_tolerance=config.eps_improve,
                   mip_gap=config.neighborhood_mip_gap, pipe_polish_enabled=False,
                   threads=config.threads, seed=config.seed, output_flag=config.output_flag,
                   gurobi_output_flag=config.gurobi_output_flag)
