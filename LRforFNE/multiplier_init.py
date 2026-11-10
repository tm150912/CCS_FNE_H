"""Initialize nodal LR prices from the full original LP, using barrier only."""
from itertools import product
from math import isfinite
from time import perf_counter
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
from solve_logging import optimize_with_log

from heuristic_stages.common import _status_name
from heuristic_stages.lp_bound import _build_original_relaxation
from read import read_pipeline_limit, read_well_capacities, read_well_injection_capacities


def _balance_rows(model, nodes, periods):
    """Retain actual constraint objects in the parent builder's node/time order.

    The original LP has no other equalities. Check both count and names rather
    than looking up names, which can collide for heterogeneous original IDs.
    """
    model.update()
    indices = list(product(nodes, periods))
    rows = [row for row in model.getConstrs() if row.Sense == '=']
    if len(rows) != len(indices):
        raise ValueError('Original LP balance row count does not match nodes x periods')
    balances = {}
    for (node, period), row in zip(indices, rows):
        if row.RHS != 0 or row.ConstrName != f'balance[{node},{period}]':
            raise ValueError('Original LP balance row order/orientation requires review')
        balances[node, period] = row
    return balances


def _read_optimal_multipliers(model, balances, nodes, periods):
    if model.Status != gp.GRB.OPTIMAL:
        return None
    try:
        pi = np.array([[balances[node, h].Pi for h in periods] for node in nodes], dtype=float)
    except (gp.GurobiError, AttributeError):
        return None
    if not np.all(np.isfinite(pi)) or np.any(np.abs(pi) >= gp.GRB.INFINITY / 2):
        return None
    # Gurobi's min-LP dual uses f - Pi*(Ax-b); our Lagrangian uses f + lambda*g.
    return -pi


def solve_multiplier_lp(network, costs, diameters, pipeline_install_limit, options):
    """Return (initial multipliers or None, diagnostics); never substitute zeros.

    The separate LP uses the existing LP budget, including its model build time.
    It is performed once per initialization, outside each method's LR deadline.
    """
    started = perf_counter()
    limit = options.lp_time_limit
    if limit is not None and (not isfinite(limit) or limit <= 0):
        raise ValueError('Multiplier LP time limit must be None or finite and positive')
    end = float('inf') if limit is None else started + limit
    nodes = list(network.sources) + list(network.sinks) + list(network.intersections)
    if len(nodes) != len(set(nodes)):
        raise ValueError('Node classes must be disjoint')
    periods = list(costs.periods)
    pipe_limit = read_pipeline_limit(pipeline_install_limit)
    metadata = dict(source='full_original_LP_barrier', status='NOT_RUN', solver_status=None,
        method=2, crossover=0, dual_attribute='Pi', sign=-1,
        residual='outflow-inflow-capture+injection', time_limit=limit,
        objective=None, lower_bound=None, runtime=0.0, solve_runtime=0.0,
        barrier_iterations=None, simplex_iterations=None)
    multipliers = None
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', int(options.gurobi_output_flag))
            env.start()
            # Reuse the authoritative continuous builder without another full
            # _Context.keys cache or a temporary integer model plus .relax copy.
            view = SimpleNamespace(n=network, c=costs, d=list(diameters), t=periods, nodes=nodes,
                qw=read_well_injection_capacities(network.well_capacity, network.sinks),
                max_wells=read_well_capacities(network, costs),
                pipe_totals={edge:pipe_limit for edge in network.ij},
                model=lambda name: gp.Model(name, env=env))
            with _build_original_relaxation(view) as model:
                balances = _balance_rows(model, nodes, periods)
                model.Params.Method = 2
                model.Params.Crossover = 0
                model.Params.Threads = options.threads
                model.Params.Seed = options.seed
                model.Params.FeasibilityTol = max(1e-9, min(1e-8, options.feasibility_tolerance/10))
                model.Params.OptimalityTol = 1e-9
                model.Params.BarConvTol = 1e-10
                metadata['build_runtime'] = perf_counter()-started
                remaining = end-perf_counter()
                if remaining <= 0:
                    metadata.update(status='TIME_LIMIT', solver_status='NOT_RUN', reason='LP_build_time_limit')
                else:
                    if limit is not None:
                        model.Params.TimeLimit = remaining
                    optimize_with_log(model)
                    metadata.update(status=_status_name(model.Status), solver_status=_status_name(model.Status),
                        solve_runtime=float(model.Runtime), barrier_iterations=int(model.BarIterCount),
                        simplex_iterations=float(model.IterCount))
                    if model.SolCount:
                        value = float(model.ObjVal)
                        if isfinite(value) and abs(value) < gp.GRB.INFINITY/2:
                            metadata['objective'] = value
                    if model.Status == gp.GRB.OPTIMAL and metadata['objective'] is not None:
                        metadata['lower_bound'] = metadata['objective']
                        multipliers = _read_optimal_multipliers(model, balances, nodes, periods)
                        if multipliers is None:
                            metadata['status'] = 'DUALS_UNAVAILABLE'
                        else:
                            metadata['multiplier_norm'] = float(np.linalg.norm(multipliers))
                    elif model.Status == gp.GRB.OPTIMAL:
                        metadata['status'] = 'SOLUTION_UNAVAILABLE'
    except gp.GurobiError as exc:
        metadata.update(status='SOLVER_ERROR', error=str(exc))
    except KeyboardInterrupt:
        metadata.update(status='INTERRUPTED', reason='user_interrupt')
    metadata['runtime'] = perf_counter()-started
    return multipliers, metadata
