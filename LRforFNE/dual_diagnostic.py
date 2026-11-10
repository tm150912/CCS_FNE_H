"""Numerical upper certificates for the optimal Lagrangian dual value.

The unregularized witness-cut master bounds max_lambda q(lambda) from above.
Its maximizing feasible objective is not an upper certificate on early exit.
This diagnostic never updates the original problem's certified lower bound.
"""
from math import fsum, isfinite
from time import perf_counter

import gurobipy as gp
import numpy as np
from solve_logging import optimize_with_log

from heuristic_stages.common import _status_name


def _certificate(cuts, weights, objective, primal_upper, quality):
    """Check the optimal LP dual certificate under explicit numerical tolerances."""
    weights = np.asarray(weights, dtype=float)
    if (len(weights) != len(cuts) + 1 or not np.all(np.isfinite(weights))
            or not isfinite(objective)):
        return None, dict(valid=False, reason='nonfinite_or_incomplete_LP_dual')
    stationarity = np.zeros_like(cuts[0][1], dtype=float) if cuts else np.zeros(1)
    for weight, (_, gradient, _, _) in zip(weights, cuts):
        stationarity += weight * gradient
    violation = np.abs(stationarity).ravel()
    dual_objective = fsum([float(w) * float(cut[0]) for w, cut in zip(weights, cuts)]
                          + [float(weights[-1]) * primal_upper])
    scale = max(1.0, abs(objective), abs(dual_objective))
    objective_tolerance = max(1e-6, 1e-8 * scale)
    checks = dict(weight_sum_error=abs(float(weights.sum()) - 1.0),
                  minimum_weight=float(weights.min()),
                  stationarity_max=float(np.max(violation, initial=0)),
                  primal_objective=objective, dual_objective=dual_objective,
                  objective_difference=abs(dual_objective - objective),
                  objective_tolerance=objective_tolerance,
                  feasibility_tolerance=1e-6, weight_tolerance=1e-9,
                  constant_cap_weight=float(weights[-1]),
                  active_witness_cuts=int(np.count_nonzero(weights[:-1] > 1e-9)),
                  **quality)
    checks['valid'] = bool(
        checks['minimum_weight'] >= -1e-9 and checks['weight_sum_error'] <= 1e-9
        and checks['stationarity_max'] <= 1e-6
        and all(isfinite(value) and value <= 1e-6 for value in quality.values())
        and checks['objective_difference'] <= objective_tolerance
        and objective <= primal_upper + objective_tolerance)
    if not checks['valid']:
        checks['reason'] = 'unreliable_optimal_LP_certificate'
        return None, checks
    # This padding addresses objective rounding, not arbitrary residuals on free
    # multipliers. The solver/tolerance certificate is numerical, not rational.
    checks['objective_padding'] = objective_tolerance
    upper = min(primal_upper, max(objective, dual_objective) + objective_tolerance)
    return upper, checks


def diagnose_dual(master, primal_upper, deadline, time_limit=5.0):
    """Solve a copied full cut LP within the remaining LR and local budget.

    ``upper_bound`` is present only for an optimal, checked LP dual certificate.
    ``primal_cap`` remains a separate, possibly uninformative upper bound on the
    dual optimum. Timeout, interruption and numerical failure never use the
    maximization LP's feasible ObjVal as a new upper bound. Solver errors and
    KeyboardInterrupt propagate to the caller; copied models are always closed.
    """
    if not isfinite(primal_upper) or abs(primal_upper) >= gp.GRB.INFINITY / 2:
        raise ValueError('Dual diagnostic needs a finite audited primal upper bound')
    if not isfinite(time_limit) or time_limit <= 0:
        raise ValueError('Dual diagnostic time_limit must be finite and positive')
    started = perf_counter()
    end = min(deadline, started + time_limit)
    result = dict(status='DEADLINE', solver_status=None, upper_bound=None,
                  primal_cap=float(primal_upper), cut_count=len(master.cuts),
                  source='unregularized_cut_master',
                  certificate_kind='solver_certified_numerical_LP_dual',
                  informative=False, build_runtime=0.0, solve_runtime=0.0,
                  solve_wall=0.0, certificate=None)
    if end <= started:
        result['runtime'] = perf_counter() - started
        return result
    # Synchronize pending witness cuts before copying. No objective, bound or
    # constraint of the active proximal master is changed by this diagnostic.
    master.model.update()
    value_index = master.v.index
    cut_indices = [row[3].index for row in master.cuts]
    incumbent_index = None if master.upper_cut is None else master.upper_cut.index
    with master.model.copy() as model:
        variables, constraints = model.getVars(), model.getConstrs()
        witness_constraints = [constraints[i] for i in cut_indices]
        if incumbent_index is not None:
            model.remove(constraints[incumbent_index])
        value = variables[value_index]
        cap = model.addConstr(value <= primal_upper, name='audited_primal_cap')
        model.setObjective(value, gp.GRB.MAXIMIZE)
        model.Params.Method = 1
        model.Params.OutputFlag = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.update()
        result['build_runtime'] = perf_counter() - started
        remaining = end - perf_counter()
        if remaining > 0:
            model.Params.TimeLimit = remaining
            solve_started = perf_counter()
            optimize_with_log(model)
            result.update(status=_status_name(model.Status), solver_status=model.Status,
                          solve_runtime=float(model.Runtime),
                          solve_wall=perf_counter() - solve_started)
            if model.Status == gp.GRB.OPTIMAL and model.SolCount:
                weights = model.getAttr('Pi', witness_constraints + [cap])
                quality = dict(primal_constraint_violation=float(model.ConstrVio),
                               primal_bound_violation=float(model.BoundVio),
                               dual_violation=float(model.DualVio))
                upper, certificate = _certificate(master.cuts, weights,
                    float(model.ObjVal), float(primal_upper), quality)
                result.update(upper_bound=upper, certificate=certificate)
                if upper is None:
                    result['status'] = 'NUMERICAL_ERROR'
                else:
                    result['informative'] = bool(
                        primal_upper - upper > certificate['objective_tolerance'])
    result['runtime'] = perf_counter() - started
    return result
