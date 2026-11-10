"""Inspect PH reference starts and let Gurobi recompute their continuous flows."""
from __future__ import annotations

from time import perf_counter

import gurobipy as gp
import numpy as np

from solve_logging import optimize_with_log


def _reference_diagnostics(model):
    """Evaluate the supplied linear-model start without copying/solving the model."""
    variables = model.getVars()
    values = np.asarray(model.getAttr('Start', variables), dtype=float)
    complete = bool(np.all(np.isfinite(values) & (np.abs(values) < gp.GRB.INFINITY)))
    result = dict(reference_objective=None, reference_feasible=False,
                  reference_complete=complete, max_constraint_violation=None,
                  max_bound_violation=None, max_integrality_violation=None,
                  feasibility_tolerance=model.Params.FeasibilityTol,
                  integrality_tolerance=model.Params.IntFeasTol, worst_constraint=None)
    if not complete:
        return result, variables
    costs = np.asarray(model.getAttr('Obj', variables), dtype=float)
    lower = np.asarray(model.getAttr('LB', variables), dtype=float)
    upper = np.asarray(model.getAttr('UB', variables), dtype=float)
    kinds = np.asarray(model.getAttr('VType', variables))
    integer = kinds != gp.GRB.CONTINUOUS
    bound_violation = float(max(0.0, np.max(lower - values, initial=0.0),
                                np.max(values - upper, initial=0.0)))
    integer_violation = float(np.max(np.abs(values[integer] - np.rint(values[integer])), initial=0.0))
    constraints = model.getConstrs()
    rhs = np.asarray(model.getAttr('RHS', constraints), dtype=float)
    senses = np.asarray(model.getAttr('Sense', constraints))
    residual = np.asarray(model.getA() @ values).reshape(-1) - rhs
    violation = np.where(senses == '=', np.abs(residual),
                         np.where(senses == '<', residual, -residual))
    constraint_violation = float(np.max(violation, initial=0.0))
    result.update(reference_objective=float(costs @ values + model.ObjCon),
        reference_feasible=(max(bound_violation, constraint_violation) <= model.Params.FeasibilityTol
                            and integer_violation <= model.Params.IntFeasTol),
        max_constraint_violation=constraint_violation, max_bound_violation=bound_violation,
        max_integrality_violation=integer_violation,
        worst_constraint=constraints[int(np.argmax(violation))].ConstrName if constraint_violation > 0 else None)
    return result, variables


def optimize_with_reference(ctx, model, end, *, stage, callback=None):
    """Optimize an already populated PH model; keep its audited reference external.

    The diagnostic objective belongs only to this restricted model. Clearing
    continuous starts repairs their numerical residuals through Gurobi's normal
    start completion; all integer starts, bounds and constraints stay unchanged.
    """
    model.update()
    if not model.IsMIP:
        if callback is None:
            ctx.optimize(model, end)
        else:
            model.Params.TimeLimit = ctx.check_time(end)
            optimize_with_log(model, callback)
        return dict(reference_objective=None, reference_feasible=False, strategy='not_mip')

    started = perf_counter()
    ctx.check_time(end)
    diagnostics, variables = _reference_diagnostics(model)
    continuous = [var for var in variables if var.VType == gp.GRB.CONTINUOUS]
    model.setAttr('Start', continuous, [gp.GRB.UNDEFINED] * len(continuous))
    diagnostics.update(strategy='integer_design', continuous_starts_cleared=len(continuous),
                       diagnostic_runtime=perf_counter() - started)
    ctx.record(stage + '_warm_start', **diagnostics)
    messages = []

    def record(current, where):
        if where == gp.GRB.Callback.MESSAGE:
            message = current.cbGet(gp.GRB.Callback.MSG_STRING).strip()
            if 'mip start' in message.lower():
                messages.append(message)
        if callback is not None:
            callback(current, where)

    output_flag, console = model.Params.OutputFlag, model.Params.LogToConsole
    model.Params.LogToConsole = console if output_flag else 0
    model.Params.OutputFlag = 1  # MESSAGE callbacks also run in quiet experiments.
    try:
        model.Params.TimeLimit = ctx.check_time(end)
        optimize_with_log(model, record)
    finally:
        model.Params.OutputFlag = output_flag
        model.Params.LogToConsole = console
        loaded = any('loaded user mip start' in message.lower() for message in messages)
        rejected = any('violates' in message.lower() for message in messages)
        diagnostics.update(warm_start_status='loaded' if loaded else 'violated' if rejected else 'not_loaded',
                           start_messages=messages)
        ctx.record(stage + '_warm_start_result', **diagnostics)
    return diagnostics
