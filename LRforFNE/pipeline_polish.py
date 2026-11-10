"""Reuse PH's audited fixed-flow pipe polish within the existing LR deadline."""
from dataclasses import replace
from time import perf_counter

from gurobipy import GRB

from heuristic_stages.common import CandidatePool, _BudgetExpired
from heuristic_stages.pipeline_polish import _polish_and_merge


_MISSING = object()


def polish_pipelines(ctx, pool, config, *, source):
    """Keep audited improvements, restore caller options, and propagate interrupts."""
    started, initial, history_start = perf_counter(), ctx.best, len(ctx.history)
    limit = config.pipeline_polish_time_limit
    if limit == 0:
        ctx.record('LR_pipeline_polish', source=source, status='DISABLED',
                   start_objective=initial, objective=ctx.best, improvement=0.0,
                   wall_runtime=perf_counter()-started, time_limit=limit)
        return ctx.incumbent
    options = ctx.options
    original_optimize = ctx.optimize
    optimize_attribute = ctx.__dict__.get('optimize', _MISSING)
    def optimize(model, end):
        result = original_optimize(model, end)
        if model.Status == GRB.INTERRUPTED:
            raise KeyboardInterrupt
        return result
    status = 'ERROR'
    try:
        ctx.check_time()
        ctx.options = replace(options, pipe_polish_enabled=True,
            pipe_polish_time_limit=limit, pipe_polish_max_arcs=len(ctx.arcs))
        ctx.optimize = optimize
        _polish_and_merge(ctx, pool, ctx.incumbent, time_limit=limit, full_sweep=True)
        status = 'COMPLETED'
        rows = ctx.history[history_start:]
        if any(row.get('solver_status') == 'INTERRUPTED'
               or row.get('solver_statuses', {}).get('INTERRUPTED', 0) for row in rows):
            raise KeyboardInterrupt
        if any(row.get('stopped') == 'time_limit' for row in rows):
            status = 'TIME_LIMIT'
        elif any('solver_error' in row or row.get('solver_statuses', {}).get('SOLVER_ERROR', 0)
                 for row in rows):
            status = 'SOLVER_ERROR'
    except _BudgetExpired:
        status = 'TIME_LIMIT'
    except KeyboardInterrupt:
        status = 'INTERRUPTED'
        raise
    finally:
        ctx.options = options
        if optimize_attribute is _MISSING:
            if 'optimize' in ctx.__dict__:
                del ctx.optimize
        else:
            ctx.optimize = optimize_attribute
        pool.merge(CandidatePool.from_plan(ctx.incumbent))
        ctx.record('LR_pipeline_polish', source=source, status=status,
                   start_objective=initial, objective=ctx.best,
                   improvement=initial-ctx.best, wall_runtime=perf_counter()-started,
                   time_limit=limit)
    return ctx.incumbent
