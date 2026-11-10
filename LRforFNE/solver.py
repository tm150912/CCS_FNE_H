"""Initialization, auditing, certified bounds and the Bundle LR loop."""
from copy import deepcopy
from dataclasses import asdict, replace
from time import perf_counter
import numpy as np
import gurobipy as gp
from heuristic_stages.common import _Context, _BudgetExpired, CandidatePool, Plan
from .config import LRConfig
from .lr_types import LRResult
from .warm_start import prepare_warm_start, primal_options
from .subproblems import LagrangianOracle
from .guidance import Guidance
from .neighborhoods import improve
from .pipeline_polish import polish_pipelines
from .bundle import run_bundle


class _StopRun(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason


def merge_evaluations(previous, current, config, precision_factor=1.0):
    """Intersect same-point certificates, retaining the complete best-u witness.

    evaluation_id and blocks belong to that witness; raw calls remain in history.
    The merged ell may come from another completed evaluation at exactly this point.
    """
    if not np.array_equal(previous.multipliers, current.multipliers):
        raise ValueError('Cannot merge oracle intervals at different multipliers')
    bounded = [r for r in (current, previous) if r.ell is not None]
    if not all(np.isfinite(r.ell) for r in bounded):
        raise FloatingPointError('Nonfinite oracle interval bound')
    bound_source = max(bounded, key=lambda r: r.ell) if bounded else None
    lower = None if bound_source is None else bound_source.ell
    bound_id = (None if bound_source is None else
                bound_source.bound_evaluation_id or bound_source.evaluation_id)
    witnesses = [r for r in (current, previous)
                 if r.plan is not None and r.residual is not None
                 and r.original_objective is not None and r.u is not None]
    if not witnesses:
        return replace(current, plan=None, residual=None, original_objective=None,
                       u=None, ell=lower, epsilon=None, accurate=False,
                       status='ORACLE_INCOMPLETE', bound_evaluation_id=bound_id)
    if not all(np.isfinite(r.u) for r in witnesses):
        raise FloatingPointError('Nonfinite oracle witness value')
    witness = min(witnesses, key=lambda r: r.u)
    epsilon = None if lower is None else witness.u - lower
    tolerance = precision_factor * (config.oracle_abs_tol
                    + config.oracle_rel_tol * max(1, abs(witness.u)))
    accurate = epsilon is not None and -tolerance <= epsilon <= tolerance
    status = ('NUMERICALLY_EXACT' if accurate else
              'BOUNDED_INEXACT' if lower is not None else 'ORACLE_INCOMPLETE')
    if epsilon is not None and epsilon < -tolerance:
        lower, accurate, status = None, False, 'NUMERICAL_ERROR'
    return replace(witness, ell=lower, epsilon=epsilon, accurate=accurate, status=status,
                   bound_evaluation_id=bound_id)


class _Run:
    def __init__(self, ctx, config, record, deadline, started):
        self.ctx, self.config, self.record = ctx, config, record
        self.deadline, self.started = deadline, started
        self.shape = (len(ctx.nodes), len(ctx.t))
        self.initial_multipliers = record.initial_multipliers.copy()
        if self.initial_multipliers.shape != self.shape or not np.all(np.isfinite(self.initial_multipliers)):
            raise ValueError('Initial LP multipliers must be finite node x period values')
        self.lower, self.origin = record.lower_bound, record.lower_bound_origin
        self.best_multipliers, self.multiplier_bound, self.multiplier_evaluation = None, None, None
        self.history, self.diagnostics = [], []
        self.evaluations = 0
        self.guidance, self.pool = Guidance(config), CandidatePool.from_plan(ctx.incumbent)
        self.last_complete_oracle, self.last_aggregate = None, None
        self.latest_oracle_usable = False
        self.primal_batches, self.primal_runtime = 0, 0.0
        self.oracle_termination = None

    def stop(self, status, reason):
        raise _StopRun(status, reason)

    def gap(self):
        return None if self.lower is None or self.ctx.best is None else (self.ctx.best-self.lower)/max(1,abs(self.ctx.best))

    def check_bounds(self):
        if self.lower is not None and self.ctx.best is not None:
            if self.lower > self.ctx.best + self.config.feasibility_tolerance * max(1,abs(self.ctx.best)):
                self.stop('NUMERICAL_ERROR', 'certified_LB_exceeds_audited_UB')

    def check_gap(self):
        self.check_bounds()
        if self.gap() is not None and self.gap() <= self.config.target_gap:
            self.stop('GAP_CERTIFIED', 'certified_relative_gap_reached')

    def ensure_budget(self):
        if perf_counter() >= self.deadline:
            self.stop('FEASIBLE_TIME_LIMIT', 'LR_stage_deadline')
        if self.evaluations >= self.config.max_iterations:
            self.stop('FEASIBLE_ITERATION_LIMIT', 'maximum_oracle_evaluations')

    def can_continue_primal(self, stop):
        # An unresolved Bundle step can still have valid interval/witness data.
        # Stop the dual algorithm honestly, while preserving the primal budget.
        bounded_ambiguity = (stop.status == 'ORACLE_INCOMPLETE' and stop.reason in (
            'ambiguous_serious_null_interval', 'ambiguous_bundle_prediction_interval',
            'bundle_oracle_precision_floor'))
        return (self.config.neighborhoods_enabled and self.latest_oracle_usable
                and (stop.status in ('FEASIBLE_ITERATION_LIMIT', 'FEASIBLE_STAGNATION')
                     or bounded_ambiguity))

    def continue_primal(self, stop):
        """Spend remaining LR time on audited N3 search, without another oracle."""
        self.oracle_termination = dict(status=stop.status, reason=stop.reason,
                                       evaluations=self.evaluations)
        self.ctx.record('LR_primal_start', **self.oracle_termination,
                        UB=self.ctx.best, LB=self.lower, gap=self.gap(),
                        guidance_source='cached_oracle',
                        guidance_oracle_id=self.last_complete_oracle.evaluation_id)
        started = perf_counter()
        try:
            while True:
                if perf_counter() >= self.deadline:
                    self.stop('FEASIBLE_TIME_LIMIT', 'LR_stage_deadline')
                self.check_gap()
                self.primal_batches += 1
                info = self.guidance.update(self.ctx, self.last_complete_oracle,
                    aggregate=self.last_aggregate, advance=False,
                    rotation_step=self.evaluations + self.primal_batches)
                before = self.ctx.best
                history_start = len(self.ctx.history)
                details = {}
                try:
                    details = improve(self.ctx, self.pool, info, self.config,
                        self.evaluations + self.primal_batches, n3_only=True)
                    self.check_bounds()
                    if any(row.get('solver_status') == 'INTERRUPTED'
                           for row in self.ctx.history[history_start:]):
                        self.stop('FEASIBLE_INTERRUPTED', 'primal_search_interrupted')
                    self.check_gap()
                finally:
                    self.ctx.record('LR_primal_search', batch=self.primal_batches,
                        oracle_evaluations=self.evaluations,
                        guidance_source='cached_oracle',
                        guidance_oracle_id=self.last_complete_oracle.evaluation_id,
                        improvement=before-self.ctx.best, UB=self.ctx.best,
                        LB=self.lower, gap=self.gap(),
                        n3_attempts=details.get('n3_attempts', 0),
                        n3_new_candidates=details.get('n3_new_candidates', 0),
                        n3_rotation_sweep_complete=details.get('n3_rotation_sweep_complete', False))
                    if self.config.output_flag:
                        print(f"[LR primal {self.primal_batches}] UB={self.ctx.best} "
                              f"LB={self.lower} gap={self.gap()} oracle_calls={self.evaluations}", flush=True)
        finally:
            self.primal_runtime = perf_counter()-started

    def evaluate(self, point, precision_factor=1.0, aggregate=None, previous=None):
        self.ensure_budget()
        self.evaluations += 1
        self.latest_oracle_usable = False
        result = self.oracle.evaluate(point, self.ctx.incumbent,
                    evaluation_id=self.evaluations, precision_factor=precision_factor)
        # Validate a refined certificate against the earlier feasible witness
        # before it can enter the global LB or trigger gap certification.
        merged = (None if previous is None else
                  merge_evaluations(previous, result, self.config, precision_factor))
        inconsistent = merged is not None and merged.status == 'NUMERICAL_ERROR'
        if result.ell is not None and not inconsistent and result.status != 'NUMERICAL_ERROR':
            if not np.isfinite(result.ell):
                self.stop('NUMERICAL_ERROR', 'nonfinite_oracle_lower_bound')
            if self.multiplier_bound is None or result.ell > self.multiplier_bound:
                self.multiplier_bound = result.ell
                self.best_multipliers = point.copy()
                self.multiplier_evaluation = self.evaluations
            if self.lower is None or result.ell > self.lower:
                self.lower, self.origin = result.ell, 'LR_oracle_ell'
        row = dict(oracle_id=self.evaluations, iteration=self.evaluations,
                   elapsed=perf_counter()-self.started, u=result.u, ell=result.ell,
                   epsilon=result.epsilon, accurate=result.accurate, oracle_status=result.status,
                   oracle_runtime=result.runtime, precision_factor=precision_factor,
                   residual_norm=None if result.residual is None else float(np.linalg.norm(result.residual)),
                   blocks={key:{k:v for k,v in block.items() if k!='arc_details'}
                           for key,block in result.blocks.items()})
        self.history.append(row)
        if merged is not None:
            row['merged_interval'] = dict(
                u=merged.u, ell=merged.ell, epsilon=merged.epsilon, accurate=merged.accurate,
                status=merged.status, precision_factor=precision_factor,
                witness_evaluation_id=merged.evaluation_id if merged.plan is not None else None,
                bound_evaluation_id=merged.bound_evaluation_id,
                previous_u=previous.u, previous_ell=previous.ell)
        try:
            if inconsistent:
                self.stop('NUMERICAL_ERROR', 'same_point_oracle_intervals_inconsistent')
            self.check_bounds()
            if result.status == 'INTERRUPTED' or any(
                    block.get('status') == 'INTERRUPTED' for block in result.blocks.values()):
                self.stop('FEASIBLE_INTERRUPTED', 'oracle_interrupted')
            if result.status == 'NUMERICAL_ERROR':
                self.stop('NUMERICAL_ERROR', 'oracle_numerical_inconsistency')
            if result.plan is not None and result.residual is not None:
                balance = result.residual
                if np.max(np.abs(balance), initial=0) <= self.config.feasibility_tolerance:
                    audit = self.ctx.audit(result.plan)
                    if audit['feasible']:
                        self.ctx.consider(result.plan, 'LR_balanced_oracle')
                self.check_gap()
                info = (self.guidance.update(self.ctx, result, aggregate=aggregate)
                        if self.config.neighborhoods_enabled else None)
                complete = result if merged is None else merged
                if (complete.ell is not None and complete.u is not None
                        and complete.original_objective is not None
                        and np.isfinite(complete.ell) and np.isfinite(complete.u)
                        and np.isfinite(complete.original_objective)
                        and np.all(np.isfinite(complete.residual))):
                    self.last_complete_oracle = complete
                    self.last_aggregate = None if aggregate is None else aggregate.copy()
                    self.latest_oracle_usable = True
                if (self.config.neighborhoods_enabled
                        and (self.evaluations == 1 or self.evaluations % self.config.neighborhood_every == 0)):
                    details = improve(self.ctx, self.pool, info, self.config, self.evaluations)
                    row['neighborhoods'] = details
                self.check_bounds()
        finally:
            row.update(UB=self.ctx.best, LB=self.lower, gap=self.gap(),
                       elapsed=perf_counter()-self.started)
            callback = getattr(self, 'progress_callback', None)
            if callback is not None:
                callback(dict(row))
            if self.config.output_flag:
                print(f"[LR {self.evaluations}] UB={self.ctx.best} LB={self.lower} gap={self.gap()} oracle={result.status}", flush=True)
        return result if merged is None else merged

    def refine_evaluation(self, previous, precision_factor=0.1, aggregate=None):
        return self.evaluate(previous.multipliers, precision_factor, aggregate, previous=previous)

    def bounded_evaluation(self, point, aggregate=None):
        """Return a complete certified interval without requiring an exact oracle."""
        factor = 1.0
        result = None
        for attempt in range(self.config.oracle_refinements+1):
            result = (self.evaluate(point, factor, aggregate) if result is None else
                      self.refine_evaluation(result, factor, aggregate))
            self.check_gap()
            if (result.plan is not None and result.residual is not None
                    and result.original_objective is not None
                    and result.u is not None and result.ell is not None):
                if not (np.isfinite(result.u) and np.isfinite(result.original_objective)
                        and np.all(np.isfinite(result.residual))):
                    self.stop('NUMERICAL_ERROR', 'nonfinite_oracle_witness')
                return result
            self.ensure_budget()
            if attempt == self.config.oracle_refinements:
                self.stop('ORACLE_INCOMPLETE', 'missing_oracle_witness_or_bound')
            factor *= 0.1
        raise AssertionError('unreachable')


def _solve_record(network, costs, diameters, pipeline_install_limit, method, config, record,
                  progress_callback=None):
    if method != 'bundle':
        raise ValueError('method must be bundle')
    config.validate()
    started = perf_counter()
    deadline = started + config.lr_time_limit
    polish_reserve = (min(config.pipeline_polish_time_limit, 0.2 * config.lr_time_limit)
                      if config.neighborhoods_enabled else 0.0)
    search_deadline = deadline - polish_reserve
    # A single context per LR version: includes construction and all audit time.
    ctx = _Context(network, costs, diameters, primal_options(record, config),
                   pipeline_install_limit=pipeline_install_limit)
    ctx.end, ctx.phase_end = deadline, search_deadline
    context_runtime = perf_counter()-started
    audit_started = perf_counter()
    initial = None
    if record.plan is not None:
        ctx.consider(record.plan, 'LR_initial_audit')
        initial = ctx.best
    initial_audit_runtime = perf_counter()-audit_started
    run = None
    final_polish_runtime = 0.0
    status, reason = 'NO_INITIAL_FEASIBLE_SOLUTION', record.status
    if (ctx.best is not None and record.lower_bound is not None and
            record.lower_bound > ctx.best + config.feasibility_tolerance * max(1, abs(ctx.best))):
        status, reason = 'NUMERICAL_ERROR', 'certified_LB_exceeds_audited_UB'
    elif record.proven_infeasible:
        status, reason = 'PROVEN_INFEASIBLE', 'certified_full_original_LP_infeasible'
    elif ctx.incumbent is not None and record.initial_multipliers is None:
        lp_status = record.multiplier_initialization.get('status')
        if lp_status == 'INTERRUPTED':
            status, reason = 'FEASIBLE_INTERRUPTED', 'multiplier_LP_interrupted'
        elif lp_status in ('INFEASIBLE', 'UNBOUNDED', 'INF_OR_UNBD'):
            status, reason = 'NUMERICAL_ERROR', 'multiplier_LP_inconsistent_with_audited_incumbent'
        else:
            status, reason = 'MULTIPLIER_INITIALIZATION_INCOMPLETE', f'multiplier_LP_{lp_status}'
    elif ctx.incumbent is not None:
        run = _Run(ctx, config, record, search_deadline, started)
        run.progress_callback = progress_callback
        try:
            run.check_gap()
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', int(config.gurobi_output_flag))
                env.start()
                ctx.env = env
                try:
                    run.ensure_budget()
                    with LagrangianOracle(network, costs, diameters, pipeline_install_limit,
                                          config, env, search_deadline) as oracle:
                        run.oracle = oracle
                        try:
                            run_bundle(run)
                        except _StopRun as stop:
                            if not run.can_continue_primal(stop):
                                raise
                            oracle_stop = stop
                    # Dispose oracle models; retain the environment and shared budget.
                    run.continue_primal(oracle_stop)
                except (_StopRun, _BudgetExpired) as stop:
                    normal_stop = (isinstance(stop, _BudgetExpired) or stop.status in (
                        'FEASIBLE_TIME_LIMIT', 'FEASIBLE_ITERATION_LIMIT', 'FEASIBLE_STAGNATION'))
                    if polish_reserve and normal_stop:
                        ctx.phase_end = deadline
                        polish_started = perf_counter()
                        try:
                            polish_pipelines(ctx, run.pool, config, source='final')
                            run.check_gap()
                        finally:
                            final_polish_runtime = perf_counter() - polish_started
                    raise
        except _StopRun as stop:
            status, reason = stop.status, stop.reason
        except _BudgetExpired:
            status, reason = 'FEASIBLE_TIME_LIMIT', 'LR_stage_deadline'
        except KeyboardInterrupt:
            status, reason = 'FEASIBLE_INTERRUPTED', 'user_interrupt'
        except (FloatingPointError, OverflowError) as exc:
            status, reason = 'NUMERICAL_ERROR', str(exc)
        except gp.GurobiError as exc:
            status, reason = 'FEASIBLE_SOLVER_ERROR', str(exc)
        finally:
            ctx.env = None
    audit = ctx.audit(ctx.incumbent) if ctx.incumbent is not None else None
    lower = run.lower if run else record.lower_bound
    gap = None if lower is None or ctx.best is None else (ctx.best-lower)/max(1,abs(ctx.best))
    runtime = perf_counter()-started
    warm_runtime = record.runtime
    timings = dict(record.timings, lr_wall=runtime, shared_warm_start_wall=record.runtime,
                   allocated_warm_start_wall=warm_runtime, primal_context_wall=context_runtime,
                   initial_audit_wall=initial_audit_runtime,
                   pipeline_polish_reserve=polish_reserve,
                   final_pipeline_polish_wall=final_polish_runtime,
                   primal_continuation_wall=run.primal_runtime if run else 0.0,
                   primal_continuation_batches=run.primal_batches if run else 0,
                   oracle_termination=run.oracle_termination if run else None)
    best = run.best_multipliers if run else None
    return LRResult(method, status, reason, initial, ctx.best, lower, gap,
        ctx.incumbent or Plan(), run.history if run else [], warm_runtime, runtime,
        warm_runtime+runtime, best,
        run.multiplier_bound if run else None, run.multiplier_evaluation if run else None,
        run.origin if run else record.lower_bound_origin, audit,
        dict(algorithm=config.to_dict(), initialization_options=asdict(record.options),
             warm_start=record.timings['warm_start_options'],
             model_parameters=deepcopy(getattr(record, 'model_parameters', {})),
             primal=ctx.configuration['options'], initial_lower_bound=record.lower_bound,
             initial_lower_bound_origin=record.lower_bound_origin,
             multiplier_initialization=record.multiplier_initialization), record.instance, timings,
        ctx.history + (run.diagnostics if run else []),
        None if record.initial_multipliers is None else record.initial_multipliers.copy(),
        list(getattr(record, 'initialization_history', [])))


def solve_lr(network, costs, diameters, *, pipeline_install_limit, method='bundle',
             config=None, heuristic_options=None, initial_result=None):
    config = (config or LRConfig()).validate()
    if method != 'bundle':
        raise ValueError('method must be bundle')
    record = prepare_warm_start(network, costs, diameters, pipeline_install_limit,
                                heuristic_options, initial_result)
    return _solve_record(network, costs, diameters, pipeline_install_limit, method, config, record)

