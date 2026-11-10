"""Concave proximal bundle: affine upper cuts, certified interval decisions."""
from time import perf_counter
import sys
import gurobipy as gp
import numpy as np
from solve_logging import optimize_with_log
from heuristic_stages.common import Plan, _objective, _BudgetExpired
from .subproblems import residual


def serious_step(center_ell, center_u, trial_ell, trial_u, model_value, eta):
    # The same center value occurs on both sides of the exact test. Combine
    # it before taking interval endpoints, rather than worsening it twice.
    if trial_ell - center_u >= eta * (model_value - center_u):
        return 'serious'
    if trial_u - center_ell < eta * (model_value - center_ell):
        return 'null'
    return 'ambiguous'


def _has_interval(result):
    return (result.plan is not None and result.residual is not None
            and result.original_objective is not None
            and result.u is not None and result.ell is not None)


def _interval_step(center, trial, model_value, eta):
    if not _has_interval(trial):
        return 'ambiguous'
    return serious_step(center.ell, center.u, trial.ell, trial.u, model_value, eta)


def _relative_gap_floor(result):
    """Reoptimizing an unchanged, gap-terminated MIP is not tighter accuracy."""
    return (_has_interval(result) and bool(result.blocks)
            and all(block.get('status') == 'ANALYTICAL_OPTIMAL'
                    or (block.get('status') == 'OPTIMAL'
                        and block.get('mip_gap') is not None
                        and block.get('mip_gap_target') is not None
                        and block['mip_gap'] <= block['mip_gap_target'])
                    for block in result.blocks.values()))


def _refine_once(run, previous, factor, stalled, aggregate=None):
    if _relative_gap_floor(previous):
        return previous, False, 'unchanged_relative_gap_target'
    key = previous.multipliers.tobytes()
    if key in stalled:
        return previous, False, 'previous_refinement_no_progress'
    current = run.refine_evaluation(previous, precision_factor=factor, aggregate=aggregate)
    run.check_gap()
    tolerance = run.config.oracle_abs_tol + run.config.oracle_rel_tol * max(1, abs(previous.u or 0))
    improved = (_has_interval(current) and (not _has_interval(previous)
        or current.ell > previous.ell + tolerance or current.u < previous.u - tolerance))
    if not improved:
        stalled.add(key)
    return current, improved, 'interval_tightened' if improved else 'interval_not_tightened'


def _adapt_rho(rho, initial_rho, center, trial, model_value, decision, config):
    if not config.bundle_adaptive_rho or not _has_interval(trial):
        return rho, 'unchanged'
    lower = max(config.bundle_rho_floor, initial_rho / config.bundle_rho_max_factor)
    upper = initial_rho * config.bundle_rho_max_factor
    prediction = model_value - center.u
    uncertainty = max(0.0, center.u-center.ell) + max(0.0, trial.u-trial.ell)
    # Increase stabilization only for a provably worse trial while the model
    # signal exceeds oracle noise. Do not chase that noise with ever larger rho.
    if decision == 'null' and trial.u < center.ell and prediction > uncertainty:
        return min(upper, rho * config.bundle_rho_growth), 'certified_worse_trial'
    if decision == 'serious':
        return max(lower, rho * config.bundle_rho_shrink), 'certified_serious_step'
    return rho, 'unchanged'


def _record_dual_diagnostic(run, master, iteration, reason):
    if (not run.config.bundle_diagnostic_every or not hasattr(run, 'diagnostics')
            or perf_counter() >= run.deadline):
        return
    from .dual_diagnostic import diagnose_dual
    try:
        result = diagnose_dual(master, run.ctx.best, run.deadline,
                               run.config.bundle_diagnostic_time_limit)
    except gp.GurobiError as exc:
        result = dict(status='SOLVER_ERROR', upper_bound=None, error=str(exc))
    lower = getattr(run, 'lower', None)
    if (result.get('upper_bound') is not None and lower is not None
            and result['upper_bound'] < lower):
        result.update(status='NUMERICAL_INCONSISTENCY',
                      rejected_upper_bound=result['upper_bound'], upper_bound=None)
    previous = [row['best_upper_bound'] for row in run.diagnostics
                if row.get('stage') == 'LR_bundle_dual_upper'
                and row.get('best_upper_bound') is not None]
    if lower is not None:
        rejected = [value for value in previous if value < lower]
        if rejected:
            result['rejected_historical_upper_bounds'] = rejected
        previous = [value for value in previous if value >= lower]
    if result.get('upper_bound') is not None:
        previous.append(result['upper_bound'])
    best_upper = min(previous) if previous else None
    run.diagnostics.append(dict(result, stage='LR_bundle_dual_upper', iteration=iteration,
        reason=reason, elapsed=perf_counter()-run.started, lower_bound=lower,
        primal_upper=run.ctx.best, best_upper_bound=best_upper,
        optimization_gap=None if best_upper is None or lower is None else best_upper-lower))
    if result.get('status') == 'INTERRUPTED':
        run.stop('FEASIBLE_INTERRUPTED', 'bundle_dual_diagnostic_interrupted')


def _stop_precision_floor(run, center, iteration, detail, refinements):
    if hasattr(run, 'diagnostics'):
        run.diagnostics.append(dict(stage='LR_bundle_precision_floor', iteration=iteration,
            elapsed=perf_counter()-run.started, reason=detail,
            center_evaluation=center.evaluation_id, center_ell=center.ell, center_u=center.u,
            refinements=refinements))
    run.stop('ORACLE_INCOMPLETE', 'bundle_oracle_precision_floor')


class BundleMaster:
    def __init__(self, config, env, shape, deadline, rho):
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError('rho must be finite and positive')
        self.config, self.shape, self.deadline, self.rho = config, shape, deadline, rho
        self.model = gp.Model('LR_proximal_bundle', env=env)
        self.model.Params.OutputFlag = int(config.gurobi_output_flag)
        self.model.Params.Threads = config.threads
        self.model.Params.FeasibilityTol = 1e-9
        self.model.Params.OptimalityTol = 1e-9
        self.model.Params.BarConvTol = 1e-10
        self.lam = self.model.addVars(int(np.prod(shape)), lb=-gp.GRB.INFINITY, name='lambda')
        self.v = self.model.addVar(lb=-gp.GRB.INFINITY, name='v')
        self.cuts, self.upper_cut = [], None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.model.dispose()

    def _affine(self, intercept, gradient):
        flat = gradient.ravel()
        indices = np.flatnonzero(flat)
        return intercept + gp.LinExpr([float(flat[i]) for i in indices], [self.lam[int(i)] for i in indices])

    def add_cut(self, result):
        if result.plan is None or result.original_objective is None or result.residual is None:
            raise ValueError('A cut requires a relaxed feasible witness')
        construction = Plan(**{name: dict(getattr(result.plan, name)) for name in ('s','r','w','y')})
        g = result.residual.copy()
        if any(f == result.original_objective and np.array_equal(old_g, g)
               for f, old_g, _, _ in self.cuts):
            return False
        constraint = self.model.addConstr(self.v <= self._affine(result.original_objective, g))
        self.cuts.append((result.original_objective, g, construction, constraint))
        return True

    def solve(self, center, incumbent, costs, nodes, periods):
        if not self.cuts:
            raise ValueError('Cannot solve an empty bundle')
        if self.upper_cut is not None:
            self.model.remove(self.upper_cut)
        upper = _objective(costs, incumbent)
        inc_g = residual(incumbent, nodes, periods)
        self.upper_cut = self.model.addConstr(self.v <= self._affine(upper, inc_g), name='incumbent_cut')
        c = center.ravel()
        self.model.setObjective(self.v - self.rho / 2 * gp.quicksum(
            (self.lam[i]-float(c[i])) * (self.lam[i]-float(c[i])) for i in range(len(c))), gp.GRB.MAXIMIZE)
        remaining = self.deadline - perf_counter()
        if remaining <= 0:
            raise _BudgetExpired()
        self.model.Params.TimeLimit = remaining
        optimize_with_log(self.model)
        if self.model.Status != gp.GRB.OPTIMAL or self.model.SolCount == 0:
            return dict(status='BUNDLE_MASTER_INCOMPLETE', solver_status=self.model.Status)
        point = np.array([self.lam[i].X for i in range(len(c))]).reshape(self.shape)
        weights = np.array([cut[3].Pi for cut in self.cuts] + [self.upper_cut.Pi])
        if (not np.all(np.isfinite(point)) or not np.all(np.isfinite(weights))
                or np.min(weights) < -1e-7 or abs(weights.sum()-1) > 1e-6):
            return dict(status='BUNDLE_MASTER_INCOMPLETE', reason='unreliable_QP_dual_weights')
        # Tiny negative numerical noise is discarded only after the sign/sum checks.
        aggregate = Plan()
        for weight, construction in zip(np.maximum(weights, 0), [cut[2] for cut in self.cuts] + [incumbent]):
            if weight == 0:
                continue
            for name in ('s','r','w','y'):
                out = getattr(aggregate, name)
                for key, value in getattr(construction, name).items():
                    out[key] = out.get(key, 0) + float(weight) * value
        old_model = min([f + float(np.sum(g*point)) for f,g,_,_ in self.cuts]
                        + [upper + float(np.sum(inc_g*point))])
        return dict(status='OPTIMAL', point=point, model_value=old_model,
                    v=float(self.v.X), qp_objective=float(self.model.ObjVal),
                    weights=weights.tolist(), aggregate=aggregate,
                    incumbent_residual_dot=float(np.sum(inc_g*point)))


def run_bundle(run):
    center = run.bounded_evaluation(run.initial_multipliers.copy())
    run.check_gap()
    norm2 = float(np.sum(center.residual**2))
    rho = run.config.bundle_rho
    if rho is None:
        rho = max(run.config.bundle_rho_floor,
                  norm2 / max(run.ctx.best-center.u, run.config.delta_cost))
    if not np.isfinite(rho):
        raise FloatingPointError('Nonfinite bundle proximal scale')
    initial_rho, iterations, uncertain_steps = rho, 0, 0
    stalled_refinements = set()
    with BundleMaster(run.config, run.ctx.env, run.shape, run.deadline, rho) as master:
        master.add_cut(center)
        try:
            while True:
                run.ensure_budget()
                master.rho = rho
                master_result = master.solve(center.multipliers, run.ctx.incumbent, run.ctx.c,
                                             run.ctx.nodes, run.ctx.t)
                if master_result['status'] != 'OPTIMAL':
                    run.stop('BUNDLE_MASTER_INCOMPLETE', str(master_result))
                iterations += 1
                if (run.config.bundle_diagnostic_every
                        and iterations % run.config.bundle_diagnostic_every == 0):
                    _record_dual_diagnostic(run, master, iterations, 'periodic')
                model_value = master_result['model_value']
                prediction = model_value - center.u
                refinement_events = []
                # An interval-overlapping model prediction is not convergence.
                # Retain the center, seek new cuts, and stop honestly at a
                # persistent oracle floor instead of repeating identical solves.
                for attempt in range(run.config.oracle_refinements + 1):
                    tolerance = run.config.bundle_tolerance * max(1, abs(center.u))
                    if model_value - center.ell <= tolerance:
                        run.stop('FEASIBLE_STAGNATION', 'bundle_predicted_improvement_tolerance')
                    if model_value - center.u > tolerance:
                        break
                    if attempt == run.config.oracle_refinements:
                        break
                    center, changed, reason = _refine_once(run, center, 0.1**(attempt+1),
                                                           stalled_refinements)
                    refinement_events.append(dict(role='center_prediction', reason=reason))
                    master.add_cut(center)
                    if not changed:
                        break
                if (model_value-center.u <= tolerance
                        and np.array_equal(master_result['point'], center.multipliers)):
                    _stop_precision_floor(run, center, iterations,
                        'unchanged_center_with_unresolved_prediction', refinement_events)
                trial = run.evaluate(master_result['point'], aggregate=master_result['aggregate'])
                run.check_gap()
                prediction_certified = model_value-center.u > tolerance
                decision = (_interval_step(center, trial, model_value, run.config.bundle_eta)
                            if prediction_certified else 'ambiguous')
                for attempt in range(run.config.oracle_refinements):
                    if decision != 'ambiguous':
                        break
                    factor = 0.1**(attempt+1)
                    trial, trial_changed, reason = _refine_once(run, trial, factor,
                        stalled_refinements, aggregate=master_result['aggregate'])
                    refinement_events.append(dict(role='trial', reason=reason))
                    decision = (_interval_step(center, trial, model_value, run.config.bundle_eta)
                                if prediction_certified else 'ambiguous')
                    center_changed = False
                    if decision == 'ambiguous' and _has_interval(trial):
                        center, center_changed, reason = _refine_once(run, center, factor,
                                                                     stalled_refinements)
                        refinement_events.append(dict(role='center', reason=reason))
                        master.add_cut(center)
                        tolerance = run.config.bundle_tolerance * max(1, abs(center.u))
                        prediction_certified = model_value-center.u > tolerance
                        decision = (_interval_step(center, trial, model_value, run.config.bundle_eta)
                                    if prediction_certified else 'ambiguous')
                    if not trial_changed and not center_changed:
                        break
                rho_next, rho_reason = _adapt_rho(rho, initial_rho, center, trial,
                                                  model_value, decision, run.config)
                run.history[-1].update(bundle_iteration=iterations,
                    bundle_center_evaluation=center.evaluation_id,
                    bundle_center_u=center.u, bundle_center_ell=center.ell,
                    model_value=model_value, qp_objective=master_result['qp_objective'],
                    bundle_v=master_result['v'], predicted_improvement=prediction,
                    prediction_basis='old_model_and_center_before_refinement',
                    prediction_certified=prediction_certified,
                    refined_predicted_improvement=model_value-center.u,
                    predicted_improvement_upper=model_value-center.ell,
                    actual_improvement=None if trial.u is None else trial.u-center.u,
                    actual_improvement_lower=None if trial.ell is None else trial.ell-center.u,
                    actual_improvement_upper=None if trial.u is None else trial.u-center.ell,
                    bundle_trial_u=trial.u, bundle_trial_ell=trial.ell,
                    bundle_trial_epsilon=trial.epsilon, bundle_trial_accurate=trial.accurate,
                    bundle_trial_evaluation=trial.evaluation_id,
                    bundle_step=decision, decision_basis='certified_shared_center_interval',
                    bundle_action='uncertain_cutting_step' if decision == 'ambiguous' else decision,
                    bundle_refinements=refinement_events, rho=rho, rho_next=rho_next,
                    rho_update_reason=rho_reason,
                    multiplier_step_norm=float(np.linalg.norm(master_result['point']-center.multipliers)),
                    trial_upper_minus_best_lower=(None if trial.u is None or getattr(run, 'lower', None) is None
                                                 else trial.u-run.lower),
                    weights=master_result['weights'],
                    incumbent_residual_dot=master_result['incumbent_residual_dot'])
                if _has_interval(trial):
                    master.add_cut(trial)
                else:
                    run.stop('ORACLE_INCOMPLETE', 'missing_trial_oracle_witness_or_bound')
                if decision == 'ambiguous':
                    uncertain_steps += 1
                    if uncertain_steps >= run.config.bundle_uncertain_limit:
                        _stop_precision_floor(run, center, iterations,
                            'consecutive_unresolved_interval_steps', refinement_events)
                else:
                    uncertain_steps = 0
                if decision == 'serious':
                    center = trial
                rho = rho_next
        finally:
            # Diagnostics share the LR deadline. Preserve interruption/errors;
            # an optional LP must not resume work after either has occurred.
            active = sys.exc_info()[1]
            if active is None or hasattr(active, 'status'):
                if getattr(active, 'status', None) not in ('FEASIBLE_INTERRUPTED',
                        'FEASIBLE_SOLVER_ERROR', 'NUMERICAL_ERROR'):
                    _record_dual_diagnostic(run, master, iterations, 'final')
