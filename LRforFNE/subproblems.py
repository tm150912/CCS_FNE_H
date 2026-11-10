"""Persistent capture/storage MILPs and the common certified LR oracle."""
from __future__ import annotations
from math import isfinite
from time import perf_counter

import gurobipy as gp
import numpy as np
from solve_logging import optimize_with_log

from heuristic_stages.common import Plan, COST_FIELDS, _objective, _status_name, _BudgetExpired
from read import read_well_capacities, read_well_injection_capacities
from .lr_types import OracleResult
from .transport import TransportOracle


def residual(plan, nodes, periods):
    index = {node: i for i, node in enumerate(nodes)}
    times = {h: i for i, h in enumerate(periods)}
    g = np.zeros((len(nodes), len(periods)))
    for (i, _, h), q in plan.a.items():
        g[index[i], times[h]] -= q
    for (j, h), q in plan.b.items():
        g[index[j], times[h]] += q
    for (u, v, _, h), q in plan.x.items():
        g[index[u], times[h]] += q
        g[index[v], times[h]] -= q
    return g


def _finite(value):
    return float(value) if isfinite(value) and abs(value) < gp.GRB.INFINITY / 2 else None


class LagrangianOracle:
    def __init__(self, network, costs, diameters, pipeline_install_limit, config, env, deadline):
        self.n, self.c, self.config = network, costs, config
        self.deadline = deadline
        self.nodes = list(network.sources) + list(network.sinks) + list(network.intersections)
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Node classes must be disjoint")
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.periods = list(costs.periods)
        self.period_index = {h: i for i, h in enumerate(self.periods)}
        self.qw = read_well_injection_capacities(network.well_capacity, network.sinks)
        self.max_wells = read_well_capacities(network, costs)
        self.models, self.variables, self.previous = {}, {}, {}
        try:
            self._build_capture(env)
            self._build_storage(env)
            self.transport = TransportOracle(network, costs, diameters, pipeline_install_limit,
                                             self.nodes, deadline)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        transport = getattr(self, 'transport', None)
        if hasattr(transport, 'close'):
            transport.close()
        for model in self.models.values():
            model.dispose()
        self.models.clear()

    def _check_time(self):
        remaining = self.deadline - perf_counter()
        if remaining <= 0:
            raise _BudgetExpired()
        return remaining

    def _model(self, name, env):
        self._check_time()
        m = gp.Model("LR_" + name, env=env)
        self.models[name] = m
        m.Params.OutputFlag = int(self.config.gurobi_output_flag)
        m.Params.Threads = self.config.threads
        m.Params.Seed = self.config.seed
        m.Params.FeasibilityTol = max(1e-9, min(1e-8, self.config.feasibility_tolerance / 10))
        m.Params.IntFeasTol = m.Params.FeasibilityTol
        return m

    def _build_capture(self, env):
        m = self._model("capture", env)
        keys = [(i, g, h) for i, g in self.n.ig for h in self.periods]
        s = m.addVars(keys, vtype=gp.GRB.BINARY, name="s", obj=self.c.source_fixed)
        a = m.addVars(keys, name="a", obj=self.c.source_variable)
        for i, g in self.n.ig:
            self._check_time()
            active = gp.LinExpr()
            for h in self.periods:
                active += s[i, g, h]
                m.addConstr(a[i, g, h] <= self.n.source_capacity[i, g] * active)
            m.addConstr(active <= 1)
        for h in self.periods:
            m.addConstr(gp.quicksum(a[i, g, h] for i, g in self.n.ig) >= self.c.capture_target[h])
        self.variables["capture"] = dict(s=s, a=a)

    def _build_storage(self, env):
        m = self._model("storage", env)
        keys = [(j, h) for j in self.n.sinks for h in self.periods]
        r = m.addVars(keys, vtype=gp.GRB.BINARY, name="r", obj=self.c.sink_fixed)
        w = m.addVars(keys, vtype=gp.GRB.INTEGER, name="w", obj=self.c.well_fixed)
        b = m.addVars(keys, name="b", obj=self.c.sink_variable)
        for j in self.n.sinks:
            self._check_time()
            active, wells = gp.LinExpr(), gp.LinExpr()
            for h in self.periods:
                active += r[j, h]
                wells += w[j, h]
                m.addConstr(wells <= self.max_wells[j] * active)
                m.addConstr(b[j, h] <= self.qw[j] * wells)
            m.addConstr(active <= 1)
        for h in self.periods:
            m.addConstr(gp.quicksum(b[j, h] for j in self.n.sinks) >= self.c.capture_target[h])
        self.variables["storage"] = dict(r=r, w=w, b=b)

    def _block(self, name, multipliers, incumbent, precision_factor):
        m, variables = self.models[name], self.variables[name]
        flow, sign = ("a", -1) if name == "capture" else ("b", 1)
        original = getattr(self.c, COST_FIELDS[flow])
        for key, var in variables[flow].items():
            var.Obj = original[key] + sign * multipliers[self.node_index[key[0]], self.period_index[key[-1]]]
        starts = [p for p in (incumbent, self.previous.get(name)) if p is not None]
        m.NumStart = len(starts)
        for number, start in enumerate(starts):
            m.Params.StartNumber = number
            for field, mapping in variables.items():
                values = getattr(start, field)
                for key, var in mapping.items():
                    var.Start = values.get(key, 0)
        m.Params.MIPGap = (self.config.capture_mip_gap if name == 'capture' else
                          self.config.storage_mip_gap)
        m.Params.MIPGapAbs = self.config.oracle_abs_tol * precision_factor / 4
        m.Params.TimeLimit = min(self.config.oracle_time_limit, self._check_time())
        optimize_with_log(m)
        bound = _finite(m.ObjBound) if m.Status != gp.GRB.LOADED else None
        metrics = dict(status=_status_name(m.Status), ell=bound, u=None,
                       solution_count=m.SolCount, runtime=m.Runtime,
                       mip_gap_target=m.Params.MIPGap,
                       mip_gap=_finite(m.MIPGap) if m.SolCount else None)
        if not m.SolCount:
            return None, metrics
        plan = Plan()
        for field, mapping in variables.items():
            # Retain all nonzero values: arbitrary flow sparsification can break a cut witness.
            setattr(plan, field, {key: float(var.X) for key, var in mapping.items() if var.X != 0})
        if not self._block_feasible(name, plan):
            metrics.update(status="NUMERICAL_ERROR", ell=None)
            return None, metrics
        self.previous[name] = plan
        lag = _objective(self.c, plan)
        for key, q in getattr(plan, flow).items():
            lag += sign * multipliers[self.node_index[key[0]], self.period_index[key[-1]]] * q
        metrics["u"] = float(lag)
        return plan, metrics

    def _block_feasible(self, name, plan):
        tol = self.config.feasibility_tolerance
        for field in self.variables[name]:
            for q in getattr(plan, field).values():
                if not isfinite(q) or q < -tol:
                    return False
                if field in ("s", "r", "w") and abs(q - round(q)) > tol:
                    return False
        if name == "capture":
            for i, g in self.n.ig:
                active = 0.0
                for h in self.periods:
                    active += plan.s.get((i, g, h), 0)
                    if active > 1 + tol or plan.a.get((i, g, h), 0) > self.n.source_capacity[i, g] * active + tol:
                        return False
            totals = {h: 0.0 for h in self.periods}
            for (_, _, h), q in plan.a.items():
                totals[h] += q
        else:
            for j in self.n.sinks:
                active, wells = 0.0, 0.0
                for h in self.periods:
                    active += plan.r.get((j, h), 0)
                    wells += plan.w.get((j, h), 0)
                    if (active > 1 + tol or wells > self.max_wells[j] * active + tol
                            or plan.b.get((j, h), 0) > self.qw[j] * wells + tol):
                        return False
            totals = {h: 0.0 for h in self.periods}
            for (_, h), q in plan.b.items():
                totals[h] += q
        return all(totals[h] >= self.c.capture_target[h] - tol for h in self.periods)

    def _nodal_multipliers(self, multipliers):
        if multipliers.shape != (len(self.nodes), len(self.periods)) or not np.all(np.isfinite(multipliers)):
            raise ValueError("Multipliers must be finite node x period values")
        return multipliers

    def _gradient(self, plan):
        return residual(plan, self.nodes, self.periods)

    def _transport_block(self, multipliers, incumbent, precision_factor):
        started = perf_counter()
        plan, bound = self.transport.solve(multipliers, incumbent, self.deadline)
        return plan, dict(status='ANALYTICAL_OPTIMAL', u=bound, ell=bound,
                          runtime=perf_counter()-started,
                          arc_details=self.transport.last_details)

    def evaluate(self, multipliers, incumbent=None, evaluation_id=0, precision_factor=1.0):
        started = perf_counter()
        multipliers = np.asarray(multipliers, dtype=float)
        nodal = self._nodal_multipliers(multipliers)
        blocks, plans = {}, {}
        try:
            for name in ("capture", "storage"):
                plans[name], blocks[name] = self._block(name, nodal, incumbent, precision_factor)
                if blocks[name]['status'] == 'INTERRUPTED':
                    return OracleResult(evaluation_id=evaluation_id, multipliers=multipliers.copy(),
                        plan=None, residual=None, original_objective=None, u=None, ell=None,
                        epsilon=None, accurate=False, blocks=blocks, status='INTERRUPTED',
                        runtime=perf_counter()-started)
            plans['transport'], blocks['transport'] = self._transport_block(nodal, incumbent, precision_factor)
        except _BudgetExpired:
            return OracleResult(evaluation_id=evaluation_id, multipliers=multipliers.copy(),
                                plan=None, residual=None, original_objective=None, u=None, ell=None,
                                epsilon=None, accurate=False, blocks=blocks, status="ORACLE_INCOMPLETE",
                                runtime=perf_counter() - started)
        ell = sum(b["ell"] for b in blocks.values()) if all(b["ell"] is not None for b in blocks.values()) else None
        plan, g, original, u, epsilon = None, None, None, None, None
        status, accurate = "ORACLE_INCOMPLETE", False
        if all(p is not None for p in plans.values()):
            plan = Plan(**{field: getattr(p, field) for p in plans.values()
                           for field in COST_FIELDS if getattr(p, field)})
            g, original = self._gradient(plan), _objective(self.c, plan)
            u = float(original + np.sum(multipliers * g))
            if ell is not None:
                epsilon = u - ell
                tolerance = precision_factor * (self.config.oracle_abs_tol + self.config.oracle_rel_tol * max(1.0, abs(u)))
                accurate = -tolerance <= epsilon <= tolerance
                status = "NUMERICALLY_EXACT" if accurate else "BOUNDED_INEXACT"
                if epsilon < -tolerance or not isfinite(u) or not isfinite(ell):
                    status, accurate, ell = "NUMERICAL_ERROR", False, None
        if any(b["status"] in ("INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "NUMERICAL_ERROR") for b in blocks.values()):
            status, accurate, ell = "NUMERICAL_ERROR", False, None
        return OracleResult(evaluation_id=evaluation_id, multipliers=multipliers.copy(), plan=plan,
                            residual=g, original_objective=original, u=u, ell=ell, epsilon=epsilon,
                            accurate=accurate, blocks=blocks, status=status, runtime=perf_counter() - started)
