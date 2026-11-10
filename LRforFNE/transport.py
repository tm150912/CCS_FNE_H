"""Exact full-horizon, integer, directed-arc Lagrangian minimization."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from heuristic_stages.common import Plan, _BudgetExpired
from read import read_pipeline_limit


@dataclass
class ArcSolution:
    value: float
    y: np.ndarray
    x: np.ndarray
    kappa: np.ndarray


def solve_arc(fixed, gamma, capacities, pipeline_install_limit):
    """Arrays are diameter x period; argmin uses supplied diameter/time order."""
    fixed, gamma = np.asarray(fixed, dtype=float), np.asarray(gamma, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    limit = read_pipeline_limit(pipeline_install_limit)
    if fixed.shape != gamma.shape or fixed.ndim != 2 or fixed.shape[0] != len(capacities):
        raise ValueError("Arc arrays must have matching diameter x period dimensions")
    if not fixed.size or not np.all(np.isfinite(capacities)) or np.any(capacities <= 0):
        raise ValueError("Arc capacities and dimensions must be positive and finite")
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(gamma)):
        raise ValueError("Arc coefficients must be finite")
    suffix = np.cumsum(np.minimum(gamma, 0.0)[:, ::-1], axis=1)[:, ::-1]
    kappa = fixed + capacities[:, None] * suffix
    y = np.zeros_like(fixed, dtype=int)
    x = np.zeros_like(fixed)
    d, h = np.unravel_index(int(np.argmin(kappa)), kappa.shape)
    minimum = float(kappa[d, h])
    if limit and minimum < 0.0:  # Do not suppress even a very small negative kappa.
        y[d, h] = limit
        x[d, h:] = np.where(gamma[d, h:] < 0.0, capacities[d] * limit, 0.0)
    return ArcSolution(limit * min(0.0, minimum), y, x, kappa)


class TransportOracle:
    """Cache original coefficients once; never build any transportation MILP."""
    def __init__(self, network, costs, diameters, pipeline_install_limit, nodes, deadline):
        self.n, self.c = network, costs
        self.d, self.t = list(diameters), list(costs.periods)
        self.node_index = {node: i for i, node in enumerate(nodes)}
        self.diameter_index = {d: i for i, d in enumerate(self.d)}
        self.period_index = {h: i for i, h in enumerate(self.t)}
        self.limit = read_pipeline_limit(pipeline_install_limit)
        self.capacities = np.array([network.pipe_capacity[d] for d in self.d])
        self.fixed = np.empty((len(network.ij), len(self.d), len(self.t)))
        self.variable = np.empty_like(self.fixed)
        for e, (u, v) in enumerate(network.ij):
            if perf_counter() >= deadline:
                raise _BudgetExpired()
            self.fixed[e] = [[costs.pipe_fixed[u, v, d, h] for h in self.t] for d in self.d]
            self.variable[e] = [[costs.pipe_variable[u, v, d, h] for h in self.t] for d in self.d]
        self.last_details = {}

    def solve(self, multipliers, incumbent, deadline):
        incumbent_y, incumbent_x = {}, {}
        if incumbent is not None:
            for destination, values in ((incumbent_y, incumbent.y), (incumbent_x, incumbent.x)):
                for (u, v, d, h), value in values.items():
                    destination.setdefault((u, v), []).append(
                        (self.diameter_index[d], self.period_index[h], value))
        plan, details, values = Plan(), {}, []
        for e, (u, v) in enumerate(self.n.ij):
            if perf_counter() >= deadline:
                self.last_details = {}
                raise _BudgetExpired()
            gamma = self.variable[e] + multipliers[self.node_index[u]] - multipliers[self.node_index[v]]
            solved = solve_arc(self.fixed[e], gamma, self.capacities, self.limit)
            for di, hi in zip(*np.nonzero(solved.y)):
                plan.y[u, v, self.d[di], self.t[hi]] = int(solved.y[di, hi])
            for di, hi in zip(*np.nonzero(solved.x)):
                plan.x[u, v, self.d[di], self.t[hi]] = float(solved.x[di, hi])
            phi = sum(solved.kappa[d, h] * q for d, h, q in incumbent_y.get((u, v), ()))
            actual = (sum(self.fixed[e, d, h] * q for d, h, q in incumbent_y.get((u, v), ()))
                      + sum(gamma[d, h] * q for d, h, q in incumbent_x.get((u, v), ())))
            details[u, v] = dict(theta=solved.value, phi=float(phi),
                                incumbent_lagrangian=float(actual),
                                build_regret=float(phi - solved.value),
                                flow_regret=float(actual - phi))
            values.append(solved.value)
        self.last_details = details
        return plan, float(sum(values))
