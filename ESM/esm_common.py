"""Static E-Select data, auditing, logging and export used by RSS and PH.

本模块不导入或选择求解算法；运行入口显式传入配置和求解函数。
各算法的模型参数、预算、搜索规则和最优性证书仍由各自文件管理。
"""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import gurobipy as gp
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.drawing.spreadsheet_drawing import SpreadsheetDrawing
from openpyxl.styles import PatternFill
from openpyxl.worksheet._writer import WorksheetWriter
from openpyxl.writer.excel import ExcelWriter
from openpyxl.xml.constants import SHEET_MAIN_NS
from openpyxl.xml.functions import xmlfile

from capture_target_sequence import generate_capture_targets
from experiment_setup import (
    CostData, NetworkData, compute_cost_data, load_network_data, period_schedule,
)
from read import (
    read_pipeline_limit, read_well_capacities, read_well_injection_capacities,
)
from result_metadata import parameter_frame
from solve_logging import optimize_with_log

GROUPS = ("s", "r", "w", "y", "a", "b", "x")
BUILD_GROUPS = ("s", "r", "w", "y")
CONSTRUCTION_GROUPS = BUILD_GROUPS
GUROBI_TOL_FLOOR = 1e-9
_SOLVER_LOG = ContextVar("esm_solver_log", default=None)


COST_FIELDS = {
    "s": "source_fixed", "r": "sink_fixed", "w": "well_fixed",
    "y": "pipe_fixed", "a": "source_variable", "b": "sink_variable", "x": "pipe_variable",
}


# ==================== 静态输入与基础审计 ====================


@dataclass(frozen=True)
class ESelectData:

    network: NetworkData
    nodes: tuple
    diameters: tuple
    indices: dict
    pipeline_limit: int
    maximum_wells: dict
    well_capacity: dict
    periods: tuple
    durations: dict
    targets: dict
    peak: float
    target: float
    weights: dict
    coefficients: dict
    model_parameters: dict = field(default_factory=dict)


def build_capture_target(period_count, curve, coefficient, maximum_capture_amount,
                         capture_amount_sequence=None, *, normalize_capture_target=False):
    """生成并校验各期捕集目标，返回以从 1 开始的时期编号为键的字典。

    优先使用显式目标序列，否则按线性、指数或对数曲线生成目标；
    目标必须有限且非负，但允许非单调变化或初期为零。
    开启归一化时，自动非线性序列的全期总量与线性基准一致。
    """
    if (isinstance(period_count, bool) or not isinstance(period_count, int)
            or period_count <= 0):
        raise ValueError("period_count must be a positive integer")
    if capture_amount_sequence is not None:
        if len(capture_amount_sequence) != period_count:
            raise ValueError("Capture sequence length must equal the period count")
        values = list(capture_amount_sequence)
    elif normalize_capture_target and curve in ("exponential", "logarithmic"):
        return generate_capture_targets(
            period_count, maximum_capture_amount, curve, coefficient,
            normalize_capture_target=True)
    else:
        if not math.isfinite(maximum_capture_amount) or maximum_capture_amount < 0:
            raise ValueError("maximum_capture_amount must be finite and nonnegative")
        if curve == "linear":
            if coefficient is not None:
                raise ValueError("The linear curve does not use a coefficient")
        elif curve in ("exponential", "logarithmic"):
            if (coefficient is None or not math.isfinite(coefficient) or coefficient == 0
                    or (curve == "logarithmic" and coefficient <= -1)):
                raise ValueError("Invalid capture curve coefficient")
        else:
            raise ValueError(f"Unknown capture curve: {curve}")
        values = []
        for t in range(1, period_count + 1):
            z = t / period_count
            if curve == "linear":
                fraction = z
            elif curve == "logarithmic":
                fraction = math.log1p(coefficient * z) / math.log1p(coefficient)
            elif coefficient > 0:
                fraction = (math.exp(coefficient * (z - 1))
                            * -math.expm1(-coefficient * z) / -math.expm1(-coefficient))
            else:
                fraction = math.expm1(coefficient * z) / math.expm1(coefficient)
            values.append(round(maximum_capture_amount * fraction, 3))
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Capture targets must be finite and nonnegative")
    return {t: float(value) for t, value in enumerate(values, 1)}


def aggregate_e_select_data(network: NetworkData, costs: CostData, diameters,
                            pipeline_install_limit, *, target) -> ESelectData:
    """校验网络与多期数据，并聚合为静态模型使用的 ESelectData。

    静态目标由调用方显式传入，公共模块不保存求解器的配置默认值。
    建设费用按各期起剩余累计需求加权，运行费用取各期算术平均；
    全部时期目标为零时费用仍设为零，不影响独立设置的静态目标。
    聚合费用保存到新字典，不修改原始网络或费用数据。
    """
    periods, diameters = tuple(costs.periods), tuple(diameters)
    if not periods or periods != tuple(range(1, len(periods) + 1)):
        raise ValueError("Periods must be consecutive integers starting at 1")
    if not diameters or len(set(diameters)) != len(diameters):
        raise ValueError("Diameters must be nonempty and unique")
    capacities = [network.pipe_capacity[d] for d in diameters]
    if (any(not math.isfinite(q) or q < 0 for q in capacities)
            or any(a >= b for a, b in zip(capacities, capacities[1:]))):
        raise ValueError("Diameter capacities must be finite, nonnegative and increasing")
    nodes = tuple(dict.fromkeys([*network.sources, *network.sinks, *network.intersections]))
    node_set = set(nodes)
    if len(set(network.ij)) != len(network.ij) or len(set(network.ig)) != len(network.ig):
        raise ValueError("Duplicate arc or source-unit indices")
    if any(u not in node_set or v not in node_set for u, v in network.ij):
        raise ValueError("Arc endpoints must belong to the physical node set")
    expected_units = [(i, g) for i in network.sources for g in network.source_units[i]]
    if len(expected_units) != len(network.ig) or set(expected_units) != set(network.ig):
        raise ValueError("Source-unit indices disagree with source_units")
    if any(not math.isfinite(network.source_capacity[k]) or network.source_capacity[k] < 0
           for k in network.ig):
        raise ValueError("Source capacities must be finite and nonnegative")
    durations = {t: float(costs.durations[t]) for t in periods}
    targets = {t: float(costs.capture_target[t]) for t in periods}
    if any(not math.isfinite(v) or v <= 0 for v in durations.values()):
        raise ValueError("Durations must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in targets.values()):
        raise ValueError("Capture targets must be finite and nonnegative")
    pipes = tuple((u, v, d) for u, v in network.ij for d in diameters)
    indices = dict(s=tuple(network.ig), r=tuple(network.sinks),
                   w=tuple(network.sinks), y=pipes, a=tuple(network.ig),
                   b=tuple(network.sinks), x=pipes)
    peak = max(targets.values())
    target = float(target)
    if not math.isfinite(target) or target < 0:
        raise ValueError("MAX_CAPTURE_AMOUNT must be finite and nonnegative")
    weights = {}
    if peak > 0:
        tails, tail = {}, 0.0
        for t in reversed(periods):
            tail += durations[t] * (targets[t] / peak)
            tails[t] = tail
        total = math.fsum(tails.values())
        weights = {t: tails[t] / total for t in periods}
    coefficients = {}
    for group in GROUPS:
        original = getattr(costs, COST_FIELDS[group])
        coefficients[group] = {}
        for key in indices[group]:
            base = (key,) if group in ("r", "w", "b") else key
            values = [original[(*base, t)] for t in periods]
            if any(not math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite {COST_FIELDS[group]} coefficient: {key}")
            # 全零时期目标时代理权重无定义，保留零费用；静态捕集目标独立设置。
            coefficients[group][key] = (math.fsum(
                (weights[t] if group in BUILD_GROUPS else 1 / len(periods)) * value
                for t, value in zip(periods, values)) if peak > 0 else 0.0)
    return ESelectData(
        network, nodes, diameters, indices, read_pipeline_limit(pipeline_install_limit),
        read_well_capacities(network, costs),
        read_well_injection_capacities(network.well_capacity, network.sinks),
        periods, durations, targets, peak, target, weights, coefficients,
        deepcopy(getattr(costs, "model_parameters", {})))


def proxy_cost_breakdown(data, solution):
    """按聚合系数与解值重算七项代理费用并返回明细，不额外乘年数或折现。"""

    return {COST_FIELDS[g]: math.fsum(data.coefficients[g][k] * solution[g][k]
                                     for k in data.indices[g]) for g in GROUPS}


def audit_e_select_solution(data, solution, *, reported_objective=None,
                            feasibility_tol=1e-7, integer_tol=1e-7, objective_ulps=8):
    """独立校验静态解的索引完整性、变量取值、全部约束及代理目标值。

    缺失索引不补零；若提供求解器目标值，还需与重新计算的费用一致。
    返回有效标志、最大约束残差和违规信息，仅在校验通过时返回费用明细。
    objective_ulps 只控制目标重算的浮点容差；物理约束仍使用 feasibility_tol。
    """
    audit = dict(valid=False, cost=None, cost_breakdown={}, max_residual=0.0,
                 violations=[])
    errors = audit["violations"]

    def fail(message):
        """记录一条审计违规信息，最多保留前 20 条。"""
        if len(errors) < 20:
            errors.append(message)

    if not isinstance(solution, dict) or set(solution) != set(GROUPS):
        fail("No complete seven-group static solution")
        return audit
    for group in GROUPS:
        values = solution[group]
        if not isinstance(values, dict) or set(values) != set(data.indices[group]):
            fail(f"Missing or unexpected indices in {group}")
            continue
        for key, value in values.items():
            if not math.isfinite(value) or value < -feasibility_tol:
                fail(f"Nonfinite or negative {group}{key}")
            elif group in BUILD_GROUPS:
                if abs(value - round(value)) > integer_tol:
                    fail(f"Noninteger {group}{key}")
                if group in ("s", "r") and value > 1 + feasibility_tol:
                    fail(f"Nonbinary {group}{key}")
    if errors:
        return audit

    def upper(lhs, rhs, label):
        """检查 lhs <= rhs，更新最大违反量，并记录超出允许误差的违规。"""
        if not math.isfinite(lhs) or not math.isfinite(rhs):
            fail(f"Nonfinite arithmetic: {label}")
            return
        residual = max(0.0, lhs - rhs)
        audit["max_residual"] = max(audit["max_residual"], residual)
        if residual > feasibility_tol:
            fail(f"{label}: {lhs} > {rhs}")

    s, r, w, y, a, b, x = (solution[g] for g in GROUPS)
    n = data.network
    balance = dict.fromkeys(data.nodes, 0.0)
    for i, g in n.ig:
        upper(a[i, g], n.source_capacity[i, g] * s[i, g], f"Capture capacity {i,g}")
        balance[i] -= a[i, g]
    for j in n.sinks:
        upper(w[j], data.maximum_wells[j] * r[j], f"Site-well link {j}")
        upper(w[j], data.maximum_wells[j], f"Maximum wells {j}")
        upper(b[j], data.well_capacity[j] * w[j], f"Injection capacity {j}")
        balance[j] += b[j]
    for u, v in n.ij:
        upper(math.fsum(y[u, v, d] for d in data.diameters),
              data.pipeline_limit, f"All-diameter pipe count {u,v}")
        for d in data.diameters:
            flow = x[u, v, d]
            upper(flow, n.pipe_capacity[d] * y[u, v, d], f"Pipe capacity {u,v,d}")
            balance[u] += flow
            balance[v] -= flow
    upper(data.target, math.fsum(a.values()), "Economic capture target")
    for node, residual in balance.items():
        audit["max_residual"] = max(audit["max_residual"], abs(residual))
        if not math.isfinite(residual) or abs(residual) > feasibility_tol:
            fail(f"Node balance {node}: {residual}")
    breakdown = proxy_cost_breakdown(data, solution)
    objective = math.fsum(breakdown.values())
    if not math.isfinite(objective):
        fail("Nonfinite proxy cost")
    if reported_objective is not None:
        if not math.isfinite(reported_objective):
            fail("Nonfinite reported proxy objective")
        elif abs(objective - reported_objective) > max(
                feasibility_tol, objective_ulps * max(math.ulp(objective), math.ulp(reported_objective))):
            fail("Reported objective disagrees with seven proxy cost sums")
    if not errors:
        audit.update(valid=True, cost=objective, cost_breakdown=breakdown)
    return audit


# ==================== 后端数据视图与求解辅助 ====================


def _coerce_esm_view(esm_data: Any, *, integer_tol: float = 1e-6) -> _ESMView:
    """Copy and validate the canonical aggregate interface without guessing aliases."""
    if isinstance(esm_data, _ESMView):
        esm_data = vars(esm_data)
    if not isinstance(esm_data, Mapping):
        raise TypeError("esm_data must be an aggregated ESM mapping")
    data = esm_data
    required = ("sources", "sinks", "intersections", "arcs", "diameters",
                "source_units", "source_capacity", "pipe_capacity", "max_wells",
                "well_capacity", "source_bounds", "well_bounds", "pipe_bounds",
                "targets", "coefficients")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"esm_data missing fields: {missing}")
    sources, sinks, intersections = (tuple(data[k]) for k in required[:3])
    if any(len(group) != len(set(group)) for group in (sources, sinks, intersections)):
        raise ValueError("source, sink and intersection sets must each be unique")
    # E-Select permits capture and injection at the same physical node.
    nodes = tuple(dict.fromkeys((*sources, *sinks, *intersections)))
    node_set = set(nodes)
    if "nodes" in data and set(data["nodes"]) != node_set:
        raise ValueError("nodes must equal the source, sink and intersection union")
    arcs = tuple(tuple(edge) for edge in data["arcs"])
    if len(arcs) != len(set(arcs)):
        raise ValueError("duplicate directed arcs")
    if any(len(e) != 2 or e[0] == e[1] or e[0] not in node_set or e[1] not in node_set for e in arcs):
        raise ValueError("every directed arc must join distinct physical nodes")
    diameters = tuple(data["diameters"])
    if not diameters or len(diameters) != len(set(diameters)):
        raise ValueError("diameters must be nonempty and unique")

    def values(name, keys, integer=False):
        raw = data[name]
        if not isinstance(raw, Mapping):
            raise TypeError(f"{name} must be a mapping")
        result = {}
        for key in keys:
            if key not in raw:
                raise ValueError(f"{name} missing index {key!r}")
            label = f"{name}[{key!r}]"
            result[key] = (_as_integer_limit(raw[key], label, integer_tol)
                           if integer else _finite_nonnegative(raw[key], label))
        return result

    source_units = {}
    for i in sources:
        if i not in data["source_units"]:
            raise ValueError(f"source_units missing source {i!r}")
        units = tuple(data["source_units"][i])
        if len(units) != len(set(units)):
            raise ValueError(f"source_units[{i!r}] contains duplicates")
        source_units[i] = units
    ig = tuple((i, g) for i in sources for g in source_units[i])
    edges = tuple((u, v, d) for u, v in arcs for d in diameters)
    source_capacity = values("source_capacity", ig)
    pipe_capacity = values("pipe_capacity", diameters)
    max_wells = values("max_wells", sinks, True)
    source_bounds = values("source_bounds", sources, True)
    well_bounds = values("well_bounds", sinks, True)
    pipe_bounds = values("pipe_bounds", arcs, True)
    if any(source_bounds[i] > len(source_units[i]) for i in sources):
        raise ValueError("source_bounds exceeds the number of candidate source units")
    if any(well_bounds[j] > max_wells[j] for j in sinks):
        raise ValueError("well_bounds exceeds max_wells")
    coefficients = {}
    raw_coefficients = data["coefficients"]
    for name, keys in (("Fs", ig), ("Vs", ig), ("Fr", sinks), ("Fw", sinks),
                       ("Vr", sinks), ("Fp", edges), ("Vp", edges)):
        if name not in raw_coefficients or not isinstance(raw_coefficients[name], Mapping):
            raise ValueError(f"coefficients missing mapping {name}")
        mapping = raw_coefficients[name]
        coefficients[name] = {}
        for key in keys:
            if key not in mapping:
                raise ValueError(f"coefficient {name} missing index {key!r}")
            coefficients[name][key] = _finite_nonnegative(mapping[key], f"{name}[{key!r}]")
    targets = _coerce_targets(data["targets"])
    return _ESMView(sources, sinks, intersections, nodes, arcs, diameters,
                    source_units, source_capacity, pipe_capacity, max_wells,
                    read_well_injection_capacities(data["well_capacity"], sinks),
                    source_bounds, well_bounds, pipe_bounds, targets, coefficients)


def _flow_tolerance(data: _ESMView, options: Any) -> float:
    return options.feasibility_tol * (min(1.0, data.K) if data.K > 0 else 1.0)


def _audit_esm_solution(data: _ESMView, solution: Mapping, *, options: Any,
                        reported_objective: Optional[float] = None,
                        require_integer: bool = True) -> dict:
    """Certify a solution against a validated data view and validated options."""
    result = {"valid": False, "cost": None, "cost_breakdown": {},
              "max_residual": None, "max_constraint_residual": None, "violations": []}
    stage = "options"
    try:
        opts = options
        if not isinstance(require_integer, bool):
            raise TypeError("require_integer must be bool")
        if reported_objective is not None:
            reported_objective = float(reported_objective)
            if not math.isfinite(reported_objective):
                raise ValueError("reported_objective must be finite")
        stage = "solution"
        sol, errors = _normalize_audit_solution(data, solution)
        if errors:
            result["violations"] = errors
            return result
        flow_tol = _flow_tolerance(data, opts)
        max_residual = 0.0

        def upper(lhs, rhs, name, keys=None, tolerance=flow_tol):
            nonlocal max_residual
            residual = lhs - rhs
            # A finite difference also implies finite float operands here.
            finite = math.isfinite(residual)
            if not finite or residual > tolerance:
                label = name if keys is None else f"{name}[{','.join(map(repr, keys))}]"
                if not finite:
                    raise ValueError(f"{label}: nonfinite arithmetic")
                errors.append(f"{label}: residual {residual:.12g} > {tolerance:.12g}")
            if residual > max_residual:
                max_residual = residual

        stage = "domains"
        for group in GROUPS:
            construction = group in CONSTRUCTION_GROUPS
            binary = group in ("s", "r")
            tol = opts.integer_tol if construction else flow_tol
            for key, value in sol[group].items():
                # Normalization has already checked every value for finiteness.
                if value < 0.0:
                    if -value > tol:
                        errors.append(f"{group}[{key!r}] nonnegative: "
                                      f"residual {-value:.12g} > {tol:.12g}")
                    errors.append(f"{group}[{key!r}]: negative values cannot certify a solution")
                if binary and value > 1.0:
                    residual = value - 1.0
                    if residual > max_residual:
                        max_residual = residual
                    if residual > opts.integer_tol:
                        errors.append(f"{group}[{key!r}] binary upper: "
                                      f"residual {residual:.12g} > {opts.integer_tol:.12g}")
                if require_integer and construction:
                    integer_value = round(value)
                    if value != integer_value:
                        residual = abs(value - integer_value)
                        if residual > opts.integer_tol:
                            errors.append(f"{group}[{key!r}] integer: "
                                          f"residual {residual:.12g} > {opts.integer_tol:.12g}")
                        errors.append(f"{group}[{key!r}]: exact integer required; "
                                      "external solutions are not implicitly rounded")
        if errors:
            result["violations"] = errors
            return result
        stage = "constraints"
        balance = {v: 0.0 for v in data.nodes}
        for i, g in data.ig:
            upper(sol["a"][i, g], data.source_capacity[i, g] * sol["s"][i, g],
                  "source_capacity", (i, g))
            balance[i] -= sol["a"][i, g]
        for i in data.sources:
            upper(sum(sol["s"][i, g] for g in data.source_units[i]),
                  data.source_bounds[i], "source_quota", (i,), opts.integer_tol)
        for j in data.sinks:
            upper(sol["b"][j], data.well_capacity[j] * sol["w"][j], "sink_capacity", (j,))
            upper(sol["w"][j], data.well_bounds[j] * sol["r"][j],
                  "well_link", (j,), opts.integer_tol)
            upper(sol["w"][j], data.well_bounds[j], "well_upper", (j,), opts.integer_tol)
            balance[j] += sol["b"][j]
        for u, v in data.arcs:
            upper(sum(sol["y"][u, v, d] for d in data.diameters), data.pipe_bounds[u, v],
                  "pipe_quota", (u, v), opts.integer_tol)
            for d in data.diameters:
                flow = sol["x"][u, v, d]
                upper(flow, data.pipe_capacity[d] * sol["y"][u, v, d],
                      "pipe_capacity", (u, v, d))
                balance[u] += flow
                balance[v] -= flow
        for v, difference in balance.items():
            upper(abs(difference), 0.0, "balance", (v,))
        total_capture = sum(sol["a"].values())
        upper(abs(total_capture - sum(sol["b"].values())), 0.0, "supply_demand")
        upper(data.K, total_capture, "capture_target")
        result["max_residual"] = result["max_constraint_residual"] = max_residual
        result["violations"] = errors
        if errors:
            return result
        stage = "cost"
        breakdown = _cost_breakdown(data, sol)
        if not all(math.isfinite(value) for value in breakdown.values()):
            raise ValueError("cost breakdown contains nonfinite arithmetic")
        cost = breakdown["total"]
        if reported_objective is not None:
            scale = max(1.0, abs(cost), abs(reported_objective))
            if abs(cost / scale - reported_objective / scale) > opts.objective_tol:
                result["violations"].append(f"objective mismatch: {cost} vs {reported_objective}")
                return result
        result.update(valid=True, cost=cost, cost_breakdown=breakdown)
    except Exception as exc:
        result["violations"].append(f"{stage}: {type(exc).__name__}: {exc}")
    return result


@dataclass(frozen=True)
class _ESMView:
    """Validated view of one aggregated terminal ESM instance."""

    sources: tuple
    sinks: tuple
    intersections: tuple
    nodes: tuple
    arcs: tuple
    diameters: tuple
    source_units: dict
    source_capacity: dict
    pipe_capacity: dict
    max_wells: dict
    well_capacity: dict
    source_bounds: dict
    well_bounds: dict
    pipe_bounds: dict
    targets: dict
    coefficients: dict

    @property
    def ig(self):
        return tuple((i, g) for i in self.sources for g in self.source_units[i])

    @property
    def edges(self):
        return tuple((v, z, d) for v, z in self.arcs for d in self.diameters)

    @property
    def K(self):
        return max(self.targets.values(), default=0.0)

    @property
    def solution_indices(self):
        return dict(s=self.ig, r=self.sinks, w=self.sinks, y=self.edges,
                    a=self.ig, b=self.sinks, x=self.edges)

    @property
    def solution_index_sets(self):
        ig, sinks, edges = frozenset(self.ig), frozenset(self.sinks), frozenset(self.edges)
        return dict(s=ig, r=sinks, w=sinks, y=edges, a=ig, b=sinks, x=edges)


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _as_integer_limit(value: Any, label: str, tol: float = 1e-6) -> int:
    number = _finite_nonnegative(value, label)
    rounded = round(number)
    if abs(number - rounded) > tol:
        raise ValueError(f"{label} must be an integer-valued quota")
    return int(rounded)


def _coerce_targets(raw: Any) -> dict:
    if raw is None:
        raise ValueError("esm_data is missing targets")
    if isinstance(raw, Mapping):
        items = list(raw.items())
    else:
        if isinstance(raw, (str, bytes)):
            raise ValueError("capture target sequence must be numeric")
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
        items = list(enumerate(values, start=1))
    if not items:
        raise ValueError("capture target sequence must not be empty")
    targets = {key: _finite_nonnegative(value, f"targets[{key!r}]")
               for key, value in items}
    try:
        ordered_keys = sorted(targets)
    except TypeError:
        ordered_keys = list(targets)
    ordered = [targets[key] for key in ordered_keys]
    if any(left > right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("targets must be nonnegative and nondecreasing")
    return targets


def _cost_breakdown(data: _ESMView, solution: Mapping) -> dict:
    c = data.coefficients
    result = {
        "source_fixed": sum(c["Fs"][i, g] * solution["s"][i, g]
                             for i, g in data.ig),
        "source_variable": sum(c["Vs"][i, g] * solution["a"][i, g]
                                for i, g in data.ig),
        "sink_fixed": sum(c["Fr"][j] * solution["r"][j] for j in data.sinks),
        "well_fixed": sum(c["Fw"][j] * solution["w"][j] for j in data.sinks),
        "sink_variable": sum(c["Vr"][j] * solution["b"][j] for j in data.sinks),
        "pipe_fixed": sum(c["Fp"][v, z, d] * solution["y"][v, z, d]
                           for v, z, d in data.edges),
        "pipe_variable": sum(c["Vp"][v, z, d] * solution["x"][v, z, d]
                              for v, z, d in data.edges),
    }
    result["construction"] = (result["source_fixed"] + result["sink_fixed"] +
                               result["well_fixed"] + result["pipe_fixed"])
    result["operation"] = (result["source_variable"] + result["sink_variable"] +
                            result["pipe_variable"])
    result["total"] = result["construction"] + result["operation"]
    result["source_construction"] = result["source_fixed"]
    result["sink_construction"] = result["sink_fixed"] + result["well_fixed"]
    result["transport_construction"] = result["pipe_fixed"]
    return result


def _resource_loads(data: _ESMView, solution: Mapping) -> dict:
    return {
        "alpha": {i: sum(float(solution["a"][i, g])
                         for g in data.source_units[i]) for i in data.sources},
        "beta": {j: float(solution["b"][j]) for j in data.sinks},
        "xi": {(v, z): sum(float(solution["x"][v, z, d])
                            for d in data.diameters) for v, z in data.arcs},
    }


def _normalize_audit_solution(data: _ESMView, solution: Mapping
                              ) -> tuple[Optional[dict], list[str]]:
    """Validate the exact index sets and copy finite values for arithmetic."""

    if not isinstance(solution, Mapping):
        return None, ["solution must be a mapping of variable groups"]
    expected = data.solution_indices
    errors = []
    groups = set(solution)
    missing = set(GROUPS) - groups
    extra = groups - set(GROUPS)
    if missing:
        errors.append(f"solution: missing groups {sorted(missing, key=repr)!r}")
    if extra:
        errors.append(f"solution: extra groups {sorted(extra, key=repr)!r}")
    normalized = {}
    for group in GROUPS:
        if group not in solution:
            continue
        values = solution[group]
        if not isinstance(values, Mapping):
            errors.append(f"solution.{group} must be an index-value mapping")
            continue
        indices = set(values)
        expected_indices = data.solution_index_sets[group]
        if indices != expected_indices:
            missing = expected_indices - indices
            extra = indices - expected_indices
            if missing:
                errors.append(f"solution.{group}: missing indices {sorted(missing, key=repr)!r}")
            if extra:
                errors.append(f"solution.{group}: extra indices {sorted(extra, key=repr)!r}")
            continue
        normalized[group] = {}
        for key in expected[group]:
            try:
                value = float(values[key])
            except (TypeError, ValueError, OverflowError):
                errors.append(f"solution.{group}[{key!r}]: nonnumeric or outside the finite float range")
                continue
            if not math.isfinite(value):
                errors.append(f"solution.{group}[{key!r}]: nonfinite")
                continue
            normalized[group][key] = value
    return (None if errors else normalized), errors


def _eta(data):
    return min(1.0, data.K) if data.K > 0 else 1.0


def _remaining(deadline):
    return max(0.0, deadline - perf_counter())


@contextmanager
def _building_model(model):
    """Release a failed build; successful models remain owned by their callers."""
    try:
        yield model
    except BaseException:
        model.dispose()
        raise


def _expired(deadline):
    return deadline is not None and perf_counter() >= deadline


def _finite_bound(value):
    import gurobipy as gp

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and abs(value) < gp.GRB.INFINITY / 2 else None


def _status_name(status):
    import gurobipy as gp

    names = {gp.GRB.OPTIMAL: "OPTIMAL", gp.GRB.INFEASIBLE: "INFEASIBLE",
             gp.GRB.INF_OR_UNBD: "INF_OR_UNBD", gp.GRB.UNBOUNDED: "UNBOUNDED",
             gp.GRB.TIME_LIMIT: "TIME_LIMIT", gp.GRB.INTERRUPTED: "INTERRUPTED",
             gp.GRB.NUMERIC: "NUMERIC", gp.GRB.SUBOPTIMAL: "SUBOPTIMAL"}
    return names.get(status, f"STATUS_{status}")


def _new_environment():
    import gurobipy as gp

    env = gp.Env(empty=True)
    try:
        env.setParam("OutputFlag", 0)
        env.start()
    except BaseException:
        env.dispose()
        raise
    return env


def with_solver_log(solve):
    """将一次求解及其子问题的消息集中到返回结果，不使用磁盘日志。"""
    @wraps(solve)
    def wrapped(*args, **kwargs):
        messages = []
        token = _SOLVER_LOG.set(messages)
        try:
            result = solve(*args, **kwargs)
            result["solver_log"] = messages
            return result
        finally:
            _SOLVER_LOG.reset(token)
    return wrapped


def optimize_esm_model(model, callback=None, *, stage=None):
    """捕获 Gurobi MESSAGE 回调，同时保留现有算法回调及控制台设置。"""
    import gurobipy as gp

    messages = _SOLVER_LOG.get()
    if messages is None:
        messages = []
    output_flag, console = model.Params.OutputFlag, model.Params.LogToConsole
    model.Params.LogToConsole = 0
    model.Params.LogFile = ""
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = console if output_flag else 0
    started = perf_counter()

    def record(current, where):
        if where == gp.GRB.Callback.MESSAGE:
            messages.append(dict(stage=stage, elapsed=perf_counter() - started,
                                 message=current.cbGet(gp.GRB.Callback.MSG_STRING)))
        if callback is not None:
            callback(current, where)

    try:
        optimize_with_log(model, record)
    finally:
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = output_flag
        model.Params.LogToConsole = console


def _configure(model, opts):
    model.Params.OutputFlag = opts.output_flag
    model.Params.LogToConsole = opts.output_flag
    model.Params.LogFile = ""
    if opts.threads is not None:
        model.Params.Threads = opts.threads
    if opts.seed is not None:
        model.Params.Seed = opts.seed
    model.Params.FeasibilityTol = max(GUROBI_TOL_FLOOR, min(1e-2, opts.feasibility_tol))
    model.Params.OptimalityTol = max(GUROBI_TOL_FLOOR, min(1e-2, opts.dual_tolerance))
    model.Params.IntFeasTol = max(GUROBI_TOL_FLOOR, min(1e-1, opts.integer_tol))


def _gap_fields(ub, lb):
    if ub is None or lb is None:
        return dict(gap=None, gap_absolute=None, gap_relative=None,
                    gap_denominator=None, bound_difference=None)
    difference = ub - lb
    absolute = max(0.0, difference)
    denominator = max(abs(ub), 1e-12)
    return dict(gap=absolute / denominator, gap_absolute=absolute,
                gap_relative=absolute / denominator, gap_denominator=denominator,
                bound_difference=difference)


def to_solver_data(data):
    """把 ESM 已聚合数据原样映射到后端的七项费用和物理容量接口。"""
    n = data.network
    return dict(
        problem="E-Select", sources=tuple(n.sources), sinks=tuple(n.sinks),
        intersections=tuple(n.intersections), nodes=data.nodes,
        arcs=tuple(n.ij), diameters=data.diameters,
        source_units={i: tuple(n.source_units[i]) for i in n.sources},
        source_capacity=dict(n.source_capacity),
        pipe_capacity={d: n.pipe_capacity[d] for d in data.diameters},
        max_wells=dict(data.maximum_wells), well_capacity=dict(data.well_capacity),
        source_bounds={i: len(n.source_units[i]) for i in n.sources},
        well_bounds=dict(data.maximum_wells),
        pipe_bounds={edge: data.pipeline_limit for edge in n.ij},
        targets={1: data.target},
        coefficients={name: dict(data.coefficients[group]) for name, group in
                      (("Fs", "s"), ("Fr", "r"), ("Fw", "w"), ("Fp", "y"),
                       ("Vs", "a"), ("Vr", "b"), ("Vp", "x"))})


def audited_result(data, result, *, audit=audit_e_select_solution):
    """算法返回结果须再次通过本地静态审计，统一代理目标与证书范围。"""
    result = dict(result)
    checked = audit(data, result.get("solution"))
    if not checked["valid"] and result.get("feasible"):
        result["status"] = "INVALID_INCUMBENT"
    objective = checked["cost"]
    result.update(
        problem="E-Select", scope="E_SELECT_PROXY_OBJECTIVE",
        objective_scope="E-Select static proxy model only; not original CCS_FNE UB/LB",
        original_lower_bound=None, original_gap=None,
        feasible=checked["valid"], objective=objective, e_select_objective=objective,
        upper_bound=objective, solution=result.get("solution") if checked["valid"] else None,
        cost_breakdown=checked["cost_breakdown"], max_constraint_residual=checked["max_residual"],
        peak_target=data.peak, economic_target=data.target)
    bound = result.get("lower_bound")
    result["gap"] = (None if objective is None or bound is None else
                     0.0 if objective == bound else
                     abs(objective - bound) / abs(objective) if objective else None)
    result.setdefault("algorithm_info", {}).update(
        scope="E_SELECT_PROXY_OBJECTIVE", e_select_audit=checked)
    return result


# ==================== 参数、日志与工作簿 ====================


MODEL_SETTINGS = (
    "INPUT_DIR", "PERIOD_NUM", "NP", "DIAMETERS", "CURVE", "COEFFICIENT",
    "MAX_CAPTURE_AMOUNT", "CAPTURE_AMOUNT_SEQUENCE", "NORMALIZE_CAPTURE_TARGET",
)


def _flatten(values, prefix=""):
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            yield from _flatten(value, name)
        else:
            yield name, value


def record_e_select_metadata(result, data):
    """Use solved data and saved options, never the exporting module's globals."""
    requested = result.get("run_parameters", {})
    model = deepcopy(getattr(data, "model_parameters", {}))
    model.update({name.lower(): requested[name] for name in MODEL_SETTINGS if name in requested})
    model.update(
        period_num=len(data.periods), periods=list(data.periods),
        durations=dict(data.durations), total_years=sum(data.durations.values()),
        np=data.pipeline_limit, diameters=list(data.diameters), D=len(data.diameters),
        capture_target=dict(data.targets), static_capture_target=data.target,
        peak_capture_target=data.peak, construction_weights=dict(data.weights),
        source_count=len(data.network.sources), sink_count=len(data.network.sinks),
        arc_count=len(data.network.ij), source_unit_count=len(data.network.ig),
        objective_scope=result.get("objective_scope"),
        flow_unit="kt_CO2/year", cost_unit="thousand_USD")
    algorithm = dict(result.get("algorithm_parameters", {}))
    algorithm.update(method=result.get("algorithm"))
    algorithm.setdefault("effective_options", result.get("options", {}))
    for name in ("requested_options", "TIME_LIMIT", "TOTAL_TIME_LIMIT", "OUTPUT_FLAG",
                 "GUROBI_OUTPUT_FLAG", "INITIAL_SOLUTION_FILE"):
        if name in requested:
            key = ("requested_" if name in ("TIME_LIMIT", "TOTAL_TIME_LIMIT") else "") + name.lower()
            algorithm[key] = requested[name]
    if "solve_mode" in result:
        algorithm["solve_mode"] = result["solve_mode"]
    if "sspair_result" in result:
        algorithm["sspair_options"] = result["sspair_result"].get("options", {})
    result["model_parameters"] = deepcopy(model)
    result["algorithm_parameters"] = deepcopy(algorithm)


def e_select_metadata_frames(result):
    """Keep raw solver messages and structured events with their stage of origin."""
    records = []
    phases = ([("sspair", result["sspair_result"]), ("full_esm", result)]
              if "sspair_result" in result else [(result.get("algorithm"), result)])
    for phase, stage_result in phases:
        for source in ("solver_log", "history", "process_log", "iteration_history", "bound_history"):
            for index, row in enumerate(stage_result.get(source, []), 1):
                values = dict(row) if isinstance(row, Mapping) else {"message": row}
                records.append(dict(values, phase=phase, record_type=source, record_index=index))
        records.append(dict(phase=phase, record_type="termination", status=stage_result.get("status"),
                            gurobi_status=stage_result.get("gurobi_status"),
                            termination_reason=stage_result.get("termination_reason"),
                            elapsed=stage_result.get("elapsed")))
    return {
        "model_parameters": parameter_frame(dict(_flatten(result.get("model_parameters", {})))),
        "algorithm_parameters": parameter_frame(dict(_flatten(result.get("algorithm_parameters", {})))),
        "solve_log": pd.DataFrame(records),
    }


_RESULT_GROUP_INDEX_COLUMNS = {
    "s": ("source", "unit"), "r": ("sink",), "w": ("sink",),
    "y": ("start", "end", "diameter"), "a": ("source", "unit"),
    "b": ("sink",), "x": ("start", "end", "diameter"),
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _excel_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        value = asdict(value)
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _excel_scalar(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if isinstance(value, (Mapping, tuple, list)):
        text = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    if ILLEGAL_CHARACTERS_RE.search(text):
        text = json.dumps(text, ensure_ascii=False)
    return text if len(text) <= EXCEL_TEXT_LENGTH else "[Full value in result_data]"


def _result_variable_frame(result: Mapping, group: str, pandas: Any) -> Any:
    columns = list(_RESULT_GROUP_INDEX_COLUMNS[group]) + ["value"]
    values = result.get("solution", {}).get(group, {}) if result.get("solution") else {}
    rows = []
    for key, value in values.items():
        parts = list(key) if isinstance(key, tuple) else [key]
        rows.append([_excel_scalar(parts[i]) if i < len(parts) else None
                     for i in range(len(columns) - 1)] + [_excel_scalar(value)])
    return pandas.DataFrame(rows, columns=columns)


def _result_summary_frame(result: Mapping, pandas: Any) -> Any:
    keys = ("problem", "algorithm", "status", "gurobi_status", "termination_reason",
            "feasible", "psp_objective", "psp_lower_bound", "lower_bound_source",
            "gap", "gap_absolute", "gap_relative", "bound_difference",
            "upper_bound", "lower_bound", "gap_denominator",
            "max_constraint_residual", "elapsed",
            "solve_elapsed", "preprocessing_elapsed", "export_elapsed", "export_errors")
    keys += tuple(key for key, value in result.items()
                  if key not in keys and not isinstance(value, (Mapping, tuple, list)))
    rows = [{"section": "run", "name": key,
             "value": _excel_scalar(result.get(key))}
            for key in keys if key in result]
    for key, value in result.get("cost_breakdown", {}).items():
        rows.append({"section": "cost_breakdown", "name": key,
                     "value": _excel_scalar(value)})
    return pandas.DataFrame(rows, columns=["section", "name", "value"])


def _diagnostic_rows(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        if not value:
            yield prefix, None
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _diagnostic_rows(item, child)
    else:
        yield prefix, _excel_scalar(value)


EXCEL_DATA_ROWS = 1048575  # 每个工作表另占一行表头。


EXCEL_TEXT_LENGTH = 32767


class _UTF8WorksheetWriter(WorksheetWriter):
    """显式使用 UTF-8，避免旧版 lxml 默认 ASCII 写入时丢失中文文本。"""

    def get_stream(self):
        with xmlfile(self.out, encoding="utf-8") as stream:
            with stream.element("worksheet", xmlns=SHEET_MAIN_NS):
                try:
                    while True:
                        element = yield
                        if element is True:
                            yield stream
                        elif element is not None:
                            stream.write(element)
                except GeneratorExit:
                    pass


class _ESelectExcelWriter(ExcelWriter):
    """仅为本次导出的普通工作表选择 UTF-8 writer，不改动全局库设置。"""

    def write_worksheet(self, worksheet):
        worksheet._drawing = SpreadsheetDrawing()
        worksheet._drawing.charts = worksheet._charts
        worksheet._drawing.images = worksheet._images
        writer = _UTF8WorksheetWriter(worksheet)
        try:
            writer.write()
            worksheet._rels = writer._rels
            self._archive.write(writer.out, worksheet.path[1:])
            self.manifest.append(worksheet)
        finally:
            writer.cleanup()


def _complete_result_rows(value, path="result"):
    """展开全部结果，保留容器类型和空值；长文本分段，避免单元格静默截断。"""
    if is_dataclass(value):
        value = asdict(value)
    kind = type(value).__name__
    if isinstance(value, Mapping):
        yield (path, kind, 1, None)
        for key, item in value.items():
            yield from _complete_result_rows(item, f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        yield (path, kind, 1, None)
        for index, item in enumerate(value):
            yield from _complete_result_rows(item, f"{path}[{index}]")
    else:
        if isinstance(value, float) and not math.isfinite(value):
            scalar = repr(value)
        elif isinstance(value, int) and abs(value) >= 10 ** 15:
            scalar = str(value)  # Excel 数字只有 15 位有效数字。
        elif isinstance(value, (str, Path)):
            scalar = str(value)
        else:
            scalar = _excel_scalar(value)
        if isinstance(scalar, str):
            if ILLEGAL_CHARACTERS_RE.search(scalar):
                scalar = json.dumps(scalar, ensure_ascii=False)
                kind += " (escaped)"
            for offset in range(0, max(1, len(scalar)), EXCEL_TEXT_LENGTH):
                yield (path, kind, offset // EXCEL_TEXT_LENGTH + 1,
                       scalar[offset:offset + EXCEL_TEXT_LENGTH])
        else:
            yield (path, kind, 1, scalar)


def _write_excel_rows(workbook, name, columns, rows):
    """自动续表，并将日志中的等号文本写成文字而非公式。"""
    worksheet = None
    for index, row in enumerate(rows):
        if index % EXCEL_DATA_ROWS == 0:
            part = index // EXCEL_DATA_ROWS + 1
            suffix = "" if part == 1 else f"_{part}"
            worksheet = workbook.create_sheet(name[:31 - len(suffix)] + suffix)
            worksheet.append(list(columns))
        worksheet.append([_excel_scalar(value) for value in row])
        row_number = index % EXCEL_DATA_ROWS + 2
        for column in range(1, len(columns) + 1):
            cell = worksheet.cell(row_number, column)
            if isinstance(cell.value, str):
                # 完整原文已在 result_data 分段保存；普通表的超长字段指向该表。
                cell.data_type = "s"
    if worksheet is None:
        worksheet = workbook.create_sheet(name)
        worksheet.append(list(columns))


def _save_e_select_workbook(result: Mapping, path: str | Path, *, extra_sheets=None, history_sheets=()) -> Path:
    """保存 E-Select 变量、诊断和完整结果，并保留现有工作簿布局。"""
    started = perf_counter()
    result = result if isinstance(result, dict) else dict(result)
    result.setdefault("export_errors", [])
    result["export_elapsed"] = 0.0
    result["export_timing"] = "workbook generation; excludes final metadata update and file write"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames = {"Summary": _result_summary_frame(result, pd)}
    frames.update(e_select_metadata_frames(result))
    for group in GROUPS:
        frames[f"variable_{group}"] = _result_variable_frame(result, group, pd)
    diagnostics = []
    for section in ("run_parameters", "options", "algorithm_info", "diagnostics"):
        for name, value in _diagnostic_rows(result.get(section, {}), section):
            diagnostics.append({"section": section, "name": name, "value": value})
    frames["diagnostics"] = pd.DataFrame(diagnostics,
                                          columns=["section", "name", "value"])
    loads = []
    for resource, values in result.get("resource_loads", {}).items():
        for key, value in values.items():
            parts = list(key) if isinstance(key, tuple) else [key]
            loads.append({"resource": resource,
                          "index_1": _excel_scalar(parts[0]) if parts else None,
                          "index_2": _excel_scalar(parts[1]) if len(parts) > 1 else None,
                          "load": _excel_scalar(value)})
    frames["resource_loads"] = pd.DataFrame(
        loads, columns=["resource", "index_1", "index_2", "load"])
    frames["bound_history"] = pd.DataFrame(
        [{key: _excel_scalar(value) for key, value in row.items()}
         for row in result.get("bound_history", [])])
    for name in (*history_sheets, "history", "process_log", "solver_log", "iteration_history",
                 "restricted_history", "arc_feedback"):
        rows = result.get(name, [])
        frames[name] = pd.DataFrame(rows if rows and isinstance(rows[0], Mapping)
                                   else [{"message": row} for row in rows])
    progress = result.get("diagnostics", {}).get("progress", [])
    if progress:
        frames["progress"] = pd.DataFrame({"message": progress})
    for name, rows in (extra_sheets or {}).items():
        frames[name] = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    try:
        workbook = Workbook()
        workbook.remove(workbook.active)
        for sheet, frame in frames.items():
            _write_excel_rows(workbook, sheet, list(frame.columns) or ["no_records"],
                              frame.itertuples(index=False, name=None))
        _write_excel_rows(workbook, "result_data", ["path", "type", "part", "value"],
                          _complete_result_rows(result))
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.sheet_view.showGridLines = False
            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                font.color = "FFFFFF"
                cell.font = font
                cell.fill = PatternFill(fill_type="solid", fgColor="1F4E78")
            for cells in worksheet.columns:
                values = ["" if cell.value is None else str(cell.value) for cell in cells]
                worksheet.column_dimensions[cells[0].column_letter].width = min(
                    60, max(10, max(map(len, values), default=0) + 2))
        result["export_elapsed"] = perf_counter() - started
        for worksheet in workbook.worksheets:
            if worksheet.title == "Summary" or worksheet.title.startswith("Summary_"):
                for row in worksheet.iter_rows(min_row=2):
                    if row[1].value == "export_elapsed":
                        row[2].value = result["export_elapsed"]
            if worksheet.title == "result_data" or worksheet.title.startswith("result_data_"):
                for row in worksheet.iter_rows(min_row=2):
                    if row[0].value == "result['export_elapsed']":
                        row[3].value = result["export_elapsed"]
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            _ESelectExcelWriter(workbook, archive).write_data()
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def save_e_select_result(result, data, output_file, *, audit=audit_e_select_solution,
                         extra_sheets=None, history_sheets=()):
    """再次校验结果后导出 Excel 工作簿，拒绝导出未经验证的变量值。

    七类变量表采用静态索引加 value 列，另附各期时长、
    捕集目标与费用聚合权重，返回工作簿保存路径。
    """
    if result["feasible"]:
        audit_result = audit(data, result["solution"],
                                        reported_objective=result["objective"])
        if not audit_result["valid"]:
            raise ValueError("Refusing invalid E-Select export: " + "; ".join(audit_result["violations"]))
    elif result.get("solution") is not None:
        raise ValueError("Unverified solutions must not be exported as variable values")
    if Path(output_file).suffix.lower() != ".xlsx":
        raise ValueError("E-Select results must use an .xlsx path")
    record_e_select_metadata(result, data)
    return _save_e_select_workbook(result, output_file, extra_sheets={
        "select_parameters": [
            dict(period=t, duration=data.durations[t], capture_target=data.targets[t],
                 construction_weight=data.weights.get(t),
                 operating_weight=1 / len(data.periods) if data.peak > 0 else None)
            for t in data.periods], **(extra_sheets or {})}, history_sheets=history_sheets)


# ==================== 独立入口的数据准备与运行流程 ====================
def prepare_e_select_data(config):
    """按调用方配置准备网络和静态数据，不读取任何求解器的全局参数。"""
    network = load_network_data(config["INPUT_DIR"])
    periods, durations = period_schedule(config["PERIOD_NUM"])
    costs = compute_cost_data(network, periods, durations, apply_shared_capture_target=False)
    costs = replace(costs, capture_target=build_capture_target(
        config["PERIOD_NUM"], config["CURVE"], config["COEFFICIENT"],
        config["MAX_CAPTURE_AMOUNT"], config["CAPTURE_AMOUNT_SEQUENCE"],
        normalize_capture_target=config.get("NORMALIZE_CAPTURE_TARGET", False)))
    data = aggregate_e_select_data(network, costs, config["DIAMETERS"], config["NP"],
                                   target=config["MAX_CAPTURE_AMOUNT"])
    return network, data


def run_e_select(config, *, method, options, solve, initial_solution=None):
    """读取调用方顶部设置；预算包含准备和求解，导出另计。"""
    started = perf_counter()
    limit = options.time_limit
    if limit is not None and (not math.isfinite(limit) or limit < 0):
        raise ValueError("TIME_LIMIT must be None or finite and nonnegative")
    network, data = prepare_e_select_data(config)
    preprocessing_elapsed = perf_counter() - started
    remaining = None if limit is None else max(0.0, limit - preprocessing_elapsed)
    names = ("INPUT_DIR", "PERIOD_NUM", "NP", "DIAMETERS", "CURVE", "COEFFICIENT",
             "MAX_CAPTURE_AMOUNT", "CAPTURE_AMOUNT_SEQUENCE", "PLOT_ENABLED",
             "RESULT_FILE", "PLOT_FILE", "BOUND_PLOT_FILE")
    run_parameters = deepcopy({name: config[name] for name in names})
    run_parameters.update(
        NORMALIZE_CAPTURE_TARGET=config.get("NORMALIZE_CAPTURE_TARGET", False),
        method=method, requested_options=asdict(options), capture_target=data.targets,
        economic_target=data.target, construction_weights=data.weights,
        cost_semantics="E-Select proxy coefficients; no additional duration/discount")
    result = solve(data,
                            options=replace(options, time_limit=remaining),
                            initial_solution=initial_solution)
    result.update(preprocessing_elapsed=preprocessing_elapsed,
                  total_elapsed=perf_counter() - started)
    result["elapsed"] = result["total_elapsed"]
    result["run_parameters"] = run_parameters
    print(f"[E-Select/{method}] status={result['status']}, feasible={result['feasible']}, "
          f"objective={result['objective']}, E_model_LB={result['lower_bound']}, "
          f"E_model_gap={result['gap']}")
    result["plots"], result["export_errors"] = {}, []
    export_started = perf_counter()
    if config["PLOT_ENABLED"]:
        if result.get("bound_history"):
            try:
                from plot_bound import plot_bound_history

                path = plot_bound_history(
                    result["bound_history"], output_file=config["BOUND_PLOT_FILE"],
                    title=f"E-Select | {method} bounds", objective_label="E-Select objective value")
                result["plots"]["bounds"] = str(path)
            except Exception as error:
                result["export_errors"].append(f"Bound plot: {type(error).__name__}: {error}")
        if result["feasible"]:
            try:
                from plot_results import plot_results

                images = plot_results(result["solution"], network,
                                      diameters=config["DIAMETERS"], output_file=config["PLOT_FILE"])
                result["plots"]["construction"] = [str(path) for path in images]
            except Exception as error:
                result["export_errors"].append(f"Construction plot: {type(error).__name__}: {error}")
    result["plot_elapsed"] = perf_counter() - export_started
    for error in result["export_errors"]:
        print(error)
    path = config["save_e_select_result"](result, data, config["RESULT_FILE"])
    print(f"E-Select variables and diagnostics saved: {path}")
    return result
