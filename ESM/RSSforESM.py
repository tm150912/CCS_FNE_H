"""ESM-RSS：共同起点、未用弧价格校正与真实费用反馈阻尼。

每轮求解源汇整数决策与连续路由，再逐弧配置整数管道并审计原模型。
初始斜率采用 NP 根最大管径管道满负荷时的平均费用 fixed/Q + variable。
首轮正流量中位数固定为参考流量；未用弧也按该流量的精确选管费用定价。
活跃弧按实际流量定价，所有弧的价格均作阻尼更新；真实费用恶化时缩小步长。
微小正流量弧本轮关闭并重新平衡，不建管；下一轮恢复全图并按未用弧规则更新价格。
以真实成本保存最好可行解；固定点与近似 MIP 的界不代表原 ESM 最优性。
共用 esm_common 的数据、审计与导出；不导入其他求解器，导入本模块不会启动求解。
"""

from __future__ import annotations

import math
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from time import monotonic, perf_counter

import gurobipy as gp
import numpy as np

# 直接运行文件时补齐包上下文，不依赖当前工作目录。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "ESM"

from experiment_setup import CostData, NetworkData
from . import esm_common as shared
from solve_logging import logged_run, optimize_with_log
from .esm_common import (
    ESelectData, GROUPS, BUILD_GROUPS, COST_FIELDS, build_capture_target,
    proxy_cost_breakdown, record_e_select_metadata,
)


# ==================== 输入输出文件 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR
RESULT_DIR = PROJECT_DIR / "output" / "RSSforESM"
RESULT_FILE = RESULT_DIR / "RSSforESM_variables.xlsx"
INITIAL_SOLUTION_FILE = None  # 可指定共用的 solution.json（七组 key/value 记录）；None 独立求首轮。
PLOT_ENABLED = True
PLOT_FILE = RESULT_DIR / "RSSforESM_results.png"

# ==================== 模型参数 ====================
PERIOD_NUM = 5
NP = 3
DIAMETERS = [
    "p4", "p6", "p10", "p12", "p16", "p20",
    "p24a", "p24b", "p30a", "p30b", "p36", "p48"]
CURVE = "linear"
COEFFICIENT = None
MAX_CAPTURE_AMOUNT = 1e6
CAPTURE_AMOUNT_SEQUENCE = None  # 仅影响费用聚合，不覆盖静态捕集目标。
NORMALIZE_CAPTURE_TARGET = False  # True 是 / False 否；仅归一化自动非线性序列。

# ==================== 算法参数 ====================
TOTAL_TIME_LIMIT = 800  # 软预算，含数据准备、构模、恢复、审计；导出/绘图另计。
ITERATION_TIME_LIMIT = 80  # 单轮近似 MIP 上限，仍受剩余总预算约束。
MAX_ITERATIONS = 1000  # 以总时间或斜率收敛为主停止，避免200轮提前耗尽迭代额度。
MIP_GAP = 3e-3  # 近似 MIP 的 gap，不是原 ESM 的全局 gap。
SLOPE_ZERO_FLOW_TOL = 1e-5  # 微小流量弧关闭并重新平衡，不建管；定价时作为未用弧。
CONVERGENCE_TOL = 1e-6  # 流量与斜率的相对变化阈值；设施整数决策须完全相同。
STEP_SIZE_INITIAL = 0.5  # 首次价格更新步长；以下系数是可检验的启发式设置。
STEP_SIZE_MIN = 0.05
STEP_SIZE_MAX = 1.0
STEP_SIZE_SHRINK = 0.5  # 本轮真实费用比上一审计轮恶化时缩小步长。
STEP_SIZE_GROWTH = 1.25  # 改善时逐步放大，持平时保持。
COST_CHANGE_TOL = 1e-6  # 真实费用相对变化判定阈值，与主问题 gap 无关。
THREADS = 0
SEED = 0
GUROBI_OUTPUT_FLAG = 0
OUTPUT_FLAG = 1

FEAS_TOL = 1e-7
INTEGER_TOL = 1e-7


# ==================== 公共数据接口与求解参数 ====================
@dataclass(frozen=True)
class ESelectOptions:

    time_limit: float = TOTAL_TIME_LIMIT
    output_flag: int = GUROBI_OUTPUT_FLAG
    threads: int = THREADS
    seed: int = SEED
    mip_gap: float = MIP_GAP
    print_progress: bool = bool(OUTPUT_FLAG)


def aggregate_e_select_data(network: NetworkData, costs: CostData, diameters,
                            pipeline_install_limit, *, target=None) -> ESelectData:
    """使用本入口默认目标，费用聚合统一交给公共模块。"""
    return shared.aggregate_e_select_data(network, costs, diameters,
        pipeline_install_limit, target=MAX_CAPTURE_AMOUNT if target is None else target)


def _options(options):
    """提供默认求解参数，校验参数类型及各项取值范围后返回参数对象。"""
    opts = ESelectOptions() if options is None else options
    if not isinstance(opts, ESelectOptions):
        raise TypeError("options must be ESelectOptions")
    if not math.isfinite(opts.time_limit) or opts.time_limit < 0:
        raise ValueError("time_limit must be finite and nonnegative")
    if opts.output_flag not in (0, 1):
        raise ValueError("output_flag must be 0 or 1")
    if not isinstance(opts.print_progress, bool):
        raise ValueError("print_progress must be boolean")
    for name in ("threads", "seed"):
        value = getattr(opts, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not math.isfinite(opts.mip_gap) or not 0 <= opts.mip_gap <= 1:
        raise ValueError("mip_gap must be in [0, 1]")
    return opts


def _check_deadline(deadline):
    """检查单调时钟截止时间；达到期限时抛出 TimeoutError，None 表示不检查。"""
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("E-Select construction/solve budget exhausted")


def _timed(items, deadline):
    """逐项生成输入元素，在首个元素及此后每隔 1024 个元素检查截止时间。"""
    for index, item in enumerate(items):
        if index % 1024 == 0:
            _check_deadline(deadline)
        yield item


def audit_e_select_solution(data, solution, *, reported_objective=None):
    """按本求解器容差调用公共静态审计。"""
    return shared.audit_e_select_solution(data, solution, reported_objective=reported_objective,
        feasibility_tol=FEAS_TOL, integer_tol=INTEGER_TOL, objective_ulps=0)


def _finite_attribute(model, name):
    """读取模型数值属性；不可用、非有限或达到 Gurobi 无穷大阈值时返回 None。"""
    try:
        value = float(getattr(model, name))
    except (AttributeError, gp.GurobiError):
        return None
    return value if math.isfinite(value) and abs(value) < gp.GRB.INFINITY else None


def _status_name(status):
    """将常见 Gurobi 状态码转换为名称，其余状态返回 STATUS_状态码。"""
    names = ("OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "TIME_LIMIT",
             "INTERRUPTED", "NUMERIC", "SUBOPTIMAL")
    return next((name for name in names if status == getattr(gp.GRB, name)),
                f"STATUS_{status}")


def _optimize_e_select_model(model, messages):
    """收集本次求解的日志，并保留原有控制台输出设置。"""
    output_flag, console = model.Params.OutputFlag, model.Params.LogToConsole
    model.Params.LogToConsole = 0
    model.Params.LogFile = ""
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = console if output_flag else 0
    started = perf_counter()

    def record(current, where):
        if where == gp.GRB.Callback.MESSAGE:
            messages.append(dict(stage="E-Select", elapsed=perf_counter() - started,
                                 message=current.cbGet(gp.GRB.Callback.MSG_STRING)))

    try:
        optimize_with_log(model, record)
    finally:
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = output_flag
        model.Params.LogToConsole = console


def audit_solution(data, solution, *, reported_objective=None):
    """复用基础审计，并补查 ESM 构模中的三类设施—管道联动约束。"""
    audit = audit_e_select_solution(data, solution, reported_objective=reported_objective)
    if not audit["valid"]:
        return audit
    n = data.network
    a, b, y = (solution[group] for group in ("a", "b", "y"))
    outgoing = {node: [] for node in data.nodes}
    incoming = {node: [] for node in data.nodes}
    for u, v in n.ij:
        outgoing[u].append(v)
        incoming[v].append(u)

    def upper(lhs, rhs, label):
        residual = max(0.0, lhs - rhs)
        audit["max_residual"] = max(audit["max_residual"], residual)
        if not math.isfinite(lhs) or not math.isfinite(rhs) or residual > FEAS_TOL:
            audit["violations"].append(f"{label}: {lhs} > {rhs}")

    captured = {i: math.fsum(a[i, g] for g in n.source_units[i]) for i in n.sources}
    for i in n.sources:
        for g in n.source_units[i]:
            upper(a[i, g] - b.get(i, 0.0), math.fsum(
                min(n.pipe_capacity[d], n.source_capacity[i, g]) * y[i, v, d]
                for v in outgoing[i] for d in data.diameters), f"Capture-unit pipe link {i,g}")
        capacity = math.fsum(n.source_capacity[i, g] for g in n.source_units[i])
        upper(captured[i] - b.get(i, 0.0), math.fsum(
            min(n.pipe_capacity[d], capacity) * y[i, v, d]
            for v in outgoing[i] for d in data.diameters), f"Source pipe link {i}")
    for j in n.sinks:
        capacity = data.well_capacity[j] * data.maximum_wells[j]
        upper(b[j] - captured.get(j, 0.0), math.fsum(
            min(n.pipe_capacity[d], capacity) * y[u, j, d]
            for u in incoming[j] for d in data.diameters), f"Sink pipe link {j}")
    if audit["violations"]:
        audit.update(valid=False, cost=None, cost_breakdown={})
    return audit


def save_workbook(result, data, output_file):
    """导出已审计结果，保留七类变量、迭代历史、完整结果和各期参数。"""
    if Path(output_file).suffix.lower() != ".xlsx":
        raise ValueError("SS results must use an .xlsx path")
    record_e_select_metadata(result, data)
    return shared._save_e_select_workbook(result, output_file, history_sheets=("pipe_details",), extra_sheets={
        "select_parameters": [
            dict(period=t, duration=data.durations[t], capture_target=data.targets[t],
                 construction_weight=data.weights.get(t),
                 operating_weight=1 / len(data.periods) if data.peak > 0 else None)
            for t in data.periods]})


def save_e_select_result(result, data, output_file=None):
    """完整审计后导出 SS 自身的变量、诊断和迭代记录。"""
    if result["feasible"]:
        audit = audit_solution(data, result["solution"], reported_objective=result["objective"])
        if not audit["valid"]:
            raise ValueError("Refusing invalid SS export: " + "; ".join(audit["violations"]))
    elif result.get("solution") is not None:
        raise ValueError("Unverified solutions must not be exported as variable values")
    return save_workbook(result, data, RESULT_FILE if output_file is None else output_file)


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class PipeCostOracle:
    """逐弧求 min sum(f_d*y_d + v_d*x_d)，满足 sum(x_d)=Q、x_d<=q_d*y_d、sum(y_d)<=NP。

    最优解可取为若干满管和至多一根未满管；按满管数量与总容量做精确 DP。
    每弧缓存满管费用和前驱，查询时补一根余量管；连续需求 Q 不作离散化。
    同管径并行管道等价，x 按管径汇总；逐根流量可无损拆分。
    """

    def __init__(self, data, deadline=None):
        self.data = data
        self.diameters = data.diameters
        self.max_diameter = max(data.diameters, key=data.network.pipe_capacity.__getitem__)
        self._pipe_capacity = np.asarray(
            [data.network.pipe_capacity[d] for d in self.diameters], dtype=float)
        self.max_flow = float(data.pipeline_limit * max(self._pipe_capacity))
        # 十进制容量精确化为整数单位，只存可达容量，避免任意网格取整。
        capacities = [Fraction(str(q)) for q in self._pipe_capacity]
        denominator = math.lcm(*(q.denominator for q in capacities))
        steps = [int(q * denominator) for q in capacities]
        divisor = math.gcd(*steps) or 1
        unit = Fraction(divisor, denominator)
        steps = [q // divisor for q in steps]
        states = [(0,)]
        self._transitions = []
        for _ in range(1, data.pipeline_limit):
            incoming = {}
            for previous, capacity in enumerate(states[-1]):
                _check_deadline(deadline)
                for d, step in enumerate(steps):
                    incoming.setdefault(capacity + step, []).append((previous, d))
            states.append(tuple(sorted(incoming)))
            self._transitions.append(tuple(tuple(incoming[s]) for s in states[-1]))
        self._full_capacity = np.asarray([float(s * unit) for layer in states for s in layer])
        self._state_positions = [(k, i) for k, layer in enumerate(states) for i in range(len(layer))]
        self._arc_costs = {}

    def configure(self, arc, flow):
        if (not math.isfinite(flow) or flow < 0 or flow > self.max_flow + FEAS_TOL
                or (flow > 0 and self.max_flow == 0)):
            raise ValueError(f"Invalid flow for integer pipe configuration: {arc}: {flow}")
        if arc not in self._arc_costs:
            keys = [(*arc, d) for d in self.diameters]
            fixed = np.asarray([self.data.coefficients["y"][key] for key in keys])
            variable = np.asarray([self.data.coefficients["x"][key] for key in keys])
            if (not np.isfinite(fixed).all() or not np.isfinite(variable).all()
                    or (fixed < 0).any() or (variable < 0).any()):
                raise ValueError("SS requires finite nonnegative pipe construction/transport costs")
            full_pipe_cost = fixed + variable * self._pipe_capacity
            values, parents = [np.asarray([0.0])], []
            # D_k(S) = min_d D_{k-1}(S-q_d) + f_d + v_d*q_d。
            for layer in self._transitions:
                previous_values = values[-1]
                predecessors = [min(choices, key=lambda pair:
                                    previous_values[pair[0]] + full_pipe_cost[pair[1]])
                                for choices in layer]
                values.append(np.asarray([previous_values[p] + full_pipe_cost[d]
                                          for p, d in predecessors]))
                parents.append(predecessors)
            self._arc_costs[arc] = (fixed, variable, np.concatenate(values), parents)
        fixed, variable, full_cost, parents = self._arc_costs[arc]
        counts = dict.fromkeys(self.diameters, 0)
        flows = dict.fromkeys(self.diameters, 0.0)
        if flow == 0:
            return 0.0, counts, flows
        # 余量管可满载，因而全满管解也被覆盖；任意正需求都至少建设一根管。
        remaining = (flow - self._full_capacity)[:, None]
        costs = full_cost[:, None] + fixed + remaining * variable
        feasible = ((remaining > 0) & (remaining <= self._pipe_capacity + FEAS_TOL)
                    & (self._pipe_capacity > 0))
        costs[~feasible] = np.inf
        best = int(np.argmin(costs))
        if not math.isfinite(float(costs.flat[best])):
            raise ValueError(f"No integer pipe combination for {arc} at flow {flow}")
        state, residual_d = divmod(best, len(self.diameters))
        k, position = self._state_positions[state]
        while k > 0:
            position, d = parents[k - 1][position]
            counts[self.diameters[d]] += 1
            k -= 1
        # 回溯满管流量，余量交给查询选中的管型，保持候选费用与实际方案一致。
        rest = flow
        for k, d in enumerate(self.diameters):
            flows[d] = min(rest, float(counts[d] * self._pipe_capacity[k]))
            rest -= flows[d]
        if rest > self._pipe_capacity[residual_d] + FEAS_TOL:
            raise ValueError(f"Pipe capacity residual on {arc}: {rest - self._pipe_capacity[residual_d]}")
        diameter = self.diameters[residual_d]
        counts[diameter] += 1
        flows[diameter] += rest
        cost = math.fsum(self.data.coefficients["y"][(*arc, d)] * counts[d]
                        + self.data.coefficients["x"][(*arc, d)] * flows[d]
                        for d in self.diameters)
        return cost, counts, flows

    def reference_slopes(self, deadline=None):
        """按 NP 根最大管径管道满负荷估价；NP 抵消，得到 fixed/Q + variable。"""
        if self.max_flow == 0:
            return {arc: 0.0 for arc in _timed(self.data.network.ij, deadline)}
        diameter = self.max_diameter
        capacity = self.data.network.pipe_capacity[diameter]
        slopes = {}
        for arc in _timed(self.data.network.ij, deadline):
            fixed = self.data.coefficients["y"][(*arc, diameter)]
            variable = self.data.coefficients["x"][(*arc, diameter)]
            if any(not math.isfinite(v) or v < 0 for v in (fixed, variable)):
                raise ValueError("SS requires finite nonnegative pipe construction/transport costs")
            slopes[arc] = fixed / capacity + variable
        return slopes


def _extract_slope_solution(variables):
    solution = {}
    for group, values in variables.items():
        solution[group] = {}
        for key, var in values.items():
            value = float(var.X)
            if not math.isfinite(value) or value < -FEAS_TOL:
                raise ValueError(f"Invalid solution value {group}{key}: {value}")
            if group in ("s", "r", "w"):
                if abs(value - round(value)) > INTEGER_TOL:
                    raise ValueError(f"Noninteger facility value {group}{key}: {value}")
                value = int(round(value))
            solution[group][key] = max(0.0, value)
    return solution


def recover_solution(data, variables_solution, oracle, deadline=None):
    """对已完成微小流量清理与平衡的解逐弧恢复整数管道。"""
    zero_tolerance = _nonnegative(SLOPE_ZERO_FLOW_TOL, "SLOPE_ZERO_FLOW_TOL")
    solution = {g: dict.fromkeys(keys, 0.0) for g, keys in data.indices.items()}
    for group in ("s", "r", "w", "a", "b"):
        solution[group] = dict(variables_solution[group])
    for arc, value in _timed(variables_solution["F"].items(), deadline):
        if value > 0:
            if value <= zero_tolerance:
                raise ValueError(f"Tiny arc flow must be removed with balance correction before integer recovery: {arc}: {value}")
            _, counts, flows = oracle.configure(arc, value)
            for d in data.diameters:
                solution["y"][(*arc, d)] = counts[d]
                solution["x"][(*arc, d)] = flows[d]
        elif value < 0 or not math.isfinite(value):
            raise ValueError(f"Invalid aggregate flow: {arc}: {value}")
    return solution


def _arc_flows(data, solution):
    result = {}
    for (u, v, d), value in solution["x"].items():
        if value > 0:
            result[u, v] = result.get((u, v), 0.0) + value
    return result


def _support(solution):
    return {key[:2] for key, value in solution["y"].items() if value > 0.5} if solution else set()


def solution_metrics(solution):
    return dict(source_units=sum(v > 0.5 for v in solution["s"].values()),
                source_nodes=len({i for (i, g), v in solution["s"].items() if v > 0.5}),
                sinks=sum(v > 0.5 for v in solution["r"].values()),
                wells=sum(round(v) for v in solution["w"].values()),
                arcs=len(_support(solution)), pipes=sum(round(v) for v in solution["y"].values()))


def individual_pipe_plan(data, solution):
    """将同管径汇总流量无损分配到每根已建管道，给出一组等价的逐管方案。"""
    rows = []
    for (u, v, diameter), value in solution["y"].items():
        count = round(value)
        remaining = solution["x"][u, v, diameter]
        capacity = data.network.pipe_capacity[diameter]
        for number in range(1, count + 1):
            flow = min(remaining, capacity) if number < count else remaining
            remaining -= flow
            rows.append(dict(start=u, end=v, diameter=diameter, pipe_number=number,
                capacity=capacity, flow=flow, fixed_cost=data.coefficients["y"][u, v, diameter],
                unit_operating_cost=data.coefficients["x"][u, v, diameter],
                operating_cost=data.coefficients["x"][u, v, diameter] * flow))
    return rows


# ==================== 全弧近似模型与纯斜率缩放 ====================
def build_slope_model(data, slopes, *, options=None, env=None, deadline=None):
    """保留全部候选弧与源汇整数变量，仅用 rho*F 近似管道费用。"""
    opts = _options(ESelectOptions(
        time_limit=ITERATION_TIME_LIMIT, output_flag=GUROBI_OUTPUT_FLAG,
        threads=THREADS, seed=SEED, mip_gap=MIP_GAP) if options is None else options)
    arcs = tuple(data.network.ij)
    if set(slopes) != set(arcs) or any(not math.isfinite(slopes[e]) or slopes[e] < 0 for e in arcs):
        raise ValueError("Every candidate arc needs a finite nonnegative slope")
    _check_deadline(deadline)
    model = gp.Model("ESM_RSS", env=env)
    try:
        model.Params.OutputFlag = opts.output_flag
        model.Params.Threads = opts.threads
        model.Params.Seed = opts.seed
        model.Params.MIPGap = opts.mip_gap
        model.Params.TimeLimit = opts.time_limit
        model.Params.FeasibilityTol = FEAS_TOL
        model.Params.IntFeasTol = INTEGER_TOL
        variables = {}
        for group in ("s", "r", "w", "a", "b"):
            variables[group] = model.addVars(data.indices[group], lb=0,
                vtype=gp.GRB.BINARY if group in ("s", "r") else
                gp.GRB.INTEGER if group == "w" else gp.GRB.CONTINUOUS, name=group)
        maximum = data.pipeline_limit * max(data.network.pipe_capacity[d] for d in data.diameters)
        variables["F"] = model.addVars(arcs, lb=0, ub=maximum, name="F")
        s, r, w, a, b, flow = (variables[g] for g in ("s", "r", "w", "a", "b", "F"))
        for i, g in _timed(data.network.ig, deadline):
            model.addConstr(a[i, g] <= data.network.source_capacity[i, g] * s[i, g],
                            name=f"capture_capacity[{i},{g}]")
        for j in _timed(data.network.sinks, deadline):
            model.addConstr(b[j] <= data.well_capacity[j] * w[j], name=f"injection_capacity[{j}]")
            model.addConstr(w[j] <= data.maximum_wells[j] * r[j], name=f"site_well_link[{j}]")
        model.addConstr(gp.quicksum(a.values()) >= data.target, name="economic_target")
        balance = {v: gp.LinExpr() for v in data.nodes}
        for (u, v), value in _timed(flow.items(), deadline):
            balance[u] += value
            balance[v] -= value
        for (i, g), value in a.items():
            balance[i] -= value
        for j, value in b.items():
            balance[j] += value
        for v in _timed(data.nodes, deadline):
            model.addConstr(balance[v] == 0, name=f"balance[{v}]")
        model.setObjective(gp.quicksum(
            data.coefficients[g][key] * var
            for g in ("s", "r", "w", "a", "b") for key, var in variables[g].items())
            + gp.quicksum(slopes[e] * var for e, var in flow.items()), gp.GRB.MINIMIZE)
        model.update()
        _check_deadline(deadline)
        return model, variables
    except Exception:
        model.dispose()
        raise


def validate_rss_parameters():
    values = {name: _nonnegative(globals()[name], name) for name in (
        "STEP_SIZE_INITIAL", "STEP_SIZE_MIN", "STEP_SIZE_MAX",
        "STEP_SIZE_SHRINK", "STEP_SIZE_GROWTH", "COST_CHANGE_TOL")}
    if not 0 < values["STEP_SIZE_MIN"] <= values["STEP_SIZE_INITIAL"] <= values["STEP_SIZE_MAX"] <= 1:
        raise ValueError("Require 0 < STEP_SIZE_MIN <= STEP_SIZE_INITIAL <= STEP_SIZE_MAX <= 1")
    if not 0 < values["STEP_SIZE_SHRINK"] < 1:
        raise ValueError("STEP_SIZE_SHRINK must be between zero and one")
    if values["STEP_SIZE_GROWTH"] <= 1:
        raise ValueError("STEP_SIZE_GROWTH must exceed one")


def adjust_step_size(step_size, previous_cost, current_cost):
    """用相邻已审计真实费用反馈；保留坏候选的探索机会，不承诺真实目标单调下降。"""
    validate_rss_parameters()
    step_size = _nonnegative(step_size, "step_size")
    if not STEP_SIZE_MIN <= step_size <= STEP_SIZE_MAX:
        raise ValueError("step_size is outside configured limits")
    current_cost = _nonnegative(current_cost, "current_cost")
    if previous_cost is None:
        return step_size, "initial"
    previous_cost = _nonnegative(previous_cost, "previous_cost")
    change = (current_cost - previous_cost) / max(1.0, abs(previous_cost))
    if change > COST_CHANGE_TOL:
        return max(STEP_SIZE_MIN, step_size * STEP_SIZE_SHRINK), "worsened"
    if change < -COST_CHANGE_TOL:
        return min(STEP_SIZE_MAX, step_size * STEP_SIZE_GROWTH), "improved"
    return step_size, "unchanged"


def reference_flow_from_solution(data, solution):
    """首轮清理后的正弧流量中位数；同点捕存等无管网情形不设置参考流量。"""
    tolerance = _nonnegative(SLOPE_ZERO_FLOW_TOL, "SLOPE_ZERO_FLOW_TOL")
    positive = [q for q in _arc_flows(data, solution).values() if q > tolerance]
    if not positive:
        return None
    maximum = data.pipeline_limit * max(data.network.pipe_capacity[d] for d in data.diameters)
    return min(float(np.median(positive)), maximum)


def reference_arc_prices(oracle, reference_flow, deadline=None):
    """在共同参考流量上，逐弧精确选管并缓存全费用平均价；不产生实际建设决策。"""
    if reference_flow is None:
        return {}
    reference_flow = _nonnegative(reference_flow, "reference_flow")
    if reference_flow == 0:
        raise ValueError("reference_flow must be positive or None")
    prices = {}
    for arc in _timed(oracle.data.network.ij, deadline):
        cost, _, _ = oracle.configure(arc, reference_flow)
        prices[arc] = cost / reference_flow
    return prices


def update_slopes(data, solution, slopes, deadline=None, *, reference_prices, step_size):
    """活跃弧按实际 C(Q)/Q，零流量弧按参考价；统一作价格阻尼，实际流量不变。"""
    zero_tolerance = _nonnegative(SLOPE_ZERO_FLOW_TOL, "SLOPE_ZERO_FLOW_TOL")
    step_size = _nonnegative(step_size, "step_size")
    if not 0 < step_size <= 1:
        raise ValueError("step_size must be in (0, 1]")
    updated = dict(slopes)
    flows = _arc_flows(data, solution)
    for arc in _timed(slopes, deadline):
        flow = flows.get(arc, 0.0)
        if flow <= zero_tolerance:
            target = reference_prices.get(arc, slopes[arc])
        else:
            fixed = math.fsum(data.coefficients["y"][(*arc, d)] * solution["y"][(*arc, d)]
                              for d in data.diameters)
            variable = math.fsum(data.coefficients["x"][(*arc, d)] * solution["x"][(*arc, d)]
                                 for d in data.diameters)
            target = (fixed + variable) / flow
        updated[arc] = slopes[arc] + step_size * (target - slopes[arc])
    return updated


def load_initial_solution(path):
    """读取共用可行解的七组 key/value 列表；模型可行性由 solve_e_select 完整审计。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != set(GROUPS):
        raise ValueError("Initial solution must contain s/r/w/y/a/b/x record lists")
    return {group: {tuple(row["key"]) if len(row["key"]) > 1 else row["key"][0]: row["value"]
                    for row in records} for group, records in payload.items()}


def _polish_continuous_solution(data, model, variables, *, deadline=None):
    """仅校正数值残差：固定本轮整数设施，以原目标和原约束重算连续流量。"""
    _check_deadline(deadline)
    selected = {}
    for group in ("s", "r", "w"):
        selected[group] = {}
        for key, var in variables[group].items():
            value = float(var.X)
            if (not math.isfinite(value) or value < -INTEGER_TOL
                    or abs(value - round(value)) > INTEGER_TOL
                    or (group in ("s", "r") and round(value) not in (0, 1))):
                raise ValueError(f"Cannot fix unreliable integer facility {group}{key}: {value}")
            selected[group][key] = int(round(value))
    model.update()
    corrected = model.copy()
    try:
        copied = {group: {key: corrected.getVarByName(var.VarName) for key, var in values.items()}
                  for group, values in variables.items()}
        for group, values in selected.items():
            for key, value in values.items():
                var = copied[group][key]
                var.LB = var.UB = value
                var.VType = gp.GRB.CONTINUOUS
        corrected.Params.FeasibilityTol = 1e-9
        corrected.Params.OptimalityTol = 1e-9
        corrected.Params.NumericFocus = 2
        corrected.Params.Method = 1
        corrected.update()
        _check_deadline(deadline)
        remaining = deadline - monotonic() if deadline is not None else model.Params.TimeLimit
        corrected.Params.TimeLimit = min(model.Params.TimeLimit,
            max(0.0, remaining - min(1.0, remaining * 0.1)))
        messages = []
        _optimize_e_select_model(corrected, messages)
        status = _status_name(corrected.Status)
        if corrected.SolCount == 0:
            _check_deadline(deadline)
            raise ValueError(f"Continuous numerical correction has no feasible solution: {status}")
        state = _extract_slope_solution(copied)
        if any(state[group] != values for group, values in selected.items()):
            raise ValueError("Continuous numerical correction changed integer facilities")
        return state, dict(gurobi_status=status, solver_runtime=float(corrected.Runtime),
                           surrogate_objective=_finite_attribute(corrected, "ObjVal"),
                           solver_log=messages)
    finally:
        corrected.dispose()


class _TinyFlowCleanupError(ValueError):
    """本轮闭弧后的清理子问题未能给出可接受解，不代表原模型不可行。"""


def _remove_tiny_arc_flows(data, model, variables, state, *, closed_arcs=(), deadline=None):
    """累计关闭本轮微小流量弧并重新求解平衡；固定设施 LP 不可行时放开设施。"""
    zero_tolerance = _nonnegative(SLOPE_ZERO_FLOW_TOL, "SLOPE_ZERO_FLOW_TOL")
    closed = set(closed_arcs) | {e for e, q in state["F"].items() if 0 < q <= zero_tolerance}
    info = dict(closed_arcs=sorted(closed), rounds=0, facility_mip_fallbacks=0,
                gurobi_status="NOT_NEEDED", solver_runtime=0.0, solver_log=[], all_optimal=True)
    if not closed:
        return state, info
    _check_deadline(deadline)
    model.update()
    corrected = model.copy()
    try:
        copied = {g: {key: corrected.getVarByName(var.VarName) for key, var in values.items()}
                  for g, values in variables.items()}
        original_bounds = {(g, key): (var.LB, var.UB, var.VType)
                           for g in ("s", "r", "w") for key, var in copied[g].items()}
        corrected.Params.FeasibilityTol = 1e-9
        corrected.Params.OptimalityTol = 1e-9
        corrected.Params.NumericFocus = 2
        corrected.Params.Method = 1

        def optimize(phase):
            _check_deadline(deadline)
            remaining = deadline - monotonic() if deadline is not None else model.Params.TimeLimit
            corrected.Params.TimeLimit = min(model.Params.TimeLimit,
                max(0.0, remaining - min(1.0, remaining * 0.1)))
            messages = []
            _optimize_e_select_model(corrected, messages)
            for message in messages:
                message.update(phase=phase)
            info["solver_log"].extend(messages)
            info["solver_runtime"] += float(corrected.Runtime)
            info["gurobi_status"] = _status_name(corrected.Status)

        while True:
            _check_deadline(deadline)
            info["rounds"] += 1
            info["closed_arcs"] = sorted(closed)
            for arc in closed:
                copied["F"][arc].LB = copied["F"][arc].UB = 0.0
            for g in ("s", "r", "w"):
                for key, var in copied[g].items():
                    var.VType = gp.GRB.CONTINUOUS
                    var.LB = var.UB = state[g][key]
            optimize("fixed-facility flow balance")
            if corrected.Status == gp.GRB.INFEASIBLE:
                info["facility_mip_fallbacks"] += 1
                for (g, key), (lb, ub, vtype) in original_bounds.items():
                    var = copied[g][key]
                    var.LB, var.UB, var.VType = lb, ub, vtype
                optimize("free-facility flow balance")
            if corrected.SolCount == 0:
                _check_deadline(deadline)
                raise _TinyFlowCleanupError(
                    f"No feasible tiny-flow cleanup with {len(closed)} arcs closed: "
                    f"{info['gurobi_status']}; this is only the current restricted subproblem")
            try:
                state = _extract_slope_solution(copied)
            except ValueError as error:
                raise _TinyFlowCleanupError(f"Invalid tiny-flow cleanup: {error}") from error
            for arc in closed:
                if state["F"][arc] > FEAS_TOL:
                    raise _TinyFlowCleanupError(f"Closed arc still carries flow: {arc}")
                state["F"][arc] = 0.0  # 固定为零的变量仅消除求解器数值残差；随后完整审计。
            info["all_optimal"] &= corrected.Status == gp.GRB.OPTIMAL
            newly_tiny = {e for e, q in state["F"].items() if 0 < q <= zero_tolerance}
            if not newly_tiny:
                return state, info
            closed.update(newly_tiny)  # 已关闭弧流量严格为零，因此每轮闭弧集合严格增大。
    finally:
        corrected.dispose()


def _relative_change(previous, current):
    return max((abs(current[key] - value) / max(1.0, abs(value), abs(current[key]))
                for key, value in previous.items()), default=0.0)


def solve_e_select(data, *, options=None, total_time_limit=None, max_iterations=None,
                   iteration_time_limit=None, initial_solution=None):
    """独立 RSS；可显式共用已审计的初始解，其生成时间不计入本次求解预算。

    未提供初始解时，使用与基础 SS 相同的首轮模型和价格；墙钟限时不保证同一首轮解。
    首轮后固定参考流量，按相邻审计轮的真实费用调整价格步长，始终保留最好可行上界。
    """
    started = monotonic()
    validate_rss_parameters()
    opts = _options(ESelectOptions(
        time_limit=TOTAL_TIME_LIMIT, output_flag=GUROBI_OUTPUT_FLAG,
        threads=THREADS, seed=SEED, mip_gap=MIP_GAP,
        print_progress=bool(OUTPUT_FLAG)) if options is None else options)
    limit = _nonnegative(opts.time_limit if total_time_limit is None else total_time_limit, "total_time_limit")
    cap = _nonnegative(ITERATION_TIME_LIMIT if iteration_time_limit is None else iteration_time_limit,
                       "iteration_time_limit")
    iterations = _integer(MAX_ITERATIONS if max_iterations is None else max_iterations, "max_iterations")
    tolerance = _nonnegative(CONVERGENCE_TOL, "CONVERGENCE_TOL")
    zero_tolerance = _nonnegative(SLOPE_ZERO_FLOW_TOL, "SLOPE_ZERO_FLOW_TOL")
    deadline = started + limit
    result = dict(problem="E-Select", algorithm="ESM-RSS: common start, inactive-arc pricing and cost-driven damping",
        status="NO_FEASIBLE_SOLUTION", gurobi_status="NOT_SOLVED", feasible=False, solution=None,
        objective=None, initial_objective=None, upper_bound=None, e_select_objective=None, lower_bound=None, gap=None,
        original_lower_bound=None, original_gap=None, lower_bound_source="not computed",
        objective_scope="original E-Select static cost after integer pipe recovery",
        cost_breakdown={}, max_constraint_residual=None, solver_runtime=0.0, solver_log=[],
        bound_history=[], history=[], options=asdict(opts), peak_target=data.peak,
        economic_target=data.target, total_time_limit=limit, termination_reason="ITERATION_LIMIT",
        algorithm_parameters=dict(
            effective_options=asdict(replace(opts, time_limit=limit)),
            total_time_limit=limit, iteration_time_limit=cap, max_iterations=iterations,
            convergence_tol=tolerance, zero_flow_threshold=zero_tolerance,
            step_size_initial=STEP_SIZE_INITIAL, step_size_min=STEP_SIZE_MIN,
            step_size_max=STEP_SIZE_MAX, step_size_shrink=STEP_SIZE_SHRINK,
            step_size_growth=STEP_SIZE_GROWTH, cost_change_tol=COST_CHANGE_TOL,
            feasibility_tolerance=FEAS_TOL, integer_tolerance=INTEGER_TOL),
        algorithm_info=dict(iteration_time_limit=cap, max_iterations=iterations,
            convergence_tol=tolerance, candidate_arcs=len(data.network.ij),
            initial_slope_rule="max-diameter full-capacity cost: fixed/Q + variable",
            initialization_mode="common_ss_model" if initial_solution is None else "supplied_audited_solution",
            initial_solution_generation_in_budget=initial_solution is None,
            reference_flow=None,
            reference_flow_rule="median of positive flows in the first audited solution; fixed thereafter",
            slope_update="damped C(Q)/Q on active arcs; damped C(reference_flow)/reference_flow on inactive arcs",
            initial_step_size=STEP_SIZE_INITIAL, min_step_size=STEP_SIZE_MIN, max_step_size=STEP_SIZE_MAX,
            step_size_shrink=STEP_SIZE_SHRINK, step_size_growth=STEP_SIZE_GROWTH,
            cost_change_tol=COST_CHANGE_TOL,
            damping_feedback="adjacent audited true costs; worsening shrinks, improvement grows, ties retain step",
            per_arc_pipe_optimization="exact structured DP over full-pipe capacity states plus one residual pipe",
            zero_flow_threshold=zero_tolerance,
            tiny_flow_policy="close tiny-flow arcs for this iteration and rebalance; restore full graph next iteration",
            bound_scope="Surrogate bounds and fixed points do not certify original ESM optimality"))

    def keep_best(candidate, audit):
        if result["initial_objective"] is None:
            result["initial_objective"] = audit["cost"]
        if result["objective"] is not None and audit["cost"] >= result["objective"] - FEAS_TOL:
            return False
        result.update(feasible=True, solution=candidate, objective=audit["cost"],
            upper_bound=audit["cost"], e_select_objective=audit["cost"],
            cost_breakdown=audit["cost_breakdown"], max_constraint_residual=audit["max_residual"])
        result["bound_history"].append(dict(stage="RSS audited integer recovery",
            elapsed=monotonic() - started, upper_bound=audit["cost"], lower_bound=None))
        return True

    shared_state = None
    if initial_solution is not None:
        initial_solution = {group: dict(values) for group, values in initial_solution.items()}
        audit = audit_solution(data, initial_solution)
        if not audit["valid"]:
            raise ValueError("Invalid initial solution: " + "; ".join(audit["violations"]))
        initial_flows = _arc_flows(data, initial_solution)
        if any(0 < q <= zero_tolerance for q in initial_flows.values()):
            raise ValueError("Initial solution contains tiny positive arc flows; provide a cleaned feasible start")
        shared_state = {group: dict(initial_solution[group]) for group in ("s", "r", "w", "a", "b")}
        shared_state["F"] = initial_flows
        keep_best(initial_solution, audit)
    elif data.target == 0:
        zero = {g: dict.fromkeys(keys, 0.0) for g, keys in data.indices.items()}
        audit = audit_solution(data, zero)
        if audit["valid"]:
            keep_best(zero, audit)
    stage = "initialization"
    iteration = None
    record = None
    try:
        _check_deadline(deadline)
        if cap == 0 or iterations == 0:
            result["termination_reason"] = "ITERATION_DISABLED"
        else:
            oracle = PipeCostOracle(data, deadline)
            slopes = oracle.reference_slopes(deadline=deadline)
            result["algorithm_info"].update(initial_diameter=oracle.max_diameter,
                initial_pipe_count=data.pipeline_limit, initial_reference_flow=oracle.max_flow)
            previous = None
            previous_cost = None
            step_size = STEP_SIZE_INITIAL
            reference_flow = None
            reference_prices = {}
            reference_ready = False
            if shared_state is not None:
                stage = "reference_price_calibration"
                reference_flow = reference_flow_from_solution(data, initial_solution)
                result["algorithm_info"]["reference_flow"] = reference_flow
                reference_prices = reference_arc_prices(oracle, reference_flow, deadline)
                reference_ready = True
                slopes = update_slopes(data, initial_solution, slopes, deadline,
                    reference_prices=reference_prices, step_size=step_size)
                previous, previous_cost = shared_state, result["initial_objective"]
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.start()
                for iteration in range(1, iterations + 1):
                    _check_deadline(deadline)
                    record = dict(iteration=iteration, candidate_arcs=len(data.network.ij),
                        gurobi_status="NOT_SOLVED", surrogate_objective=None, surrogate_bound=None,
                        surrogate_gap=None, true_objective=None, best_objective=result["objective"],
                        improved=False, max_flow_change=None, max_slope_change=None,
                        max_slope_before=max(slopes.values(), default=0.0))
                    result["history"].append(record)
                    model = None
                    stop = False
                    try:
                        stage = "build_slope_model"
                        model, variables = build_slope_model(data, slopes,
                            options=replace(opts, time_limit=cap), env=env, deadline=deadline)
                        remaining = deadline - monotonic()
                        _check_deadline(deadline)
                        model.Params.TimeLimit = min(cap, remaining - min(2.0, remaining * 0.1))
                        messages = []
                        stage = "surrogate_optimization"
                        _optimize_e_select_model(model, messages)
                        for message in messages:
                            message.update(stage="RSS surrogate", iteration=iteration)
                        result["solver_log"].extend(messages)
                        result["solver_runtime"] += float(model.Runtime)
                        status = _status_name(model.Status)
                        result["gurobi_status"] = record["gurobi_status"] = status
                        record["surrogate_bound"] = _finite_attribute(model, "ObjBound")
                        if model.SolCount == 0:
                            result["termination_reason"] = "NO_SURROGATE_INCUMBENT"
                            stop = True
                        else:
                            record["surrogate_objective"] = _finite_attribute(model, "ObjVal")
                            record["surrogate_gap"] = _finite_attribute(model, "MIPGap")
                            stage = "extract_recover_audit"
                            correction_optimal = True
                            closed_arcs = set()

                            def clean_tiny_flows(raw_state):
                                nonlocal stage, correction_optimal
                                stage = "tiny_flow_removal"
                                clean_state, cleanup = _remove_tiny_arc_flows(
                                    data, model, variables, raw_state, closed_arcs=closed_arcs, deadline=deadline)
                                closed_arcs.update(cleanup["closed_arcs"])
                                if cleanup["rounds"]:
                                    correction_optimal &= cleanup["all_optimal"]
                                    for message in cleanup.pop("solver_log"):
                                        message.update(stage="RSS tiny-flow removal", iteration=iteration)
                                        result["solver_log"].append(message)
                                    result["solver_runtime"] += cleanup["solver_runtime"]
                                    record.setdefault("tiny_flow_cleanup", []).append(cleanup)
                                stage = "extract_recover_audit"
                                return clean_state

                            try:
                                state = clean_tiny_flows(_extract_slope_solution(variables))
                                candidate = recover_solution(data, state, oracle, deadline)
                                audit = audit_solution(data, candidate)
                                if not audit["valid"]:
                                    raise ValueError("Invalid recovered incumbent: " + "; ".join(audit["violations"]))
                            except _TinyFlowCleanupError:
                                raise
                            except ValueError as numerical_error:
                                record["numerical_repair"] = dict(attempted=True, trigger=str(numerical_error))
                                stage = "continuous_numerical_correction"
                                state, correction = _polish_continuous_solution(
                                    data, model, variables, deadline=deadline)
                                correction_log = correction.pop("solver_log", [])
                                for message in correction_log:
                                    message.update(stage="RSS continuous numerical correction", iteration=iteration)
                                result["solver_log"].extend(correction_log)
                                result["solver_runtime"] += correction["solver_runtime"]
                                record["numerical_repair"].update(correction)
                                correction_optimal &= correction["gurobi_status"] == "OPTIMAL"
                                state = clean_tiny_flows(state)
                                candidate = recover_solution(data, state, oracle, deadline)
                                audit = audit_solution(data, candidate)
                                if not audit["valid"]:
                                    raise ValueError("Numerically corrected solution fails original audit: "
                                                     + "; ".join(audit["violations"]))
                                record["numerical_repair"].update(valid=True, max_residual=audit["max_residual"])
                                if opts.print_progress:
                                    print(f"[ESM-RSS] iteration={iteration}, numerical correction passed: "
                                          f"{numerical_error}")
                            improved = keep_best(candidate, audit)
                            # 定价超时时也保留已完成的审计记录和可行上界。
                            record.update(true_objective=audit["cost"], best_objective=result["objective"],
                                improved=improved, metrics=solution_metrics(candidate),
                                cost_breakdown=audit["cost_breakdown"],
                                max_constraint_residual=audit["max_residual"], pricing_completed=False)
                            if not reference_ready:
                                stage = "reference_price_calibration"
                                reference_flow = reference_flow_from_solution(data, candidate)
                                result["algorithm_info"]["reference_flow"] = reference_flow
                                reference_prices = reference_arc_prices(oracle, reference_flow, deadline)
                                reference_ready = True
                            stage = "update_slopes"
                            step_size, feedback = adjust_step_size(step_size, previous_cost, audit["cost"])
                            targets = update_slopes(data, candidate, slopes, deadline,
                                reference_prices=reference_prices, step_size=1.0)
                            new_slopes = {arc: price + step_size * (targets[arc] - price)
                                          for arc, price in slopes.items()}
                            raw_slope_change = _relative_change(slopes, targets)
                            slope_change = _relative_change(slopes, new_slopes)
                            flow_change = (max(_relative_change(previous[g], state[g]) for g in ("a", "b", "F"))
                                           if previous is not None else None)
                            facilities_same = (previous is not None and all(
                                previous[g] == state[g] for g in ("s", "r", "w")))
                            record.update(true_objective=audit["cost"], best_objective=result["objective"],
                                improved=improved, metrics=solution_metrics(candidate),
                                max_constraint_residual=audit["max_residual"],
                                max_flow_change=flow_change, max_slope_change=slope_change,
                                raw_slope_change=raw_slope_change, damping_step_size=step_size,
                                cost_feedback=feedback, reference_flow=reference_flow, pricing_completed=True,
                                cost_change_relative=((audit["cost"] - previous_cost) / max(1.0, abs(previous_cost))
                                                      if previous_cost is not None else None),
                                inactive_priced_arcs=sum(q <= zero_tolerance and e in reference_prices
                                                        for e, q in state["F"].items()),
                                added_arcs=(sum(q > zero_tolerance and previous["F"].get(e, 0) <= zero_tolerance
                                                for e, q in state["F"].items()) if previous is not None else None),
                                removed_arcs=(sum(q <= zero_tolerance and previous["F"].get(e, 0) > zero_tolerance
                                                  for e, q in state["F"].items()) if previous is not None else None),
                                max_slope_after=max(new_slopes.values(), default=0.0),
                                tiny_positive_flow_arcs=sum(0 < flow <= zero_tolerance for flow in state["F"].values()),
                                facility_changes={g: sum(state[g][key] != previous[g][key]
                                    for key in state[g]) if previous else None for g in ("s", "r", "w")})
                            if (status == "OPTIMAL" and correction_optimal and facilities_same and flow_change <= tolerance
                                    and raw_slope_change <= tolerance):
                                result["termination_reason"] = "SLOPE_FIXED_POINT"
                                stop = True
                            slopes, previous = new_slopes, state
                            previous_cost = audit["cost"]
                    finally:
                        record["elapsed"] = monotonic() - started
                        if model is not None:
                            model.dispose()
                    record["elapsed"] = monotonic() - started
                    if opts.print_progress:
                        print(f"[ESM-RSS] iteration={iteration}, arcs={len(data.network.ij)}, "
                              f"status={record['gurobi_status']}, actual={record['true_objective']}, "
                              f"best={result['objective']}, elapsed={record['elapsed']:.2f}s")
                    if stop:
                        break
    except TimeoutError:
        result["termination_reason"] = "TOTAL_TIME_LIMIT"
    except (gp.GurobiError, ValueError) as error:
        result["termination_reason"] = "ERROR"
        result["error"] = str(error)
        result["error_iteration"] = iteration
        result["error_stage"] = stage
        if record is not None:
            record.update(error=str(error), error_stage=stage)
    result["status"] = "FEASIBLE_RSS" if result["feasible"] else "NO_FEASIBLE_SOLUTION"
    if result["termination_reason"] == "ERROR":
        result["status"] = "FEASIBLE_WITH_ERROR" if result["feasible"] else "ERROR"
    if result["feasible"]:
        result["metrics"] = solution_metrics(result["solution"])
        result["pipe_details"] = individual_pipe_plan(data, result["solution"])
    result["total_budget_exhausted"] = monotonic() >= deadline
    if result["total_budget_exhausted"] and result["termination_reason"] != "ERROR":
        result["termination_reason"] = "TOTAL_TIME_LIMIT"
    result["solve_elapsed"] = result["elapsed"] = monotonic() - started
    result["algorithm_info"].update(iterations=len(result["history"]),
        time_remaining=max(0.0, deadline - monotonic()))
    record_e_select_metadata(result, data)
    return result


# 完整运行日志保存为结果文件旁的同名 .log，Gurobi 控制台开关不影响保存。
@logged_run(lambda: Path(RESULT_FILE).with_suffix(".log"))
def main():
    """按顶部配置独立求解并保存至 output/RSSforESM。"""
    started = monotonic()
    _nonnegative(TOTAL_TIME_LIMIT, "TOTAL_TIME_LIMIT")
    _nonnegative(MAX_CAPTURE_AMOUNT, "MAX_CAPTURE_AMOUNT")
    network, data = shared.prepare_e_select_data(globals())
    initial_solution = load_initial_solution(INITIAL_SOLUTION_FILE) if INITIAL_SOLUTION_FILE is not None else None
    preparation = monotonic() - started
    run_parameters = deepcopy({key: globals()[key] for key in (
        "PERIOD_NUM", "NP", "DIAMETERS", "CURVE", "COEFFICIENT", "MAX_CAPTURE_AMOUNT",
        "CAPTURE_AMOUNT_SEQUENCE", "NORMALIZE_CAPTURE_TARGET",
        "TOTAL_TIME_LIMIT", "ITERATION_TIME_LIMIT", "MAX_ITERATIONS",
        "MIP_GAP", "SLOPE_ZERO_FLOW_TOL", "CONVERGENCE_TOL", "THREADS", "SEED",
        "STEP_SIZE_INITIAL", "STEP_SIZE_MIN", "STEP_SIZE_MAX", "STEP_SIZE_SHRINK",
        "STEP_SIZE_GROWTH", "COST_CHANGE_TOL",
        "GUROBI_OUTPUT_FLAG", "OUTPUT_FLAG", "PLOT_ENABLED")})
    run_parameters.update(INPUT_DIR=str(INPUT_DIR), RESULT_FILE=str(RESULT_FILE),
        INITIAL_SOLUTION_FILE=str(INITIAL_SOLUTION_FILE) if INITIAL_SOLUTION_FILE is not None else None,
        PLOT_FILE=str(PLOT_FILE), capture_target=data.targets, construction_weights=data.weights)
    result = solve_e_select(data, total_time_limit=max(0.0, TOTAL_TIME_LIMIT - preparation),
                            initial_solution=initial_solution)
    result.update(preprocessing_elapsed=preparation, total_time_limit=run_parameters["TOTAL_TIME_LIMIT"],
                  total_elapsed=monotonic() - started)
    result["elapsed"] = result["total_elapsed"]
    result["run_parameters"] = run_parameters
    result["plots"], result["export_errors"] = {}, []
    if OUTPUT_FLAG:
        print(f"[ESM-RSS] status={result['status']}, objective={result['objective']}, "
              f"reason={result['termination_reason']}, total={result['total_elapsed']:.3f}s")
        if result["feasible"]:
            print(f"[ESM-RSS] {result['metrics']}")
        if result.get("error"):
            print(f"[ESM-RSS] error_iteration={result.get('error_iteration')}, "
                  f"error_stage={result.get('error_stage')}: {result['error']}")
    if PLOT_ENABLED and result["feasible"]:
        try:
            import plot_results

            result["plots"]["construction"] = [str(p) for p in plot_results.plot_results(
                result["solution"], network, diameters=DIAMETERS, output_file=PLOT_FILE)]
        except Exception as error:
            result["export_errors"].append(f"Construction plot: {type(error).__name__}: {error}")
            print(result["export_errors"][-1])
    path = save_e_select_result(result, data, RESULT_FILE)
    print(f"ESM-RSS variables and diagnostics saved: {path}")
    return result


if __name__ == "__main__":
    main()
