"""用 Gurobi 求解仅保留设施全期建设数量限制的 CCS_FNE 单算例。

捕集、封存、运输均不设置逐期建设数量限制。
每条有向弧全期所有管径合计的管道条数上限由本文件的 NP 独立配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from math import expm1, isfinite, log1p
from pathlib import Path
from typing import Optional

import gurobipy as gp
import pandas as pd

from plot_results import plot_results_from_excel
from capture_target_sequence import generate_capture_targets

from experiment_setup import (
    CostData,
    D_LIST,
    NetworkData,
    compute_cost_data,
    load_network_data,
    period_schedule,
)
from read import (read_pipeline_limit, read_well_capacities,
                  read_well_injection_capacities)
from result_metadata import build_model_parameters, parameter_frame
from solve_logging import logged_run, optimize_with_log


# ==================== 输入输出文件 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "output"
RESULT_FILE = RESULT_DIR / "CCS_FNE_gurobi_variables.xlsx"
# 包含七类变量，以及模型参数、算法参数、solve_summary、solve_log 工作表。

# 单案例图片输出路径；绘图开关、样式、地图边界和容差统一在 plot_results.py 修改。
# 绘图文件目录（单案例场景固定）；与 RESULT_FILE 同目录。
PLOT_OUTPUT_DIR = RESULT_DIR
PLOT_OUTPUT_IMAGE_NAME = "CCS_FNE_gurobi_results.png"
# 绘图总览图路径（单案例可直接修改文件名）。
PLOT_FILE = PLOT_OUTPUT_DIR / PLOT_OUTPUT_IMAGE_NAME
# 仅输出各期演化图目录。
PLOT_PERIOD_OUTPUT_DIR = PLOT_OUTPUT_DIR / f"{Path(PLOT_OUTPUT_IMAGE_NAME).stem}_periods"

# ==================== 模型参数 ====================
# 单次运行参数。
PERIOD_NUM = 15
NP = 3
DIAMETERS = D_LIST[-1]
CURVE = "linear"
COEFFICIENT = None
MAX_CAPTURE_AMOUNT = 1e6
NORMALIZE_CAPTURE_TARGET = False  # True（是）：非线性曲线与线性基准等总量；False（否）。
CAPTURE_AMOUNT_SEQUENCE = None

# ==================== 算法参数 ====================
TIME_LIMIT = 15000
GUROBI_OUTPUT_FLAG = 1  # 控制台日志：0 隐藏，1 显示；Excel 始终保存求解日志。
# 单算例同名 .log 保存完整运行日志；Gurobi 日志开关只影响终端显示。

def _capture_fraction(
    period: int,
    period_count: int,
    curve: str,
    coefficient: Optional[float] = None) -> float:
    """Return one normalized point on this single case's local curve."""

    if curve == "linear":
        if coefficient is not None:
            raise ValueError("The linear curve does not use a coefficient")
        return period / period_count
    if curve == "exponential":
        if (coefficient is None or not isfinite(coefficient)
                or coefficient == 0):
            raise ValueError(
                "The exponential coefficient must be finite and non-zero")
        return expm1(coefficient * period / period_count) / expm1(coefficient)
    if curve == "logarithmic":
        if (coefficient is None or not isfinite(coefficient)
                or coefficient <= -1 or coefficient == 0):
            raise ValueError(
                "The logarithmic coefficient must be finite, greater than "
                "-1, and non-zero")
        return log1p(coefficient * period / period_count) / log1p(coefficient)
    raise ValueError(f"Unknown capture-target curve: {curve}")


def build_capture_target(
    period_count: int,
    curve: str,
    coefficient: Optional[float],
    maximum_capture_amount: Optional[float],
    capture_amount_sequence: Optional[list[float] | tuple[float, ...]] = None,
    *, normalize_capture_target: bool = False,
) -> dict[int, float]:
    """Build targets, optionally matching the linear total; explicit sequence wins."""

    if period_count <= 0:
        raise ValueError("period_count must be positive")
    if capture_amount_sequence is not None:
        if len(capture_amount_sequence) != period_count:
            raise ValueError(
                f"CAPTURE_AMOUNT_SEQUENCE must have {period_count} values, "
                f"got {len(capture_amount_sequence)}")
        target = {}
        for period, value in enumerate(capture_amount_sequence, start=1):
            if (not isfinite(value) or value < 0):
                raise ValueError(
                    "CAPTURE_AMOUNT_SEQUENCE must contain finite, "
                    "non-negative values")
            target[period] = float(value)
        return target

    if (maximum_capture_amount is None or not isfinite(maximum_capture_amount)
            or maximum_capture_amount < 0):
        raise ValueError(
            "maximum_capture_amount must be finite and non-negative")
    if normalize_capture_target and curve != "linear":
        return generate_capture_targets(
            period_count, maximum_capture_amount, curve, coefficient,
            normalize_capture_target=True)
    return {
        period: round(
            maximum_capture_amount * _capture_fraction(
                period, period_count, curve, coefficient),
            3,
        )
        for period in range(1, period_count + 1)
    }


def with_local_capture_target(
    costs: CostData,
    curve: str,
    coefficient: Optional[float],
    maximum_capture_amount: Optional[float],
    capture_amount_sequence: Optional[list[float] | tuple[float, ...]] = None,
    *, normalize_capture_target: bool = False,
) -> CostData:
    """Replace only the target using parameters declared in this script."""

    period_count = len(costs.periods)
    if costs.periods != list(range(1, period_count + 1)):
        raise ValueError("cost periods must be consecutive and start at 1")
    target = build_capture_target(
        period_count,
        curve,
        coefficient,
        maximum_capture_amount,
        capture_amount_sequence,
        normalize_capture_target=normalize_capture_target,
    )
    return replace(costs, capture_target=target, model_parameters={
        **costs.model_parameters, "curve": curve, "coefficient": coefficient,
        "max_capture_amount": maximum_capture_amount,
        "normalize_capture_target": normalize_capture_target,
        "capture_amount_sequence": (None if capture_amount_sequence is None
                                    else list(capture_amount_sequence)),
        "target_source": ("generated_curve" if capture_amount_sequence is None
                          else "explicit_sequence")})


@dataclass(frozen=True)
class GurobiResult:
    """原模型指标、详细变量及 Gurobi 求解日志。"""

    status: str
    objective: Optional[float]
    lower_bound: Optional[float]
    gap: Optional[float]
    s: dict
    r: dict
    w: dict
    y: dict
    a: dict
    b: dict
    x: dict
    runtime: Optional[float] = None
    solver_log: list[str] = field(default_factory=list)
    model_parameters: dict = field(default_factory=dict)
    algorithm_parameters: dict = field(default_factory=dict)


def _status_name(status: int) -> str:
    names = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"STATUS_{status}")


def _integer_values(variables: gp.tupledict) -> dict:
    return {key: int(round(variable.X))
            for key, variable in variables.items()}


def _continuous_values(variables: gp.tupledict) -> dict:
    return {
        key: 0.0 if abs(variable.X) <= 1e-9 else float(variable.X)
        for key, variable in variables.items()
    }


def solve_original_model(
    network: NetworkData,
    costs: CostData,
    diameters: list[str],
    *,
    pipeline_install_limit: Optional[int] = None) -> GurobiResult:
    """按照当前 ``CCS_FNE.py`` 构建并求解原始混合整数模型。

    pipeline_install_limit 为每弧全期所有管径合计的最大建设条数，
    必须为非负整数，缺省读取本文件的 NP，不按年数换算。
    无可行解时返回空变量及求解日志，objective 和 gap 为 None。
    """


    well_capacity = read_well_injection_capacities(network.well_capacity, network.sinks)
    maximum_wells = read_well_capacities(network, costs)
    missing_diameters = sorted(set(diameters).difference(network.pipe_capacity))
    if missing_diameters:
        raise ValueError(f"Unknown pipeline diameters: {missing_diameters}")
    capacities = [network.pipe_capacity[d] for d in diameters]
    if any(left >= right for left, right in zip(capacities, capacities[1:])):
        raise ValueError("Pipeline diameters must be ordered by increasing capacity")
    pipeline_limit = read_pipeline_limit(
        NP if pipeline_install_limit is None else pipeline_install_limit)

    upper_capacity = {d: network.pipe_capacity[d] for d in diameters}
    selected_ijd = [(i, j, d) for i, j in network.ij for d in diameters]
    selected_ijdh = [(i, j, d, h) for i, j, d in selected_ijd
                     for h in costs.periods]

    # ------------------------------ 决策变量 ------------------------------
    model = gp.Model("CCS_FNE")
    s = model.addVars(costs.source_periods, name="s", vtype=gp.GRB.BINARY)
    r = model.addVars(costs.sink_periods, name="r", vtype=gp.GRB.BINARY)
    w = model.addVars(costs.sink_periods, name="w", vtype=gp.GRB.INTEGER)
    y = model.addVars(selected_ijdh, name="y", vtype=gp.GRB.INTEGER, lb=0)
    a = model.addVars(costs.source_periods, name="a", lb=0.0)
    b = model.addVars(costs.sink_periods, name="b", lb=0.0)
    x = model.addVars(selected_ijdh, name="x", lb=0.0)

    # ------------------------------ 排放源约束 ------------------------------
    model.addConstrs(a[i, g, h] <= network.source_capacity[i, g] *
                     gp.quicksum(s[i, g, t] for t in range(1, h + 1))
                     for i, g, h in costs.source_periods)
    model.addConstrs(
        gp.quicksum(s[i, g, h] for h in costs.periods) <= 1
        for i, g in network.ig)
    model.addConstrs(
        gp.quicksum(a[i, g, h]
                    for i in network.sources
                    for g in network.source_units[i])
        >= costs.capture_target[h]
        for h in costs.periods)

    # ------------------------------ 封存汇约束 ------------------------------
    model.addConstrs(b[j, h] <= well_capacity[j] *
                     gp.quicksum(w[j, t] for t in range(1, h + 1))
                     for j, h in costs.sink_periods)
    model.addConstrs(
        gp.quicksum(w[j, t]
                    for t in range(1, h + 1)) <= maximum_wells[j] *
        gp.quicksum(r[j, t] for t in range(1, h + 1))
        for j, h in costs.sink_periods)
    model.addConstrs(
        gp.quicksum(r[j, h] for h in costs.periods) <= 1
        for j in network.sinks)
    # ------------------------------ 管输约束 ------------------------------
    model.addConstrs(x[i, j, d, h] <= upper_capacity[d] *
                     gp.quicksum(y[i, j, d, t] for t in range(1, h + 1))
                     for i, j, d, h in selected_ijdh)

    model.addConstrs((
        gp.quicksum(y[i, j, d, h] for d in diameters for h in costs.periods)
        <= pipeline_limit
        for i, j in network.ij), name="pipe_total_limit")

    # ------------------------------ 节点流量平衡 ------------------------------
    model.addConstrs(
        gp.quicksum(x[i, j, d, h]
                    for j in network.successors[i] for d in diameters) -
        gp.quicksum(x[j, i, d, h]
                    for j in network.predecessors[i] for d in diameters) -
        gp.quicksum(a[i, g, h] for g in network.source_units[i]) == 0
        for i in network.sources for h in costs.periods)
    model.addConstrs(
        gp.quicksum(x[j, i, d, h]
                    for i in network.successors[j] for d in diameters) -
        gp.quicksum(x[i, j, d, h]
                    for i in network.predecessors[j] for d in diameters) +
        b[j, h] == 0
        for j in network.sinks for h in costs.periods)
    model.addConstrs(
        gp.quicksum(x[i, j, d, h]
                    for j in network.successors[i] for d in diameters) -
        gp.quicksum(x[j, i, d, h]
                    for j in network.predecessors[i] for d in diameters) == 0
        for i in network.intersections for h in costs.periods)

    # ------------------------------ 目标函数 ------------------------------
    objective = (gp.quicksum(costs.source_fixed[i, g, h] * s[i, g, h]
                             for i, g, h in costs.source_periods) +
                 gp.quicksum(costs.sink_fixed[j, h] * r[j, h]
                             for j, h in costs.sink_periods) +
                 gp.quicksum(costs.well_fixed[j, h] * w[j, h]
                             for j, h in costs.sink_periods) +
                 gp.quicksum(costs.pipe_fixed[i, j, d, h] * y[i, j, d, h]
                             for i, j, d, h in selected_ijdh) +
                 gp.quicksum(costs.pipe_variable[i, j, d, h] * x[i, j, d, h]
                             for i, j, d, h in selected_ijdh) +
                 gp.quicksum(costs.source_variable[i, g, h] * a[i, g, h]
                             for i, g, h in costs.source_periods) +
                 gp.quicksum(costs.sink_variable[j, h] * b[j, h]
                             for j, h in costs.sink_periods))
    model.setObjective(objective, gp.GRB.MINIMIZE)
    # OutputFlag=0 会禁用 MESSAGE 回调；仅通过 LogToConsole 控制屏幕输出。
    model.Params.LogToConsole = GUROBI_OUTPUT_FLAG
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.MIPFocus=1
    model.Params.Heuristics=1
    model.Params.Method = 2  # 根节点初始松弛使用 barrier
    model.Params.NodeMethod = 2  # 后续节点松弛使用 barrier
    model.Params.Crossover = 0  # 关闭 crossover
    model_parameters = build_model_parameters(network, costs, diameters, pipeline_limit)
    algorithm_parameters = {
        name: getattr(model.Params, name) for name in (
            "TimeLimit", "MIPGap", "MIPFocus", "Heuristics", "Method", "NodeMethod",
            "Crossover", "Threads", "Seed", "FeasibilityTol", "IntFeasTol",
            "OptimalityTol", "NumericFocus", "OutputFlag", "LogToConsole")}
    algorithm_parameters.update(algorithm="Gurobi", time_limit_scope="optimization",
                                gurobi_version=".".join(map(str, gp.gurobi.version())))
    solver_log = []

    def record_log(current, where):
        if where == gp.GRB.Callback.MESSAGE:
            solver_log.extend(current.cbGet(gp.GRB.Callback.MSG_STRING).splitlines())

    try:
        optimize_with_log(model, record_log)
        has_solution = model.SolCount > 0
        try:
            lower_bound = float(model.ObjBound)
        except (AttributeError, gp.GurobiError):
            lower_bound = None
        if lower_bound is not None and not isfinite(lower_bound):
            lower_bound = None
        return GurobiResult(
            status=_status_name(model.Status),
            objective=float(model.ObjVal) if has_solution else None,
            lower_bound=lower_bound,
            gap=float(model.MIPGap) if has_solution else None,
            s=_integer_values(s) if has_solution else {},
            r=_integer_values(r) if has_solution else {},
            w=_integer_values(w) if has_solution else {},
            y=_integer_values(y) if has_solution else {},
            a=_continuous_values(a) if has_solution else {},
            b=_continuous_values(b) if has_solution else {},
            x=_continuous_values(x) if has_solution else {},
            runtime=float(model.Runtime),
            solver_log=solver_log,
            model_parameters=model_parameters,
            algorithm_parameters=algorithm_parameters,
        )
    finally:
        model.dispose()


def _variable_frame(index_columns: list[str], index_rows: list[tuple],
                    periods: list[int], values: dict) -> pd.DataFrame:
    period_columns = [f"period{period}" for period in periods]
    rows = [
        [*indices,
         *(values.get((*indices, period), 0) for period in periods)]
        for indices in index_rows
    ]
    return pd.DataFrame(rows, columns=[*index_columns, *period_columns])


def save_gurobi_results(result: GurobiResult,
                        network: NetworkData,
                        costs: CostData,
                        diameters: list[str]) -> Path:
    """将变量、求解摘要和原始求解日志写入 Excel；无解时仅保存摘要与日志。"""

    frames = {
        "variable_s": _variable_frame(
            ["source", "unit"], list(network.ig), costs.periods, result.s),
        "variable_a": _variable_frame(
            ["source", "unit"], list(network.ig), costs.periods, result.a),
        "variable_r": _variable_frame(
            ["sink"], [(sink,) for sink in network.sinks],
            costs.periods, result.r),
        "variable_w": _variable_frame(
            ["sink"], [(sink,) for sink in network.sinks],
            costs.periods, result.w),
        "variable_b": _variable_frame(
            ["sink"], [(sink,) for sink in network.sinks],
            costs.periods, result.b),
        "variable_x": _variable_frame(
            ["start", "end", "diameter"],
            [(start, end, diameter)
             for start, end in network.ij for diameter in diameters],
            costs.periods, result.x),
        "variable_y": _variable_frame(
            ["start", "end", "diameter"],
            [(start, end, diameter)
             for start, end in network.ij for diameter in diameters],
            costs.periods, result.y),
    } if result.objective is not None else {}

    metrics = ("status", "objective", "lower_bound", "gap", "runtime")
    frames["solve_summary"] = pd.DataFrame(
        [(name, getattr(result, name)) for name in metrics], columns=["metric", "value"])
    frames["solve_log"] = pd.DataFrame(
        enumerate(result.solver_log, start=1), columns=["line", "message"])
    frames["model_parameters"] = parameter_frame(result.model_parameters)
    frames["algorithm_parameters"] = parameter_frame(result.algorithm_parameters)

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_FILE.with_name(
        f".{RESULT_FILE.stem}.tmp{RESULT_FILE.suffix}")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    os.replace(temporary, RESULT_FILE)
    return RESULT_FILE


@logged_run(lambda: RESULT_FILE.with_suffix('.log'))
def main() -> None:
    """读取数据，使用本文件配置的全期建设上限求解一个 CCS_FNE 算例。"""

    if CAPTURE_AMOUNT_SEQUENCE is None:
        print(f"period_num={PERIOD_NUM}, NP={NP} (per arc, full horizon), "
              f"curve={CURVE}, coef={COEFFICIENT}, "
              f"max_capture={MAX_CAPTURE_AMOUNT}, D={DIAMETERS}")
    else:
        print(f"period_num={PERIOD_NUM}, NP={NP} (per arc, full horizon), "
              f"curve={CURVE}, coef={COEFFICIENT}, "
              f"capture_sequence={CAPTURE_AMOUNT_SEQUENCE}, "
              f"D={DIAMETERS}")
    network = load_network_data(SCRIPT_DIR)
    periods, durations = period_schedule(PERIOD_NUM)
    base_costs = compute_cost_data(
        network,
        periods,
        durations,
        apply_shared_capture_target=False,
    )
    costs = with_local_capture_target(
        base_costs,
        CURVE,
        COEFFICIENT,
        MAX_CAPTURE_AMOUNT,
        CAPTURE_AMOUNT_SEQUENCE,
        normalize_capture_target=NORMALIZE_CAPTURE_TARGET,
    )
    costs = replace(costs, model_parameters={**costs.model_parameters,
                                           "data_dir": str(SCRIPT_DIR)})
    print(f"normalize_capture_target={NORMALIZE_CAPTURE_TARGET}")
    print(f"capture_target={costs.capture_target}")
    result = solve_original_model(
        network, costs, DIAMETERS, pipeline_install_limit=NP)
    print(f"status={result.status}")
    print(f"objective={result.objective}")
    print(f"lower bound={result.lower_bound}")
    print(f"gap={result.gap}")
    print(f"runtime={result.runtime}s")
    output_file = save_gurobi_results(
        result, network, costs, DIAMETERS)
    print(f"Gurobi results saved: {output_file}")
    if result.objective is None:
        print("未取得可行解，已保存求解摘要与日志，不导出建设方案或绘图。")
        return
    image_files = plot_results_from_excel(
        output_file, network, diameters=DIAMETERS,
        output_file=PLOT_FILE, period_output_dir=PLOT_PERIOD_OUTPUT_DIR,
        evolution_only=True,
    )
    for image_file in image_files:
        print(f"result_plot={image_file}")


if __name__ == "__main__":
    main()
