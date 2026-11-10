"""使用内点法批量统计多周期 CCS 模型的初始根松弛求解时间。

本脚本独立实现与 CCS_FNE.py 相同的参数网格和建模顺序，
保留 s/r 的二进制类型和 w/y 的整数类型，以 NodeLimit=0 在初始根松弛后停止。
Method=2、NodeMethod=2、Crossover=0，使用 Barrier 且不额外构造基解。
NumericFocus=2，加强数值稳定性处理。
root_lp_runtime 仅取 MIP 日志 Root relaxation 行中的 seconds，精度与日志一致，
不使用包含预处理等开销的 Model.Runtime，也不提取或保存对偶乘子。
root_lp_objective 在最优时取同一日志行的 objective，次优时取根节点 Barrier 的
Sub-optimal termination 行中的 objective，精度与日志一致；未报告目标值时留空。
没有该日志时，时间留空；根松弛超时或中断时，若日志有时间则记录已耗秒数。
root_optimal/root_infeasible/root_unbounded/root_cutoff 表示根松弛的结束状态，
root_suboptimal 表示根松弛未满足最优性容差，此时记录次优目标值；
其余状态表示 MIP 的结束原因，suboptimal 表示未满足最优性容差；
依据正常结果表跳过当前网格中 optimal/time_limit 的组合；中断时保留已输出结果。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import gurobipy as gp
import pandas as pd

from experiment_setup import (CURVE_SPECS, D_LIST, MAX_CAPTURE_AMOUNT_LIST,
                              NP_LIST,
                              PERIOD_NUM_LIST, TY, CostData, NetworkData,
                              compute_cost_data,
                              load_network_data, period_schedule,
                              with_capture_target)
from read import (read_pipeline_limit,
                  read_well_capacities, read_well_injection_capacities)

from read import grid_instance_key, read_result_rows, result_instance_key

NORMALIZE_CAPTURE_TARGET = False  # True：是，非线性序列与线性基准等总量；False：否。

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "output" / ("normalized" if NORMALIZE_CAPTURE_TARGET else "")
RESULT_FILE = RESULT_DIR / f"CCS_FNE_RLP_IM_{TY}Y_result.csv"

# 单个 MIP 的总优化时限（秒，含预处理），不是 root_lp_runtime 的统计口径。
TIME_LIMIT = 3600
GUROBI_OUTPUT_FLAG = 1  # 控制台日志：0 隐藏，1 显示；MESSAGE 回调始终启用。
RESULT_COLUMNS = [
    "period_num", "D", "Np", "max_capture_amount", "coef", "curve", "status",
    "root_lp_runtime", "root_lp_objective"]
ROOT_RELAXATION_PATTERN = re.compile(
    r"Root relaxation:\s*([^,\n]+),\s*[\d,]+ iterations,\s*"
    r"([\d.eE+-]+) seconds")
SUBOPTIMAL_OBJECTIVE_PATTERN = re.compile(
    r"Sub-optimal termination\s*-\s*objective\s+([\d.eE+-]+)")


@dataclass(frozen=True)
class SolveResult:
    """单个算例的状态、根松弛日志时间和最优或次优目标值；未报告的值为 None。"""

    status: str
    root_lp_runtime: Optional[float]
    root_lp_objective: Optional[float] = None


def _status_name(status: int, root_outcome: Optional[str]) -> str:
    """优先报告根松弛的结束状态；NODE_LIMIT 不代表整个 MIP 已达到最优。"""

    if root_outcome is not None and root_outcome.startswith("objective "):
        return "root_optimal"
    if root_outcome == "sub-optimal":
        return "root_suboptimal"
    if root_outcome in {"infeasible", "unbounded", "cutoff"}:
        return f"root_{root_outcome}"
    names = {
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.SUBOPTIMAL: "suboptimal",
        gp.GRB.TIME_LIMIT: "time_limit",
        gp.GRB.INTERRUPTED: "interrupted",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.INF_OR_UNBD: "inf_or_unbd",
    }
    if status in names:
        return names[status]
    # 包括到达节点限制却未捕获根松弛日志，不能伪装为已完成的时间统计。
    raise RuntimeError(
        f"Unexpected root LP solver status {status}; root outcome: {root_outcome!r}")


def solve_relaxation(network: NetworkData, costs: CostData, diameters: list[str],
                     pipeline_install_limit: int) -> SolveResult:
    """保留整数类型，用无 crossover 的内点法求解初始根松弛并读取日志时间。

    捕集单元全期至多建设一次，封存井保留全期累计井数上限及开场联动。
    每条有向弧的全期管道总数（所有管径合计）不超过 pipeline_install_limit，
    该上限必须显式传入；批量运行从 NP_LIST 逐项取值，不按周期年数换算。
    """

    well_capacity = read_well_injection_capacities(network.well_capacity, network.sinks)
    maximum_wells = read_well_capacities(network, costs)
    pipeline_limit = read_pipeline_limit(pipeline_install_limit)

    missing_diameters = sorted(set(diameters).difference(network.pipe_capacity))
    if missing_diameters:
        raise ValueError(f"Unknown pipeline diameters: {missing_diameters}")
    capacities = [network.pipe_capacity[d] for d in diameters]
    if any(left >= right for left, right in zip(capacities, capacities[1:])):
        raise ValueError("Pipeline diameters must be ordered by increasing capacity")
    upper_capacity = {d: network.pipe_capacity[d] for d in diameters}
    selected_ijd = [(i, j, d) for i, j in network.ij for d in diameters]
    selected_ijdh = [(i, j, d, h) for i, j, d in selected_ijd
                     for h in costs.periods]

    # ------------------------------ 决策变量 ------------------------------
    model = gp.Model("CCS_FNE_RLP_IM")
    s = model.addVars(costs.source_periods, name="s", vtype=gp.GRB.BINARY)
    r = model.addVars(costs.sink_periods, name="r", vtype=gp.GRB.BINARY)
    w = model.addVars(costs.sink_periods, name="w", vtype=gp.GRB.INTEGER)
    y = model.addVars(selected_ijdh, name="y", vtype=gp.GRB.INTEGER)
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

    model.addConstrs(
        (gp.quicksum(y[i, j, d, h] for d in diameters for h in costs.periods)
         <= pipeline_limit for i, j in network.ij), name="pipe_total_limit")

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
    model.Params.LogToConsole = GUROBI_OUTPUT_FLAG
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.NodeLimit = 0
    model.Params.Method = 2  # 根松弛使用内点法（Barrier）。
    model.Params.NodeMethod = 2  # MIP 关闭 crossover 要求两种 Method 均为 Barrier。
    model.Params.Crossover = 0  # 保留内点解，不额外构造基解。

    root_outcome = None
    root_lp_runtime = None
    root_lp_objective = None
    root_barrier_started = False
    root_suboptimal_objective = None

    def record_root_relaxation(model: gp.Model, where: int) -> None:
        nonlocal root_outcome, root_lp_runtime, root_lp_objective
        nonlocal root_barrier_started, root_suboptimal_objective
        if where != gp.GRB.Callback.MESSAGE or root_outcome is not None:
            return
        message = model.cbGet(gp.GRB.Callback.MSG_STRING)
        if "Root barrier log..." in message:
            root_barrier_started = True
        if root_barrier_started:
            match = SUBOPTIMAL_OBJECTIVE_PATTERN.search(message)
            if match is not None:
                root_suboptimal_objective = float(match.group(1))
        match = ROOT_RELAXATION_PATTERN.search(message)
        if match is not None:
            root_outcome = match.group(1).strip()
            root_lp_runtime = float(match.group(2))
            if root_outcome.startswith("objective "):
                root_lp_objective = float(root_outcome.split()[1])
            elif root_outcome == "sub-optimal":
                root_lp_objective = root_suboptimal_objective

    try:
        try:
            model.optimize(record_root_relaxation)
        except KeyboardInterrupt:
            return SolveResult("interrupted", root_lp_runtime, root_lp_objective)

        status = _status_name(model.Status, root_outcome)
        if (root_lp_objective is None
                and status in {"root_suboptimal", "suboptimal", "time_limit"}):
            root_lp_objective = root_suboptimal_objective
        return SolveResult(status, root_lp_runtime, root_lp_objective)
    finally:
        model.dispose()


def _case_grid():
    """按当前 experiment_setup 配置生成本次批量求解的参数网格。"""

    return product(PERIOD_NUM_LIST, MAX_CAPTURE_AMOUNT_LIST, NP_LIST,
                   CURVE_SPECS, D_LIST)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """先写同目录临时文件再原子替换，避免中断留下半个 CSV。"""

    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_results(rows: list[dict]) -> None:
    """原子保存已有记录与本次新增结果合并后的结果表。"""

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(pd.DataFrame(rows, columns=RESULT_COLUMNS), RESULT_FILE)


def run_case(network: NetworkData, costs: CostData, diameters: list[str],
             period_num: int, curve: str, coefficient,
             np: int, *, maximum_capture_amount: float) -> dict:
    """求解单个算例；np 是每弧全期管道总数上限，不使用逐期建设额度。"""

    print(f"period_num={period_num}, Np={np}, curve={curve}, "
          f"coef={coefficient}, "
          f"max_capture_amount={maximum_capture_amount:.3f}, "
          f"capture_target_max={max(costs.capture_target.values()):.3f}, "
          f"D={diameters}")
    result = solve_relaxation(
        network, costs, diameters,
        pipeline_install_limit=np)
    print(f"status={result.status}, root_lp_runtime={result.root_lp_runtime} seconds, "
          f"root_lp_objective={result.root_lp_objective}")
    return {
        "period_num": period_num,
        "D": len(diameters),
        "Np": np,
        "max_capture_amount": maximum_capture_amount,
        "coef": coefficient,
        "curve": curve,
        "status": result.status,
        "root_lp_runtime": result.root_lp_runtime,
        "root_lp_objective": result.root_lp_objective}


def main() -> None:
    """根据正常结果表跳过当前网格中 optimal/time_limit/root_optimal 的算例。"""

    cases = list(_case_grid())
    rows, completed = read_result_rows(RESULT_FILE, cases)
    if all(grid_instance_key(case) in completed for case in cases):
        print(f"All current cases already completed. Results: {RESULT_FILE}")
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    network = load_network_data(SCRIPT_DIR)
    active_period_num = None
    printed_case = False
    try:
        for case in cases:
            key = grid_instance_key(case)
            if key in completed:
                continue
            period_num, maximum_capture_amount, np, (curve, coefficient), diameters = case
            # 同一周期划分共用本次运行内的折现成本。
            if period_num != active_period_num:
                periods, durations = period_schedule(period_num)
                base_costs = compute_cost_data(
                    network, periods, durations, apply_shared_capture_target=False)
                active_period_num = period_num
            costs = with_capture_target(
                base_costs, curve, coefficient,
                maximum_capture_amount=maximum_capture_amount,
                normalize_capture_target=NORMALIZE_CAPTURE_TARGET)
            if printed_case:
                print()
            row = run_case(network, costs, diameters, period_num,
                           curve, coefficient, np,
                           maximum_capture_amount=maximum_capture_amount)
            rows = [old for old in rows if result_instance_key(old) != key]
            rows.append(row)
            save_results(rows)
            print(f"Results saved: {RESULT_FILE}")
            if row["status"].upper() == "INTERRUPTED":
                print(f"Interrupted. Current results: {RESULT_FILE}")
                return
            printed_case = True
    except KeyboardInterrupt:
        print(f"Interrupted. Completed results: {RESULT_FILE}")
        raise
    print(f"All cases completed. Results: {RESULT_FILE}")


if __name__ == "__main__":
    main()
