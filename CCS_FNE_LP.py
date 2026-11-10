"""批量求解多周期 CCS 混合整数规划模型的线性松弛。

本脚本独立实现与 CCS_FNE.py 相同的参数网格和建模顺序，
不导入或调用其他求解器。仅将 s/r 松弛到 [0,1]、w/y 松弛为非负连续变量，
并在最优时额外保存节点流量平衡和捕集目标约束的乘子。
结果表记录求解状态、求解时间（秒）和 LP 最优目标值，不记录相对间隙或中间快照。
仅 optimal 状态填写 lp_objective，其他状态留空。
在时限内达到最优记为 optimal，超时或未满足最优性容差记为 suboptimal，中断记为 interrupted。
依据正常结果表跳过当前网格中 optimal/time_limit 的组合；中断时保留已输出结果。
沿用拉格朗日约定：节点乘子为 -Pi，捕集目标乘子为 Pi（非负）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
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
MULTIPLIERS_DIR = SCRIPT_DIR / "multipliers" / ("normalized" if NORMALIZE_CAPTURE_TARGET else "")
RESULT_DIR = SCRIPT_DIR / "output" / ("normalized" if NORMALIZE_CAPTURE_TARGET else "")
RESULT_FILE = RESULT_DIR / f"CCS_FNE_LP_{TY}Y_result.csv"

# 单个 LP 的总优化时限（秒）；runtime 使用 Gurobi Runtime，不含建模和文件写入。
TIME_LIMIT = 3600
GUROBI_OUTPUT_FLAG = 1  # Gurobi 求解日志：0 关闭，1 开启。
RESULT_COLUMNS = [
    "period_num", "D", "Np", "max_capture_amount", "coef", "curve", "status",
    "runtime", "lp_objective"]


@dataclass(frozen=True)
class DualMultipliers:
    """节点流量平衡的自由乘子 -Pi 和捕集目标的非负乘子 Pi。"""

    node_multipliers: dict[tuple, float]
    target_multipliers: dict[int, float]


@dataclass(frozen=True)
class SolveResult:
    """单个 LP 算例的状态、实际优化秒数、最优乘子和最优目标值。"""

    status: str
    runtime: Optional[float]
    multipliers: Optional[DualMultipliers] = None
    lp_objective: Optional[float] = None


def _status_name(status: int, runtime: float) -> str:
    """将求解状态映射为批处理状态；suboptimal 表示未获最优证明。"""

    if status == gp.GRB.INTERRUPTED:
        return "interrupted"
    if status == gp.GRB.OPTIMAL:
        return "optimal" if runtime <= TIME_LIMIT else "suboptimal"
    if status == gp.GRB.TIME_LIMIT:
        return "suboptimal"
    if status == gp.GRB.SUBOPTIMAL:
        return "suboptimal"
    if status == gp.GRB.INFEASIBLE:
        return "infeasible"
    if status == gp.GRB.UNBOUNDED:
        return "unbounded"
    if status == gp.GRB.INF_OR_UNBD and runtime >= TIME_LIMIT:
        return "suboptimal"
    # 数值错误等异常不能伪装为最优或超时，必须向调用方报告。
    raise RuntimeError(
        f"Unexpected LP solver status {status} after {runtime:.6f} seconds")


def _target_multiplier(pi_value: float, period: int,
                       tolerance: float) -> float:
    """校验 F4 对偶值，仅投影数值容差内的微小负值。"""

    value = float(pi_value)
    if not math.isfinite(value):
        raise RuntimeError(
            f"F4 dual multiplier for period {period} is not finite: {value}")
    if value < -tolerance:
        raise RuntimeError(
            f"F4 dual multiplier for period {period} is negative: {value}")
    return max(0.0, value)


def solve_relaxation(network: NetworkData, costs: CostData, diameters: list[str],
                     pipeline_install_limit: int) -> SolveResult:
    """求解一个算例的线性松弛。

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
    model = gp.Model("CCS_FNE_LP")
    s = model.addVars(costs.source_periods, name="s", lb=0.0, ub=1.0)
    r = model.addVars(costs.sink_periods, name="r", lb=0.0, ub=1.0)
    w = model.addVars(costs.sink_periods, name="w", lb=0.0)
    y = model.addVars(selected_ijdh, name="y", lb=0.0)
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
    capture_target_constraints = model.addConstrs(
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
    source_balance_constraints = model.addConstrs(
        gp.quicksum(x[i, j, d, h]
                    for j in network.successors[i] for d in diameters) -
        gp.quicksum(x[j, i, d, h]
                    for j in network.predecessors[i] for d in diameters) -
        gp.quicksum(a[i, g, h] for g in network.source_units[i]) == 0
        for i in network.sources for h in costs.periods)
    sink_balance_constraints = model.addConstrs(
        gp.quicksum(x[j, i, d, h]
                    for i in network.successors[j] for d in diameters) -
        gp.quicksum(x[i, j, d, h]
                    for i in network.predecessors[j] for d in diameters) +
        b[j, h] == 0
        for j in network.sinks for h in costs.periods)
    intersection_balance_constraints = model.addConstrs(
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
    model.Params.OutputFlag = GUROBI_OUTPUT_FLAG
    model.Params.TimeLimit = TIME_LIMIT
    # model.Params.Method = 2  # 内点法（Barrier）。
    # model.Params.Crossover = 0  # 保留内点解，不额外构造基解。
    # model.Params.NumericFocus = 2  # 加强数值稳定性处理。

    runtime = 0.0
    started = perf_counter()
    try:
        try:
            model.optimize()
            runtime += float(model.Runtime)
            # 若预处理无法区分不可行和无界，仅用剩余时限重新求解。
            if model.Status == gp.GRB.INF_OR_UNBD and runtime < TIME_LIMIT:
                model.Params.DualReductions = 0
                model.Params.TimeLimit = TIME_LIMIT - runtime
                model.reset()
                model.optimize()
                runtime += float(model.Runtime)
        except KeyboardInterrupt:
            # Python 中断可能使 Runtime 尚不可读，此时记录实际优化经过的秒数。
            return SolveResult("interrupted", perf_counter() - started)

        status = _status_name(model.Status, runtime)
        multipliers = None
        lp_objective = None
        if status == "optimal":
            lp_objective = float(model.ObjVal)
            tolerance = max(1e-9, float(model.Params.OptimalityTol))
            multipliers = DualMultipliers(
                node_multipliers={
                    key: -float(row.Pi)
                    for rows in (source_balance_constraints, sink_balance_constraints,
                                 intersection_balance_constraints)
                    for key, row in rows.items()},
                target_multipliers={
                    h: _target_multiplier(row.Pi, h, tolerance)
                    for h, row in capture_target_constraints.items()})
        return SolveResult(status, runtime, multipliers=multipliers,
                           lp_objective=lp_objective)
    finally:
        model.dispose()


def _case_grid():
    """按当前 experiment_setup 配置生成本次批量求解的参数网格。"""

    return product(PERIOD_NUM_LIST, MAX_CAPTURE_AMOUNT_LIST, NP_LIST,
                   CURVE_SPECS, D_LIST)


def _case_key(period_num: int, curve: str, coefficient,
              diameters: list[str], np: int, *,
              maximum_capture_amount: float,
              normalize_capture_target: Optional[bool] = None) -> str:
    """按算例参数生成稳定标识，用于区分乘子输出文件。"""

    if normalize_capture_target is None:
        normalize_capture_target = NORMALIZE_CAPTURE_TARGET

    return json.dumps(
        {
            "construction_limits": "full_horizon_v1",
            "capture_target": ("linear_total_v1" if normalize_capture_target
                               else "common_max_v1"),
            "period_num": period_num,
            "Np": np,
            "max_capture_amount": maximum_capture_amount,
            "curve": curve,
            "coef": coefficient,
            "diameters": diameters,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"))


def _coefficient_token(coefficient) -> str:
    """生成稳定且适合文件名的曲线参数文本。"""

    return (format(coefficient, "g")
            if isinstance(coefficient, float) else str(coefficient))


def multiplier_paths(period_num: int, curve: str,
                     coefficient, diameters: list[str], np: int, *,
                     maximum_capture_amount: float) -> dict[str, Path]:
    """目录包含完整算例参数；文件摘要区分同数量不同管径等组合。"""
    if curve not in ("linear", "exponential", "logarithmic"):
        raise ValueError(f"Unknown capture-target curve: {curve}")
    directory_name = f"period{period_num}_NP{np}_{curve}"
    if curve != "linear":
        directory_name += f"_coef{_coefficient_token(coefficient)}"
    directory_name += f"_D{len(diameters)}"
    key = _case_key(period_num, curve, coefficient, diameters, np,
                    maximum_capture_amount=maximum_capture_amount)
    digest = sha256(key.encode("utf-8")).hexdigest()[:16]
    directory = MULTIPLIERS_DIR / directory_name
    suffix = f"{TY}_{period_num}_p{len(diameters)}_{digest}.csv"
    return {name: directory / f"{name}_{suffix}" for name in ("node", "target")}


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """先写同目录临时文件再原子替换，避免中断留下半个 CSV。"""

    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_results(rows: list[dict]) -> None:
    """原子保存已有记录与本次新增结果合并后的结果表。"""

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(pd.DataFrame(rows, columns=RESULT_COLUMNS), RESULT_FILE)


def save_multipliers(multipliers: DualMultipliers,
                     paths: dict[str, Path]) -> None:
    """把两组初始乘子分别保存成结构明确的 CSV 文件。"""

    node_frame = pd.DataFrame(
        ((node, h, value)
         for (node, h), value in multipliers.node_multipliers.items()),
        columns=["node", "h", "node_multiplier"],
    )
    target_frame = pd.DataFrame(
        ((h, value)
         for h, value in multipliers.target_multipliers.items()),
        columns=["h", "target_multiplier"],
    )

    paths["node"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(node_frame, paths["node"])
    _atomic_write_csv(target_frame, paths["target"])


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
    if result.multipliers is not None:
        paths = multiplier_paths(period_num, curve, coefficient,
                                 diameters, np,
                                 maximum_capture_amount=maximum_capture_amount)
        save_multipliers(result.multipliers, paths)
        print(f"Multipliers saved: node={paths['node']}, target={paths['target']}")
    print(f"status={result.status}, runtime={result.runtime} seconds, "
          f"lp_objective={result.lp_objective}")
    return {
        "period_num": period_num,
        "D": len(diameters),
        "Np": np,
        "max_capture_amount": maximum_capture_amount,
        "coef": coefficient,
        "curve": curve,
        "status": result.status,
        "runtime": result.runtime,
        "lp_objective": result.lp_objective}


def main() -> None:
    """根据正常结果表跳过当前网格中 optimal/time_limit/root_optimal 的算例。"""

    cases = list(_case_grid())
    rows, completed = read_result_rows(RESULT_FILE, cases)
    if all(grid_instance_key(case) in completed for case in cases):
        print(f"All current cases already completed. Results: {RESULT_FILE}")
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MULTIPLIERS_DIR.mkdir(parents=True, exist_ok=True)
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
