"""批量求解多周期 CCS 混合整数规划模型。

脚本遍历周期数、捕集目标上限、每弧全期管道总数上限 NP_LIST、捕集目标曲线和候选管径集合。
依据正常结果 CSV 跳过当前网格中 optimal/time_limit 的组合，每完成一例就原子更新结果。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Optional

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

# ==================== 输入输出文件 ====================
NORMALIZE_CAPTURE_TARGET = False  # True：是，非线性序列与线性基准等总量；False：否。

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "output" / ("normalized" if NORMALIZE_CAPTURE_TARGET else "")
RESULT_FILE = RESULT_DIR / f"CCS_FNE_{TY}Y_result.csv"

# ==================== 模型参数 ====================
# 规划期、管径、捕集目标及管道数量等模型参数从 experiment_setup 导入。

# ==================== 算法参数 ====================
# 时限和快照时刻均以秒配置；2h/5h 为兼容历史文件的列名，实际阈值为 1h/3h。
TIME_LIMIT = 11_000
GUROBI_OUTPUT_FLAG = 1  # Gurobi 求解日志：0 关闭，1 开启。
GUROBI_THREADS = 0  # Gurobi 求解线程数：0 自动选择，正整数指定线程数。
RECORD_TIMES = (("1h", 3_600), ("3h", 10_800))

# 内部常量：结果字段与状态记录。
METRIC_COLUMNS = [
    "UB", "LB", "gap", "2hUB", "2hLB", "2hgap", "5hUB", "5hLB", "5hgap"]
RESULT_COLUMNS = [
    "period_num", "D", "Np", "max_capture_amount", "coef", *METRIC_COLUMNS,
    "curve", "status"]


@dataclass
class Snapshot:
    """求解运行到指定时刻时的上界、下界和 Gap 快照。"""

    label: str
    time: float
    recorded: bool = False
    ub: Optional[float] = None
    lb: Optional[float] = None
    gap: Optional[float] = None


@dataclass(frozen=True)
class SolveResult:
    """单个参数算例的最终状态和两个中间时刻的求解指标。"""

    status: str
    ub: Optional[float]
    lb: Optional[float]
    gap: Optional[float]
    ub_2h: Optional[float]
    lb_2h: Optional[float]
    gap_2h: Optional[float]
    ub_5h: Optional[float]
    lb_5h: Optional[float]
    gap_5h: Optional[float]

    @classmethod
    def empty(cls, status: str) -> "SolveResult":
        """为不可行或无界算例生成数值字段全部为 "None" 的结果。"""

        return cls(status, None, None, None, None, None, None, None, None, None)

    def metrics(self) -> list[Optional[float]]:
        return [
            self.ub, self.lb, self.gap, self.ub_2h, self.lb_2h, self.gap_2h,
            self.ub_5h, self.lb_5h, self.gap_5h]


def _finite_gurobi_value(value: float) -> Optional[float]:
    """把 Gurobi 用大数表示的无穷值转换为 "None"，防止其进入最终结果。"""

    if not math.isfinite(value) or abs(value) >= gp.GRB.INFINITY / 2:
        return None
    return float(value)


def _relative_gap(ub: Optional[float], lb: Optional[float]) -> Optional[float]:
    """按 |UB-LB|/|UB| 计算相对 Gap；缺少可行解或 UB=0 时不强行计算。"""

    if ub is None or lb is None:
        return None
    if ub == lb:
        return 0.0
    if ub == 0:
        return None
    return abs(ub - lb) / abs(ub)


class Recorder:
    """通过 MIP 回调记录指定运行时刻的 incumbent、最优下界和相对 Gap。"""
    def __init__(self) -> None:
        self.snapshots = [Snapshot(label, time) for label, time in RECORD_TIMES]

    def record(self, model: gp.Model, runtime: float) -> None:
        # 每个时刻只记录一次。如果该时刻尚无 incumbent，UB 和 Gap 记为 None，但仍保留有效的下界。
        for snapshot in self.snapshots:
            if snapshot.recorded or runtime < snapshot.time:
                continue
            snapshot.ub = _finite_gurobi_value(
                model.cbGet(gp.GRB.Callback.MIP_OBJBST))
            snapshot.lb = _finite_gurobi_value(
                model.cbGet(gp.GRB.Callback.MIP_OBJBND))
            snapshot.gap = _relative_gap(snapshot.ub, snapshot.lb)
            snapshot.recorded = True


def make_callback(recorder: Recorder):
    """创建只响应 MIP 节点事件的 Gurobi 回调函数。"""
    def callback(model: gp.Model, where: int) -> None:
        if where == gp.GRB.Callback.MIP:
            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
            recorder.record(model, runtime)

    return callback


def _status_name(status: int) -> str:
    """把 Gurobi 数字状态码转换为便于写入结果表的稳定文本。"""

    names = {
        gp.GRB.LOADED: "LOADED",
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.CUTOFF: "CUTOFF",
        gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        gp.GRB.NODE_LIMIT: "NODE_LIMIT",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        gp.GRB.WORK_LIMIT: "WORK_LIMIT",
        gp.GRB.MEM_LIMIT: "MEM_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def build_milp(network: NetworkData, costs: CostData, diameters: list[str],
               pipeline_install_limit: int):
    """构建原多期模型，返回模型和七组详细变量，不启动求解。

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
    model = gp.Model("CCS_FNE")
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
    return model, {"s": s, "a": a, "r": r, "w": w, "b": b, "x": x, "y": y}


def apply_candidate_restrictions(variables, candidates) -> None:
    """仅禁用候选集以外的设施功能，保留所有节点的原流量平衡。"""

    for name in ("s", "a"):
        for key, variable in variables[name].items():
            if key[0] not in candidates["sources"]:
                variable.UB = 0.0
    for name in ("r", "w", "b"):
        for key, variable in variables[name].items():
            if key[0] not in candidates["sinks"]:
                variable.UB = 0.0
    for name in ("x", "y"):
        for key, variable in variables[name].items():
            if key[:2] not in candidates["arcs"]:
                variable.UB = 0.0


def apply_node_bounds(variables, node_bounds) -> None:
    """Intersect a model's variable bounds with a Bundle search node.

    ``node_bounds`` is intentionally a plain nested mapping so this helper
    remains usable by recovery code without importing the Bundle driver.  A
    value is either ``(lb, ub)`` or ``{"lb": ..., "ub": ...}``.
    """
    if node_bounds is None:
        return
    if not isinstance(node_bounds, Mapping):
        raise ValueError("node_bounds must be a mapping")
    for name, group in node_bounds.items():
        if name not in variables:
            # Flattened {(name, key): bound} form is handled below.
            continue
        if not isinstance(group, Mapping):
            raise ValueError(f"node_bounds[{name!r}] must be a mapping")
        for key, bound in group.items():
            if isinstance(bound, Mapping):
                lower = bound.get("lb", bound.get("lower"))
                upper = bound.get("ub", bound.get("upper"))
            elif isinstance(bound, (tuple, list)) and len(bound) == 2:
                lower, upper = bound
            else:
                raise ValueError(f"invalid node bound for {name}[{key!r}]")
            if lower is None or upper is None:
                raise ValueError(f"invalid node bound for {name}[{key!r}]")
            try:
                lower, upper = float(lower), float(upper)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"invalid node bound for {name}[{key!r}]")
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError(f"invalid node bound for {name}[{key!r}]")
            if key not in variables[name]:
                raise ValueError(f"unknown {name} index in node_bounds: {key!r}")
            variable = variables[name][key]
            variable.LB = max(float(variable.LB), lower)
            variable.UB = min(float(variable.UB), upper)
    # Support {(name, index): (lb, ub)} as a convenience for node stores that
    # flatten variable keys.  Nested entries above remain the preferred API.
    for key, bound in node_bounds.items():
        if not isinstance(key, tuple) or len(key) < 2:
            continue
        name = key[0]
        index = key[1] if len(key) == 2 else key[1:]
        if name not in variables:
            raise ValueError(f"unknown variable group in node_bounds: {name!r}")
        if isinstance(bound, Mapping):
            lower = bound.get("lb", bound.get("lower"))
            upper = bound.get("ub", bound.get("upper"))
        elif isinstance(bound, (tuple, list)) and len(bound) == 2:
            lower, upper = bound
        else:
            raise ValueError(f"invalid node bound for {name}[{index!r}]")
        if lower is None or upper is None:
            raise ValueError(f"invalid node bound for {name}[{index!r}]")
        try:
            lower, upper = float(lower), float(upper)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"invalid node bound for {name}[{index!r}]")
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"invalid node bound for {name}[{index!r}]")
        if index not in variables[name]:
            raise ValueError(f"unknown {name} index in node_bounds: {index!r}")
        variable = variables[name][index]
        variable.LB = max(float(variable.LB), lower)
        variable.UB = min(float(variable.UB), upper)


def apply_start(variables, start) -> None:
    """将已有详细解写入 MIP Start，不修改变量上下界。"""

    for name, group in variables.items():
        for key, value in start.get(name, {}).items():
            group[key].Start = float(value)


def solve_milp(network: NetworkData, costs: CostData, diameters: list[str],
               pipeline_install_limit: int, *, candidates=None, start=None,
               return_solution: bool = False, time_limit=None,
               output_flag: int = GUROBI_OUTPUT_FLAG, node_bounds=None):
    """求解原多期模型；默认保留旧结果结构、时限和日志行为。

    candidates 可限制 sources/sinks/arcs 集合；start 仅提供详细初始解。
    return_solution=True 时返回 (SolveResult, 七组解字典或 None)。
    """

    model, variables = build_milp(
        network, costs, diameters, pipeline_install_limit)
    try:
        model.Params.OutputFlag = output_flag
        model.Params.Threads = GUROBI_THREADS
        model.Params.TimeLimit = TIME_LIMIT if time_limit is None else time_limit
        if candidates is not None:
            apply_candidate_restrictions(variables, candidates)
        apply_node_bounds(variables, node_bounds)
        if start is not None:
            apply_start(variables, start)

        # Recorder 只保存指定时刻的指标，不改变求解过程。
        recorder = Recorder()
        model.optimize(make_callback(recorder))

        status = _status_name(model.Status)
        # 没有 incumbent 时不读取变量或 ObjVal，但仍保留认证下界。
        ub = _finite_gurobi_value(model.ObjVal) if model.SolCount > 0 else None
        gap = _finite_gurobi_value(model.MIPGap) if model.SolCount > 0 else None
        try:
            lb = _finite_gurobi_value(model.ObjBound)
        except (AttributeError, gp.GurobiError):
            lb = None
        for snapshot in recorder.snapshots:
            if not snapshot.recorded:
                snapshot.ub, snapshot.lb, snapshot.gap = ub, lb, gap

        snapshots = {
            snapshot.label: (snapshot.ub, snapshot.lb, snapshot.gap)
            for snapshot in recorder.snapshots}
        result = SolveResult(
            status, ub, lb, gap, *snapshots["1h"], *snapshots["3h"])
        if return_solution:
            solution = ({name: {key: variable.X for key, variable in group.items()}
                         for name, group in variables.items()}
                        if model.SolCount > 0 else None)
            return result, solution
        return result
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
    result = solve_milp(
        network, costs, diameters,
        pipeline_install_limit=np,
        output_flag=GUROBI_OUTPUT_FLAG)
    print(f"status={result.status}, 1h_gap={result.gap_2h}, "
          f"3h_gap={result.gap_5h}, gap={result.gap}")
    return {
        "period_num": period_num,
        "D": len(diameters),
        "Np": np,
        "max_capture_amount": maximum_capture_amount,
        "coef": coefficient,
        **dict(zip(METRIC_COLUMNS, result.metrics())), "curve": curve,
        "status": result.status}


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
