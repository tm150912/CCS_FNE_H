"""逐期求解 CCS_FNE（Myopic Heuristic，MH）。

每次只优化当期，固定此前全部建设和运营决策，直到规划期结束。
某期不可行时记录 INFEASIBLE 并结束当前算例；批量任务继续下一算例。
复用 CCS_FNE 的约束、原期号和全期折现费用，不重置管道/井的累计上限。
批量算例与原脚本一致；算法参数在本文件顶部设置。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Optional

import gurobipy as gp
import pandas as pd

from CCS_FNE import (_atomic_write_csv,
                     _finite_gurobi_value, _relative_gap, _status_name, build_milp)
from experiment_setup import (CURVE_SPECS, D_LIST, MAX_CAPTURE_AMOUNT_LIST,
                              NP_LIST, PERIOD_NUM_LIST, TY,
                              CostData, NetworkData, compute_cost_data,
                              load_network_data, period_schedule,
                              with_capture_target)
from read import read_pipeline_limit
from result_metadata import build_model_parameters
from solve_logging import optimize_with_log

from read import grid_instance_key, read_result_rows, result_instance_key

# ==================== 输入输出文件 ====================
NORMALIZE_CAPTURE_TARGET = False  # True：是，非线性序列与线性基准等总量；False：否。

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "output" / ("normalized" if NORMALIZE_CAPTURE_TARGET else "")
RESULT_FILE = RESULT_DIR / f"CCS_FNE_MH_{TY}Y_result.csv"

# ==================== 模型参数 ====================
# 规划期、管径、捕集目标及管道数量等模型参数从 experiment_setup 导入。

# ==================== 算法参数 ====================
TIME_LIMIT = 10800  # 单个算例的总时限（秒），含建模；在操作边界检查。
SUBPROBLEM_TIME_LIMIT = None  # 默认按剩余期数均分剩余时间；正数为额外单期上限（秒）。
MIP_GAP = 0.001  # 当期 MILP 的相对 gap 阈值，不是全期启发式 gap。
THREADS = 0  # Gurobi 线程数，0 自动。
SEED = 0  # Gurobi 随机种子。
OUTPUT_FLAG = 1  # 是否输出各期子问题目标值、gap、求解时间及算例进度。
GUROBI_OUTPUT_FLAG = 0  # 0 关闭求解器日志，1 开启。

# 内部常量：结果字段与状态记录。
RESULT_COLUMNS = [
    "period_num", "D", "Np", "max_capture_amount", "coef", "UB", "runtime",
    "avg_subproblem_gap", "completed_periods", "failed_period", "curve", "status"]


@dataclass(frozen=True)
class SolveResult:
    """完整方案才有 UB；窗口失败不代表原全期模型不可行。

    INFEASIBLE 表示固定历史后的当前期/窗口不可行，并立即结束当前算例。
    solution 包含七组原模型变量。history 的 ub/lb/gap 仅属于各自窗口，
    不得作为全期上下界，也不得累加重叠窗口的目标值。
    """

    status: str
    ub: Optional[float]
    runtime: float
    completed_periods: int
    failed_period: Optional[int]
    solution: Optional[dict]
    history: list[dict]
    solver_log: list[str] = field(default_factory=list)
    model_parameters: dict = field(default_factory=dict)
    algorithm_parameters: dict = field(default_factory=dict)

    @property
    def avg_subproblem_gap(self) -> Optional[float]:
        """子问题有限 gap 的算术平均值（非全期 gap）；无有效值返回 None。"""

        gaps = [row["gap"] for row in self.history
                if row["gap"] is not None and math.isfinite(row["gap"])]
        return math.fsum(gaps) / len(gaps) if gaps else None


def _validate_options(costs, window_size, time_limit, subproblem_time_limit,
                      mip_gap, threads, seed, output_flag) -> None:
    if not costs.periods or list(costs.periods) != list(range(1, len(costs.periods) + 1)):
        raise ValueError("costs.periods must be consecutive integers starting at 1")
    for label, value, minimum in (("window_size", window_size, 1),
                                   ("threads", threads, 0), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    for label, value in (("time_limit", time_limit),
                         ("subproblem_time_limit", subproblem_time_limit)):
        if label == "subproblem_time_limit" and value is None:
            continue
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    if isinstance(mip_gap, bool) or not math.isfinite(mip_gap) or mip_gap < 0:
        raise ValueError("mip_gap must be finite and nonnegative")
    if output_flag not in (0, 1):
        raise ValueError("output_flag must be 0 or 1")


def _build_window(network, costs, diameters, pipeline_install_limit,
                  period, window_end, committed):
    """保留 1..window_end 的原模型，固定 period 以前的七组变量。

    原模型累计容量按 range(1, h+1) 索引，因此使用前缀而非重新编号的窗口。
    已固定的历史变量由预处理消去，真正自由的决策仅在当前窗口内。
    费用字典保持完整规划的原系数，不能按窗口长度重新摊销。
    """

    prefix = replace(
        costs,
        periods=[h for h in costs.periods if h <= window_end],
        durations={h: v for h, v in costs.durations.items() if h <= window_end},
        source_periods=[key for key in costs.source_periods if key[-1] <= window_end],
        sink_periods=[key for key in costs.sink_periods if key[-1] <= window_end],
        arc_periods=[key for key in costs.arc_periods if key[-1] <= window_end],
        capture_target={h: v for h, v in costs.capture_target.items() if h <= window_end})
    model, variables = build_milp(network, prefix, diameters, pipeline_install_limit)
    try:
        for name, group in variables.items():
            for key, variable in group.items():
                if key[-1] < period:
                    variable.LB = variable.UB = committed[name][key]
                    # 历史费用已发生，移出目标以免沉没成本稀释窗口 MIPGap。
                    variable.Obj = 0.0
        return model, variables
    except BaseException:
        model.dispose()
        raise


def _solve_windows(network: NetworkData, costs: CostData, diameters: list[str],
                   pipeline_install_limit: int, *, window_size: int,
                   time_limit: float, subproblem_time_limit: Optional[float],
                   mip_gap: float, threads: int, seed: int,
                   output_flag: int, log_output: bool = False,
                   capture_solver_log: bool = False) -> SolveResult:
    """MH/RH 共用的滚动内核，全部算法参数由入口脚本显式传入。

    每次只提交窗口首期，未来期只是前瞻计划。单窗到时已有可行解时继续；
    无解或总预算耗尽则停止，未完成的前缀不作为全期可行解返回。
    建模后按剩余子问题数均分剩余时间，提前结束节省的时间留给后续窗口；
    子问题数等于规划期数，与窗口长度无关。可另设单窗上限。
    建模等不可中断操作可能使总时限略有超出。
    capture_solver_log 为 True 时保留各窗口的原始 Gurobi 日志。
    """

    started = perf_counter()
    _validate_options(costs, window_size, time_limit, subproblem_time_limit,
                      mip_gap, threads, seed, output_flag)
    pipeline_install_limit = read_pipeline_limit(pipeline_install_limit)
    model_parameters = build_model_parameters(network, costs, diameters, pipeline_install_limit)
    algorithm_parameters = dict(
        window_size=window_size, time_limit=time_limit,
        subproblem_time_limit=subproblem_time_limit, mip_gap=mip_gap,
        threads=threads, seed=seed, gurobi_output_flag=output_flag,
        output_flag=bool(log_output), capture_solver_log=capture_solver_log,
        time_limit_scope="case_including_model_build",
        window_budget_rule="remaining_time / remaining_periods, capped by subproblem_time_limit",
        gurobi_version=".".join(map(str, gp.gurobi.version())))
    coefficients = {
        "s": costs.source_fixed, "r": costs.sink_fixed, "w": costs.well_fixed,
        "y": costs.pipe_fixed, "a": costs.source_variable,
        "b": costs.sink_variable, "x": costs.pipe_variable}
    committed = {name: {} for name in coefficients}
    history = []
    solver_log = []
    completed_periods = 0
    total_cost = 0.0

    def record_log(model, where):
        if where == gp.GRB.Callback.MESSAGE:
            solver_log.extend(model.cbGet(gp.GRB.Callback.MSG_STRING).splitlines())

    def finish(status, failed_period=None):
        complete = completed_periods == len(costs.periods)
        return SolveResult(status, total_cost if complete else None,
                           perf_counter() - started, completed_periods,
                           failed_period, committed if complete else None,
                           history, solver_log, model_parameters, algorithm_parameters)

    for period in costs.periods:
        if perf_counter() - started >= time_limit:
            return finish("TIME_LIMIT", period)
        window_end = min(len(costs.periods), period + window_size - 1)
        model, variables = _build_window(
            network, costs, diameters, pipeline_install_limit,
            period, window_end, committed)
        try:
            remaining = time_limit - (perf_counter() - started)
            if remaining <= 0:
                return finish("TIME_LIMIT", period)
            remaining_subproblems = len(costs.periods) - completed_periods
            allocated_time = remaining / remaining_subproblems
            if subproblem_time_limit is not None:
                allocated_time = min(allocated_time, subproblem_time_limit)
            if capture_solver_log:
                # OutputFlag=0 会禁用 MESSAGE 回调，仅用 LogToConsole 控制屏幕输出。
                model.Params.LogToConsole = output_flag
                model.Params.OutputFlag = 1
            else:
                model.Params.OutputFlag = output_flag
            model.Params.TimeLimit = allocated_time
            model.Params.MIPGap = mip_gap
            model.Params.Threads = threads
            model.Params.Seed = seed
            if capture_solver_log:
                solver_log.append(f"[period={period}, window_end={window_end}]")
                optimize_with_log(model, record_log)
            else:
                optimize_with_log(model)

            status = _status_name(model.Status)
            ub = _finite_gurobi_value(model.ObjVal) if model.SolCount > 0 else None
            try:
                lb = _finite_gurobi_value(model.ObjBound)
            except (AttributeError, gp.GurobiError):
                lb = None
            row = {
                "period": period, "window_end": window_end, "status": status,
                "ub": ub, "lb": lb, "gap": _relative_gap(ub, lb),
                "runtime": model.Runtime, "time_limit": allocated_time,
                "committed_cost": None}
            history.append(row)
            if log_output:
                objective_text = f"{ub:.6f}" if ub is not None else "N/A"
                gap_text = f"{row['gap']:.4%}" if row["gap"] is not None else "N/A"
                print(f"period={period}/{len(costs.periods)}, "
                      f"window={period}-{window_end}, status={status}, "
                      f"subproblem_objective={objective_text}, "
                      f"subproblem_gap={gap_text}, "
                      f"subproblem_runtime={row['runtime']:.3f}s", flush=True)
            if model.SolCount == 0:
                if model.Status == gp.GRB.INFEASIBLE:
                    return finish("INFEASIBLE", period)
                if model.Status in (gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED):
                    return finish(status, period)
                return finish("NO_SOLUTION", period)

            # 只保存当期七组变量，未来窗口可以重新排期；历史运营量也不可回写。
            current = {name: {key: variable.X for key, variable in group.items()
                              if key[-1] == period}
                       for name, group in variables.items()}
            period_cost = math.fsum(
                coefficients[name][key] * value
                for name, group in current.items() for key, value in group.items())
            for name, group in current.items():
                committed[name].update(group)
            row["committed_cost"] = period_cost
            total_cost += period_cost
            completed_periods += 1
            if model.Status == gp.GRB.INTERRUPTED:
                return finish("INTERRUPTED", period if completed_periods < len(costs.periods) else None)
        finally:
            model.dispose()

    # 最后一个窗口覆盖完整原模型，固定历史后仍可行；不宣称全局最优。
    return finish("FEASIBLE")


def solve_mh(network: NetworkData, costs: CostData, diameters: list[str],
             pipeline_install_limit: int, *, time_limit=None,
             subproblem_time_limit=None, mip_gap=None, threads=None,
             seed=None, output_flag=None) -> SolveResult:
    """逐期求解一个算例；顶部 OUTPUT_FLAG 控制摘要，output_flag 控制 Gurobi 日志。"""

    return _solve_windows(
        network, costs, diameters, pipeline_install_limit, window_size=1,
        time_limit=TIME_LIMIT if time_limit is None else time_limit,
        subproblem_time_limit=(SUBPROBLEM_TIME_LIMIT if subproblem_time_limit is None
                               else subproblem_time_limit),
        mip_gap=MIP_GAP if mip_gap is None else mip_gap,
        threads=THREADS if threads is None else threads,
        seed=SEED if seed is None else seed,
        output_flag=GUROBI_OUTPUT_FLAG if output_flag is None else output_flag,
        log_output=OUTPUT_FLAG)


def _case_grid():
    """与 CCS_FNE 使用同一模型参数网格。"""

    return product(PERIOD_NUM_LIST, MAX_CAPTURE_AMOUNT_LIST,
                   NP_LIST, CURVE_SPECS, D_LIST)


def save_results(rows: list[dict]) -> None:
    """原子保存已有记录与本次新增结果合并后的结果表。"""

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(pd.DataFrame(rows, columns=RESULT_COLUMNS), RESULT_FILE)


def run_case(network: NetworkData, costs: CostData, diameters: list[str],
             period_num: int, curve: str, coefficient, np: int, *,
             maximum_capture_amount: float) -> dict:
    """求解一组参数；失败期及最后窗口状态用于区分算法死路和求解时限。"""

    if OUTPUT_FLAG:
        print(f"period_num={period_num}, Np={np}, curve={curve}, coef={coefficient}, "
              f"max_capture_amount={maximum_capture_amount:.3f}, "
              f"capture_target_max={max(costs.capture_target.values()):.3f}, D={diameters}")
    result = solve_mh(network, costs, diameters, pipeline_install_limit=np)
    last_status = result.history[-1]["status"] if result.history else None
    if OUTPUT_FLAG:
        print(f"status={result.status}, UB={result.ub}, runtime={result.runtime:.3f}s, "
              f"avg_subproblem_gap={result.avg_subproblem_gap}, "
              f"completed_periods={result.completed_periods}, "
              f"failed_period={result.failed_period}, last_window_status={last_status}")
    return {
        "period_num": period_num, "D": len(diameters), "Np": np,
        "max_capture_amount": maximum_capture_amount,
        "coef": coefficient, "UB": result.ub, "runtime": result.runtime,
        "avg_subproblem_gap": result.avg_subproblem_gap,
        "completed_periods": result.completed_periods,
        "failed_period": result.failed_period, "curve": curve, "status": result.status}


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
            if printed_case and OUTPUT_FLAG:
                print()
            row = run_case(network, costs, diameters, period_num,
                           curve, coefficient, np,
                           maximum_capture_amount=maximum_capture_amount)
            rows = [old for old in rows if result_instance_key(old) != key]
            rows.append(row)
            save_results(rows)
            if OUTPUT_FLAG:
                print(f"Results saved: {RESULT_FILE}")
            if row["status"].upper() == "INTERRUPTED":
                print(f"Interrupted. Current results: {RESULT_FILE}")
                return
            printed_case = True
    except KeyboardInterrupt:
        print(f"Interrupted. Completed results: {RESULT_FILE}")
        raise
    if OUTPUT_FLAG:
        print(f"All cases completed. Results: {RESULT_FILE}")


if __name__ == "__main__":
    main()
