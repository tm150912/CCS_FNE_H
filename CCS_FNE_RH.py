"""批量运行多期 CCS 网络规划的滚动时域启发式算法。

每次优化包含当期的有限前瞻窗口，只提交窗口第一期的建设和运营决策；
历史决策固定后向前滚动一期。算法参数在本文件设置，实验参数沿用
experiment_setup。只有完成全部规划期，才报告完整可行方案的目标值 UB。
某期窗口不可行时记录 INFEASIBLE 并结束当前算例；批量任务继续下一算例。
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd

from CCS_FNE import _atomic_write_csv
from CCS_FNE_MH import SolveResult, _solve_windows
from experiment_setup import (CURVE_SPECS, D_LIST, MAX_CAPTURE_AMOUNT_LIST,
                              NP_LIST, PERIOD_NUM_LIST, TY,
                              CostData, NetworkData, compute_cost_data,
                              load_network_data, period_schedule,
                              with_capture_target)

from read import grid_instance_key, read_result_rows, result_instance_key

# ==================== 输入输出文件 ====================
NORMALIZE_CAPTURE_TARGET = False  # True：是，非线性序列与线性基准等总量；False：否。

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "output" / ("normalized" if NORMALIZE_CAPTURE_TARGET else "")
RESULT_FILE = RESULT_DIR / f"CCS_FNE_RH_{TY}Y_result.csv"

# ==================== 模型参数 ====================
# 规划期、管径、捕集目标及管道数量等模型参数从 experiment_setup 导入。

# ==================== 算法参数 ====================
# 算法参数在本文件设置；窗口含当期，每次只提交一期。
WINDOW_SIZE = 3
TIME_LIMIT = 10800  # 单个算例的总时限（秒），包含建模时间。
SUBPROBLEM_TIME_LIMIT = None  # 默认按剩余期数均分剩余时间；正数为额外单窗上限（秒）。
MIP_GAP = 0.001
THREADS = 0
SEED = 0
OUTPUT_FLAG = 1  # 逐期子问题目标值、gap 和求解时间：0 关闭，1 开启。
GUROBI_OUTPUT_FLAG = 1

# 内部常量：结果字段与状态记录。
RESULT_COLUMNS = [
    "period_num", "D", "Np", "max_capture_amount", "coef", "window_size", "UB", "runtime",
    "avg_subproblem_gap", "completed_periods", "failed_period", "curve", "status"]


def solve_rh(network: NetworkData, costs: CostData, diameters: list[str],
             pipeline_install_limit: int, *,
             window_size: Optional[int] = None,
             time_limit: Optional[float] = None,
             subproblem_time_limit: Optional[float] = None,
             mip_gap: Optional[float] = None,
             threads: Optional[int] = None,
             seed: Optional[int] = None,
             output_flag: Optional[int] = None) -> SolveResult:
    """滚动求解并逐期提交决策；未显式传入的参数读取本文件当前配置。

    WINDOW_SIZE=1 时退化为逐期求解，窗口覆盖全期时第一窗求解完整模型。
    窗口末端以原规划终点为限，捕集目标和折现成本始终使用原周期编号。
    每次只推进一期，按剩余期数均分总剩余时间，与 WINDOW_SIZE 无关。
    """

    return _solve_windows(
        network, costs, diameters, pipeline_install_limit,
        window_size=WINDOW_SIZE if window_size is None else window_size,
        time_limit=TIME_LIMIT if time_limit is None else time_limit,
        subproblem_time_limit=(SUBPROBLEM_TIME_LIMIT
                               if subproblem_time_limit is None
                               else subproblem_time_limit),
        mip_gap=MIP_GAP if mip_gap is None else mip_gap,
        threads=THREADS if threads is None else threads,
        seed=SEED if seed is None else seed,
        output_flag=GUROBI_OUTPUT_FLAG if output_flag is None else output_flag,
        log_output=OUTPUT_FLAG)


def _case_grid():
    """按当前 experiment_setup 配置生成本次批量求解的参数网格。"""

    return product(PERIOD_NUM_LIST, MAX_CAPTURE_AMOUNT_LIST,
                   NP_LIST, CURVE_SPECS, D_LIST)


def save_results(rows: list[dict]) -> None:
    """原子保存已有记录与本次新增结果合并后的结果表。"""

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(pd.DataFrame(rows, columns=RESULT_COLUMNS), RESULT_FILE)


def run_case(network: NetworkData, costs: CostData, diameters: list[str],
             period_num: int, curve: str, coefficient,
             np: int, *, maximum_capture_amount: float) -> dict:
    """运行一个滚动时域算例；np 仍为每弧全期所有管径的管道总数上限。"""

    print(f"period_num={period_num}, Np={np}, curve={curve}, "
          f"coef={coefficient}, window_size={WINDOW_SIZE}, "
          f"max_capture_amount={maximum_capture_amount:.3f}, "
          f"capture_target_max={max(costs.capture_target.values()):.3f}, D={diameters}")
    result = solve_rh(network, costs, diameters, pipeline_install_limit=np)
    last_window_status = (result.history[-1].get("status")
                          if result.history else None)
    print(f"status={result.status}, UB={result.ub}, runtime={result.runtime:.3f}, "
          f"avg_subproblem_gap={result.avg_subproblem_gap}, "
          f"completed_periods={result.completed_periods}, "
          f"failed_period={result.failed_period}, "
          f"last_window_status={last_window_status}")
    return {
        "period_num": period_num,
        "D": len(diameters),
        "Np": np,
        "max_capture_amount": maximum_capture_amount,
        "coef": coefficient,
        "window_size": WINDOW_SIZE,
        "UB": result.ub,
        "runtime": result.runtime,
        "avg_subproblem_gap": result.avg_subproblem_gap,
        "completed_periods": result.completed_periods,
        "failed_period": result.failed_period,
        "curve": curve,
        "status": result.status,
    }


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
