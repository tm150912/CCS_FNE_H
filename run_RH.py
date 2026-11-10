"""用 Rolling Horizon（RH）求解一个 CCS_FNE 算例。

沿用 run_gurobi 的单算例参数、变量 Excel 格式和逐期绘图流程。
每次优化包含当期的前瞻窗口，只提交首期决策并固定历史。
始终保存摘要与日志，仅完整方案导出变量和绘图；窗口 gap 不代表全期 gap。
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from CCS_FNE_MH import SolveResult, _solve_windows
from experiment_setup import (
    CostData,
    D_LIST,
    NetworkData,
    compute_cost_data,
    load_network_data,
    period_schedule,
)
from plot_results import plot_results_from_excel
from run_gurobi import with_local_capture_target
from run_MH import _save_window_results
from solve_logging import logged_run


# ==================== 输入输出文件 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "output"
RESULT_FILE = RESULT_DIR / "CCS_FNE_RH_variables.xlsx"

# 绘图开关、样式、地图边界和容差统一在 plot_results.py 修改。
PLOT_OUTPUT_DIR = RESULT_DIR
PLOT_OUTPUT_IMAGE_NAME = "CCS_FNE_RH_results.png"
PLOT_FILE = PLOT_OUTPUT_DIR / PLOT_OUTPUT_IMAGE_NAME
PLOT_PERIOD_OUTPUT_DIR = PLOT_OUTPUT_DIR / f"{Path(PLOT_OUTPUT_IMAGE_NAME).stem}_periods"

# ==================== 模型参数 ====================
# 单次运行参数；NP 为每条有向弧全期所有管径合计的管道条数上限。
PERIOD_NUM = 5
NP = 3
DIAMETERS = D_LIST[-1]
CURVE = "linear"
COEFFICIENT = None
MAX_CAPTURE_AMOUNT = 2e6
NORMALIZE_CAPTURE_TARGET = False  # True（是）：非线性曲线与线性基准等总量；False（否）。
CAPTURE_AMOUNT_SEQUENCE = None

# ==================== 算法参数 ====================
WINDOW_SIZE = 3  # 含当期的前瞻窗口期数；1 时退化为 MH。
TIME_LIMIT = 10800  # 单个算例的总时限（秒），含建模。
SUBPROBLEM_TIME_LIMIT = None  # 默认按剩余期数均分剩余时间。
MIP_GAP = 0.001  # 窗口 MILP 的相对 gap 阈值，不是全期 gap。
THREADS = 0  # Gurobi 线程数，0 自动。
SEED = 0  # Gurobi 随机种子。
OUTPUT_FLAG = 1  # 逐期子问题目标值、gap 和求解时间：0 关闭，1 开启。
GUROBI_OUTPUT_FLAG = 1  # 控制台日志：0 关闭，1 开启；Excel 始终保存求解日志。
# 单算例同名 .log 保存完整运行日志；Gurobi 日志开关只影响终端显示。


def solve_rh(network: NetworkData, costs: CostData, diameters: list[str],
             pipeline_install_limit: int, *, window_size=None, time_limit=None,
             subproblem_time_limit=None, mip_gap=None, threads=None,
             seed=None, output_flag=None) -> SolveResult:
    """使用本文件参数滚动求解；某期窗口不可行时立即返回 INFEASIBLE。"""

    result = _solve_windows(
        network, costs, diameters, pipeline_install_limit,
        window_size=WINDOW_SIZE if window_size is None else window_size,
        time_limit=TIME_LIMIT if time_limit is None else time_limit,
        subproblem_time_limit=(SUBPROBLEM_TIME_LIMIT if subproblem_time_limit is None
                               else subproblem_time_limit),
        mip_gap=MIP_GAP if mip_gap is None else mip_gap,
        threads=THREADS if threads is None else threads,
        seed=SEED if seed is None else seed,
        output_flag=GUROBI_OUTPUT_FLAG if output_flag is None else output_flag,
        log_output=OUTPUT_FLAG, capture_solver_log=True)
    return replace(result, algorithm_parameters={**result.algorithm_parameters,
                                                  "algorithm": "RH"})


def save_rh_results(result: SolveResult,
                    network: NetworkData,
                    costs: CostData,
                    diameters: list[str]) -> Path:
    """保存 RH 单算例结果；无完整方案时仅保存摘要、窗口历史与日志。"""

    return _save_window_results(result, network, costs, diameters, RESULT_FILE)


@logged_run(lambda: RESULT_FILE.with_suffix('.log'))
def main() -> None:
    """读取数据，滚动求解并保存结果；完整方案另绘制各期演化图。"""

    if CAPTURE_AMOUNT_SEQUENCE is None:
        print(f"period_num={PERIOD_NUM}, NP={NP} (per arc, full horizon), "
              f"curve={CURVE}, coef={COEFFICIENT}, "
              f"max_capture={MAX_CAPTURE_AMOUNT}, D={DIAMETERS}")
    else:
        print(f"period_num={PERIOD_NUM}, NP={NP} (per arc, full horizon), "
              f"curve={CURVE}, coef={COEFFICIENT}, "
              f"capture_sequence={CAPTURE_AMOUNT_SEQUENCE}, "
              f"D={DIAMETERS}")
    print(f"window_size={WINDOW_SIZE}")
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
    result = solve_rh(
        network, costs, DIAMETERS, pipeline_install_limit=NP,
        window_size=WINDOW_SIZE, time_limit=TIME_LIMIT,
        subproblem_time_limit=SUBPROBLEM_TIME_LIMIT,
        mip_gap=MIP_GAP, threads=THREADS, seed=SEED,
        output_flag=GUROBI_OUTPUT_FLAG,
    )
    print(f"status={result.status}")
    print(f"UB={result.ub}")
    print(f"runtime={result.runtime:.6f}s")
    print(f"avg_subproblem_gap={result.avg_subproblem_gap} (not full-horizon gap)")
    print(f"completed_periods={result.completed_periods}")
    print(f"failed_period={result.failed_period}")
    output_file = save_rh_results(result, network, costs, DIAMETERS)
    print(f"RH results saved: {output_file}")
    if result.solution is None:
        print("未取得完整可行方案，已保存求解摘要、窗口历史与日志，不导出变量或绘图。")
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
