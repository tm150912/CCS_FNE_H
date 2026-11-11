"""按 PH 共享参数网格批量运行拉格朗日松弛，逐例保存 CSV 并支持续跑。

直接运行本文件，或从父项目执行 python -m LRforFNE.CCS_FNE_LR。
算法核心复用 solver；逐例 JSON 保存后原子更新 CSV，不生成工作簿或图片。
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

# 直接文件启动和 IDE 启动均按脚本位置定位父项目。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "LRforFNE"

import pandas as pd
from CCS_FNE import _atomic_write_csv
from experiment_setup import (CURVE_SPECS, D_LIST, MAX_CAPTURE_AMOUNT_LIST,
                              NP_LIST, PERIOD_NUM_LIST, TY, compute_cost_data,
                              load_network_data, period_schedule, with_capture_target)
from read import grid_instance_key, read_result_rows, result_instance_key

from .config import LRConfig
from .result_io import write_result
from .solver import solve_lr
from .warm_start import WarmStartOptions


# ==================== 输入输出文件 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
RESULT_DIR = SCRIPT_DIR / "output" / "batch"
NORMALIZE_CAPTURE_TARGET = False  # True：非线性序列与线性基准等总量。
# 每个方法分别写 CCS_FNE_LR_<method>_<TY>Y_result.csv；归一化结果写入 normalized/。
# 续跑按模型参数匹配；更换数据或算法设置进行新实验时，应更换 RESULT_DIR。

# ==================== 模型参数 ====================
# 与 CCS_FNE_PH.py 共用 experiment_setup 的五维网格：
# PERIOD_NUM_LIST、MAX_CAPTURE_AMOUNT_LIST、NP_LIST、CURVE_SPECS、D_LIST。
# NP 是每条有向弧全期、全部管径合计的管道条数上限。
# 以下算法参数独立于 run_lrh.py；修改单算例设置不会影响本入口。

# ==================== 算法参数 ====================
METHOD = "bundle"  # The paper uses proximal Bundle optimization.

# 初始化：跳过启发式中仅求下界的完整 LP，直接由 P1 和后续搜索取得原可行解。
# Initialize multipliers once from the original LP with Method=2 and Crossover=0.
PH_TIME_LIMIT = 2400  # LR 前的 PH 搜索预算（秒，含 context 构建，不含乘子 LP 和 LR）。
PH_SP_SOLVER_METHOD = "RSS"  # P1 uses the RSS E-Select solver.
PH_P1_TIME_LIMIT = 500.0  # 全部 P1 剩余预算交给 E-Select 后端，无额外 MILP 精修；受 PH_TIME_LIMIT 限制。
PH_MAX_ROUNDS = 20  # 初始化启发式 P3/P4/P5 的最多轮数。

# PH 初解的完整搜索配置；只复用 run_PH 的求解逻辑，不读取其顶部参数。
# 以下 PH_* 仅控制初解，与后续 LR 阶段的同类设置分开。
PH_SCHEDULE_TIME_LIMIT = 60.0  # 每次 P3 全期设施排期时限（秒）。
PH_FIRST_PIPE_POLISH_TIME_LIMIT = 30.0  # 首轮 P3 后管线精修时限；None 沿用普通精修。
PH_PIPE_TIME_LIMIT = 90.0  # 每次 P4 管道建设与流量联合优化时限（秒）。
PH_FLOW_TIME_LIMIT = 20.0  # 初解阶段单期流量重算时限（秒）。
PH_NO_IMPROVEMENT_ROUNDS = 6  # 连续无改善后释放全部候选设施的轮数。
PH_LOCAL_SEARCH_MOVES = 20  # 每轮最多尝试的推迟一期动作数；0 关闭。
PH_NEIGHBORHOOD_SIZE = 12  # P3 走廊邻域的设施节点上限。
PH_EXPANSION_BATCH = 3  # 每类候选扩充的批量上限；0 关闭扩充。
PH_PIPE_COST_WEIGHT = 1.0  # P3 管道固定费代理项权重，不改变真实成本。

# PH 管径/建设时序精修。
PH_PIPE_POLISH_ENABLED = True
PH_PIPE_POLISH_TIME_LIMIT = 10.0  # 每次管道精修总预算（秒）。
PH_PIPE_ARC_TIME_LIMIT = 2.0  # 每条多管弧的小 MILP 时限（秒）。
PH_PIPE_POLISH_MAX_ARCS = 50  # 每次最多处理的多管弧数，不限制单管枚举。

# PH 替代路线与联合邻域。
PH_REROUTE_ENABLED = True
PH_REROUTE_PATHS = 3  # 路径费用权重及备选汇数量的上限。
PH_REROUTE_BATCH = 8  # 每轮走廊邻域数量上限；0 关闭通道扩充。
PH_REROUTE_TIME_LIMIT = 20.0  # 路径筛选总预算（秒）。
PH_JOINT_LNS_ENABLED = True
PH_JOINT_LNS_TIME_LIMIT = 60.0  # 每个联合邻域的求解时限（秒）。
PH_JOINT_NEIGHBORHOOD_SIZE = 12  # 联合邻域最多释放的源汇设施节点数。
PH_FINAL_POLISH_RESERVE = 120.0  # 初解尾部精修预留；最多为剩余预算的 20%。

# PH 求解精度、审计和日志；不读取下方 LR 阶段的同名设置。
PH_MIP_GAP = 0.001  # 初解 MILP 子问题相对 gap，不是原问题全局 gap。
PH_FEASIBILITY_TOLERANCE = 1e-6  # 初解原模型可行性审计的绝对容差。
PH_IMPROVEMENT_TOLERANCE = 1e-6  # 接受目标改进的相对阈值。
PH_THREADS = 0  # 初解及乘子初始化 LP 的线程数；0 自动。
PH_SEED = 0  # 初解及乘子初始化 LP 的随机种子。
PH_OUTPUT_FLAG = True  # 初解阶段摘要。
PH_GUROBI_OUTPUT_FLAG = False  # 初解及乘子 LP 的 Gurobi 详细控制台日志。

# 独立的乘子初始化 LP；PH 初解不执行完整下界 LP。
LR_LP_TIME_LIMIT = None  # 乘子初始化 LP 的独立预算（秒，含建模）；None 不限，不能为 0。

# 拉式主循环、全局 gap 与子问题精度。
LR_TIME_LIMIT = 1200.0  # 对偶迭代与后续上界搜索共用总预算；不含初始化。
LR_MAX_ITERATIONS = 100  # 仅限制 oracle 评估（含细化）；达到后用剩余时间继续 N3 上界搜索。
LR_TARGET_GAP = 1e-4  # 仅用已认证 LB 和已审计 UB 判断。
LR_ORACLE_TIME_LIMIT = 90.0  # 捕集/封存每块每次求解的局部时限，同时受 LR 剩余预算限制。
LR_CAPTURE_MIP_GAP = 1e-4  # 捕集 MILP 相对 gap=0.01%；不随 oracle 精度因子收紧。
LR_STORAGE_MIP_GAP = 1e-4  # 封存 MILP 同样使用固定相对 gap=0.01%。
LR_ORACLE_ABS_TOL = 1e-6
LR_ORACLE_REL_TOL = 1e-9
LR_ORACLE_REFINEMENTS = 2  # 最多额外细化次数；Bundle 跳过已满足固定相对 gap 的无效重求。

# Proximal Bundle：近端尺度、serious/null 判定与模型停止容差。
LR_BUNDLE_RHO = None  # None 根据首轮残差和费用尺度计算；显式值必须严格为正。
LR_BUNDLE_RHO_FLOOR = 1e-8
LR_BUNDLE_ETA = 0.1
LR_BUNDLE_TOLERANCE = 1e-8
LR_BUNDLE_ADAPTIVE_RHO = True  # 根据试探质量调整近端系数，不改变子问题精度。
LR_BUNDLE_RHO_GROWTH = 2.0
LR_BUNDLE_RHO_SHRINK = 0.7
LR_BUNDLE_RHO_MAX_FACTOR = 1e4  # 相对初始近端系数的上限倍数。
LR_BUNDLE_UNCERTAIN_LIMIT = 5  # 连续无法认证的探索次数上限；不伪报收敛或 null step。
LR_BUNDLE_DIAGNOSTIC_EVERY = 20  # 每若干次 QP 及结束时求对偶上界；0 关闭诊断。
LR_BUNDLE_DIAGNOSTIC_TIME_LIMIT = 5.0  # 诊断独立上限，仍计入 LR 总预算。
LR_DELTA_COST = 1.0

# ==================== LR 公共求解参数 ====================
# 后续 LR 阶段的审计与求解器日志设置；PH 初解使用上方 PH_*。
LR_FEASIBILITY_TOLERANCE = 1e-6
LR_THREADS = 0  # 0 自动选择线程数。
LR_SEED = 0
LR_OUTPUT_FLAG = True
LR_GUROBI_OUTPUT_FLAG = False

# ==================== LR 邻域搜索参数 ====================
# LR 引导的 N1 延迟、N2 时间窗和 N3 完整通道联合替换。

# 通用控制：开关、调用频率、求解预算与接受容差。
LR_NEIGHBORHOODS_ENABLED = True
LR_NEIGHBORHOOD_EVERY = 5
LR_NEIGHBORHOOD_TIME_LIMIT = 90
LR_NEIGHBORHOOD_MIP_GAP = 1e-4  # 0.01%；收紧邻域求解精度，与 oracle 精度独立。
LR_FLOW_TIME_LIMIT = 10.0  # LR 邻域单期运行 LP 时限。
LR_EPS_IMPROVE = 1e-7  # 接受原可行解改进的相对阈值。

# LR 信息引导。
LR_GUIDANCE_WINDOW = 5
LR_GUIDANCE_TOP_K = 3

# N1：延迟建设。
LR_N1_MAX_MOVES = 8

# N2：建设时间窗。
LR_N2_TAU_MINUS = 1  # 最多提前期数。
LR_N2_TAU_PLUS = 1  # 最多延后期数。

# N3：完整通道联合替换、邻域扩展与候选精化。
LR_N3_MAX_ARCS = 240  # 完整路径闭包超限时跳过，不截断路径。
LR_N3_MAX_FACILITIES = 12
LR_N3_CANDIDATE_LIMIT = 6  # 每批择优及探索候选数；联合替换独立获得搜索机会。
LR_N3_EXPAND_CORRIDORS = 4  # 每个 N3 域最多加入四条相关完整走廊；0 恢复原域。
LR_N3_EXPANDED_TIME_LIMIT = 120  # 扩展域每次时限；开启扩展时每批最多四个候选。
LR_N3_REFINE_CANDIDATES = 1  # 每批至多对这些高潜力候选进行同模型精化；0 关闭。
LR_N3_REFINE_TIME_LIMIT = 1.0  # 精化额外秒数，仍受邻域及 LR 总截止时间限制。
LR_N3_REFINE_ABSOLUTE_GAP = 300.0  # N3 精化的绝对费用容差；不改变捕集/封存精度。

# 管道精修。
LR_PIPELINE_POLISH_TIME_LIMIT = 120  # 全网固定流量管道精修；0 关闭，全部计入 LR 总预算。


def build_config(**overrides) -> LRConfig:
    """读取顶部 LR_* 参数；覆盖实参仍使用 LRConfig 字段名。"""
    values = {name: globals()["LR_" + name.removeprefix("lr_").upper()]
              for name in LRConfig.__dataclass_fields__}
    values.update(overrides)
    return LRConfig(**values).validate()


def build_heuristic_options():
    """仅从本入口构造全部初解参数；不调用 run_PH.build_options。"""
    options = WarmStartOptions(
        time_limit=PH_TIME_LIMIT, sp_solver_method=PH_SP_SOLVER_METHOD,
        p1_time_limit=PH_P1_TIME_LIMIT, schedule_time_limit=PH_SCHEDULE_TIME_LIMIT,
        pipe_time_limit=PH_PIPE_TIME_LIMIT,
        first_pipe_polish_time_limit=PH_FIRST_PIPE_POLISH_TIME_LIMIT,
        flow_time_limit=PH_FLOW_TIME_LIMIT,
        # PH 初解禁用完整 LP；此处保留独立乘子 LP 的预算和固定算法设置。
        lp_time_limit=LR_LP_TIME_LIMIT, lp_method=2, lp_crossover=0,
        max_rounds=PH_MAX_ROUNDS, no_improvement_rounds=PH_NO_IMPROVEMENT_ROUNDS,
        local_search_moves=PH_LOCAL_SEARCH_MOVES, neighborhood_size=PH_NEIGHBORHOOD_SIZE,
        expansion_batch=PH_EXPANSION_BATCH, pipe_cost_weight=PH_PIPE_COST_WEIGHT,
        pipe_polish_enabled=PH_PIPE_POLISH_ENABLED,
        pipe_polish_time_limit=PH_PIPE_POLISH_TIME_LIMIT,
        pipe_arc_time_limit=PH_PIPE_ARC_TIME_LIMIT,
        pipe_polish_max_arcs=PH_PIPE_POLISH_MAX_ARCS,
        reroute_enabled=PH_REROUTE_ENABLED, reroute_paths=PH_REROUTE_PATHS,
        reroute_batch=PH_REROUTE_BATCH, reroute_time_limit=PH_REROUTE_TIME_LIMIT,
        joint_lns_enabled=PH_JOINT_LNS_ENABLED, joint_lns_time_limit=PH_JOINT_LNS_TIME_LIMIT,
        joint_neighborhood_size=PH_JOINT_NEIGHBORHOOD_SIZE,
        final_polish_reserve=PH_FINAL_POLISH_RESERVE, mip_gap=PH_MIP_GAP,
        threads=PH_THREADS, seed=PH_SEED,
        feasibility_tolerance=PH_FEASIBILITY_TOLERANCE,
        improvement_tolerance=PH_IMPROVEMENT_TOLERANCE,
        output_flag=PH_OUTPUT_FLAG, gurobi_output_flag=PH_GUROBI_OUTPUT_FLAG)
    options.validate()
    return options


RESULT_COLUMNS = [
    "period_num", "D", "Np", "max_capture_amount", "coef",
    "UB", "LB", "gap", "runtime", "curve", "status", "method",
    "termination_reason", "initial_UB", "warm_start_runtime", "lr_runtime",
    "shared_warm_start_runtime", "lp_status", "lp_objective", "lp_runtime",
    "lower_bound_origin", "iterations", "max_violation", "result_dir"]
# 达到时限、迭代上限或正常停滞视为本次实验完成，不表示全局最优。
# 初始化失败、oracle 未完成、数值/求解器错误和人工中断下次重试。
COMPLETED_STATUSES = (
    "GAP_CERTIFIED", "FEASIBLE_TIME_LIMIT", "FEASIBLE_ITERATION_LIMIT",
    "FEASIBLE_STAGNATION", "PROVEN_INFEASIBLE")


def _methods(method=None):
    method = METHOD if method is None else method
    if method != "bundle":
        raise ValueError("METHOD must be bundle")
    return (method,)


def _case_grid():
    return product(PERIOD_NUM_LIST, MAX_CAPTURE_AMOUNT_LIST,
                   NP_LIST, CURVE_SPECS, D_LIST)


def result_file(method, *, result_dir=None, normalize_capture_target=None):
    directory = Path(RESULT_DIR if result_dir is None else result_dir)
    normalize = (NORMALIZE_CAPTURE_TARGET if normalize_capture_target is None
                 else normalize_capture_target)
    if normalize:
        directory /= "normalized"
    return directory / f"CCS_FNE_LR_{method}_{TY}Y_result.csv"


def save_results(rows, method, *, result_dir=None, normalize_capture_target=None):
    target = result_file(method, result_dir=result_dir,
                         normalize_capture_target=normalize_capture_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(pd.DataFrame(rows, columns=RESULT_COLUMNS), target)


def run_case(network, costs, diameters, period_num, curve, coefficient, np,
             *, maximum_capture_amount, methods=None, config=None,
             heuristic_options=None, result_dir=None, normalize_capture_target=None):
    """Solve one instance with the Bundle algorithm and its PH initialization."""
    methods = _methods() if methods is None else tuple(methods)
    if methods != ("bundle",):
        raise ValueError("methods must contain only bundle")
    config = build_config() if config is None else config.validate()
    options = build_heuristic_options() if heuristic_options is None else heuristic_options
    options.validate()
    if config.output_flag:
        print(f"method={methods}, period_num={period_num}, Np={np}, curve={curve}, "
              f"coef={coefficient}, max_capture_amount={maximum_capture_amount}, D={diameters}")
    kwargs = dict(pipeline_install_limit=np, config=config, heuristic_options=options)
    results = {methods[0]: solve_lr(network, costs, diameters, method=methods[0], **kwargs)}
    rows = []
    for method, result in results.items():
        shape = "none" if coefficient is None else f"{coefficient:.17g}"
        case_name = (f"T{period_num}_D{len(diameters)}_NP{np}_"
                     f"C{maximum_capture_amount:.17g}_{curve}_{shape}")
        directory = result_file(method, result_dir=result_dir,
            normalize_capture_target=normalize_capture_target).parent
        # A failed JSON export leaves the CSV unchanged, so the case is retried.
        detail = write_result(result, Path("cases") / case_name, output_root=directory)
        lp = result.configuration.get("multiplier_initialization", {})
        rows.append(dict(
            period_num=period_num, D=len(diameters), Np=np,
            max_capture_amount=maximum_capture_amount, coef=coefficient,
            UB=result.objective, LB=result.lower_bound, gap=result.gap,
            runtime=result.total_runtime, curve=curve, status=result.status, method=method,
            termination_reason=result.termination_reason, initial_UB=result.initial_objective,
            warm_start_runtime=result.warm_start_runtime, lr_runtime=result.lr_runtime,
            shared_warm_start_runtime=result.timings.get("shared_warm_start_wall"),
            lp_status=lp.get("status"), lp_objective=lp.get("objective"),
            lp_runtime=lp.get("runtime"), lower_bound_origin=result.lower_bound_origin,
            iterations=len(result.history),
            max_violation=result.audit["max_violation"] if result.audit else None,
            result_dir=str(detail)))
        if config.output_flag:
            print(f"{method}: status={result.status}, reason={result.termination_reason}, "
                  f"UB={result.objective}, LB={result.lower_bound}, gap={result.gap}, "
                  f"runtime={result.total_runtime:.3f}s")
    return rows


def run_batch(*, config=None, heuristic_options=None, data_dir=None, result_dir=None,
              normalize_capture_target=None, method=None):
    """Save each case's JSON before its CSV row; resume normally completed cases."""
    methods = _methods(method)
    config = build_config() if config is None else config.validate()
    options = build_heuristic_options() if heuristic_options is None else heuristic_options
    options.validate()
    if options.lp_time_limit == 0:
        raise ValueError("Multiplier LP time limit must be None or positive")
    normalize = (NORMALIZE_CAPTURE_TARGET if normalize_capture_target is None
                 else normalize_capture_target)
    paths = {method: result_file(method, result_dir=result_dir,
                                normalize_capture_target=normalize) for method in methods}
    cases = list(_case_grid())
    rows, completed = {}, {}
    for method in methods:
        rows[method], completed[method] = read_result_rows(
            paths[method], cases, completed_statuses=COMPLETED_STATUSES)
        if config.output_flag:
            print(f"{method}: {len(cases)} cases, {len(completed[method])} completed, "
                  f"{len(cases) - len(completed[method])} pending.")
    if all(grid_instance_key(case) in completed[method] for case in cases for method in methods):
        if config.output_flag:
            print(f"All current cases already completed. Results: {[str(paths[m]) for m in methods]}")
        return rows
    network = load_network_data(DATA_DIR if data_dir is None else data_dir)
    active_period_num = None
    try:
        for case in cases:
            key = grid_instance_key(case)
            pending = [method for method in methods if key not in completed[method]]
            if not pending:
                continue
            period_num, maximum, np, (curve, coefficient), diameters = case
            if period_num != active_period_num:
                periods, durations = period_schedule(period_num)
                base_costs = compute_cost_data(
                    network, periods, durations, apply_shared_capture_target=False)
                active_period_num = period_num
            costs = with_capture_target(
                base_costs, curve, coefficient, maximum_capture_amount=maximum,
                normalize_capture_target=normalize)
            current = run_case(network, costs, diameters, period_num, curve, coefficient,
                np, maximum_capture_amount=maximum, methods=pending, config=config,
                heuristic_options=options, result_dir=result_dir,
                normalize_capture_target=normalize)
            for row in current:
                method = row["method"]
                rows[method] = [old for old in rows[method] if result_instance_key(old) != key]
                rows[method].append(row)
                save_results(rows[method], method, result_dir=result_dir,
                             normalize_capture_target=normalize)
                if config.output_flag:
                    print(f"Results saved: {paths[method]}")
                if row["status"] == "FEASIBLE_INTERRUPTED":
                    print("Interrupted. Current results saved; rerun to retry unfinished cases.")
                    return rows
    except KeyboardInterrupt:
        print("Interrupted. Previously completed results remain saved.")
        raise
    if config.output_flag:
        print(f"Batch finished. Results: {[str(paths[m]) for m in methods]}")
    return rows


def main() -> None:
    run_batch()


if __name__ == "__main__":
    main()

