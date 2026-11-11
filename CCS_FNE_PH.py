"""按共享参数网格批量运行 CCS_FNE 原始启发式（PH, primal heuristic）。

直接调用 heuristic_stages，按 run_PH 的流程独立调度 LP -> P1 -> P3/P4/P5 -> 最终精修 -> 按条件恢复搜索。
依据汇总 CSV 跳过当前网格中已正常完成的组合，兼容旧检查点缺失记录；不输出逐组变量和图片。
UB、LB、gap 和状态直接采用启发式返回值，不将受限子问题的界作为全局下界。
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Optional

import gurobipy as gp
import pandas as pd

from CCS_FNE import _atomic_write_csv
from experiment_setup import (CURVE_SPECS, D_LIST, MAX_CAPTURE_AMOUNT_LIST,
                              NP_LIST, PERIOD_NUM_LIST, TY,
                              CostData, NetworkData, compute_cost_data,
                              load_network_data, period_schedule,
                              with_capture_target)
from heuristic_stages.common import (CandidatePool, HeuristicOptions, HeuristicResult,
                                     _BudgetExpired, _Context, _attempt, _result, _round_mip_gaps)
from heuristic_stages.lp_bound import run_lp
from heuristic_stages.p1_candidates import run_p1
from heuristic_stages.p3_schedule import run_p3
from heuristic_stages.p4_pipelines import run_p4
from heuristic_stages.p5_improvement import (run_p5, prepare_neighborhoods,
                                            _expand_pool, final_polish)


from read import (grid_instance_key, normalize_result_status, read_result_rows,
                  result_instance_key)

# ==================== 输入输出文件 ====================
NORMALIZE_CAPTURE_TARGET = False  # True：是，非线性序列与线性基准等总量；False：否。

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
RESULT_DIR = SCRIPT_DIR / "output" / ("normalized" if NORMALIZE_CAPTURE_TARGET else "")
RESULT_FILE = RESULT_DIR / f"CCS_FNE_PH_{TY}Y_result.csv"
CHECKPOINT_FILE = RESULT_DIR / f"CCS_FNE_H_{TY}Y_checkpoint.csv"  # 仅兼容读取旧版记录。

# ==================== 模型参数 ====================
# 与 CCS_FNE.py 共用 experiment_setup.py 中的五维参数网格：
# PERIOD_NUM_LIST、MAX_CAPTURE_AMOUNT_LIST、NP_LIST、CURVE_SPECS、D_LIST。
# NP 是每条有向弧全期、所有管径合计的建设条数上限，不按周期年数换算。

# ==================== 算法参数 ====================
# 批量运行的独立配置，直接在本文件修改；LP 独立计时，其余子时限受启发式及当前搜索阶段剩余预算限制。
TIME_LIMIT = 3600  # 完整 LP 结束后的启发式总预算（秒），包含后续建模、审计、搜索和最终精修。

# P1: use RSS for E-Select within the remaining stage budget.
P1_TIME_LIMIT = 600.0  # P1 总预算，含准备、ESM 求解和审计，受全局剩余预算限制；0 跳过 P1。

# 排期、管道和流量子问题时限（秒）。
SCHEDULE_TIME_LIMIT = 200.0  # 每次 P3 源机组、汇站点及注射井排期子问题的时限（秒，含建模）。
FIRST_PIPE_POLISH_TIME_LIMIT = 200.0  # 首轮 P3 后精修时限；预算内遍历全部候选弧，None 沿用普通精修。
PIPE_TIME_LIMIT = 100.0  # 每次固定设施后，重优化管道建设与全部流量的时限（秒，含建模）。
FLOW_TIME_LIMIT = 50.0  # 每次固定建设方案后，单期流量重算的时限（秒，含建模）。

# 完整原问题 LP；仅最优 LP 目标值用于全局下界。
LP_TIME_LIMIT = None  # None 不限制 LP 时间；正数为独立 LP 预算（秒，含建模），不占 TIME_LIMIT，不允许为 0。
LP_METHOD = 2  # 完整 LP 的求解算法：2 内点法；-1 自动、0 原始单纯形、1 对偶单纯形。
LP_CROSSOVER = 0  # 内点法后的基解交叉转换：0 关闭、-1 自动；只求 LP 下界时可关闭。
LP_FORMULATION = "cumulative"  # 完整 LP 表达：original 原增量变量；cumulative 等价累计建设变量。

# P3/P4/P5 迭代和候选扩充。
MAX_ROUNDS = 200  # 预算驱动搜索的轮次尝试保护上限，恢复搜索沿用计数；0 跳过该循环。
NO_IMPROVEMENT_ROUNDS = 6  # 连续无改善达到此轮数时释放全部候选设施；这是搜索策略，不宣称邻域已穷尽。
LOCAL_SEARCH_MOVES = 20  # 每轮 P5 最多尝试的推迟一期建设动作数；0 关闭延迟搜索。
NEIGHBORHOOD_SIZE = 12  # P3 走廊邻域的设施节点上限；首次、停滞重启及恢复搜索首轮释放全部。
EXPANSION_BATCH = 3  # 无改善时扩池的批量：新增源机组、备选汇尝试、已有源补路各自的上限；0 关闭该扩池。
PIPE_COST_WEIGHT = 1.0  # P3 管道固定费代理项的非负权重；0 去掉该项，不改变真实目标成本的计算。

# 管径/建设时序精修。
PIPE_POLISH_ENABLED = True  # 启用固定设施与逐期弧总流量下的管径/建设期精修，以及最终逐期流量重优化。
PIPE_POLISH_TIME_LIMIT = 10.0  # 每次管道精修的总时限（秒），含枚举、建模与审计；0 跳过管道精修。
PIPE_ARC_TIME_LIMIT = 2.0  # 单条弧多管配置 MILP 的时限（秒）；0 跳过此类 MILP，仍可做单管枚举。
PIPE_POLISH_MAX_ARCS = 50  # 每次精修最多处理的多管配置 MILP 弧数；0 跳过此类 MILP，不限制单管枚举。

# 替代路线与联合邻域。
REROUTE_ENABLED = True  # 启用替代路线生成，将完整替代通道及相关设施加入候选池。
REROUTE_PATHS = 3  # 对原汇尝试至多 min(本值, 3) 种费用权重路径，另尝试至多本值个替代汇；须为正整数。
REROUTE_BATCH = 8  # 每次路线生成最多选择的现用源节点数，每个源选择一条走廊；0 跳过生成。
REROUTE_TIME_LIMIT = 20.0  # 每次替代路线与联合邻域生成的总时限（秒），含流分解与路径搜索；0 跳过生成。
JOINT_LNS_ENABLED = True  # 启用联合邻域 MILP，同时调整局部源机组/汇/井/管道建设并重算全网流量。
JOINT_LNS_TIME_LIMIT = 60.0  # 每个联合邻域 MILP 的时限（秒，含建模）；0 跳过联合求解。
JOINT_NEIGHBORHOOD_SIZE = 12  # 每个联合邻域可调整建设决策的源汇节点数上限，至少为 2；不限制弧数。
FINAL_POLISH_RESERVE = 120.0  # 尾部先重算运营与小精修；最多预留剩余预算的 20%，0 不预留。

# 求解器、审计和日志。
SP_SOLVER_METHOD = "RSS"  # P1 uses the paper's RSS solver.
MIP_GAP = 0.001  # MILP 子问题的相对 gap 阈值，0.001 表示 0.1%；不是启发式最终全局 gap。
THREADS = 0  # Gurobi 线程数；0 自动选择，P1 四种后端均接收此设置。
SEED = 0  # 求解器使用的非负随机种子；0 也是固定种子。
FEASIBILITY_TOLERANCE = 1e-6  # 原模型可行性审计的绝对容差；求解器内部可行性及整数容差会进一步收紧。
IMPROVEMENT_TOLERANCE = 1e-6  # 认定有效改进所需的目标降幅须大于本值 × max(1, abs(旧目标值))。
OUTPUT_FLAG = True  # 阶段、迭代及批量摘要；P3/P4 实际 gap 按百分比突出显示，不可用时显示 N/A。
GUROBI_OUTPUT_FLAG = False  # 是否输出各 Gurobi 子问题的详细求解日志。

RESULT_COLUMNS = [
    "period_num", "D", "Np", "max_capture_amount", "coef",
    "UB", "LB", "gap", "runtime", "curve", "status",
    "lp_status", "lp_objective", "lp_runtime", "lower_bound_origin",
    "max_violation"]
# 有可行解且达到预算或轮次保护上限，视为本次实验已完成，不代表全局最优。
# 保留旧版停滞状态名称；当前停滞仅触发重新释放设施，不终止搜索。
# 无解、求解器错误和人工中断不列为完成状态，下次运行重试。
COMPLETED_STATUSES = (
    "optimal", "time_limit", "root_optimal",
    "FEASIBLE_HEURISTIC", "FEASIBLE_STAGNATION", "FEASIBLE_SEARCH_LIMIT",
    "FEASIBLE_TIME_LIMIT")


def build_options(**overrides) -> HeuristicOptions:
    """调用时读取本文件顶部配置，显式实参只覆盖本次调用。"""

    values = {name: globals()[name.upper()]
              for name in HeuristicOptions.__dataclass_fields__}
    values.update(overrides)
    options = HeuristicOptions(**values)
    options.validate()
    if options.lp_time_limit == 0:
        raise ValueError("CCS_FNE_PH requires LP first; lp_time_limit must be None or positive")
    return options


def _search_heuristic(ctx):
    """从 P1 候选启动搜索，未取得可行解时保留完整 LP 的结果。"""

    pool = run_p1(ctx)
    if ctx.incumbent is None:
        ctx.record("search_stop", reason="NO_SOLUTION_P1")
        return "NO_SOLUTION_P1"
    return _search_rounds(ctx, pool)


def _search_rounds(ctx, pool):
    """在当前预算内继续 P3/P4/P5；轮号跨精修恢复保留，停滞触发全设施排期。"""

    ctx.search_pool = pool
    stall, open_all = 0, True
    for number in range(getattr(ctx, "search_round", 0), ctx.options.max_rounds):
        ctx.check_time()
        ctx.search_round = number + 1  # 中途预算截止的轮次也计入，恢复后继续下一个邻域。
        previous_best, history_start = ctx.best, len(ctx.history)
        reference = ctx.incumbent.copy()
        neighborhoods, changed_pool = _prepare_ph_round(ctx, pool, reference, number)
        scheduled = run_p3(ctx, pool, reference, number, open_all)
        run_p4(ctx, pool, scheduled)
        run_p5(ctx, pool, neighborhoods)
        improved = ctx.best < previous_best - ctx.options.improvement_tolerance * max(1.0, abs(previous_best))
        # 未完成不代表邻域穷尽；改进模式按无收益轮数触发更广搜索。
        unfinished = any(row.get("solver_status") in {"TIME_LIMIT", "INTERRUPTED", "SUBOPTIMAL"}
                         or row.get("stopped") == "time_limit"
                         or row.get("budget_exhausted", False)
                         for row in ctx.history[history_start:])
        stall = 0 if improved else stall + 1
        open_all = False
        expanded = False
        # 最后一轮不再生成来不及评估的候选；新增项保留到下一轮。
        if not improved and number + 1 < ctx.options.max_rounds:
            before_expansion = _copy_pool(pool)
            expanded = _expand_pool(ctx, pool, round_number=number)
            if expanded:
                # Carry new candidates into the next round before recycling old corridors.
                additions = _pool_additions(pool, before_expansion)
                if getattr(ctx, "ph_expansion_support", None) is not None:
                    additions.merge(ctx.ph_expansion_support)
                if getattr(ctx, "ph_pending_pool", None) is not None:
                    additions.merge(ctx.ph_pending_pool)
                ctx.ph_pending_pool = additions
        ctx.record("round", number=number + 1, stalled=stall, reference_objective=ctx.best,
                   new_candidates=changed_pool or expanded, unfinished_subproblems=unfinished,
                   **_round_mip_gaps(ctx.history[history_start:]))
        if stall >= ctx.options.no_improvement_rounds and number + 1 < ctx.options.max_rounds:
            ctx.record("search_restart", reason="stagnation", next_round=number + 2,
                       free_facilities="all", remaining=ctx.check_time())
            stall, open_all = 0, True
    ctx.record("search_stop", reason="max_rounds", rounds=getattr(ctx, "search_round", 0))
    return "FEASIBLE_HEURISTIC"


def _copy_pool(pool):
    return CandidatePool(set(pool.units), set(pool.sinks), dict(pool.wells), dict(pool.pipes))


def _pool_additions(pool, previous):
    return CandidatePool(pool.units - previous.units, pool.sinks - previous.sinks,
        {k: v for k, v in pool.wells.items() if v > previous.wells.get(k, 0)},
        {k: v for k, v in pool.pipes.items() if v > previous.pipes.get(k, 0)})


def _prepare_ph_round(ctx, pool, reference, number):
    """Keep the initial candidates, incumbent and current corridors in the active pool."""
    if not hasattr(ctx, "ph_base_pool"):
        ctx.ph_base_pool = _copy_pool(pool)
    active = _copy_pool(ctx.ph_base_pool)
    active.merge(CandidatePool.from_plan(reference))
    pending = getattr(ctx, "ph_pending_pool", None)
    new_nodes = set()
    new_facilities = set()
    if pending is not None:
        new_facilities = {i for i, _ in pending.units - active.units} | (pending.sinks - active.sinks)
        new_nodes = new_facilities | {
            j for j, count in pending.wells.items() if count > active.wells.get(j, 0)}
        active.merge(pending)
        ctx.ph_pending_pool = None
    old_pipes = set(pool.pipes)
    pool.units, pool.sinks, pool.wells, pool.pipes = (
        active.units, active.sinks, active.wells, active.pipes)
    neighborhoods, changed = prepare_neighborhoods(ctx, pool, reference, number)
    selected = set(neighborhoods[number % len(neighborhoods)].facilities) if neighborhoods else set()
    deferred_first = [node for node in getattr(ctx, "ph_deferred_nodes", [])
                      if node in new_nodes - new_facilities]
    nodes = sorted(new_facilities) + deferred_first + sorted(
        new_nodes - new_facilities - set(deferred_first)) + sorted(selected - new_nodes)
    # Keep the small explicit source/sink expansion group together, even if a
    # custom neighborhood limit is smaller than that group. Old well increases rotate.
    ctx.ph_schedule_nodes = set(nodes[:max(ctx.options.neighborhood_size, len(new_facilities))])
    ctx.ph_deferred_nodes = [node for node in nodes if node in new_nodes - ctx.ph_schedule_nodes]
    if pending is not None and ctx.ph_deferred_nodes:
        deferred = set(ctx.ph_deferred_nodes)
        ctx.ph_pending_pool = CandidatePool(
            {key for key in pending.units if key[0] in deferred}, pending.sinks & deferred,
            {j: count for j, count in pending.wells.items() if j in deferred}, dict(pending.pipes))
    ctx.record("candidate_pool", pipe_types=len(pool.pipes), units=len(pool.units),
               sinks=len(pool.sinks), removed_pipe_types=len(old_pipes - set(pool.pipes)),
               selected_facilities=len(ctx.ph_schedule_nodes))
    return neighborhoods, changed


def solve_heuristic(network: NetworkData, costs: CostData, diameters: list[str],
                    options: Optional[HeuristicOptions] = None, *,
                    pipeline_install_limit: int) -> HeuristicResult:
    """独立调度一个算例；配置缺省读取本文件，管道上限由网格显式传入。

    完整 LP 先行，只有其最优目标值用于全局下界；P1 未找到解时保留下界。
    TIME_LIMIT 从 LP 结束后计时；runtime 保留全流程耗时，lp_runtime 单列 LP 耗时。
    返回的完整方案须通过原模型审计，受限子问题的界不用于全局 gap。
    """

    ctx = _Context(network, costs, diameters, options or build_options(),
                   pipeline_install_limit=pipeline_install_limit)
    ctx.configuration["options"]["ph_search_improvements"] = True
    ctx.configuration["options"]["lp_formulation"] = LP_FORMULATION
    if ctx.options.lp_time_limit == 0:
        raise ValueError("CCS_FNE_PH requires LP first; lp_time_limit must be None or positive")
    ctx.configuration["initialization"] = "LP_then_P1"
    ctx.configuration["time_limit_scope"] = "post_lp"
    status = "FEASIBLE_HEURISTIC"
    try:
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", int(ctx.options.gurobi_output_flag))
            environment.start()
            ctx.env = environment
            # LP 仅受自身时限控制；保留 ctx.started 用于完整运行时间及日志。
            ctx.end = ctx.phase_end = float("inf")
            _attempt(ctx, "global_LP", run_lp, formulation=LP_FORMULATION)
            ctx.end = perf_counter() + ctx.options.time_limit
            ctx.phase_end = ctx.end
            if ctx.lp_status == "INFEASIBLE":
                ctx.record("stop", reason="INFEASIBLE_LP")
                return _result(ctx, "INFEASIBLE_LP")
            ctx.check_time()
            reserve = min(ctx.options.final_polish_reserve, 0.2 * ctx.check_time())
            ctx.phase_end = ctx.end - reserve
            ctx.record("search_budget", reserve=reserve, search_remaining=ctx.check_time())
            try:
                status = _search_heuristic(ctx)
            except _BudgetExpired:
                if perf_counter() >= ctx.end:
                    raise
                status = "FEASIBLE_SEARCH_LIMIT" if ctx.incumbent is not None else "SEARCH_LIMIT_NO_SOLUTION"
                ctx.record("search_stop", reason="final_polish_reserve")
            finally:
                ctx.phase_end = ctx.end
            final_polish(ctx)
            if (status == "FEASIBLE_SEARCH_LIMIT" and ctx.incumbent is not None
                    and getattr(ctx, "search_round", 0) < ctx.options.max_rounds):
                ctx.record("search_resume", reason="unused_final_polish_budget",
                           next_round=getattr(ctx, "search_round", 0) + 1,
                           remaining=ctx.check_time())
                status = _search_rounds(ctx, ctx.search_pool)
    except _BudgetExpired:
        status = "FEASIBLE_TIME_LIMIT" if ctx.incumbent is not None else "TIME_LIMIT_NO_SOLUTION"
        ctx.record("stop", reason=status)
    except gp.GurobiError as exc:
        status = "FEASIBLE_SOLVER_ERROR" if ctx.incumbent is not None else "SOLVER_ERROR_NO_SOLUTION"
        ctx.record("stop", reason=status, solver_error=str(exc))
    finally:
        ctx.env = None
    return _result(ctx, status)


def _case_grid():
    """与 CCS_FNE.py 使用相同的模型参数和遍历顺序。"""

    return product(PERIOD_NUM_LIST, MAX_CAPTURE_AMOUNT_LIST,
                   NP_LIST, CURVE_SPECS, D_LIST)


def save_results(rows: list[dict]) -> None:
    """原子保存已有记录与本次新增结果合并后的结果表。"""

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(pd.DataFrame(rows, columns=RESULT_COLUMNS), RESULT_FILE)


def run_case(network: NetworkData, costs: CostData, diameters: list[str],
             period_num: int, curve: str, coefficient,
             np: int, *, maximum_capture_amount: float) -> dict:
    """调用本文件的阶段调度，完整保留状态、可行上界及认证下界。"""

    if OUTPUT_FLAG:
        print(f"period_num={period_num}, Np={np}, curve={curve}, "
              f"coef={coefficient}, max_capture_amount={maximum_capture_amount:.3f}, "
              f"capture_target_max={max(costs.capture_target.values()):.3f}, "
              f"D={diameters}")
    result = solve_heuristic(
        network, costs, diameters, options=build_options(),
        pipeline_install_limit=np)
    if OUTPUT_FLAG:
        print(f"status={result.status}, UB={result.objective}, LB={result.lower_bound}, "
              f"original_global_gap={format(result.gap, '.4%') if result.gap is not None else 'N/A'}, runtime={result.runtime:.3f}s, "
              f"lp_status={result.lp_status}, lower_bound_origin={result.lower_bound_origin}")
    return {
        "period_num": period_num, "D": len(diameters), "Np": np,
        "max_capture_amount": maximum_capture_amount, "coef": coefficient,
        "UB": result.objective, "LB": result.lower_bound, "gap": result.gap,
        "runtime": result.runtime, "curve": curve, "status": result.status,
        "lp_status": result.lp_status, "lp_objective": result.lp_objective,
        "lp_runtime": result.lp_runtime, "lower_bound_origin": result.lower_bound_origin,
        "max_violation": result.max_violation}


def main() -> None:
    """跳过当前网格中已完成的算例，旧检查点只补充结果表尚无的记录。"""

    build_options()
    cases = list(_case_grid())
    rows, completed = read_result_rows(
        RESULT_FILE, cases, completed_statuses=COMPLETED_STATUSES)
    if not all(grid_instance_key(case) in completed for case in cases):
        checkpoint_rows, checkpoint_completed = read_result_rows(
            CHECKPOINT_FILE, cases, completed_statuses=COMPLETED_STATUSES)
        existing_keys = {result_instance_key(row) for row in rows}
        accepted_statuses = {normalize_result_status(status) for status in COMPLETED_STATUSES}
        imported = 0
        for row in checkpoint_rows:
            key = result_instance_key(row)
            if (key in checkpoint_completed and key not in existing_keys
                    and normalize_result_status(row["status"]) in accepted_statuses):
                rows.append(row)
                existing_keys.add(key)
                completed.add(key)
                imported += 1
        if imported:
            save_results(rows)
            if OUTPUT_FLAG:
                print(f"Recovered {imported} completed cases from legacy checkpoint: {CHECKPOINT_FILE}")
    if OUTPUT_FLAG:
        print(f"Resume: {len(cases)} current cases, {len(completed)} completed cases skipped, "
              f"{len(cases) - len(completed)} cases pending.")
    if all(grid_instance_key(case) in completed for case in cases):
        print(f"All current cases already completed. Results: {RESULT_FILE}")
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    network = load_network_data(DATA_DIR)
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
