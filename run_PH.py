"""CCS_FNE 原始启发式（PH, primal heuristic）单案例入口：配置、阶段调度与结果输出。

各阶段实现均位于 heuristic_stages，P1 使用其中的 p1_candidates。
阶段顺序：LP -> P1 -> 循环（走廊准备 -> P3/P4/P5）-> 最终精修 -> 按条件恢复搜索。
停滞触发下一轮全候选设施排期；精修后恢复搜索沿用候选池、最好解和轮号。
所有单案例运行参数保留在本文件顶部；LP 独立计时，后续阶段共享启发式预算。
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import gurobipy as gp

from experiment_setup import (CostData, D_LIST, NetworkData, compute_cost_data,
    load_network_data, period_schedule)
from plot_results import plot_results
from solve_logging import logged_run
from heuristic_stages.common import (CandidatePool, HeuristicOptions, HeuristicResult,
    Plan, VARIABLES, _BudgetExpired, _Context, _attempt, _result, _round_mip_gaps)
from heuristic_stages.data_io import (build_capture_target, with_local_capture_target,
    save_heuristic_results as _save_heuristic_results)
from heuristic_stages.lp_bound import run_lp
from heuristic_stages.p1_candidates import run_p1
from heuristic_stages.p3_schedule import run_p3
from heuristic_stages.p4_pipelines import run_p4
from heuristic_stages.p5_improvement import (RerouteNeighborhood, run_p5, prepare_neighborhoods,
    _expand_pool, final_polish)


# ==================== 输入输出文件 ====================
# 文件输出包括含求解日志的最终工作簿和逐期演化图。
SCRIPT_DIR = Path(__file__).resolve().parent
# 本脚本所在目录，作为默认路径基准。
DATA_DIR = SCRIPT_DIR
# 网络输入数据目录，传给 load_network_data。
RESULT_FILE = SCRIPT_DIR / "output" / "CCS_FNE_PH_variables.xlsx"
# 输出七类变量，以及 model_parameters、algorithm_parameters、solve_summary、solve_log 工作表。

# 最终方案绘图；当前 main 使用 evolution_only=True，仅输出逐期累计建设图。
# 单案例图片输出路径；绘图开关、样式、地图边界和容差统一在 plot_results.py 修改。
PLOT_OUTPUT_DIR = RESULT_FILE.parent  # 绘图文件目录（单案例固定）；与输出工作簿共享目录。
PLOT_OUTPUT_IMAGE_NAME = "CCS_FNE_PH_results.png"
PLOT_FILE = PLOT_OUTPUT_DIR / PLOT_OUTPUT_IMAGE_NAME  # 绘图接口路径；evolution_only=True 时不生成此汇总图。
PLOT_PERIOD_OUTPUT_DIR = PLOT_OUTPUT_DIR / f"{Path(PLOT_OUTPUT_IMAGE_NAME).stem}_periods"  # 各期 construction_cumulative_period_XX.png 所在目录。

# ==================== 模型参数 ====================
# 单次运行参数，直接在此修改。
PERIOD_NUM = 30  # 规划期数；须整除 experiment_setup.TY，总年数按此均分（当前 TY=30）。
NP = 3  # 每条有向弧全期、所有管径合计的最大建设条数；须为非负整数。
DIAMETERS = D_LIST[-1]  # 默认使用网格中的最后一组；须按容量递增，可改索引或直接填写管径名称列表。
CURVE = "linear"  # 目标曲线类型：linear（线性）、exponential（指数）、logarithmic（对数）。
COEFFICIENT = None  # 曲线系数：线性设 None；指数为非零有限数；对数须大于 -1 且非零。
MAX_CAPTURE_AMOUNT = 2e6  # 归一化关闭时为曲线末期年目标；开启时为线性基准末期年目标。
NORMALIZE_CAPTURE_TARGET = False  # True（是）：非线性曲线与线性基准等总量；False（否）。
CAPTURE_AMOUNT_SEQUENCE = None  # 自定义逐期年捕集目标列表，长度须等于 PERIOD_NUM；非 None 时覆盖曲线参数。

# ==================== 算法参数 ====================
# 启发式总预算从完整 LP 结束后开始，包含后续建模、审计、流量处理和优化。
# 图算法/建模/审计在操作边界检查时间，单次不可中断操作可能超时。
TIME_LIMIT = 3600  # 完整 LP 结束后的启发式总预算（秒），包括最终精修，不含 LP。

# P1: solve E-Select with RSS and lift its design to a feasible multi-period plan.
# 建设费按各期起剩余累计需求加权，运行费取各期均值；静态目标取本次逐期目标最大值。
# RSS 调用 RSSforESM，以真实费用反馈更新斜率并恢复整数管道；近似模型界不作全局界。
# 全部剩余预算交给所选后端，不再额外进行 MILP 精修；首期建设并按各期目标缩放峰值流量。
# 静态、逐期和建设总量耦合审计均须通过，始终使用 E-Select 目标。
P1_TIME_LIMIT = 750.0  # P1 总预算（秒），含准备、ESM 求解和审计，受全局剩余预算限制；0 跳过 P1。

# P3/P4 与流量重算子问题时限。
# P3 在当前阶段截止前预留 5 秒供见证恢复、原模型审计与管线精修；这是经验余量，不保证严格按时完成。
SCHEDULE_TIME_LIMIT = 60.0  # 每次 P3 全期 source/sink/井排期 MILP 的时限（秒）。
FIRST_PIPE_POLISH_TIME_LIMIT = 30.0  # 首轮 P3 后精修时限；预算内遍历全部候选弧，None 沿用普通精修。
PIPE_TIME_LIMIT = 90.0  # 每次 P4 固定设施后，优化管道与全部流量的时限（秒）。
FLOW_TIME_LIMIT = 20.0  # 单次单期流量重算的时限（秒）。

# 完整原问题 LP：首先求解以取得全局下界，再由 P1 生成初始可行解。
LP_TIME_LIMIT = None  # None 不限制 LP 时间；正数为独立 LP 预算（秒，含建模），不占 TIME_LIMIT，不允许为 0。
LP_METHOD = 2  # LP 算法：2 内点法；-1 自动、0 原始单纯形、1 对偶单纯形，其余取值同 Gurobi。
LP_CROSSOVER = 0  # 内点法后的基解交叉转换：0 关闭、-1 自动；只求 LP 下界时可关闭。
LP_FORMULATION = "cumulative"  # 完整 LP 表达：original 原增量变量；cumulative 等价累计建设变量。

# P3～P5 迭代、局部改进和候选扩充。
MAX_ROUNDS = 200  # 预算驱动搜索的轮次尝试保护上限，恢复搜索沿用计数；0 跳过该循环。
NO_IMPROVEMENT_ROUNDS = 6  # 连续无改善达到此轮数时释放全部候选设施；这是搜索策略，不宣称邻域已穷尽。
LOCAL_SEARCH_MOVES = 20  # 每轮最多尝试的推迟一期动作数；0 关闭延迟搜索。
NEIGHBORHOOD_SIZE = 12  # P3 走廊邻域的设施节点上限；首次、停滞重启及恢复搜索首轮释放全部。
EXPANSION_BATCH = 3  # 停滞扩充时每类候选的批量上限；0 关闭候选扩充。
PIPE_COST_WEIGHT = 1.0  # 管道固定费代理项权重 lambda；非负，0 去掉该代理项，不改变真实成本计算。

# 管径/时序精修：保持逐期弧总流量，逐弧独立优化，随后可重算全网运营流量。
PIPE_POLISH_ENABLED = True  # 开启确定性单管枚举及多管小 MILP；新管径仍限本次 DIAMETERS。
PIPE_POLISH_TIME_LIMIT = 10.0  # 单次精修总预算（秒），包含枚举、建模与审计；0 关闭。
PIPE_ARC_TIME_LIMIT = 2.0  # 每条多管弧的小 MILP 时限（秒）；0 仅做不需要 MILP 的枚举。
PIPE_POLISH_MAX_ARCS = 50  # 每次最多优化的多管弧数；0 跳过多管 MILP，不限制单管枚举。

# 实际正流量走廊的替代路线与全期联合邻域。
REROUTE_ENABLED = True  # 按实际流量/费用及轮换顺序生成替代走廊，允许池外的原网络弧入池。
REROUTE_PATHS = 3  # 原汇费用路径数（最多 3 种权重）及备选汇数量的上限，须为正整数。
REROUTE_BATCH = 8  # 每轮最多生成的走廊邻域数；0 关闭通道扩充。
REROUTE_TIME_LIMIT = 20.0  # 每次路径筛选总预算（秒），包含流分解与路径搜索；0 关闭。
JOINT_LNS_ENABLED = True  # 同时释放局部 source/sink/井/管线建设，连续流量允许全网重算。
# 改进模式把邻域外固定建设代入常量；全部运行流量和转运节点平衡仍保留。
JOINT_LNS_TIME_LIMIT = 60.0  # 每个联合邻域时限（秒），包括建模；0 关闭联合求解。
JOINT_NEIGHBORHOOD_SIZE = 12  # 每个联合邻域最多释放的设施节点数，源与汇合计，须至少为 2。
FINAL_POLISH_RESERVE = 120.0  # 尾部先重算运营与小精修；最多预留剩余预算的 20%，0 不预留。

# 求解器、可行性检查及日志。
SP_SOLVER_METHOD = "RSS"  # P1 uses the paper's RSS solver.
MIP_GAP = 0.001  # MILP 子问题的相对 gap 阈值，0.01 表示 1%；不是启发式最终全局 gap。
THREADS = 0  # Gurobi 线程数；0 自动，四个静态后端均原样传递。
SEED = 0  # 非负随机种子，用于求解器；0 也是固定种子。
FEASIBILITY_TOLERANCE = 1e-6  # 原模型可行性审计的绝对容差；求解器内部容差会进一步收紧。
IMPROVEMENT_TOLERANCE = 1e-6  # 接受改进所需的降幅须大于该值 × max(1, abs(旧目标值))。
OUTPUT_FLAG = True  # 阶段、迭代及改进摘要；P3/P4 实际 gap 按百分比突出显示，不可用时显示 N/A。
GUROBI_OUTPUT_FLAG = False  # 是否输出各 Gurobi 子问题的详细求解日志。
# 单算例同名 .log 保存完整运行日志；Gurobi 日志开关只影响终端显示。

def build_options(**overrides) -> HeuristicOptions:
    """调用时读取顶部配置；显式实参只覆盖本次调用。"""
    values = {name: globals()[name.upper()] for name in HeuristicOptions.__dataclass_fields__}
    values.update(overrides)
    result = HeuristicOptions(**values)
    result.validate()
    return result


def audit_solution(network, costs, diameters, solution, tolerance=1e-6, *,
                   pipeline_install_limit: Optional[int] = None):
    """独立核验实际建设日期及七类变量；不以最早建设替换输入排期。"""
    ctx = _Context(network, costs, diameters,
                   HeuristicOptions(feasibility_tolerance=tolerance, output_flag=False),
                   pipeline_install_limit=NP if pipeline_install_limit is None else pipeline_install_limit)
    plan = Plan(**{n: dict(solution[n] if isinstance(solution, dict) else getattr(solution, n))
                   for n in VARIABLES})
    return ctx.audit(plan)


def _search_heuristic(ctx):
    """在搜索预算内由 P1 建立初解和候选池，再启动 P3/P4/P5 外循环。"""
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
        # 最后一轮不再制造来不及评估的候选；真正扩充后重置计数并保证后续一轮。
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
                    pipeline_install_limit: Optional[int] = None) -> HeuristicResult:
    """先求原问题 LP 下界，再从无初解的 P1 启动；options 缺省读取顶部配置。

    不设置源、井或管道的逐期建设数量限制。
    TIME_LIMIT 从 LP 结束后计时；runtime 保留全流程耗时，lp_runtime 单列 LP 耗时。
    pipeline_install_limit 为每条有向弧全期、所有管径合计的建设条数上限，
    仅接受非负整数，缺省读取顶部 NP，不按周期年数换算。
    LP 未求得最优解时不采用其目标值作为下界；预算允许时仍尝试 P1。
    P1 未取得可行解时保留已求得的 LP 下界，返回无解状态，gap 为 None。
    仅返回经原模型约束审计的完整解。SP/P3 的代理目标和 P4 的受限界
    不充当原问题的目标值或全局下界；本入口不返回专门的全局最优状态。
    停滞仅触发全候选设施排期；阶段截止后的精修若正常完成且仍有预算和
    可用轮次，则沿用候选池及轮号恢复搜索，不重复 P1 或再次预留精修时间。
    """
    ctx = _Context(network, costs, diameters, options or build_options(),
                   pipeline_install_limit=NP if pipeline_install_limit is None else pipeline_install_limit)
    ctx.configuration["options"]["ph_search_improvements"] = True
    ctx.configuration["options"]["lp_formulation"] = LP_FORMULATION
    if ctx.options.lp_time_limit == 0:
        raise ValueError("run_PH requires LP first; lp_time_limit must be None or positive")
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


def save_heuristic_results(result: HeuristicResult, network: NetworkData,
                           costs: CostData, diameters: list[str],
                           output_file: Optional[Path] = None) -> Path:
    """输出七张变量宽表、模型/算法参数、求解摘要和日志；无解时仍保存参数与日志。"""
    return _save_heuristic_results(result, network, costs, diameters,
                                  RESULT_FILE if output_file is None else output_file,
                                  include_logs=True)


@logged_run(lambda: RESULT_FILE.with_suffix('.log'))
def main() -> None:
    data_dir = str(Path(DATA_DIR).resolve())
    print(f"period_num={PERIOD_NUM}, Np={NP}, curve={CURVE}, coef={COEFFICIENT}, "
          f"max_capture={MAX_CAPTURE_AMOUNT}, capture_sequence={CAPTURE_AMOUNT_SEQUENCE}, D={DIAMETERS}")
    network = load_network_data(DATA_DIR)
    periods, durations = period_schedule(PERIOD_NUM)
    costs = with_local_capture_target(
        compute_cost_data(network, periods, durations, apply_shared_capture_target=False),
        CURVE, COEFFICIENT, MAX_CAPTURE_AMOUNT, CAPTURE_AMOUNT_SEQUENCE,
        normalize_capture_target=NORMALIZE_CAPTURE_TARGET)
    print(f"normalize_capture_target={NORMALIZE_CAPTURE_TARGET}")
    print(f"capture_target={costs.capture_target}")
    result = solve_heuristic(network, costs, DIAMETERS, pipeline_install_limit=NP)
    result.configuration.setdefault("model_parameters", {})["data_dir"] = data_dir
    print(f"status={result.status}, objective={result.objective}, lower_bound={result.lower_bound}, "
          f"original_global_gap={format(result.gap, '.4%') if result.gap is not None else 'N/A'}, runtime={result.runtime:.3f}s")
    print(f"lp_status={result.lp_status}, lp_objective={result.lp_objective}, "
          f"lower_bound_origin={result.lower_bound_origin}, lp_runtime={result.lp_runtime:.3f}s")
    print(f"capture_by_period={result.capture_by_period}, total_capture={result.total_capture}")
    output = save_heuristic_results(result, network, costs, DIAMETERS)
    print(f"Heuristic results saved: {output}")
    if result.objective is None:
        print("未取得可行解，已保存模型/算法参数、求解摘要与日志，不导出建设方案或绘图。")
        return
    if result.objective is not None:
        for image in plot_results(result, network, periods=costs.periods, diameters=DIAMETERS,
                output_file=PLOT_FILE, period_output_dir=PLOT_PERIOD_OUTPUT_DIR, evolution_only=True):
            print(f"result_plot={image}")


if __name__ == "__main__":
    main()
