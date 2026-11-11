"""拉格朗日松弛单案例入口：顶部配置、求解调度与结果输出。

结构参考父目录 run_PH.py；只运行下面指定的一个模型算例。
可直接运行本文件，也可从项目根目录执行：python -m LRforFNE.run_lrh
"""
from __future__ import annotations

from copy import deepcopy
import sys
sys.dont_write_bytecode = True  # 复用父模块时不更新父目录字节码缓存。

from pathlib import Path
from time import perf_counter

# 直接运行文件时补齐包上下文；路径由本文件定位，不依赖 IDE 或当前目录。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "LRforFNE"

from experiment_setup import (D_LIST, compute_cost_data, load_network_data,
                              period_schedule)
from heuristic_stages.data_io import with_local_capture_target
from solve_logging import logged_run

from .config import LRConfig
from .result_io import OUTPUT_ROOT, write_result, write_workbook
from .solver import solve_lr
from .warm_start import WarmStartOptions


# ==================== 输入输出文件 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent  # 默认读取父项目根目录的六个 CSV。
RESULT_DIR = SCRIPT_DIR / "output" / "run_lrh"
# 结果写入 RESULT_DIR/<method>/；开启归一化时增加 normalized/ 子目录，须在 LRforFNE/output 内。
# 同一路径再次运行会覆盖该方法的结果；保留历史时先修改 RESULT_DIR。
# 输出 variables.xlsx、result.json、初始化及迭代 history JSONL、方案和乘子 JSON。

# ==================== 模型参数 ====================
PERIOD_NUM = 5  # 期数，须整除 experiment_setup.TY；总年限和费用参数沿用父项目。
NP = 3  # 每条有向弧全期、全部管径合计的最大管道建设数量。
DIAMETERS = list(D_LIST[-1])  # 默认使用网格中的最后一组；可改索引或填写按容量递增的管径名称列表。
CURVE = "linear"  # linear / exponential / logarithmic。
COEFFICIENT = None  # linear 使用 None；其余曲线填写对应系数。
MAX_CAPTURE_AMOUNT = 1e6  # 末期年捕集目标，单位 kt/年；默认不启用等总量归一化。
NORMALIZE_CAPTURE_TARGET = False  # True：是，非线性曲线与线性基准等总量；False：否。
CAPTURE_AMOUNT_SEQUENCE = None  # 可填写长度为 PERIOD_NUM 的非负序列，允许非单调；优先于曲线。

# ==================== 算法参数 ====================
METHOD = "bundle"  # The Lagrangian interfaces support proximal Bundle only.

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
LR_ORACLE_TIME_LIMIT = 30.0  # 捕集/封存每块每次求解的局部时限，同时受 LR 剩余预算限制。
LR_CAPTURE_MIP_GAP = 2e-4  # 捕集 MILP 相对 gap=0.02%；不随 oracle 精度因子收紧。
LR_STORAGE_MIP_GAP = 2e-4  # 封存 MILP 同样使用固定相对 gap=0.02%。
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
LR_NEIGHBORHOOD_TIME_LIMIT = 30.0
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
LR_N3_MAX_ARCS = 80  # 完整路径闭包超限时跳过，不截断路径。
LR_N3_MAX_FACILITIES = 12
LR_N3_CANDIDATE_LIMIT = 6  # 每批择优及探索候选数；联合替换独立获得搜索机会。
LR_N3_EXPAND_CORRIDORS = 4  # 每个 N3 域最多加入四条相关完整走廊；0 恢复原域。
LR_N3_EXPANDED_TIME_LIMIT = 45.0  # 扩展域每次时限；开启扩展时每批最多四个候选。
LR_N3_REFINE_CANDIDATES = 1  # 每批至多对这些高潜力候选进行同模型精化；0 关闭。
LR_N3_REFINE_TIME_LIMIT = 1.0  # 精化额外秒数，仍受邻域及 LR 总截止时间限制。
LR_N3_REFINE_ABSOLUTE_GAP = 100.0  # N3 精化的绝对费用容差；不改变捕集/封存精度。

# 管道精修。
LR_PIPELINE_POLISH_TIME_LIMIT = 40.0  # 全网固定流量管道精修；0 关闭，全部计入 LR 总预算。


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


def _log_path():
    directory = Path(RESULT_DIR).resolve()
    directory.relative_to(OUTPUT_ROOT.resolve())
    return directory / ('normalized' if NORMALIZE_CAPTURE_TARGET else '') / 'solve.log'


@logged_run(_log_path)
def main() -> None:
    if METHOD != "bundle":
        raise ValueError('METHOD must be bundle')
    # 在加载全国数据/启动求解前核对输出位置和算法配置。
    case_name = Path(RESULT_DIR).resolve().relative_to(OUTPUT_ROOT.resolve())
    if NORMALIZE_CAPTURE_TARGET:
        case_name /= 'normalized'
    config, options = build_config(), build_heuristic_options()
    if options.lp_time_limit == 0:
        raise ValueError('multiplier LP time limit must be None or positive')
    diameters = list(DIAMETERS)
    print(f"method={METHOD}, period_num={PERIOD_NUM}, NP={NP}, D={diameters}, "
          f"curve={CURVE}, coefficient={COEFFICIENT}, max_capture={MAX_CAPTURE_AMOUNT}, "
          f"capture_sequence={CAPTURE_AMOUNT_SEQUENCE}, normalize_capture_target={NORMALIZE_CAPTURE_TARGET}")
    started = perf_counter()
    network = load_network_data(DATA_DIR)
    data_runtime = perf_counter()-started
    prepared = perf_counter()
    periods, durations = period_schedule(PERIOD_NUM)
    costs = with_local_capture_target(
        compute_cost_data(network, periods, durations, apply_shared_capture_target=False),
        CURVE, COEFFICIENT, MAX_CAPTURE_AMOUNT, CAPTURE_AMOUNT_SEQUENCE,
        normalize_capture_target=NORMALIZE_CAPTURE_TARGET)
    preparation_runtime = perf_counter()-prepared
    print(f"capture_target={costs.capture_target}")
    invocation = deepcopy(dict(
        entrypoint='LRforFNE.run_lrh', data_dir=str(Path(DATA_DIR).resolve()),
        method=METHOD, period_num=PERIOD_NUM, pipeline_install_limit=NP,
        diameters=diameters, curve=CURVE, coefficient=COEFFICIENT,
        max_capture_amount=MAX_CAPTURE_AMOUNT,
        normalize_capture_target=NORMALIZE_CAPTURE_TARGET,
        capture_amount_sequence=None if CAPTURE_AMOUNT_SEQUENCE is None
            else list(CAPTURE_AMOUNT_SEQUENCE)))
    kwargs = dict(pipeline_install_limit=NP, config=config, heuristic_options=options)
    results = {METHOD: solve_lr(network, costs, diameters, method=METHOD, **kwargs)}
    for method, result in results.items():
        result.configuration['normalize_capture_target'] = invocation['normalize_capture_target']
        result.configuration['invocation'] = deepcopy(invocation)
        result.timings.update(data_read_wall=data_runtime, cost_preparation_wall=preparation_runtime)
        output = write_result(result, case_name)
        print(f"{method}: status={result.status}, reason={result.termination_reason}, "
              f"UB={result.objective}, LB={result.lower_bound}, gap={result.gap}, "
              f"runtime={result.total_runtime:.3f}s\nresults={output}")
        workbook = write_workbook(result, network, costs, diameters,
                                  output / "variables.xlsx")
        print(f"workbook={workbook}")
        if result.objective is None:
            print("未取得原问题可行解，已保存参数、求解摘要及日志，不导出建设变量或绘图。")


if __name__ == "__main__":
    main()
