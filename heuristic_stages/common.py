"""共享数据结构、原模型审计、流量子问题与预算控制。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from time import perf_counter
from typing import Optional

import gurobipy as gp

from read import read_pipeline_limit, read_well_capacities, read_well_injection_capacities
from result_metadata import build_model_parameters
from solve_logging import optimize_with_log


VARIABLES = ("s", "r", "w", "y", "a", "b", "x")  # 四类建设变量与三类年流量变量。
COST_FIELDS = dict(s="source_fixed", r="sink_fixed", w="well_fixed",  # 变量对应的 CostData 成本字段。
                   y="pipe_fixed", a="source_variable", b="sink_variable",
                   x="pipe_variable")


def _status_name(status: int) -> str:
    names = {
        gp.GRB.OPTIMAL: "OPTIMAL", gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD", gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT", gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC", gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"STATUS_{status}")


@dataclass(frozen=True)
class HeuristicOptions:
    time_limit: float = 15000.0
    sp_solver_method: str = "RSS"
    p1_time_limit: float = 120.0
    schedule_time_limit: float = 60.0
    pipe_time_limit: float = 120.0
    first_pipe_polish_time_limit: Optional[float] = None
    flow_time_limit: float = 10.0
    lp_time_limit: Optional[float] = None
    lp_method: int = 2
    lp_crossover: int = 0
    max_rounds: int = 10
    no_improvement_rounds: int = 3
    local_search_moves: int = 20
    neighborhood_size: int = 12
    expansion_batch: int = 3
    pipe_cost_weight: float = 1.0
    pipe_polish_enabled: bool = True
    pipe_polish_time_limit: float = 30.0
    pipe_arc_time_limit: float = 1.0
    pipe_polish_max_arcs: int = 100
    reroute_enabled: bool = True
    reroute_paths: int = 3
    reroute_batch: int = 5
    reroute_time_limit: float = 10.0
    joint_lns_enabled: bool = True
    joint_lns_time_limit: float = 60.0
    joint_neighborhood_size: int = 12
    final_polish_reserve: float = 600.0
    mip_gap: float = 0.01
    threads: int = 1
    seed: int = 0
    feasibility_tolerance: float = 1e-6
    improvement_tolerance: float = 1e-7
    output_flag: bool = True
    gurobi_output_flag: bool = False

    def validate(self):
        if self.sp_solver_method != "RSS":
            raise ValueError("sp_solver_method must be RSS")
        for name in ("time_limit", "flow_time_limit", "schedule_time_limit",
                     "pipe_time_limit", "feasibility_tolerance"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("p1_time_limit", "pipe_cost_weight",
                     "improvement_tolerance", "mip_gap", "pipe_polish_time_limit",
                     "pipe_arc_time_limit", "reroute_time_limit", "joint_lns_time_limit",
                     "final_polish_reserve"):
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lp_time_limit is not None and (not isfinite(self.lp_time_limit) or self.lp_time_limit < 0):
            raise ValueError("lp_time_limit must be None or finite and nonnegative")
        if (self.first_pipe_polish_time_limit is not None
                and (not isfinite(self.first_pipe_polish_time_limit) or self.first_pipe_polish_time_limit <= 0)):
            raise ValueError("first_pipe_polish_time_limit must be None or finite and positive")
        if not isinstance(self.lp_method, int) or self.lp_method not in range(-1, 6):
            raise ValueError("lp_method must be an integer between -1 and 5")
        if not isinstance(self.lp_crossover, int) or self.lp_crossover not in range(-1, 6):
            raise ValueError("lp_crossover must be an integer between -1 and 5")
        for name in ("max_rounds",
                     "local_search_moves", "expansion_batch", "threads", "seed",
                     "pipe_polish_max_arcs", "reroute_batch"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("no_improvement_rounds", "neighborhood_size", "reroute_paths"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.joint_neighborhood_size, int) or self.joint_neighborhood_size < 2:
            raise ValueError("joint_neighborhood_size must be an integer >= 2")
        for name in ("pipe_polish_enabled", "reroute_enabled", "joint_lns_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass
class Plan:
    s: dict = field(default_factory=dict)
    r: dict = field(default_factory=dict)
    w: dict = field(default_factory=dict)
    y: dict = field(default_factory=dict)
    a: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    x: dict = field(default_factory=dict)

    def copy(self):
        return Plan(**{name: dict(getattr(self, name)) for name in VARIABLES})


@dataclass
class CandidatePool:
    units: set
    sinks: set
    wells: dict
    pipes: dict

    @classmethod
    def from_plan(cls, plan):
        wells, pipes = defaultdict(int), defaultdict(int)
        for (j, _), value in plan.w.items():
            wells[j] += value
        for (*ed, _), value in plan.y.items():
            pipes[tuple(ed)] += value
        return cls({(i, g) for (i, g, _), v in plan.s.items() if v},
                   {j for (j, _), v in plan.r.items() if v},
                   dict(wells), dict(pipes))

    def merge(self, other):
        self.units.update(other.units)
        self.sinks.update(other.sinks)
        for name in ("wells", "pipes"):
            dest = getattr(self, name)
            for key, value in getattr(other, name).items():
                dest[key] = max(dest.get(key, 0), value)

    def covers(self, plan):
        other = CandidatePool.from_plan(plan)
        return (other.units <= self.units and other.sinks <= self.sinks
                and all(v <= self.wells.get(k, 0) for k, v in other.wells.items())
                and all(v <= self.pipes.get(k, 0) for k, v in other.pipes.items()))


@dataclass(frozen=True)
class HeuristicResult:
    status: str
    objective: Optional[float]
    lower_bound: Optional[float]
    gap: Optional[float]
    runtime: float
    s: dict
    r: dict
    w: dict
    y: dict
    a: dict
    b: dict
    x: dict
    pa: dict
    pb: dict
    px: dict
    history: list
    max_violation: Optional[float]
    capture_by_period: dict
    total_capture: Optional[float]
    lower_bound_origin: str
    lp_status: str
    lp_objective: Optional[float]
    lp_runtime: float
    configuration: dict = field(default_factory=dict)


class _BudgetExpired(TimeoutError):
    pass


def _values(variables, integer=False):
    result = {}
    for key, variable in variables.items():
        value = int(round(variable.X)) if integer else max(0.0, float(variable.X))
        if value > 1e-10:
            result[key] = value
    return result


def _cumulative(values, periods):
    cumulative = defaultdict(float)
    for key, value in values.items():
        for t in periods:
            if t >= key[-1]:
                cumulative[(*key[:-1], t)] += value
    return cumulative


def _objective(costs, plan):
    return sum(getattr(costs, COST_FIELDS[name])[key] * value
               for name in VARIABLES for key, value in getattr(plan, name).items())


class _Context:
    def __init__(self, network, costs, diameters, options,
                 *, pipeline_install_limit: int):
        self.started = perf_counter()
        self.options = options
        options.validate()
        self.end = self.started + options.time_limit
        self.phase_end = self.end  # 搜索阶段可提前截止；self.end 始终是全局硬预算。
        self.n, self.c, self.d = network, costs, list(diameters)
        self.t = list(costs.periods)
        if self.t != list(range(1, len(self.t) + 1)) or not self.t:
            raise ValueError("periods must be consecutive and start at 1")
        if not self.d or len(set(self.d)) != len(self.d):
            raise ValueError("diameters must be nonempty and unique")
        if any(d not in network.pipe_capacity for d in self.d):
            raise ValueError("Unknown pipeline diameter")
        caps = [network.pipe_capacity[d] for d in self.d]
        if any(not isfinite(v) or v <= 0 for v in caps):
            raise ValueError("Pipeline capacities must be finite and positive")
        if any(a >= b for a, b in zip(caps, caps[1:])):
            raise ValueError("Pipeline diameters must be ordered by increasing capacity")
        self.qw = read_well_injection_capacities(network.well_capacity, network.sinks)
        if any(value <= 0 for value in self.qw.values()):
            raise ValueError("well_capacity must be positive for every sink")
        for values in (network.source_capacity, network.max_wells):
            if any(not isfinite(v) or v < 0 for v in values.values()):
                raise ValueError("Facility capacities must be finite and nonnegative")
        if set(costs.capture_target) != set(self.t) or set(costs.durations) != set(self.t):
            raise ValueError("Targets and durations must define every period")
        if any(not isfinite(v) or v < 0 for v in costs.capture_target.values()):
            raise ValueError("Capture targets must be finite and nonnegative")
        if any(not isfinite(v) or v <= 0 for v in costs.durations.values()):
            raise ValueError("Durations must be finite and positive")
        self.max_wells = read_well_capacities(network, costs)
        self.pipeline_install_limit = read_pipeline_limit(pipeline_install_limit)
        self.pipe_totals = {edge: self.pipeline_install_limit for edge in network.ij}
        self.nodes = list(network.sources) + list(network.sinks) + list(network.intersections)
        if len(set(self.nodes)) != len(self.nodes) or len(set(network.ij)) != len(network.ij):
            raise ValueError("Node classes must be disjoint and arcs unique")
        uncovered = {node for edge in network.ij for node in edge} - set(self.nodes)
        if uncovered:
            raise ValueError(f"Network contains nodes without flow-balance constraints: {uncovered}")
        self.ig = set(network.ig)
        self.arcs = set(network.ij)
        self.keys = {
            "s": {(i, g, t) for i, g in network.ig for t in self.t},
            "r": {(j, t) for j in network.sinks for t in self.t},
            "y": {(u, v, d, t) for u, v in network.ij for d in self.d for t in self.t},
        }
        self.keys.update(w=self.keys["r"], a=self.keys["s"],
                         b=self.keys["r"], x=self.keys["y"])
        for name in VARIABLES:
            mapping = getattr(costs, COST_FIELDS[name])
            for key in self.keys[name]:
                if key not in mapping or not isfinite(mapping[key]) or mapping[key] < 0:
                    raise ValueError(f"Missing, negative or nonfinite {COST_FIELDS[name]}[{key}]")
        self.history = []
        self.incumbent = None
        self.best = None
        self.lower_bound = 0.0  # valid because nonnegative costs were checked above
        self.bound_origin = "nonnegative_costs"
        self.lp_status = "DISABLED" if options.lp_time_limit == 0 else "NOT_RUN"
        self.lp_objective = None
        self.lp_runtime = 0.0
        self.env = None
        self.configuration = dict(options=asdict(options), periods=list(self.t), diameters=list(self.d),
            durations=dict(costs.durations), capture_target=dict(costs.capture_target),
            max_wells=dict(self.max_wells), construction_limits="full_horizon_v1",
            pipeline_install_limit=self.pipeline_install_limit,
            model_parameters=build_model_parameters(
                network, costs, self.d, self.pipeline_install_limit))

    def check_time(self, end=None):
        remaining = min(self.end, self.phase_end, end if end is not None else self.end) - perf_counter()
        if remaining <= 0:
            raise _BudgetExpired()
        return remaining

    def deadline(self, cap):
        return min(self.end, self.phase_end, perf_counter() + cap)

    def model(self, name):
        self.check_time()
        model = gp.Model(name, env=self.env)
        self.configure(model)
        return model

    def configure(self, model):
        model.Params.OutputFlag = int(self.options.gurobi_output_flag)
        model.Params.Threads = self.options.threads
        model.Params.FeasibilityTol = max(1e-9, min(1e-8, self.options.feasibility_tolerance / 10))
        model.Params.IntFeasTol = max(1e-9, min(1e-8, self.options.feasibility_tolerance / 10))
        model.Params.MIPGap = self.options.mip_gap
        model.Params.Seed = self.options.seed

    def optimize(self, model, end):
        model.Params.TimeLimit = self.check_time(end)
        optimize_with_log(model)

    def record(self, stage, *, message=None, **details):
        if stage in ("P3_schedule", "P4_pipelines"):
            details.setdefault(f"{stage[:2].lower()}_mip_gap", details.get("restricted_gap"))
        if stage == "round" or stage.endswith("_coupling_round"):
            details["original_global_gap"] = (max(0.0, self.best - self.lower_bound)
                / max(1.0, abs(self.best)) if self.incumbent is not None else None)
        row = dict(stage=stage, elapsed=perf_counter() - self.started,
                   best_objective=self.best, **details)
        self.history.append(row)
        if self.options.output_flag:
            if message is None:
                display = dict(details)
                highlights = []
                for name in ("p3_mip_gap", "p4_mip_gap"):
                    if name in display:
                        value = display.pop(name)
                        gap = f"{value:.4%}" if value is not None else "N/A"
                        highlights.append(f">>> {name[:2].upper()} SUBPROBLEM MIP GAP={gap} <<<")
                if "original_global_gap" in display:
                    value = display.pop("original_global_gap")
                    gap = f"{value:.4%}" if value is not None else "N/A"
                    highlights.append(f"ORIGINAL GLOBAL GAP={gap}")
                prefix = " ".join(highlights) + " " if highlights else ""
                print(f"[{stage}] {prefix}elapsed={row['elapsed']:.2f}s best={self.best} "
                      + " ".join(f"{k}={v}" for k, v in display.items()), flush=True)
            elif message:
                print(message, flush=True)

    def audit(self, plan, flows=True):
        violations = {}
        maximum_violation = 0.0
        tol = self.options.feasibility_tolerance
        def check(name, value):
            nonlocal maximum_violation
            maximum_violation = max(maximum_violation, value)
            if value > tol:
                violations[name] = float(value)
        for name in VARIABLES:
            for key, value in getattr(plan, name).items():
                if key not in self.keys[name] or not isfinite(value):
                    raise ValueError(f"Invalid {name}[{key}]={value}")
                check(f"nonnegative_{name}[{key}]", -value)
                if name in ("s", "r", "w", "y"):
                    check(f"integer_{name}[{key}]", abs(value - round(value)))
                if name in ("s", "r"):
                    check(f"binary_{name}[{key}]", value - 1)
        sc, rc, wc, yc = (_cumulative(getattr(plan, name), self.t) for name in ("s", "r", "w", "y"))
        pipes = defaultdict(float)
        for (u, v, d, t), value in plan.y.items():
            pipes[u, v] += value
        for ig in self.n.ig:
            check(f"source_once[{ig}]", sc[(*ig, self.t[-1])] - 1)
        for key, value in pipes.items():
            check(f"pipe_total[{key}]", value - self.pipe_totals[key])
        for j in self.n.sinks:
            check(f"sink_once[{j}]", rc[j, self.t[-1]] - 1)
            for t in self.t:
                check(f"well_total[{j},{t}]", wc[j, t] - self.max_wells[j] * rc[j, t])
        if flows:
            balance, capture = defaultdict(float), defaultdict(float)
            for (i, g, t), value in plan.a.items():
                check(f"source_capacity[{i},{g},{t}]", value - self.n.source_capacity[i, g] * sc[i, g, t])
                balance[i, t] -= value
                capture[t] += value
            for (j, t), value in plan.b.items():
                check(f"well_capacity[{j},{t}]", value - self.qw[j] * wc[j, t])
                balance[j, t] += value
            for (u, v, d, t), value in plan.x.items():
                check(f"pipe_capacity[{u},{v},{d},{t}]", value - self.n.pipe_capacity[d] * yc[u, v, d, t])
                balance[u, t] += value
                balance[v, t] -= value
            for key, value in balance.items():
                check(f"balance[{key}]", abs(value))
            for t in self.t:
                check(f"capture[{t}]", self.c.capture_target[t] - capture[t])
        return dict(feasible=not violations, violations=violations,
                    max_violation=maximum_violation,
                    objective=_objective(self.c, plan) if flows else None)

    def consider(self, plan, stage):
        audit = self.audit(plan)
        if not audit["feasible"]:
            raise RuntimeError(f"{stage} failed original-model audit: {audit['violations']}")
        value = audit["objective"]
        threshold = self.options.improvement_tolerance * max(1.0, abs(self.best or 0))
        if self.best is None or value < self.best - threshold:
            self.incumbent, self.best = plan.copy(), value
            self.record(stage, accepted=True, objective=value)
            return True
        return False


def _trim(ctx, plan):
    """只删除现有流量见证不需要的建设，保持剩余建设的实际日期。"""
    plan = plan.copy()
    used_units = {(i, g) for (i, g, _), v in plan.a.items() if v > 0}
    used_sinks = {j for (j, _), v in plan.b.items() if v > 0}
    plan.s = {k: v for k, v in plan.s.items() if k[:2] in used_units and v}
    plan.r = {k: v for k, v in plan.r.items() if k[0] in used_sinks and v}
    required_wells, required_pipes = defaultdict(int), defaultdict(int)
    for (j, t), value in plan.b.items():
        required_wells[j] = max(required_wells[j], ceil(max(0, value) / ctx.qw[j]))
    for (u, v, d, t), value in plan.x.items():
        required_pipes[u, v, d] = max(required_pipes[u, v, d], ceil(max(0, value) / ctx.n.pipe_capacity[d]))
    for name, requirements in (("w", required_wells), ("y", required_pipes)):
        retained = {}
        for key in sorted(getattr(plan, name), key=lambda k: k[-1]):
            prefix = key[0] if name == "w" else key[:-1]
            number = min(getattr(plan, name)[key], requirements[prefix])
            if number:
                retained[key] = number
                requirements[prefix] -= number
        setattr(plan, name, retained)
    return plan


def _flow_constraints(model, nodes, periods, targets, a, b, x):
    """x 可按管径或按弧聚合；所有节点（包括未开放设施）仍可转运。"""
    balance, capture = defaultdict(gp.LinExpr), defaultdict(gp.LinExpr)
    for (i, g, t), var in a.items():
        balance[i, t] -= var
        capture[t] += var
    for (j, t), var in b.items():
        balance[j, t] += var
    for key, var in x.items():
        balance[key[0], key[-1]] += var
        balance[key[1], key[-1]] -= var
    model.addConstrs((balance[node, t] == 0 for node in nodes for t in periods),
                     name="balance")
    return {t: model.addConstr(capture[t] >= targets[t], name=f"target[{t}]") for t in periods}


def _warm_start(variables, values):
    for key, var in variables.items():
        var.Start = values.get(key, 0)


def _verified(ctx, plan, stage):
    audit = ctx.audit(plan)
    if not audit["feasible"]:
        ctx.record(stage, rejected="original_model_audit", max_violation=audit["max_violation"])
    return audit["feasible"]


def _period_flow(ctx, plan, t):
    """固定实际建设方案的单期 LP；满足捕集目标后最小化运行成本。"""
    end = ctx.deadline(ctx.options.flow_time_limit)
    sc, wc, yc = (_cumulative(getattr(plan, n), [t]) for n in ("s", "w", "y"))
    with ctx.model(f"flow_{t}") as model:
        a = model.addVars([key for key, v in sc.items() if v > 0], lb=0, name="a")
        b = model.addVars([key for key, v in wc.items() if v > 0], lb=0, name="b")
        x = model.addVars([key for key, v in yc.items() if v > 0], lb=0, name="x")
        for (i, g, h), var in a.items():
            var.UB = ctx.n.source_capacity[i, g] * sc[i, g, h]
        for key, var in b.items():
            var.UB = ctx.qw[key[0]] * wc[key]
        for key, var in x.items():
            var.UB = ctx.n.pipe_capacity[key[2]] * yc[key]
        _flow_constraints(model, ctx.nodes, [t], ctx.c.capture_target, a, b, x)
        operation = gp.quicksum(getattr(ctx.c, COST_FIELDS[name])[k] * v
            for name, variables in (("a", a), ("b", b), ("x", x)) for k, v in variables.items())
        model.setObjective(operation, gp.GRB.MINIMIZE)
        ctx.optimize(model, end)
        status = _status_name(model.Status)
        if not model.SolCount:
            return dict(status=status, plan=None)
        result = plan.copy()
        for name, variables in (("a", a), ("b", b), ("x", x)):
            values = {key: value for key, value in getattr(result, name).items() if key[-1] != t}
            values.update(_values(variables))
            setattr(result, name, values)
        return dict(status=status, plan=result)


def _facility_constraints(ctx, model, pool, s, r, w, a, b):
    for i, g in sorted(pool.units):
        model.addConstr(gp.quicksum(s[i, g, t] for t in ctx.t) <= 1, name=f"source_once[{i},{g}]")
        active = gp.LinExpr()
        for t in ctx.t:
            active += s[i, g, t]
            model.addConstr(a[i, g, t] <= ctx.n.source_capacity[i, g] * active)
    for j in sorted(pool.sinks):
        model.addConstr(gp.quicksum(r[j, t] for t in ctx.t) <= 1)
        active, wells = gp.LinExpr(), gp.LinExpr()
        for t in ctx.t:
            active += r[j, t]
            wells += w[j, t]
            model.addConstr(wells <= min(ctx.max_wells[j], pool.wells.get(j, 0)) * active)
            model.addConstr(b[j, t] <= ctx.qw[j] * wells)


def _solver_metrics(model):
    """受限子问题的搜索诊断；这些界不能用作原问题全局下界。"""
    bound = float(model.ObjBound) if model.IsMIP else None
    gap = float(model.MIPGap) if model.IsMIP and model.SolCount else None
    return dict(restricted_bound=bound if bound is not None and isfinite(bound) else None,
                restricted_gap=gap if gap is not None and isfinite(gap) else None,
                solution_count=int(model.SolCount), node_count=float(model.NodeCount),
                model_variables=int(model.NumVars), binary_variables=int(model.NumBinVars),
                solver_runtime=float(model.Runtime))


def _round_mip_gaps(history):
    """取本轮各阶段最后一次尝试的 gap；失败时不沿用更早的数值。"""
    return {name: next((row.get(name) for row in reversed(history) if row["stage"] == stage), None)
            for stage, name in (("P3_schedule", "p3_mip_gap"), ("P4_pipelines", "p4_mip_gap"))}


def _attempt(ctx, stage, function, *args, **kwargs):
    """子问题未完成不等于不可行；到达全局预算时交给总入口保留 incumbent。"""
    try:
        return function(ctx, *args, **kwargs)
    except _BudgetExpired:
        ctx.record(stage, stopped="time_limit")
        ctx.check_time()
        return None
    except gp.GurobiError as exc:
        ctx.record(stage, solver_error=str(exc))
        return None


@dataclass(frozen=True)
class _Delay:
    kind: str
    key: tuple
    saving: float


def _delay_actions(ctx, plan, kinds=("s", "r", "w", "y")):
    actions = []
    for name in kinds:
        fixed = getattr(ctx.c, COST_FIELDS[name])
        for key, number in getattr(plan, name).items():
            t = key[-1]
            if not number or t == ctx.t[-1]:
                continue
            saving = fixed[key] - fixed[(*key[:-1], t + 1)]
            if name == "r":
                wells = plan.w.get((key[0], t), 0)
                saving += wells * (ctx.c.well_fixed[key] - ctx.c.well_fixed[key[0], t + 1])
            actions.append(_Delay(name, key, saving))
    return sorted(actions, key=lambda action: (-action.saving, action.kind, str(action.key)))


def _delayed_plan(ctx, plan, action):
    """延后一期间；sink 与开放当期的井联动，施工冲突先行过滤。"""
    name, key = action.kind, action.key
    if not getattr(plan, name).get(key, 0) or key[-1] == ctx.t[-1]:
        return None
    result = plan.copy()
    mapping = getattr(result, name)
    mapping[key] -= 1
    if not mapping[key]:
        del mapping[key]
    future = (*key[:-1], key[-1] + 1)
    mapping[future] = mapping.get(future, 0) + 1
    if name == "r":
        wells = result.w.pop(key, 0)
        if wells:
            result.w[future] = result.w.get(future, 0) + wells
    return result if ctx.audit(result, flows=False)["feasible"] else None


def _check_lower_bound(ctx, lower_bound):
    if not isfinite(lower_bound):
        raise RuntimeError(f"Nonfinite full LP lower bound: {lower_bound}")
    tolerance = ctx.options.feasibility_tolerance * max(1.0, abs(ctx.best or 0))
    if ctx.best is not None and lower_bound > ctx.best + tolerance:
        raise RuntimeError(f"Full LP lower bound {lower_bound} exceeds feasible objective {ctx.best}")


def _result(ctx, status):
    plan = ctx.incumbent or Plan()
    pa, px = defaultdict(float), defaultdict(float)
    for (i, g, t), value in plan.a.items():
        pa[i, t] += value
    for (u, v, d, t), value in plan.x.items():
        px[u, v, t] += value
    captures = {t: sum(pa[i, t] for i in ctx.n.sources) for t in ctx.t} if ctx.incumbent else {}
    audit = ctx.audit(plan) if ctx.incumbent else None
    if audit and not audit["feasible"]:
        raise RuntimeError(f"Final original-model audit failed: {audit['violations']}")
    if ctx.incumbent is not None:
        _check_lower_bound(ctx, ctx.lower_bound)
    has_lower_bound = ctx.incumbent is not None or ctx.lp_status == "OPTIMAL"
    return HeuristicResult(status=status, objective=ctx.best,
        lower_bound=ctx.lower_bound if has_lower_bound else None,
        gap=max(0.0, ctx.best - ctx.lower_bound) / max(1.0, abs(ctx.best)) if ctx.incumbent else None,
        runtime=perf_counter() - ctx.started, **{name: dict(getattr(plan, name)) for name in VARIABLES},
        pa=dict(pa), pb=dict(plan.b), px=dict(px), history=list(ctx.history),
        max_violation=audit["max_violation"] if audit else None, capture_by_period=captures,
        total_capture=sum(ctx.c.durations[t] * captures[t] for t in ctx.t) if ctx.incumbent else None,
        lower_bound_origin=ctx.bound_origin if has_lower_bound else "unavailable",
        lp_status=ctx.lp_status, lp_objective=ctx.lp_objective, lp_runtime=ctx.lp_runtime,
        configuration=ctx.configuration)


def _pool_signature(pool):
    return (frozenset(pool.units), frozenset(pool.sinks),
            frozenset(pool.wells.items()), frozenset(pool.pipes.items()))


def _facility_signature(plan):
    return tuple(frozenset(getattr(plan, name).items()) for name in ("s", "r", "w"))
