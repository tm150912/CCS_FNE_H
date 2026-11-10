"""读取并规范化 CCS 网络数据、算例参数和设施全期容量。

本模块读取 CSV 并检查字段与索引的一致性，完成单位换算后写入 "CCS" 类级命名空间；
同时通过各个 ``read_*`` 接口检查并规范化算例参数、建设额度和封存能力。
优化模型调用一次 ``read_data()`` 后，即可通过 ``CCS.属性名`` 使用网络和成本数据。
"""

from __future__ import annotations

import json
from math import floor, isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from CCS import CCS
from cost_amortize import normalize_storage_lifecycle


DATA_DIR = Path(__file__).resolve().parent
NODES_FILE = "nodes.csv"
ARCS_FILE = "arcs.csv"
SOURCE_COST_FILE = "hub_unit_location_cost.csv"
SINK_COST_FILE = "sink_subsite_cost.csv"
PIPE_CAPACITY_FILE = "pipeline_capacity.csv"
PIPE_COST_FILE = "pipeline_cost.csv"
NETWORK_INPUT_FILES = (
    NODES_FILE, ARCS_FILE, SOURCE_COST_FILE,
    SINK_COST_FILE, PIPE_CAPACITY_FILE, PIPE_COST_FILE,
)
# 原始投资成本以“货币单位”记录，而模型以“千货币单位”计价；捕集量和管输量也由 t 换算为 kt。
MONEY_SCALE = 1e3
STORAGE_FIXED_OM_COLUMNS = {
    "sink_fixed_om": "M_r_USD2020_per_year",
    "well_fixed_om": "M_w_USD2020_per_well_year",
}
STORAGE_SCHEDULE_COLUMNS = {
    "sink_extra_capital": "Site_Extra_CAPEX_USD2020_by_age",
    "sink_extra_opex": "Site_Extra_OPEX_USD2020_by_age",
    "well_extra_capital": "Well_Extra_CAPEX_USD2020_by_age",
    "well_extra_opex": "Well_Extra_OPEX_USD2020_by_age",
}
STORAGE_LIFECYCLE_COLUMN = "Storage_Lifecycle_USD2020"
STORAGE_SCHEMA_COLUMNS = (
    *STORAGE_FIXED_OM_COLUMNS.values(), *STORAGE_SCHEDULE_COLUMNS.values(),
    "Storage_Cost_Schema_Version", "Cost_Schedule_Years", STORAGE_LIFECYCLE_COLUMN,
)
STORAGE_SCHEDULE_YEARS = 30


def read_well_injection_capacities(capacities, sinks: Iterable) -> dict:
    """Normalize annual per-well capacities by sink, in the caller's units.

    CSV loading supplies a mapping; scalar inputs remain supported for existing
    programmatically constructed instances with uniform injection capacity.
    """

    result = {}
    for sink in sinks:
        try:
            raw = capacities[sink] if isinstance(capacities, Mapping) else capacities
        except KeyError as error:
            raise ValueError(f"Missing well_capacity for sink {sink!r}") from error
        if isinstance(raw, bool):
            raise ValueError(f"well_capacity[{sink!r}] must be finite and nonnegative")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"well_capacity[{sink!r}] must be finite and nonnegative") from error
        if not isfinite(value) or value < 0:
            raise ValueError(f"well_capacity[{sink!r}] must be finite and nonnegative")
        result[sink] = value
    return result


def read_pipeline_limit(limit) -> int:
    """Read the full-horizon pipe count per directed arc, without annual conversion."""

    if isinstance(limit, bool) or not isinstance(limit, Real):
        raise TypeError("NP must be a nonnegative integer")
    if not isfinite(limit) or limit < 0 or limit != floor(limit):
        raise ValueError("NP must be a finite nonnegative integer")
    return int(limit)


def read_period_limits(limits: Mapping, keys: Iterable[tuple], label: str) -> dict[tuple, int]:
    """Legacy quota reader retained for external callers using period limits.

    Source indices (source, unit, period) share one quota per (source, period).
    Source-period pairs can also be supplied directly.
    """

    if not isinstance(limits, Mapping):
        raise TypeError(f"{label} period limits must be a mapping")
    if label == "source":
        keys = dict.fromkeys(
            (key[0], key[-1]) if len(key) == 3 else key for key in keys)
    result = {}
    for key in keys:
        try:
            value = float(limits[key])
        except KeyError as error:
            raise KeyError(f"Missing {label} installation limit for {key}") from error
        if not isfinite(value) or value < 0.0:
            raise ValueError(
                f"{label} installation limit {key} must be finite and nonnegative")
        result[key] = floor(value)
    return result


def read_installation_inputs(costs, annual_limit, label: str) -> tuple[float, dict]:
    """Read annual rates for legacy callers using period limits.

    Source, well and pipeline rates may be fractional; callers floor the rate
    multiplied by each period's length when producing integer quotas.
    """
    if isinstance(annual_limit, bool) or not isinstance(annual_limit, Real):
        raise TypeError(f"{label} annual installation limit must be a numeric scalar")
    rate = float(annual_limit)
    if not isfinite(rate) or rate < 0.0:
        raise ValueError(f"{label} installation limit must be finite and nonnegative")
    durations = {}
    for period in costs.periods:
        try:
            duration = float(costs.durations[period])
        except KeyError as error:
            raise KeyError(f"Missing period duration for period {period}") from error
        if not isfinite(duration) or duration <= 0.0:
            raise ValueError(
                f"Period duration for period {period} must be finite and positive")
        if not isfinite(rate * duration):
            raise ValueError(
                f"{label} installation limit for period {period} must be finite and nonnegative")
        durations[period] = duration
    return rate, durations


def read_well_capacities(network, costs=None) -> dict:
    """Read each sink's full-horizon maximum well count, without period quotas."""

    if costs is not None:
        expected_sink_periods = {
            (sink, period)
            for sink in network.sinks
            for period in costs.periods
        }
        actual_sink_periods = list(costs.sink_periods)
        if (len(actual_sink_periods) != len(expected_sink_periods)
                or set(actual_sink_periods) != expected_sink_periods):
            missing = sorted(
                expected_sink_periods.difference(actual_sink_periods), key=str)
            extra = sorted(
                set(actual_sink_periods).difference(expected_sink_periods), key=str)
            raise ValueError(
                "costs.sink_periods must contain every (sink, period) "
                "exactly once: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    capacities = {}
    for sink in network.sinks:
        maximum = float(network.max_wells[sink])
        if not isfinite(maximum) or maximum < 0.0:
            raise ValueError("max_wells must be finite and nonnegative")
        capacities[sink] = floor(maximum)
    return capacities


def read_period_schedule(project_years: int, period_num: int) -> tuple[list[int], dict[int, int]]:
    """Read an equal-length schedule that covers the complete project."""

    if (isinstance(project_years, bool) or not isinstance(project_years, Integral)
            or project_years <= 0):
        raise ValueError("project_years must be a positive integer")
    if (isinstance(period_num, bool) or not isinstance(period_num, Integral)
            or period_num <= 0):
        raise ValueError("period_num must be a positive integer")
    if project_years % period_num:
        raise ValueError(f"TY={project_years} is not divisible by period_num={period_num}")
    periods = list(range(1, period_num + 1))
    return periods, {period: int(project_years // period_num) for period in periods}


def read_capture_curve(curve: str, coefficient=None) -> tuple[str, float | None]:
    """Read a supported capture curve and its finite, admissible coefficient."""

    if curve not in ("linear", "exponential", "logarithmic"):
        raise ValueError(f"Unknown capture-target curve: {curve}")
    if coefficient is None:
        if curve == "linear":
            return curve, None
        if curve == "exponential":
            raise ValueError("The exponential curve requires a non-zero coefficient")
        raise ValueError(
            "The logarithmic curve requires a coefficient greater than -1 and not 0")
    if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
        raise TypeError("Capture-curve coefficient must be a numeric scalar")
    value = float(coefficient)
    if not isfinite(value):
        raise ValueError("Capture-curve coefficient must be finite")
    if curve == "exponential" and value == 0:
        raise ValueError("The exponential curve requires a non-zero coefficient")
    if curve == "logarithmic" and (value <= -1 or value == 0):
        raise ValueError(
            "The logarithmic curve requires a coefficient greater than -1 and not 0")
    return curve, value


def read_capture_periods(period_count: int, reference_coefficients: Iterable,
                         durations: Mapping | None = None) -> dict[int, float]:
    """Read consecutive capture periods and their finite, positive lengths."""

    if (isinstance(period_count, bool) or not isinstance(period_count, Integral)
            or period_count <= 0):
        raise ValueError("period_count must be a positive integer")
    if not list(reference_coefficients):
        raise ValueError("E_LIST must contain at least one exponential curve")
    periods = range(1, period_count + 1)
    if durations is None:
        return {period: 1.0 for period in periods}
    if set(durations) != set(periods):
        raise ValueError("durations must define every consecutive period exactly once")
    period_durations = {period: float(durations[period]) for period in periods}
    if any(not isfinite(duration) or duration <= 0.0
           for duration in period_durations.values()):
        raise ValueError("Every period duration must be positive and finite")
    return period_durations


def result_instance_key(row: Mapping) -> tuple | None:
    """按结果表六个参数列识别算例；无效或不完整参数不匹配。"""
    try:
        numbers = tuple(float(row[name]) for name in
                        ("period_num", "D", "Np", "max_capture_amount"))
        curve = str(row["curve"]).strip().lower()
        coefficient = row["coef"]
        if (coefficient is None or pd.isna(coefficient)
                or str(coefficient).strip().lower() in ("", "none", "nan")):
            coefficient = None
        else:
            coefficient = float(coefficient)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not all(isfinite(value) for value in numbers)
            or curve not in ("linear", "exponential", "logarithmic")
            or (coefficient is not None and not isfinite(coefficient))
            or (curve != "linear" and coefficient is None)):
        return None
    return (*numbers, curve, coefficient)


def grid_instance_key(case) -> tuple:
    """将 experiment_setup 参数网格中的一组参数转为结果表标识。"""
    period_num, maximum, np, (curve, coefficient), diameters = case
    key = result_instance_key(dict(period_num=period_num, D=len(diameters),
                                   Np=np, max_capture_amount=maximum,
                                   curve=curve, coef=coefficient))
    if key is None:
        raise ValueError(f"Invalid experiment-grid case: {case!r}")
    return key


def normalize_result_status(status) -> str:
    """完成状态匹配忽略大小写、空白、下划线和连字符。"""
    return "".join(str(status).lower().split()).replace("_", "").replace("-", "")


def read_result_rows(
        path: str | Path, cases: Iterable, *,
        completed_statuses: Iterable[str] = ("optimal", "time_limit", "root_optimal"),
) -> tuple[list[dict], set[tuple]]:
    """保留已有结果，按入口指定的结束状态标记当前网格中的完成算例。"""
    current_keys = {grid_instance_key(case) for case in cases}
    accepted_statuses = {normalize_result_status(status) for status in completed_statuses}
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return [], set()
    results = pd.read_csv(path)
    required = {"period_num", "D", "Np", "max_capture_amount", "curve", "coef", "status"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Result file {path} lacks required columns: {sorted(missing)}")
    rows = results.to_dict("records")
    completed = set()
    for row in rows:
        key = result_instance_key(row)
        status = normalize_result_status(row["status"])
        if key in current_keys and status in accepted_statuses:
            completed.add(key)
    return rows, completed


def _read_csv(
    filename: str, *, data_dir: Path, encoding: str | None = None) -> pd.DataFrame:
    """从指定目录读取 CSV 文件，并清理列名两侧的空白。"""

    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    frame = pd.read_csv(path, encoding=encoding)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], filename: str) -> None:
    """在进入建模阶段前检查必需字段，避免稍后出现含义不清楚的 KeyError。"""

    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, columns: list[str], filename: str) -> None:
    """检查组成模型字典键的字段是否唯一，防止重复行被 ``to_dict`` 静默覆盖。"""

    if frame.duplicated(columns).any():
        raise ValueError(f"{filename} contains duplicate keys in columns {columns}")


def _read_storage_cost_extensions(frame: pd.DataFrame, filename: str) -> None:
    """Read operating costs and the complete optional lifecycle record."""

    present = set(STORAGE_SCHEMA_COLUMNS).intersection(frame.columns)
    frame.attrs["storage_cost_schema"] = "basic_costs"
    if not present:
        return
    required = {*STORAGE_FIXED_OM_COLUMNS.values(), *STORAGE_SCHEDULE_COLUMNS.values()}
    lifecycle_present = STORAGE_LIFECYCLE_COLUMN in frame.columns
    if lifecycle_present:
        required.update(("Storage_Cost_Schema_Version", "Cost_Schedule_Years"))
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{filename} contains a partial storage cost schema; "
            f"missing columns: {sorted(missing)}")

    version = 3 if lifecycle_present else 2
    for column, expected in (("Storage_Cost_Schema_Version", version),
                             ("Cost_Schedule_Years", STORAGE_SCHEDULE_YEARS)):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if not (values == expected).all():
            raise ValueError(f"{filename} requires {column}={expected} on every row")

    for column in STORAGE_FIXED_OM_COLUMNS.values():
        values = pd.to_numeric(frame[column], errors="coerce")
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"{filename} {column} must be finite and nonnegative")
        frame[column] = values

    for column in STORAGE_SCHEDULE_COLUMNS.values():
        parsed = []
        for sink, raw in zip(frame["j"], frame[column]):
            try:
                values = json.loads(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{filename} {column}[{sink}] must be a JSON array") from error
            if (not isinstance(values, list)
                    or len(values) != STORAGE_SCHEDULE_YEARS
                    or not all(isinstance(value, (int, float))
                               and not isinstance(value, bool)
                               and isfinite(value) and value >= 0
                               for value in values)):
                raise ValueError(
                    f"{filename} {column}[{sink}] requires exactly "
                    f"{STORAGE_SCHEDULE_YEARS} finite nonnegative numbers")
            parsed.append(tuple(float(value) for value in values))
        frame[column] = parsed
    if lifecycle_present:
        records = []
        for sink, raw in zip(frame["j"], frame[STORAGE_LIFECYCLE_COLUMN]):
            label = f"{filename} {STORAGE_LIFECYCLE_COLUMN}[{sink}]"
            try:
                value = json.loads(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} must be a JSON object") from error
            records.append(normalize_storage_lifecycle(value, label))
        frame[STORAGE_LIFECYCLE_COLUMN] = records
    frame.attrs["storage_cost_schema"] = "lifecycle_costs" if lifecycle_present else "operating_costs"


def _read_sink_subsite_cost(
    *, data_dir: Path, sink_nodes: Iterable[str]) -> pd.DataFrame:
    """Read and validate one cost/capacity row for every sink-subsite node.

    The current network represents each sink subsite as its own ``P...-S...``
    node. ``Well_Annual_Capacity_t`` supplies annual capacity in tonnes per well.
    Cost and well-count header aliases are normalized before validation.
    """

    filename = SINK_COST_FILE
    sink_cost = _read_csv(filename, data_dir=data_dir)

    well_limit_aliases = [
        column
        for column in ("Max_Well_Num", "Max_Well Num")
        if column in sink_cost.columns
    ]
    if not well_limit_aliases:
        raise ValueError(
            f"{filename} is missing required column: Max_Well_Num "
            "(or Max_Well Num)"
        )
    if len(well_limit_aliases) > 1:
        raise ValueError(
            f"{filename} contains both Max_Well_Num and Max_Well Num; "
            "keep only one"
        )
    if well_limit_aliases[0] != "Max_Well_Num":
        sink_cost = sink_cost.rename(
            columns={well_limit_aliases[0]: "Max_Well_Num"}
        )

    sink_id_aliases = [
        column for column in ("j", "ju", "Subsite_ID") if column in sink_cost.columns
    ]
    if not sink_id_aliases:
        raise ValueError(f"{filename} is missing required column: j (or ju/Subsite_ID)")
    if len(sink_id_aliases) > 1:
        raise ValueError(f"{filename} contains multiple sink ID aliases; keep only one")

    cost_aliases = {
        "Vj": ("Vj", "Vju", "V_r_USD2020_per_tCO2"),
        "Fj": ("Fj", "Fju", "F_r_USD2020"),
        "Fwj": ("Fwj", "Fwju", "F_w_USD2020_per_well"),
    }
    rename_columns = {sink_id_aliases[0]: "j"}
    for current, accepted in cost_aliases.items():
        aliases = [
            column for column in accepted
            if column in sink_cost.columns
        ]
        if not aliases:
            raise ValueError(
                f"{filename} is missing required column: {current} "
                f"(accepted aliases: {accepted})")
        if len(aliases) > 1:
            raise ValueError(
                f"{filename} contains multiple aliases for {current}: {aliases}; "
                "keep only one")
        rename_columns[aliases[0]] = current
    sink_cost = sink_cost.rename(columns=rename_columns)

    cost_columns = list(cost_aliases)
    required_columns = ["j", *cost_columns, "Max_Well_Num", "Well_Annual_Capacity_t"]
    _require_columns(sink_cost, required_columns, filename)

    sink_cost = sink_cost.copy()
    sink_cost["j"] = sink_cost["j"].astype("string").str.strip()
    if sink_cost["j"].isna().any() or (sink_cost["j"] == "").any():
        raise ValueError(f"{filename} contains missing sink-subsite IDs")
    _require_unique(sink_cost, ["j"], filename)

    numeric_columns = [*cost_columns, "Max_Well_Num", "Well_Annual_Capacity_t"]
    sink_cost[numeric_columns] = sink_cost[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric_data = sink_cost[numeric_columns]
    if (not all(isfinite(value) for value in numeric_data.to_numpy().flat)
            or (numeric_data < 0).any().any()):
        raise ValueError(
            f"{filename} contains missing, non-finite, non-numeric, or negative "
            "costs/well limits/injection capacities"
        )
    if (sink_cost["Max_Well_Num"] % 1 != 0).any():
        raise ValueError(f"{filename} contains non-integer Max_Well_Num values")
    sink_cost["Max_Well_Num"] = sink_cost["Max_Well_Num"].astype(int)
    _read_storage_cost_extensions(sink_cost, filename)

    expected_sinks = {str(sink).strip() for sink in sink_nodes}
    cost_sinks = set(sink_cost["j"])
    missing_sinks = sorted(expected_sinks.difference(cost_sinks), key=str)
    extra_sinks = sorted(cost_sinks.difference(expected_sinks), key=str)
    if missing_sinks or extra_sinks:
        raise ValueError(
            f"{filename} sink-subsite set does not match nodes.csv: "
            f"missing={missing_sinks[:10]}, extra={extra_sinks[:10]}"
        )

    return sink_cost


def read_data(data_dir: str | Path | None = None) -> None:
    """读取全部模型输入并填充 :class:`CCS` 命名空间。

    Parameters
    -----
    "I/S/R" 分别表示全体节点、排放源节点和封存节点；"Np/Nn" 是每个节点的出弧/入弧邻接表。
    ``ig`` 和 ``ijd`` 分别是源-捕集单元、弧-管径索引。封存参数直接以物理封存地
    ``j`` 为键。所有投资成本在此处统一
    除以 ``MONEY_SCALE``，后续成本摊销函数不再重复处理单位。
    """

    base_dir = Path(data_dir).resolve() if data_dir is not None else DATA_DIR

    # ------------------------------------------------------------------
    # 1. 网络节点和有向弧
    # nodes.csv 记录节点编号、经度和纬度；H/P/I 前缀分别表示源、汇和交汇点。
    # arcs.csv 记录起点、终点和距离，并据此构造正向邻接表 Np 与反向邻接表 Nn。
    # ------------------------------------------------------------------
    node_filename = NODES_FILE
    arc_filename = ARCS_FILE
    nodes = _read_csv(node_filename, data_dir=base_dir)
    arcs = _read_csv(arc_filename, data_dir=base_dir)
    node_columns = ["I", "Longitude", "Latitude"]
    arc_columns = ["i", "j", "distance"]
    _require_columns(nodes, node_columns, node_filename)
    _require_columns(arcs, arc_columns, arc_filename)

    node_id, longitude, latitude = node_columns
    arc_from, arc_to, distance = arc_columns
    _require_unique(nodes, [node_id], node_filename)
    if nodes[[node_id, longitude, latitude]].isna().any().any():
        raise ValueError(f"{node_filename} contains missing node IDs or coordinates")

    CCS.I = {row[node_id]: [row[longitude], row[latitude]] for _, row in nodes.iterrows()}
    CCS.S = {node: coord for node, coord in CCS.I.items() if str(node).startswith("H")}
    CCS.R = {node: coord for node, coord in CCS.I.items() if str(node).startswith("P")}
    CCS.X = {node: coord for node, coord in CCS.I.items() if str(node).startswith("I")}
    classified_nodes = set(CCS.S) | set(CCS.R) | set(CCS.X)
    unclassified_nodes = sorted(set(CCS.I).difference(classified_nodes), key=str)
    if unclassified_nodes:
        raise ValueError(
            f"{node_filename} contains unsupported node IDs; expected H/P/I prefixes: "
            f"{unclassified_nodes[:10]}"
        )
    if not CCS.S or not CCS.R or not CCS.X:
        raise ValueError(
            f"{node_filename} must contain source hubs (H), sinks (P), and intersections (I)"
        )

    CCS.Np = {node: [] for node in CCS.I}
    CCS.Nn = {node: [] for node in CCS.I}
    _require_unique(arcs, [arc_from, arc_to], arc_filename)
    if arcs[[arc_from, arc_to, distance]].isna().any().any():
        raise ValueError(f"{arc_filename} contains missing arc endpoints or distances")
    if (arcs[distance] <= 0).any():
        raise ValueError(f"{arc_filename} contains non-positive distances")
    arc_pairs = list(arcs[[arc_from, arc_to]].itertuples(index=False, name=None))
    for start, end in arc_pairs:
        if start not in CCS.I or end not in CCS.I:
            raise ValueError(f"{arc_filename} references an unknown node: ({start}, {end})")
        if start == end:
            raise ValueError(f"{arc_filename} contains a self-loop: ({start}, {end})")
        CCS.Np[start].append(end)
        CCS.Nn[end].append(start)
    CCS.ij = arc_pairs

    arc_endpoints = {node for arc in CCS.ij for node in arc}
    missing_sinks = sorted(set(CCS.R).difference(arc_endpoints), key=str)
    if missing_sinks:
        raise ValueError(
            "Every physical sink must appear as an arc endpoint: "
            f"missing={missing_sinks[:5]}"
        )

    # ------------------------------------------------------------------
    # 2. 排放源及捕集单元参数
    # FC 为一次性投资成本，VC 为单位捕集可变成本，cap 为年捕集能力。
    # 接受聚类输出的原始成本列名；字典键统一使用 (source, unit)。
    # ------------------------------------------------------------------
    source_filename = SOURCE_COST_FILE
    source_cost = _read_csv(source_filename, data_dir=base_dir)
    source_aliases = {
        "hub": "source",
        "Operation_cost($/tCO2)": "VC",
        "Total_Investment_cost($)": "FC",
    }
    source_cost = source_cost.rename(columns={
        original: normalized for original, normalized in source_aliases.items()
        if original in source_cost.columns and normalized not in source_cost.columns
    })
    source_columns = ["source", "unit", "VC", "cap", "FC"]
    _require_columns(source_cost, source_columns, source_filename)
    _require_unique(source_cost, ["source", "unit"], source_filename)
    if (source_cost[["VC", "cap", "FC"]] < 0).any().any():
        raise ValueError(f"{source_filename} contains negative costs or capacities")

    CCS.ig = list(source_cost[["source", "unit"]].itertuples(index=False, name=None))
    CCS.G = {source: [] for source in CCS.S}
    for source, unit in CCS.ig:
        if source not in CCS.G:
            raise ValueError(f"{source_filename} references unknown source hub {source}")
        CCS.G[source].append(unit)
    missing_source_hubs = sorted(
        (source for source, units in CCS.G.items() if not units), key=str
    )
    if missing_source_hubs:
        raise ValueError(
            f"{source_filename} has no capture-unit rows for source hubs: "
            f"{missing_source_hubs[:10]}"
        )

    # 投资成本由原始货币单位换算为千货币单位，捕集能力由 t/年换算为 kt/年；VC 保持货币单位/t。
    source_rows = source_cost.set_index(["source", "unit"])
    CCS.Tig = {key: value / MONEY_SCALE for key, value in source_rows["FC"].items()}
    CCS.Vig = source_rows["VC"].to_dict()
    CCS.Qig = {key: value / MONEY_SCALE for key, value in source_rows["cap"].items()}

    # ------------------------------------------------------------------
    # 3. 物理封存地成本、最大井数与各子场的单井年能力
    # arcs.csv 中的 P...-S... 编号是直接出发或到达的物理封存地 j。
    # sink_subsite_cost.csv 提供成本、Max_Well_Num 和 Well_Annual_Capacity_t；不再读取或分摊
    # 旧的 sink_cost.csv 与 sink_subsite_wellnum.csv。
    # ------------------------------------------------------------------
    sink_cost = _read_sink_subsite_cost(data_dir=base_dir, sink_nodes=CCS.R)
    sink_rows = sink_cost.set_index("j")
    sink_ids = list(CCS.R)
    CCS.Pj = {j: int(sink_rows.at[j, "Max_Well_Num"]) for j in sink_ids}
    CCS.Tj = {j: sink_rows.at[j, "Fj"] / MONEY_SCALE for j in sink_ids}
    CCS.Vj = {j: sink_rows.at[j, "Vj"] for j in sink_ids}
    CCS.Twj = {j: sink_rows.at[j, "Fwj"] / MONEY_SCALE for j in sink_ids}
    CCS.storage_cost_schema = sink_cost.attrs["storage_cost_schema"]
    for attribute, column in STORAGE_FIXED_OM_COLUMNS.items():
        values = ({j: sink_rows.at[j, column] / MONEY_SCALE for j in sink_ids}
                  if CCS.storage_cost_schema in {"operating_costs", "lifecycle_costs"} else {})
        setattr(CCS, attribute, values)
    for attribute, column in STORAGE_SCHEDULE_COLUMNS.items():
        values = ({j: tuple(value / MONEY_SCALE for value in sink_rows.at[j, column])
                   for j in sink_ids}
                  if CCS.storage_cost_schema in {"operating_costs", "lifecycle_costs"} else {})
        setattr(CCS, attribute, values)
    CCS.storage_lifecycle = ({
        j: normalize_storage_lifecycle(
            sink_rows.at[j, STORAGE_LIFECYCLE_COLUMN],
            f"{SINK_COST_FILE} {STORAGE_LIFECYCLE_COLUMN}[{j}]", MONEY_SCALE)
        for j in sink_ids} if CCS.storage_cost_schema == "lifecycle_costs" else {})
    # 输入已是 t/年/井，仅换算为 kt/年/井，不再乘 365 或项目年限。
    CCS.Qw = read_well_injection_capacities(
        {j: sink_rows.at[j, "Well_Annual_Capacity_t"] / MONEY_SCALE for j in sink_ids},
        sink_ids,
    )

    # ------------------------------------------------------------------
    # 4. 候选管径、输送能力及弧-管径成本
    # 模型使用 pipeline_capacity.csv。Qd[d] 由 t/年换算为 kt/年；pipeline_cost
    # 按列名读取起点、终点、管径、一次性投资成本与可变成本。
    # ij 只保留网络弧，ijd 保留每条弧的全部候选管径，二者分别服务于流量变量和建造变量索引。
    # ------------------------------------------------------------------
    pipe_capacity = _read_csv(PIPE_CAPACITY_FILE, data_dir=base_dir)
    if len(pipe_capacity.columns) < 2:
        raise ValueError("pipeline_capacity.csv has an invalid layout")
    diameter, capacity = pipe_capacity.columns[:2]
    _require_unique(pipe_capacity, [diameter], PIPE_CAPACITY_FILE)
    CCS.D = pipe_capacity[diameter].tolist()
    CCS.Qd = {row[diameter]: row[capacity] / MONEY_SCALE for _, row in pipe_capacity.iterrows()}

    pipeline_cost_filename = PIPE_COST_FILE
    pipeline_cost = _read_csv(pipeline_cost_filename, data_dir=base_dir)
    pipeline_columns = [
        "node1_ID",
        "node2_ID",
        "CO2 flow rate(kg/s)",
        "Operation_cost($/tCO2)",
        "Investment_cost($)",
    ]
    _require_columns(pipeline_cost, pipeline_columns, pipeline_cost_filename)
    start, end, pipe, variable, fixed = pipeline_columns
    _require_unique(pipeline_cost, [start, end, pipe], pipeline_cost_filename)
    pipeline_numeric = pipeline_cost[[variable, fixed]]
    if (pipeline_cost[[start, end, pipe]].isna().any().any()
            or pipeline_numeric.isna().any().any()
            or (pipeline_numeric < 0).any().any()):
        raise ValueError(
            f"{pipeline_cost_filename} contains missing keys/costs or negative costs"
        )
    unknown_diameters = sorted(set(pipeline_cost[pipe]).difference(CCS.D))
    if unknown_diameters:
        raise ValueError(
            f"{pipeline_cost_filename} contains unknown diameters: {unknown_diameters}"
        )

    cost_arc_pairs = set(
        pipeline_cost[[start, end]].drop_duplicates().itertuples(index=False, name=None)
    )
    network_arc_pairs = set(CCS.ij)
    missing_cost_arcs = sorted(network_arc_pairs.difference(cost_arc_pairs), key=str)
    extra_cost_arcs = sorted(cost_arc_pairs.difference(network_arc_pairs), key=str)
    if missing_cost_arcs or extra_cost_arcs:
        raise ValueError(
            f"{pipeline_cost_filename} arc set does not match {arc_filename}: "
            f"missing={missing_cost_arcs[:5]}, extra={extra_cost_arcs[:5]}"
        )
    pipe_count_by_arc = pipeline_cost.groupby([start, end], sort=False)[pipe].nunique()
    if not (pipe_count_by_arc == len(CCS.D)).all():
        bad_arcs = pipe_count_by_arc[pipe_count_by_arc != len(CCS.D)].head(5).to_dict()
        raise ValueError(
            f"{pipeline_cost_filename} must provide all {len(CCS.D)} pipe types per arc; "
            f"invalid counts: {bad_arcs}"
        )

    CCS.ijd = list(pipeline_cost[[start, end, pipe]].itertuples(index=False, name=None))
    pipeline_rows = pipeline_cost.set_index([start, end, pipe])
    CCS.Vijd = pipeline_rows[variable].to_dict()
    CCS.Tijd = {key: value / MONEY_SCALE for key, value in pipeline_rows[fixed].items()}
