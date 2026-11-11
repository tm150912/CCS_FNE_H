"""Shared experiment grid and data preparation for CCS_FNE solvers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import expm1, floor, log1p
from pathlib import Path
from typing import Mapping, Optional, Union

from CCS import CCS
from capture_target_sequence import generate_capture_targets
from cost_amortize import (DISCOUNT_RATE, INTEREST_RATE,
                           TECHNICAL_LIFETIME, cost_amortize)
from read import (read_capture_curve, read_capture_periods, read_data,
                  read_installation_inputs, read_period_schedule,
                  read_well_capacities)


# ==================== 输入输出文件 ====================
PROJECT_DIR = Path(__file__).resolve().parent

# ==================== 模型参数 ====================
# Shared batch grid. CCS_FNE and CCS_FNE_LP use only full-horizon limits.
# Algorithm-specific solver limits and output settings remain in the corresponding
# solver module.

# Data_Suplementary.tex: 2 catalogs x 4 period counts x 2 limits x 2 scales x 7 profiles.
TY = 30
NP_LIST = [1, 3]
MAX_CAPTURE_AMOUNT_LIST = [0.5e6, 2e6]  # kt/year: 0.5 and 2 Gt/year.
PERIOD_NUM_LIST = [5, 10, 15, 30]
E_LIST = [1, 2.5, 4]
L_LIST = [2, 15, 500]

D_LIST = [
    ["p4", "p6", "p20", "p48"],
    ["p4", "p6", "p10", "p12", "p16", "p20",
     "p24a", "p24b", "p30a", "p30b", "p36", "p48"]]

CURVE_SPECS = [
    ("linear", None),
    *(("exponential", coefficient) for coefficient in E_LIST),
    *(("logarithmic", coefficient) for coefficient in L_LIST)]

# ==================== 算法参数 ====================
# 本模块仅提供共享参数与数据准备；算法参数在各求解器模块中设置。


def _capture_fraction(
        period: int,
        period_count: int,
        curve: str,
        coefficient: Optional[float] = None) -> float:
    """Return the normalized target from curve inputs prepared by read.py."""

    curve, coefficient = read_capture_curve(curve, coefficient)

    if curve == "exponential":
        return expm1(coefficient * period / period_count) / expm1(coefficient)

    if curve == "logarithmic":
        return log1p(coefficient * period / period_count) / log1p(coefficient)

    return period / period_count


def capture_max_amount(
        period_count: int,
        curve: str,
        coefficient: Optional[float] = None,
        durations: Optional[Mapping[int, float]] = None,
        *, maximum_capture_amount: float) -> float:
    """Use the current grid maximum as every curve's terminal annual target."""

    read_capture_periods(period_count, E_LIST, durations)
    read_capture_curve(curve, coefficient)
    return maximum_capture_amount


@dataclass(frozen=True)
class NetworkData:
    """Network sets and raw parameters shared by every experiment case."""

    ij: list
    ijd: list
    ig: list
    sources: dict
    sinks: dict
    intersections: dict
    successors: dict
    predecessors: dict
    source_units: dict
    source_capacity: dict
    pipe_capacity: dict
    max_wells: dict
    well_capacity: dict
    source_fixed_cost: dict
    pipe_fixed_cost: dict
    sink_fixed_cost: dict
    well_fixed_cost: dict
    source_variable_cost: dict
    pipe_variable_cost: dict
    sink_variable_cost: dict
    sink_fixed_om: dict = field(default_factory=dict)
    well_fixed_om: dict = field(default_factory=dict)
    sink_extra_capital: dict = field(default_factory=dict)
    sink_extra_opex: dict = field(default_factory=dict)
    well_extra_capital: dict = field(default_factory=dict)
    well_extra_opex: dict = field(default_factory=dict)
    storage_lifecycle: dict = field(default_factory=dict)

    @classmethod
    def from_ccs(cls) -> "NetworkData":
        return cls(
            ij=CCS.ij,
            ijd=CCS.ijd,
            ig=CCS.ig,
            sources=CCS.S,
            sinks=CCS.R,
            intersections=CCS.X,
            successors=CCS.Np,
            predecessors=CCS.Nn,
            source_units=CCS.G,
            source_capacity=CCS.Qig,
            pipe_capacity=CCS.Qd,
            max_wells=CCS.Pj,
            well_capacity=CCS.Qw,
            source_fixed_cost=CCS.Tig,
            pipe_fixed_cost=CCS.Tijd,
            sink_fixed_cost=CCS.Tj,
            well_fixed_cost=CCS.Twj,
            source_variable_cost=CCS.Vig,
            pipe_variable_cost=CCS.Vijd,
            sink_variable_cost=CCS.Vj,
            sink_fixed_om=getattr(CCS, "sink_fixed_om", None) or {},
            well_fixed_om=getattr(CCS, "well_fixed_om", None) or {},
            sink_extra_capital=getattr(CCS, "sink_extra_capital", None) or {},
            sink_extra_opex=getattr(CCS, "sink_extra_opex", None) or {},
            well_extra_capital=getattr(CCS, "well_extra_capital", None) or {},
            well_extra_opex=getattr(CCS, "well_extra_opex", None) or {},
            storage_lifecycle=getattr(CCS, "storage_lifecycle", None) or {})


@dataclass(frozen=True)
class CostData:
    """Period indices, amortized costs and capture targets for one case."""

    periods: list[int]
    durations: dict[int, int]
    source_periods: list
    sink_periods: list
    arc_periods: list
    source_fixed: dict
    sink_fixed: dict
    well_fixed: dict
    pipe_fixed: dict
    source_variable: dict
    sink_variable: dict
    pipe_variable: dict
    capture_target: dict
    model_parameters: dict = field(default_factory=dict)


def load_network_data(data_dir: str | Path = PROJECT_DIR) -> NetworkData:
    """Read the shared CSV inputs once and return an immutable data view."""

    read_data(data_dir)
    return NetworkData.from_ccs()


def period_schedule(period_num: int) -> tuple[list[int], dict[int, int]]:
    """Read the equal-length period schedule for one experiment case."""

    return read_period_schedule(TY, period_num)


def compute_cost_data(
    network: NetworkData,
    periods: list[int],
    durations: dict[int, int],
    curve: str = "linear",
    coefficient: Optional[float] = None,
    *,
    maximum_capture_amount: Optional[float] = None,
    apply_shared_capture_target: bool = True,
    normalize_capture_target: bool = False,
    source_lifetime: float = TECHNICAL_LIFETIME,
    sink_lifetime: float = TECHNICAL_LIFETIME,
    well_lifetime: float = TECHNICAL_LIFETIME,
    pipe_lifetime: float = TECHNICAL_LIFETIME,
    interest_rate: float = INTEREST_RATE,
    discount_rate: float = DISCOUNT_RATE) -> CostData:
    """Amortize costs and optionally apply the shared experiment curve."""

    if apply_shared_capture_target and maximum_capture_amount is None:
        raise ValueError("maximum_capture_amount is required for shared capture targets")

    values = cost_amortize(
        TY,
        durations,
        periods,
        network.ig,
        network.sinks,
        network.ij,
        network.ijd,
        network.source_fixed_cost,
        network.pipe_fixed_cost,
        network.sink_fixed_cost,
        network.well_fixed_cost,
        network.source_variable_cost,
        network.sink_variable_cost,
        network.pipe_variable_cost,
        maximum_capture_amount=(0.0 if maximum_capture_amount is None
                                else maximum_capture_amount),
        source_lifetime=source_lifetime,
        sink_lifetime=sink_lifetime,
        well_lifetime=well_lifetime,
        pipe_lifetime=pipe_lifetime,
        interest_rate=interest_rate,
        discount_rate=discount_rate,
        sink_fixed_om=network.sink_fixed_om or None,
        well_fixed_om=network.well_fixed_om or None,
        sink_extra_capital=network.sink_extra_capital or None,
        sink_extra_opex=network.sink_extra_opex or None,
        well_extra_capital=network.well_extra_capital or None,
        well_extra_opex=network.well_extra_opex or None,
        storage_lifecycle=network.storage_lifecycle or None)

    (
        source_fixed,
        sink_fixed,
        well_fixed,
        pipe_fixed,
        source_variable,
        sink_variable,
        pipe_variable,
        source_periods,
        sink_periods,
        arc_periods,
        capture_target
    ) = values

    costs = CostData(
        periods,
        durations,
        source_periods,
        sink_periods,
        arc_periods,
        source_fixed,
        sink_fixed,
        well_fixed,
        pipe_fixed,
        source_variable,
        sink_variable,
        pipe_variable,
        capture_target,
        model_parameters=dict(
            planning_years=TY, interest_rate=interest_rate, discount_rate=discount_rate,
            source_lifetime=source_lifetime, sink_lifetime=sink_lifetime,
            well_lifetime=well_lifetime, pipe_lifetime=pipe_lifetime))

    if not apply_shared_capture_target:
        return replace(costs, capture_target={})
    return with_capture_target(
        costs, curve, coefficient, maximum_capture_amount=maximum_capture_amount,
        normalize_capture_target=normalize_capture_target)


def with_capture_target(
    costs: CostData,
    curve: str,
    coefficient: Optional[float] = None,
    *, maximum_capture_amount: float,
    normalize_capture_target: bool = False) -> CostData:
    """Replace targets; optionally match the linear total over equal periods."""

    period_count = len(costs.periods)
    maximum = capture_max_amount(
        period_count, curve, coefficient, costs.durations,
        maximum_capture_amount=maximum_capture_amount)

    if normalize_capture_target and curve != "linear":
        target = generate_capture_targets(
            period_count, maximum, curve, coefficient, normalize_capture_target=True)
    else:
        target = {period: round(_capture_fraction(period, period_count,
                                                  curve, coefficient) * maximum, 3)
                  for period in costs.periods}
    return replace(costs, capture_target=target, model_parameters={
        **costs.model_parameters, "curve": curve, "coefficient": coefficient,
        "max_capture_amount": maximum_capture_amount,
        "normalize_capture_target": normalize_capture_target,
        "capture_amount_sequence": None, "target_source": "generated_curve"})


# Legacy period-limit helpers retained for external callers.
# Current FNE, heuristic and PSP solvers use full-horizon limits instead.
def _period_install_limits(costs, annual_limit, keys, label) -> dict[tuple, int]:
    """Convert inputs prepared by read.py into integer period quotas."""

    rate, durations = read_installation_inputs(costs, annual_limit, label)
    by_period = {period: floor(rate * durations[period])
        for period in costs.periods}
    return {key: by_period[key[-1]] for key in keys}


def period_unit_limits(
    network: NetworkData,
    costs: CostData,
    v: Union[int, float]) -> dict[tuple, int]:
    """Return legacy units-per-period quotas from a caller-supplied annual rate.

    Each quota is floor(Ns * period years), keyed by (source, period).
    """

    source_periods = [(i, period)
        for i in network.sources for period in costs.periods]
    limits = _period_install_limits(costs, v, source_periods, "source")

    return {(i, period): limits[i, period]
        for i, period in source_periods}


def period_well_limits(
    network: NetworkData,
    costs: CostData,
    v: Union[int, float]) -> dict[tuple, int]:
    """Return legacy wells-per-period quotas from a caller-supplied annual rate.

    Each quota is floor(Nr * period years), capped by the sink's maximum wells.
    """

    limits = _period_install_limits(costs, v, costs.sink_periods, "well")
    maximum_wells = read_well_capacities(network, costs)

    return {(j, period): min(limits[j, period], maximum_wells[j])
        for j, period in costs.sink_periods}


def period_pipe_limits(
    network: NetworkData,
    costs: CostData,
    v: Union[int, float]) -> dict[tuple, int]:
    """Return legacy pipes-per-period quotas from a caller-supplied annual rate.

    Each quota is floor(Np * period years), counting all diameters together.
    Np may be fractional: 0.3 over a five-year period permits one new pipe.
    The annual rate is never rounded before multiplying by the period length.
    """

    limits = _period_install_limits(costs, v, costs.arc_periods, "pipeline")

    return {(i, j, period): limits[i, j, period]
        for i, j, period in costs.arc_periods}
