"""计算多周期 CCS 模型的折现成本，并生成带周期下标的变量索引。"""

from __future__ import annotations

from math import expm1, isfinite, log1p
from typing import Callable, Iterable, Mapping

# 利率仅用于 CRF，折现率仅用于年度现金流现值；二者保持独立。
DISCOUNT_RATE = 0.07
INTEREST_RATE = 0.09
# 本研究的统一默认寿命；四类设施均可通过函数关键字参数分别覆盖。
TECHNICAL_LIFETIME = 30
STORAGE_LIFECYCLE_AGES = 30


def normalize_storage_lifecycle(value, label, money_scale=1.0):
    """Validate a lifecycle cash-flow record and convert its monetary amounts."""
    fields = {
        "closure_basis", "pisc_years", "site_pre_opex_by_offset",
        "well_pre_opex_by_offset", "site_terminal_capital_by_age",
        "site_terminal_opex_by_age", "well_terminal_capital",
        "well_terminal_opex",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")
    if value["closure_basis"] != "common_project_end":
        raise ValueError(f"{label} requires closure_basis=common_project_end")
    years = value["pisc_years"]
    if isinstance(years, bool) or not isinstance(years, int) or years < 0:
        raise ValueError(f"{label} pisc_years must be a nonnegative integer")

    def amount(raw):
        if (isinstance(raw, bool) or not isinstance(raw, (int, float))
                or not isfinite(raw) or raw < 0):
            raise ValueError(f"{label} requires finite nonnegative cash amounts")
        return float(raw) / money_scale

    def vector(raw):
        if not isinstance(raw, (list, tuple)) or len(raw) != years + 1:
            raise ValueError(f"{label} terminal vectors require pisc_years + 1 amounts")
        return tuple(amount(item) for item in raw)

    result = {"closure_basis": value["closure_basis"], "pisc_years": years}
    for field in ("site_pre_opex_by_offset", "well_pre_opex_by_offset"):
        raw = value[field]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} {field} must map year offsets to cash amounts")
        offsets = {}
        for offset, cash in raw.items():
            try:
                year = int(offset)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} requires nonpositive integer offsets") from error
            if not isinstance(offset, str) or str(year) != offset or year > 0:
                raise ValueError(f"{label} requires nonpositive integer offsets")
            offsets[offset] = amount(cash)
        result[field] = offsets
    for field in ("site_terminal_capital_by_age", "site_terminal_opex_by_age"):
        rows = value[field]
        if not isinstance(rows, (list, tuple)) or len(rows) != STORAGE_LIFECYCLE_AGES:
            raise ValueError(f"{label} {field} requires 30 operating-age rows")
        result[field] = tuple(vector(row) for row in rows)
    for field in ("well_terminal_capital", "well_terminal_opex"):
        result[field] = vector(value[field])
    return result


def _validate_schedule(project_years: int, Oh: Mapping[int, int],
                       H: Iterable[int]) -> list[int]:
    periods = list(H)
    if not periods:
        raise ValueError("H must contain at least one period")
    if periods != list(range(1, len(periods) + 1)):
        raise ValueError("H must be consecutive and start at 1")
    if set(Oh) != set(periods):
        raise ValueError("Oh must define a duration for every period in H")
    if any(Oh[h] <= 0 for h in periods):
        raise ValueError("Every period duration in Oh must be positive")
    if sum(Oh[h] for h in periods) != project_years:
        raise ValueError(
            "The durations in Oh must add up to the total project_years")
    return periods


def _capital_recovery_factor(lifetime: float, interest_rate: float) -> float:
    """Return CRF based on a facility's full technical lifetime."""

    if lifetime <= 0:
        raise ValueError("Technical lifetime must be positive")
    if interest_rate < 0:
        raise ValueError("Interest rate must be non-negative")
    if interest_rate == 0:
        return 1 / lifetime
    growth = (1 + interest_rate)**lifetime
    return interest_rate * growth / (growth - 1)


def _residual_capital_value(capital_cost: float, technical_lifetime: float,
                           years_in_project: float, interest_rate: float) -> float:
    """Return undiscounted residual value if project horizon ends before lifetime."""

    if years_in_project >= technical_lifetime:
        return 0.0
    if interest_rate == 0:
        return capital_cost * (1 - years_in_project / technical_lifetime)
    growth = (1 + interest_rate)**technical_lifetime
    return capital_cost * (
        growth - (1 + interest_rate)**years_in_project) / (growth - 1)


def _storage_cost_input(values, sink_nodes, label, project_years=None):
    """Validate optional annual amounts or age-indexed cash-flow sequences."""
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping by sink")
    result = {}
    for j in sink_nodes:
        if j not in values:
            raise ValueError(f"{label} is missing sink {j!r}")
        raw = values[j]
        if project_years is not None:
            if not isinstance(raw, (tuple, list)) or len(raw) < project_years:
                raise ValueError(
                    f"{label}[{j!r}] must contain at least {project_years} "
                    "amounts for ages 1 through project_years")
            amounts = raw
        else:
            amounts = [raw]
        converted = []
        for amount in amounts:
            try:
                value = float(amount)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{label}[{j!r}] must contain finite nonnegative amounts") from error
            if isinstance(amount, bool) or not isfinite(value) or value < 0:
                raise ValueError(
                    f"{label}[{j!r}] must contain finite nonnegative amounts")
            converted.append(value)
        result[j] = tuple(converted) if project_years is not None else converted[0]
    return result


def _amortize(project_years, Oh, H, ig, sink_nodes, ij, ijd,
               Tig, Tijd, Tj, Twj, Vig, Vj, Vijd,
               target_fraction: Callable[[int, int], float],
               *,
               maximum_capture_amount: float,
               source_lifetime=TECHNICAL_LIFETIME,
               sink_lifetime=TECHNICAL_LIFETIME,
               well_lifetime=TECHNICAL_LIFETIME,
               pipe_lifetime=TECHNICAL_LIFETIME,
               interest_rate=INTEREST_RATE,
               discount_rate=DISCOUNT_RATE,
               sink_fixed_om=None,
               well_fixed_om=None,
               sink_extra_capital=None,
               sink_extra_opex=None,
               well_extra_capital=None,
               well_extra_opex=None,
               storage_lifecycle=None,
               ):
    periods = _validate_schedule(project_years, Oh, H)
    sink_nodes = list(sink_nodes)
    storage_inputs = {}
    for name, values, age_indexed in (
            ("sink_fixed_om", sink_fixed_om, False),
            ("well_fixed_om", well_fixed_om, False),
            ("sink_extra_capital", sink_extra_capital, True),
            ("sink_extra_opex", sink_extra_opex, True),
            ("well_extra_capital", well_extra_capital, True),
            ("well_extra_opex", well_extra_opex, True)):
        storage_inputs[name] = _storage_cost_input(
            values, sink_nodes, name, project_years if age_indexed else None)
    lifecycle = {}
    if storage_lifecycle is not None:
        if not isinstance(storage_lifecycle, Mapping):
            raise ValueError("storage_lifecycle must be a mapping by sink")
        if project_years > STORAGE_LIFECYCLE_AGES:
            raise ValueError("storage_lifecycle supports at most 30 operating years")
        for j in sink_nodes:
            if j not in storage_lifecycle:
                raise ValueError(f"storage_lifecycle is missing sink {j!r}")
            lifecycle[j] = normalize_storage_lifecycle(
                storage_lifecycle[j], f"storage_lifecycle[{j!r}]")
    lifetimes = {
        "source": source_lifetime,
        "sink": sink_lifetime,
        "well": well_lifetime,
        "pipe": pipe_lifetime,
    }
    for facility, lifetime in lifetimes.items():
        if lifetime < project_years:
            raise ValueError(
                f"{facility} technical_lifetime={lifetime} must be >= "
                f"project_years={project_years};"
                "the model has no retirement or replacement decisions")
    if discount_rate <= -1:
        raise ValueError("Discount rate must be greater than -1")

    start_year = {}
    end_year = {}
    elapsed = 0
    for h in periods:
        start_year[h] = elapsed + 1
        elapsed += Oh[h]
        end_year[h] = elapsed

    fixed_present_value_factor = {}
    operating_factor = {}
    for h in periods:
        # Annualized investment payments run from the build year through the project end;
        # variable costs cover every year represented by the current period.
        fixed_present_value_factor[h] = sum(
            (1 + discount_rate)**-year
            for year in range(start_year[h], project_years + 1))
        operating_factor[h] = sum(
            (1 + discount_rate)**-year
            for year in range(start_year[h], end_year[h] + 1))

    years_of_operation_in_project = {
        h: project_years - start_year[h] + 1
        for h in periods
    }
    project_end_discount = (1 + discount_rate) ** -project_years
    residual_factor = {}
    for facility, lifetime in lifetimes.items():
        residual_factor[facility] = {
            h: project_end_discount * _residual_capital_value(
                1.0, lifetime, years_of_operation_in_project[h], interest_rate)
            for h in periods
        }

    recovery_factor = {
        facility: _capital_recovery_factor(lifetime, interest_rate)
        for facility, lifetime in lifetimes.items()
    }

    igh = [(i, g, h) for i, g in ig for h in periods]
    jh = [(j, h) for j in sink_nodes for h in periods]
    ijh = [(i, j, h) for i, j in ij for h in periods]
    ijdh = [(i, j, d, h) for i, j, d in ijd for h in periods]

    target_co2 = {
        h: round(target_fraction(h, len(periods)) * maximum_capture_amount, 3)
        for h in periods
    }
    Figh = {
        (i, g, h):
        Tig[i, g] * recovery_factor["source"]
        * fixed_present_value_factor[h]
        + Tig[i, g] * residual_factor["source"][h]
        for i, g, h in igh
    }
    Fjh = {
        (j, h):
        Tj[j] * recovery_factor["sink"]
        * fixed_present_value_factor[h]
        + Tj[j] * residual_factor["sink"][h]
        for j, h in jh
    }
    Fwjh = {
        (j, h):
        Twj[j] * recovery_factor["well"]
        * fixed_present_value_factor[h]
        + Twj[j] * residual_factor["well"][h]
        for j, h in jh
    }

    def add_storage_cashflows(costs, facility):
        """Age 1 is the cohort's build year; only in-horizon events are charged.

        All inputs use the same currency units as Tj/Twj. Annual fixed O&M and
        extra OPEX receive discounting only. Extra capital follows the original
        CRF plus positive ending-capital-balance rule from its actual event year.
        Site monitoring investments use sink_lifetime; well events use
        well_lifetime. The streams must exclude capital and O&M already in the
        corresponding base coefficients.
        """
        fixed_om = storage_inputs[f"{facility}_fixed_om"]
        extra_capital = storage_inputs[f"{facility}_extra_capital"]
        extra_opex = storage_inputs[f"{facility}_extra_opex"]
        if fixed_om is None and extra_capital is None and extra_opex is None:
            return
        discounts = {
            year: (1 + discount_rate) ** -year
            for year in range(1, project_years + 1)}
        capital_factors = {}
        if extra_capital is not None:
            for year in discounts:
                capital_factors[year] = (
                    recovery_factor[facility]
                    * sum(discounts[later] for later in range(year, project_years + 1))
                    + project_end_discount * _residual_capital_value(
                        1.0, lifetimes[facility], project_years - year + 1,
                        interest_rate))
        for j, h in jh:
            addition = (fixed_om[j] * fixed_present_value_factor[h]
                        if fixed_om is not None else 0.0)
            for age_index, year in enumerate(range(start_year[h], project_years + 1)):
                if extra_opex is not None:
                    addition += extra_opex[j][age_index] * discounts[year]
                if extra_capital is not None:
                    addition += extra_capital[j][age_index] * capital_factors[year]
            costs[j, h] += addition

    add_storage_cashflows(Fjh, "sink")
    add_storage_cashflows(Fwjh, "well")
    # Pre-operation expenses and closure/PISC are direct cash payments. Only
    # operating capital above uses CRF and a positive ending loan balance.
    for j, h in jh:
        if j not in lifecycle:
            continue
        record = lifecycle[j]
        age_index = years_of_operation_in_project[h] - 1
        for scope, costs in (("site", Fjh), ("well", Fwjh)):
            costs[j, h] += sum(
                cash * (1 + discount_rate) ** -(start_year[h] + int(offset))
                for offset, cash in record[f"{scope}_pre_opex_by_offset"].items())
            for kind in ("capital", "opex"):
                cashflows = (record[f"site_terminal_{kind}_by_age"][age_index]
                             if scope == "site" else record[f"well_terminal_{kind}"])
                costs[j, h] += sum(
                    cash * (1 + discount_rate) ** -(project_years + offset)
                    for offset, cash in enumerate(cashflows))
    Fijdh = {
        (i, j, d, h):
        Tijd[i, j, d] * recovery_factor["pipe"]
        * fixed_present_value_factor[h]
        + Tijd[i, j, d] * residual_factor["pipe"][h]
        for i, j, d, h in ijdh
    }
    Vigh = {(i, g, h): Vig[i, g] * operating_factor[h] for i, g, h in igh}
    Vjh = {(j, h): Vj[j] * operating_factor[h] for j, h in jh}
    Vijdh = {(i, j, d, h): Vijd[i, j, d] * operating_factor[h]
             for i, j, d, h in ijdh}

    return (Figh, Fjh, Fwjh, Fijdh, Vigh, Vjh, Vijdh, igh, jh, ijh,
            target_co2)


def cost_amortize_l(TY, Oh, H, ig, sink_nodes, ij, ijd,
                     Tig, Tijd, Tj, Twj, Vig, Vj,
                     Vijd, l, *,
                     maximum_capture_amount: float,
                     source_lifetime=TECHNICAL_LIFETIME,
                     sink_lifetime=TECHNICAL_LIFETIME,
                     well_lifetime=TECHNICAL_LIFETIME,
                     pipe_lifetime=TECHNICAL_LIFETIME,
                     interest_rate=INTEREST_RATE,
                     discount_rate=DISCOUNT_RATE,
                     sink_fixed_om=None,
                     well_fixed_om=None,
                     sink_extra_capital=None,
                     sink_extra_opex=None,
                     well_extra_capital=None,
                     well_extra_opex=None,
                     storage_lifecycle=None):
    """采用参数为 "l" 的对数型捕集目标曲线，捕集上限由调用方显式传入。"""

    if l <= -1 or l == 0:
        raise ValueError("k must be greater than -1 and different from 0")
    denominator = log1p(l)
    return _amortize(TY, Oh, H, ig, sink_nodes, ij, ijd,
                     Tig, Tijd, Tj, Twj, Vig, Vj,
                     Vijd, lambda h, count: log1p(l * h / count) / denominator,
                     maximum_capture_amount=maximum_capture_amount,
                     source_lifetime=source_lifetime,
                     sink_lifetime=sink_lifetime,
                     well_lifetime=well_lifetime,
                     pipe_lifetime=pipe_lifetime,
                     interest_rate=interest_rate,
                     discount_rate=discount_rate,
                     sink_fixed_om=sink_fixed_om,
                     well_fixed_om=well_fixed_om,
                     sink_extra_capital=sink_extra_capital,
                     sink_extra_opex=sink_extra_opex,
                     well_extra_capital=well_extra_capital,
                     well_extra_opex=well_extra_opex,
                     storage_lifecycle=storage_lifecycle)


def cost_amortize_e(TY, Oh, H, ig, sink_nodes, ij, ijd,
                     Tig, Tijd, Tj, Twj, Vig, Vj,
                     Vijd, e, *,
                     maximum_capture_amount: float,
                     source_lifetime=TECHNICAL_LIFETIME,
                     sink_lifetime=TECHNICAL_LIFETIME,
                     well_lifetime=TECHNICAL_LIFETIME,
                     pipe_lifetime=TECHNICAL_LIFETIME,
                     interest_rate=INTEREST_RATE,
                     discount_rate=DISCOUNT_RATE,
                     sink_fixed_om=None,
                     well_fixed_om=None,
                     sink_extra_capital=None,
                     sink_extra_opex=None,
                     well_extra_capital=None,
                     well_extra_opex=None,
                     storage_lifecycle=None):
    """采用参数为 "e" 的指数型捕集目标曲线，捕集上限由调用方显式传入。"""

    if e == 0:
        raise ValueError("n must be different from 0")
    denominator = expm1(e)
    return _amortize(TY, Oh, H, ig, sink_nodes, ij, ijd,
                     Tig, Tijd, Tj, Twj, Vig, Vj,
                     Vijd, lambda h, count: expm1(e * h / count) / denominator,
                     maximum_capture_amount=maximum_capture_amount,
                     source_lifetime=source_lifetime,
                     sink_lifetime=sink_lifetime,
                     well_lifetime=well_lifetime,
                     pipe_lifetime=pipe_lifetime,
                     interest_rate=interest_rate,
                     discount_rate=discount_rate,
                     sink_fixed_om=sink_fixed_om,
                     well_fixed_om=well_fixed_om,
                     sink_extra_capital=sink_extra_capital,
                     sink_extra_opex=sink_extra_opex,
                     well_extra_capital=well_extra_capital,
                     well_extra_opex=well_extra_opex,
                     storage_lifecycle=storage_lifecycle)


def cost_amortize(TY, Oh, H, ig, sink_nodes, ij, ijd,
                   Tig, Tijd, Tj, Twj, Vig, Vj,
                   Vijd, *,
                   maximum_capture_amount: float,
                   source_lifetime=TECHNICAL_LIFETIME,
                   sink_lifetime=TECHNICAL_LIFETIME,
                   well_lifetime=TECHNICAL_LIFETIME,
                   pipe_lifetime=TECHNICAL_LIFETIME,
                   interest_rate=INTEREST_RATE,
                   discount_rate=DISCOUNT_RATE,
                   sink_fixed_om=None,
                   well_fixed_om=None,
                   sink_extra_capital=None,
                   sink_extra_opex=None,
                   well_extra_capital=None,
                   well_extra_opex=None,
                   storage_lifecycle=None):
    """采用线性捕集目标曲线，捕集上限由调用方显式传入。"""

    return _amortize(TY, Oh, H, ig, sink_nodes, ij, ijd,
                     Tig, Tijd, Tj, Twj, Vig, Vj,
                     Vijd, lambda h, count: h / count,
                     maximum_capture_amount=maximum_capture_amount,
                     source_lifetime=source_lifetime,
                     sink_lifetime=sink_lifetime,
                     well_lifetime=well_lifetime,
                     pipe_lifetime=pipe_lifetime,
                     interest_rate=interest_rate,
                     discount_rate=discount_rate,
                     sink_fixed_om=sink_fixed_om,
                     well_fixed_om=well_fixed_om,
                     sink_extra_capital=sink_extra_capital,
                     sink_extra_opex=sink_extra_opex,
                     well_extra_capital=well_extra_capital,
                     well_extra_opex=well_extra_opex,
                     storage_lifecycle=storage_lifecycle)
