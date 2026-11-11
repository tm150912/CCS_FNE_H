"""Independent checks of raw storage profiles and multi-period coefficients.

No network or injection-capacity assumptions are required. Monetary inputs are
converted from USD2020 to kUSD2020; variable USD/t equals kUSD/kt numerically.
"""

from collections import defaultdict
import csv
from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from cost_amortize import cost_amortize, normalize_storage_lifecycle

RESULTS = ROOT / "storage_cost_results"
YEARS = 30
M_FIELDS = {"sink": "M_r_USD2020_per_year", "well": "M_w_USD2020_per_well_year"}
F_FIELDS = {"sink": "F_r_USD2020", "well": "F_w_USD2020_per_well"}
PROFILE_FIELDS = {
    "sink_extra_capital": "Site_Extra_CAPEX_USD2020_by_age",
    "sink_extra_opex": "Site_Extra_OPEX_USD2020_by_age",
    "well_extra_capital": "Well_Extra_CAPEX_USD2020_by_age",
    "well_extra_opex": "Well_Extra_OPEX_USD2020_by_age",
}
CHECKS = {}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compare(label, actual, expected, context):
    metric = CHECKS.setdefault(label, {"comparisons": 0, "max_absolute_error": 0.0,
                                      "max_relative_error": 0.0, "worst_case": None})
    error = abs(actual - expected)
    relative = error / max(abs(expected), 1e-300)
    metric["comparisons"] += 1
    if error > metric["max_absolute_error"]:
        metric.update(max_absolute_error=error, worst_case=context)
    metric["max_relative_error"] = max(relative, metric["max_relative_error"])
    if not isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise AssertionError(f"{label}: {context}: {actual} != {expected}")


def lifecycle_cashflows(lifecycle, scope, opening_year, discount):
    """Direct payments, independent of the production discounting loops."""
    prefix = "site" if scope == "sink" else "well"
    pre = fsum(amount / (1 + discount) ** (opening_year + int(offset))
               for offset, amount in lifecycle[f"{prefix}_pre_opex_by_offset"].items())
    terminal = 0.0
    for kind in ("capital", "opex"):
        amounts = (lifecycle[f"site_terminal_{kind}_by_age"][YEARS - opening_year]
                   if scope == "sink" else lifecycle[f"well_terminal_{kind}"])
        terminal += fsum(amount / (1 + discount) ** (YEARS + offset)
                         for offset, amount in enumerate(amounts))
    return {"pre_injection_expense_pv_kUSD2020": pre,
            "closure_and_pisc_cash_pv_kUSD2020": terminal}


def reference_cashflows(base, annual_om, capitals, opex, opening_year, interest, discount,
                        lifecycle, scope):
    """Explicit annual payments and recursively remaining principal per vintage.

    A vintage pays the same installment over its 30-year life; its residual loan
    balance is paid at year 30 of the project. Future vintages outside that
    project are never incurred. No implementation amortization helper is used.
    """
    operating_years = YEARS - opening_year + 1
    payments = [0.0] * YEARS
    residual = 0.0
    installments_denominator = fsum((1 + interest) ** -age for age in range(1, YEARS + 1))
    vintages = [(opening_year, base)] + [
        (opening_year + age, value) for age, value in enumerate(capitals[:operating_years])]
    for event_year, principal in vintages:
        installment = principal / installments_denominator
        balance = principal
        for year in range(event_year, YEARS + 1):
            payments[year - 1] += installment
            balance = balance * (1 + interest) - installment
        residual += balance
    expenses = [0.0] * (opening_year - 1) + [annual_om + value for value in opex[:operating_years]]
    capital_pv = fsum(amount / (1 + discount) ** year for year, amount in enumerate(payments, 1))
    om_pv = fsum(annual_om / (1 + discount) ** year for year in range(opening_year, YEARS + 1))
    extra_opex_pv = fsum(opex[age] / (1 + discount) ** (opening_year + age)
                        for age in range(operating_years))
    terminal_pv = residual / (1 + discount) ** YEARS
    added = lifecycle_cashflows(lifecycle, scope, opening_year, discount)
    return {"total_kUSD2020": capital_pv + terminal_pv + om_pv + extra_opex_pv + fsum(added.values()),
            **added,
            "capital_installments_pv_kUSD2020": capital_pv,
            "terminal_principal_pv_kUSD2020": terminal_pv,
            "annual_fixed_om_pv_kUSD2020": om_pv,
            "extra_opex_pv_kUSD2020": extra_opex_pv,
            "capital_installments_by_project_year_kUSD2020": payments,
            "operating_expenses_by_project_year_kUSD2020": expenses,
            "terminal_principal_undiscounted_kUSD2020": residual}


def main():
    paths = {"costs": RESULTS / "sink_subsite_cost.csv",
             "source_subsites": ROOT / "sink_subsite/subsite_result/sink_subsite.csv",
             "components": RESULTS / "operating_components.json",
             "initial_components": RESULTS / "cost_components.csv",
             "details": RESULTS / "calculation_details.json",
             "assumptions": RESULTS / "assumptions.json",
             "schedule": RESULTS / "storage_cost_schedule.csv",
             "electricity_tariffs": ROOT / "electricity_tariffs.csv",
             "price_references": ROOT / "storage_reference_data.csv",
             "implementation": ROOT.parent / "cost_amortize.py"}
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    rows = read_csv(paths["costs"])
    by_id = {row["Subsite_ID"]: row for row in rows}
    source_rows = read_csv(paths["source_subsites"])
    source_ids = {row["Subsite_ID"] for row in source_rows}
    initial_components = defaultdict(list)
    initial_records = defaultdict(list)
    for component in read_csv(paths["initial_components"]):
        key = component["Subsite_ID"], component["Parameter"]
        initial_components[key].append(float(component["Value_USD2020"]))
        initial_records[key].append(component)
    details_list = json.loads(paths["details"].read_text(encoding="utf-8"))
    details = {row["subsite"]["Subsite_ID"]: row for row in details_list}
    components = json.loads(paths["components"].read_text(encoding="utf-8"))
    assumptions = json.loads(paths["assumptions"].read_text(encoding="utf-8"))
    cpi = assumptions["price_conversion"]["CPI2020"] / assumptions["price_conversion"]["CPI2008"]
    price_references = {(row["Source_ID"], int(row["Record_Key"])): float(row["Value"])
                        for row in read_csv(paths["price_references"])
                        if row["Source_ID"] in {"FX_SOURCE", "CPI_SOURCE"}}
    tariff_records = read_csv(paths["electricity_tariffs"])
    tariffs = {(row["Grid_Region"], f"{int(row['Year']):04d}-{int(row['Month']):02d}"): row
               for row in tariff_records}
    if len(tariffs) != len(tariff_records):
        raise AssertionError("Duplicate grid/month electricity tariff snapshots")
    if (len(rows) != len(source_rows) or len(by_id) != len(rows) or len(source_rows) != len(source_ids)
            or set(by_id) != source_ids or set(by_id) != set(details)):
        raise AssertionError("All current, source and detail inputs must cover the same unique IDs")
    grouped = defaultdict(list)
    for component in components:
        grouped[component["Subsite_ID"]].append(component)
    if set(grouped) != set(by_id):
        raise AssertionError("Operating component IDs differ from the cost CSV")
    schedules = {(row["Subsite_ID"], int(row["Operating_Age"])): row
                 for row in read_csv(paths["schedule"])}
    if len(schedules) != len(rows) * YEARS:
        raise AssertionError("Expected one schedule record per subsite and operating age")

    inputs = {f"{scope}_fixed_om": {} for scope in M_FIELDS}
    inputs.update({name: {} for name in PROFILE_FIELDS})
    inputs["storage_lifecycle"] = {}
    base = {scope: {} for scope in F_FIELDS}
    variable = {}
    for sid, row in by_id.items():
        if assumptions["storage_cost_schema_version"] != 3 or assumptions["cost_schedule_years"] != 30:
            raise AssertionError(f"{sid}: unsupported schema")
        raw_lifecycle = json.loads(row["Storage_Lifecycle_USD2020"])
        inputs["storage_lifecycle"][sid] = normalize_storage_lifecycle(
            raw_lifecycle, sid, money_scale=1000)
        for prefix, parameter in (("site", "Site_Pre_OPEX"), ("well", "Well_Pre_OPEX")):
            compare("pre_injection_expense_components_USD",
                    fsum(raw_lifecycle[f"{prefix}_pre_opex_by_offset"].values()),
                    fsum(initial_components[sid, parameter]), f"{sid}/{prefix}")
        for key in ("site_terminal_capital_by_age", "site_terminal_opex_by_age"):
            for age, values in enumerate(raw_lifecycle[key]):
                for offset, value in enumerate(values):
                    compare("terminal_profiles_details_USD", value,
                            details[sid]["lifecycle"][key][age][offset] * cpi,
                            f"{sid}/{key}/age{age + 1}/offset{offset}")
        for key in ("well_terminal_capital", "well_terminal_opex"):
            for offset, value in enumerate(raw_lifecycle[key]):
                compare("terminal_profiles_details_USD", value,
                        details[sid]["lifecycle"][key][offset] * cpi,
                        f"{sid}/{key}/offset{offset}")
        operation = details[sid]["operations"]
        tariff = details[sid]["tariff"]
        dates = tariff["Tariff_Year_Months"].split(";")
        demand_cny, demand_usd = [], []
        for date in dates:
            snapshot = tariffs[tariff["Grid_Region"], date]
            year = int(snapshot["Year"])
            rate = float(snapshot["Demand_CNY_per_kW_month"])
            if not isfinite(rate) or rate < 0:
                raise AssertionError(f"{sid}/{date}: invalid source demand rate")
            demand_cny.append(rate)
            demand_usd.append(rate / price_references["FX_SOURCE", year]
                              * price_references["CPI_SOURCE", 2020]
                              / price_references["CPI_SOURCE", year])
        demand_rate = fsum(demand_usd) / len(dates)
        compare("demand_tariff_CNY_per_kW_month", tariff["Demand_CNY_per_kW_month"],
                fsum(demand_cny) / len(dates), sid)
        compare("demand_tariff_USD2020_per_kW_month", tariff["Demand_USD2020_per_kW_month"],
                demand_rate, sid)
        engineering = details[sid]["engineering"]
        physical = engineering["physical"]
        peak_kw = (physical["design_annual_t"] * assumptions["engineering_constants"]["peak_average_ratio"]
                   * engineering["totals"]["electricity_kwh_per_t"] / 8760.0)
        compare("pump_peak_power_kW", physical["pump_peak_kw"], peak_kw, sid)
        annual_demand = 12.0 * peak_kw * demand_rate
        compare("annual_demand_details_USD2020", details[sid]["costs"]["M_r_Demand_USD2020_per_year"],
                annual_demand, sid)
        demand_parts = [part for part in initial_records[sid, "M_r"]
                        if part["Component"] == "electricity_demand_charge"]
        if len(demand_parts) != 1 or demand_parts[0]["Value_USD2008"].strip():
            raise AssertionError(f"{sid}: demand charge must be one USD2020-only M_r component")
        compare("annual_demand_component_USD2020", float(demand_parts[0]["Value_USD2020"]),
                annual_demand, sid)
        flat_parts = [{k: v for k, v in part.items() if k != "Subsite_ID"} for part in grouped[sid]]
        if flat_parts != operation["components"]:
            raise AssertionError(f"{sid}: flattened operation details differ")
        for scope, field in F_FIELDS.items():
            parameter = "F_r" if scope == "sink" else "F_w"
            compare("initial_F_components_USD", float(row[field]),
                    fsum(initial_components[sid, parameter]), f"{sid}/{field}")
            base[scope][sid] = float(row[field]) / 1000
        for scope, field in M_FIELDS.items():
            bucket = "site_fixed_om" if scope == "sink" else "well_fixed_om"
            total = fsum(part["annual_cost_usd2008"] for part in grouped[sid] if part["bucket"] == bucket)
            demand = annual_demand if scope == "sink" else 0.0
            compare("annual_M_components_USD", float(row[field]), total * cpi + demand, f"{sid}/{field}")
            compare("annual_M_details_USD", float(row[field]),
                    operation[bucket + "_usd2008_per_year"] * cpi + demand, f"{sid}/{field}")
            parameter = "M_r" if scope == "sink" else "M_w"
            compare("annual_M_export_components_USD", float(row[field]),
                    fsum(initial_components[sid, parameter]), f"{sid}/{field}")
            netl_parts = [part for part in initial_records[sid, parameter]
                         if part["Component"] == "annual_fixed_costs_see_operating_components"]
            if len(netl_parts) != 1:
                raise AssertionError(f"{sid}/{field}: expected one raw NETL fixed-O&M component")
            compare("annual_NETL_M_component_USD2008", float(netl_parts[0]["Value_USD2008"]),
                    total, f"{sid}/{field}")
            compare("annual_NETL_M_component_USD2020", float(netl_parts[0]["Value_USD2020"]),
                    total * cpi, f"{sid}/{field}")
            inputs[f"{scope}_fixed_om"][sid] = float(row[field]) / 1000
        for name, field in PROFILE_FIELDS.items():
            bucket = name.replace("sink_", "site_") + "_age_usd2008"
            values = json.loads(row[field])
            if len(values) != YEARS or any(not isfinite(value) or value < 0 for value in values):
                raise AssertionError(f"{sid}/{field}: invalid profile")
            selected = [part for part in grouped[sid] if part["bucket"] == bucket]
            for age, actual in enumerate(values):
                total = fsum(part["age_costs_usd2008"][age] for part in selected)
                context = f"{sid}/{field}/age{age + 1}"
                compare("age_profiles_components_USD", actual, total * cpi, context)
                compare("age_profiles_details_USD", actual, operation[bucket][age] * cpi, context)
                compare("age_profiles_schedule_USD", actual,
                        float(schedules[sid, age + 1][field.removesuffix("_by_age")]), context)
            inputs[name][sid] = tuple(value / 1000 for value in values)
        for part in grouped[sid]:
            if "age_costs_usd2008" not in part:
                continue
            removed = part.get("annual_baseline_removed_usd2008", 0.0)
            if removed:
                matches = [item for item in grouped[sid] if item["backend_row"] == part["backend_row"]
                           and item["bucket"] == part["bucket"].split("_extra_")[0] + "_fixed_om"]
                if len(matches) != 1:
                    raise AssertionError(f"{sid}: annual baseline not recorded exactly once")
                compare("baseline_split_USD2008", matches[0]["annual_cost_usd2008"], removed, f"{sid}/row{part['backend_row']}")
            for age, cost in enumerate(part["age_costs_usd2008"]):
                raw = part["base_unit_usd2008"] * part["loading_factor"] * part["quantities_age"][age]
                compare("individual_component_arithmetic_USD2008", cost + removed, raw,
                        f"{sid}/row{part['backend_row']}/age{age + 1}")
        variable[sid] = float(row["V_r_USD2020_per_tCO2"])
        compare("V_components_USD_per_t", variable[sid], fsum(float(details[sid]["costs"][field]) for field in (
            "V_r_Electricity_USD2020_per_tCO2", "V_r_Pore_USD2020_per_tCO2",
            "V_r_Oversight_USD2020_per_tCO2", "V_r_ERR_USD2020_per_tCO2",
            "V_r_Stewardship_USD2020_per_tCO2")), sid)

    plans = {f"equal_{count}": [YEARS // count] * count for count in (1, 2, 3, 5, 6, 10, 15, 30)}
    plans["unequal_3_2_25"] = [3, 2, 25]

    def run_plan(durations, interest=0.09, discount=0.07):
        periods = list(range(1, len(durations) + 1))
        result = cost_amortize(YEARS, dict(zip(periods, durations)), periods, [], list(by_id), [], [],
                               {}, {}, base["sink"], base["well"], {}, variable, {},
                               maximum_capture_amount=0, interest_rate=interest,
                               discount_rate=discount, **inputs)
        return result[1], result[2], result[5]

    annual = run_plan([1] * YEARS)
    for sid in by_id:
        for opening_year in range(1, YEARS + 1):
            for index, scope in enumerate(("sink", "well")):
                expected = reference_cashflows(base[scope][sid], inputs[f"{scope}_fixed_om"][sid],
                                               inputs[f"{scope}_extra_capital"][sid], inputs[f"{scope}_extra_opex"][sid],
                                               opening_year, 0.09, 0.07,
                                               inputs["storage_lifecycle"][sid], scope)
                compare(f"independent_{scope}_cashflow_kUSD", annual[index][sid, opening_year],
                        expected["total_kUSD2020"], f"{sid}/open{opening_year}")
    for name, durations in plans.items():
        actual, zero = run_plan(durations), run_plan(durations, 0, 0)
        opening_year = 1
        for period, duration in enumerate(durations, 1):
            remaining = YEARS - opening_year + 1
            for sid in by_id:
                for index, scope in enumerate(("sink", "well")):
                    context = f"{name}/{sid}/open{opening_year}"
                    compare(f"partition_{scope}_invariance_kUSD", actual[index][sid, period],
                            annual[index][sid, opening_year], context)
                    cash = (base[scope][sid] + inputs[f"{scope}_fixed_om"][sid] * remaining
                            + fsum(inputs[f"{scope}_extra_capital"][sid][:remaining])
                            + fsum(inputs[f"{scope}_extra_opex"][sid][:remaining])
                            + fsum(lifecycle_cashflows(inputs["storage_lifecycle"][sid],
                                                       scope, opening_year, 0).values()))
                    compare(f"zero_rates_{scope}_direct_cash_kUSD", zero[index][sid, period], cash, context)
                compare("V_period_annual_sum_USD_per_t", actual[2][sid, period],
                        fsum(variable[sid] / 1.07 ** year for year in range(opening_year, opening_year + duration)),
                        f"{name}/{sid}/period{period}")
            opening_year += duration
        for sid in by_id:
            compare("V_full_horizon_invariance_USD_per_t", fsum(actual[2][sid, h] for h in range(1, len(durations) + 1)),
                    fsum(variable[sid] / 1.07 ** year for year in range(1, YEARS + 1)), f"{name}/{sid}")

    examples = []
    sid = next(iter(by_id))
    for opening_year in (1, 6, 30):
        example = {"Subsite_ID": sid, "opening_year": opening_year, "interest_rate": 0.09, "discount_rate": 0.07,
                   "quantity_basis": "One site and one injection well, reported separately; no injection volume assumed."}
        for scope in ("sink", "well"):
            example[scope] = reference_cashflows(base[scope][sid], inputs[f"{scope}_fixed_om"][sid],
                                                 inputs[f"{scope}_extra_capital"][sid], inputs[f"{scope}_extra_opex"][sid],
                                                 opening_year, 0.09, 0.07,
                                                 inputs["storage_lifecycle"][sid], scope)
        example["variable_pv_per_annual_kt_kUSD2020"] = fsum(variable[sid] / 1.07 ** year for year in range(opening_year, YEARS + 1))
        examples.append(example)
    if hashes != {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}:
        raise AssertionError("An input changed during validation")
    report = {"passed": True, "subsites": len(rows), "plans": plans, "subsite_plan_pairs": len(rows) * len(plans),
              "subsite_opening_plan_combinations": len(rows) * sum(map(len, plans.values())),
              "independent_subsite_opening_year_cases": len(rows) * YEARS,
              "operating_component_records": len(components),
              "initial_F_components_verified": True,
              "electricity_demand_charge_verified": True,
              "demand_charge_basis": "Each selected tariff snapshot is converted with its own year FX/CPI, then averaged. Annual demand charge is 12 times design peak pump kW times USD2020/kW/month, added once to site fixed O&M without NETL contingency or USD2008 escalation.",
              "tolerance": {"relative": 1e-10, "absolute": 1e-7}, "checks": CHECKS,
              "method": "Independent annual loan-balance recursion with positive horizon debt settlement. Operating ages restart at opening and stop at the horizon. Pre-injection expenses use signed opening offsets. Closure and PISC capital/expenses are paid and discounted at the common horizon plus their terminal offsets, without additional loans.",
              "no_double_count_check": "Each removed constant baseline is restored to exactly one matching annual M component; residual profile plus baseline equals original quantity times unit cost and loading.",
              "scope": "Storage coefficients only; no network, well capacity, feasibility, or model optimum was constructed.",
              "examples": examples, "inputs_unchanged": True, "input_sha256": hashes}
    target = RESULTS / "integration_validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "subsites": len(rows), "combinations": report["subsite_opening_plan_combinations"],
                      "output": str(target), "checks": {label: value["max_absolute_error"] for label, value in CHECKS.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
