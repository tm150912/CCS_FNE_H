"""Estimate raw CCS_FNE storage costs and operating-age schedules.

Run with the bundled Python runtime. Costs are a NETL analogue scenario in
constant 2020 USD, without discounting or annualization. Operating schedules
cover 30 ages; separate cash flows cover pre-injection expenses, common-horizon
closure and post-injection care under a fixed design scenario.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import zipfile

from storage_geography import build_tariff_assignments


ROOT = Path(__file__).resolve().parent

# Input and output files (paths are relative to this script's directory).
INPUT_FILES = {
    "subsites": ROOT / "sink_subsite/subsite_result/sink_subsite.csv",
    "subsite_audit": ROOT / "sink_subsite/subsite_result/sink_subsite_audit.json",
    "references": ROOT / "storage_reference_data.csv",
    "tariffs": ROOT / "electricity_tariffs.csv",
    "workbook": ROOT / "SalineStorageCostModel.xlsm",
    "basins": ROOT / "sink.xlsx",
}
OUTPUT_DIR = ROOT / "storage_cost_results"
OUTPUT_FILES = {
    "costs": OUTPUT_DIR / "sink_subsite_cost.csv",
    "basin_capacity": OUTPUT_DIR / "basin_capacity.csv",
    "subsite_capacity": OUTPUT_DIR / "subsite_capacity.csv",
    "cost_components": OUTPUT_DIR / "cost_components.csv",
    "schedule": OUTPUT_DIR / "storage_cost_schedule.csv",
    "operating_components": OUTPUT_DIR / "operating_components.json",
    "lifecycle_components": OUTPUT_DIR / "lifecycle_components.json",
    "calculation_details": OUTPUT_DIR / "calculation_details.json",
    "assumptions": OUTPUT_DIR / "assumptions.json",
    "validation": OUTPUT_DIR / "validation.json",
}

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CPI_2008 = 215.303
CPI_2008_SOURCE = "https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202107.pdf"
MODEL_COST_COLUMNS = (
    "Subsite_ID", "Max_Well_Num", "Well_Annual_Capacity_t", "Area_km2",
    "Longitude", "Latitude",
    "F_r_USD2020", "F_w_USD2020_per_well", "V_r_USD2020_per_tCO2",
    "M_r_USD2020_per_year", "M_w_USD2020_per_well_year",
    "Site_Extra_CAPEX_USD2020_by_age", "Site_Extra_OPEX_USD2020_by_age",
    "Well_Extra_CAPEX_USD2020_by_age", "Well_Extra_OPEX_USD2020_by_age",
    "Storage_Cost_Schema_Version", "Cost_Schedule_Years",
    "Storage_Lifecycle_USD2020",
)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_workbook_cells(path):
    """Read selected original XML cells; never run macros or alter the workbook."""
    selected = {"Key_Inputs", "Activity_Inputs", "Back-End_Cost Items",
                "Plume&Well Schedule", "Surf Eq Cost",
                "Fin_Resp_Inputs"}
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(t.text or "" for t in s.findall(".//s:t", NS))
                   for s in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in
                ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        result = {}
        for sheet in book.find("s:sheets", NS):
            name = sheet.attrib["name"]
            if name not in selected:
                continue
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rels[rid]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            cells = {}
            for cell in ET.fromstring(archive.read(target)).findall(".//s:sheetData/s:row/s:c", NS):
                value = cell.findtext("s:v", default="", namespaces=NS)
                if cell.attrib.get("t") == "s":
                    value = strings[int(value)]
                cells[cell.attrib["r"]] = {
                    "v": value, "f": cell.findtext("s:f", default="", namespaces=NS)
                }
            result[name] = cells
    return result


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=("quota", "netl"), default="quota",
                        help="quota clips the basin annual limit to mainland area, then allocates it by subsite area; netl uses 3.2 Mt/year")
    parser.add_argument("--pore-fee-usd2008", type=float, default=None,
                        help="override the workbook analogue pore-space fee per tonne")
    args = parser.parse_args()
    if args.pore_fee_usd2008 is not None and (
            not math.isfinite(args.pore_fee_usd2008) or args.pore_fee_usd2008 < 0):
        raise ValueError("Pore-space fee must be finite and nonnegative")

    from storage_engineering import (compute_engineering, DEFAULT_CONSTANTS, SOURCE_CELLS,
                                     validate_reference_case, validate_pump_operating_case)
    from storage_site import compute_site
    from storage_operations import compute_operations
    from storage_lifecycle import compute_lifecycle
    from storage_capacity import load_basin_capacities, allocate_subsite_capacity

    hashes = {str(p.relative_to(ROOT)): source_hash(p) for p in INPUT_FILES.values()}
    subsites = read_csv(INPUT_FILES["subsites"])
    source_audit = json.loads(INPUT_FILES["subsite_audit"].read_text(encoding="utf-8"))
    if len(subsites) != source_audit["Subsite_Count"]:
        raise ValueError("Subsite CSV count differs from the source audit")
    if len({r["Subsite_ID"] for r in subsites}) != len(subsites):
        raise ValueError("Subsite IDs must be unique")
    basin_capacities = load_basin_capacities(INPUT_FILES["basins"], subsites)
    refs = read_csv(INPUT_FILES["references"])
    cpi2020 = next(float(r["Value"]) for r in refs
                   if r["Source_ID"] == "CPI_SOURCE" and r["Record_Key"] == "2020")
    conversion = cpi2020 / CPI_2008
    workbook = load_workbook_cells(INPUT_FILES["workbook"])
    pump_check = validate_pump_operating_case(workbook)
    tariffs = {r["Subsite_ID"]: r for r in build_tariff_assignments(subsites, INPUT_FILES["references"], INPUT_FILES["tariffs"])}
    reference_check = validate_reference_case()
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows, details, components, schedules, operation_components = [], [], [], [], []
    lifecycle_components = []
    schedule_fields = {
        "Site_Extra_CAPEX_USD2020_by_age": "site_extra_capital_age_usd2008",
        "Site_Extra_OPEX_USD2020_by_age": "site_extra_opex_age_usd2008",
        "Well_Extra_CAPEX_USD2020_by_age": "well_extra_capital_age_usd2008",
        "Well_Extra_OPEX_USD2020_by_age": "well_extra_opex_age_usd2008",
    }

    for subsite in subsites:
        sid = subsite["Subsite_ID"]
        capacity = allocate_subsite_capacity(subsite, basin_capacities[subsite["Sink_ID"]])
        if not math.isclose(capacity["Subsite_Max_Annual_Injection_t"],
                            float(subsite["Allocated_Annual_Quota_t"]), rel_tol=1e-12):
            raise ValueError(f"{sid}: refresh the subsite annual quota using mainland reduction and subsite area")
        annual = capacity["Subsite_Max_Annual_Injection_t"] if args.design == "quota" else 3200000.0
        eng = compute_engineering(subsite, annual)
        site = compute_site(subsite, eng, workbook)
        operations = compute_operations(subsite, eng, site, workbook)
        lifecycle = compute_lifecycle(eng, site, operations, workbook)
        phys, tariff = eng["physical"], tariffs[sid]
        site_extra = site["site_initial_usd2008"]
        well_extra = site["per_injection_well_initial_usd2008"]
        fr = (eng["totals"]["site_surface_usd2008"] + site_extra) * conversion
        # Backend 394/405 are construction-time Expense, not financed capital.
        engineering_well_capital = {
            name: value for name, value in eng["well_components_loaded_usd2008"].items()
            if name not in {"wellhead_equipment", "other_completion"}}
        engineering_well_expense = sum(
            eng["well_components_loaded_usd2008"][name]
            for name in ("wellhead_equipment", "other_completion"))
        fw = (sum(engineering_well_capital.values()) + well_extra) * conversion
        pre_well = {str(offset): value for offset, value in site["well_pre_opex_by_offset"].items()}
        pre_well["0"] = pre_well.get("0", 0.0) + engineering_well_expense
        lifecycle_cash = {
            "closure_basis": lifecycle["closure_basis"],
            "pisc_years": lifecycle["pisc_years"],
            "site_pre_opex_by_offset": {
                str(offset): value * conversion
                for offset, value in site["site_pre_opex_by_offset"].items()},
            "well_pre_opex_by_offset": {offset: value * conversion for offset, value in pre_well.items()},
            **{key: [[amount * conversion for amount in values] for values in lifecycle[key]]
               for key in ("site_terminal_capital_by_age", "site_terminal_opex_by_age")},
            **{key: [amount * conversion for amount in lifecycle[key]]
               for key in ("well_terminal_capital", "well_terminal_opex")},
        }
        energy = eng["totals"]["electricity_kwh_per_t"]
        vr_electricity = energy * tariff["Electricity_USD2020_per_kWh"]
        fees = {part["backend_row"]: part["unit_usd2008_per_t"]
                for part in operations["variable_components"]}
        pore_fee = fees[89] if args.pore_fee_usd2008 is None else args.pore_fee_usd2008
        oversight_fee, err_fee, stewardship_fee = fees[91], fees[495], fees[90]
        vr = vr_electricity + (pore_fee + oversight_fee + err_fee + stewardship_fee) * conversion
        mr_operating = operations["site_fixed_om_usd2008_per_year"] * conversion
        # Fixed contract demand equals the installed pump's design peak kW.
        # This local tariff is already USD2020: no NETL loading or CPI2008 uplift.
        mr_demand = 12.0 * phys["pump_peak_kw"] * tariff["Demand_USD2020_per_kW_month"]
        mr = mr_operating + mr_demand
        mw = operations["well_fixed_om_usd2008_per_year"] * conversion
        if any(not math.isfinite(c) or c <= 0 for c in (fr, fw, vr)):
            raise ValueError(f"{sid}: costs must be finite and positive")
        if annual * DEFAULT_CONSTANTS["injection_years"] > float(subsite["Allocated_Total_CO2_t"]):
            raise ValueError(f"{sid}: design injection exceeds allocated geological resource")
        if phys["plume_area_km2"] > float(subsite["Area_km2"]):
            raise ValueError(f"{sid}: proxy plume exceeds the subsite resource area")
        row = {
            "Subsite_ID": sid, "Sink_ID": subsite["Sink_ID"], "Sink_Name": subsite["Sink_Name"],
            "Province": tariff["Province"],
            "F_r_USD2020": fr, "F_w_USD2020_per_well": fw, "V_r_USD2020_per_tCO2": vr,
            "M_r_USD2020_per_year": mr, "M_w_USD2020_per_well_year": mw,
            "M_r_Operating_USD2020_per_year": mr_operating,
            "M_r_Demand_USD2020_per_year": mr_demand,
            "Storage_Cost_Schema_Version": 3, "Cost_Schedule_Years": 30,
            "Storage_Lifecycle_USD2020": json.dumps(lifecycle_cash, separators=(",", ":"), allow_nan=False),
            "Max_Well_Num": capacity["Max_Well_Num"],
            "Well_Annual_Capacity_t": capacity["Well_Annual_Capacity_t"],
            "Area_km2": float(subsite["Area_km2"]),
            "Longitude": float(subsite["Longitude"]),
            "Latitude": float(subsite["Latitude"]),
            **{field: json.dumps([value * conversion for value in operations[key]],
                                 separators=(",", ":"), allow_nan=False)
               for field, key in schedule_fields.items()},
            "V_r_Electricity_USD2020_per_tCO2": vr_electricity,
            "V_r_Pore_USD2020_per_tCO2": pore_fee * conversion,
            "V_r_Oversight_USD2020_per_tCO2": oversight_fee * conversion,
            "V_r_ERR_USD2020_per_tCO2": err_fee * conversion,
            "V_r_Stewardship_USD2020_per_tCO2": stewardship_fee * conversion,
            "Cost_Status": "NETL_analogue_scenario_not_local_engineering_quote",
            "Variable_Cost_Scope": "electricity_plus_NETL_analogue_pore_oversight_ERR_stewardship",
            "Cost_Schedule_Basis": "fixed_design_site_age_and_individual_injection_well_age",
            "Design_Basis": "mainland_reduced_parent_maximum_annual_injection_by_subsite_area" if args.design == "quota" else "uniform_NETL_3.2Mt_per_year",
            "Design_Annual_Injection_t": annual,
            "Design_Injection_Years": DEFAULT_CONSTANTS["injection_years"],
            "Design_Active_Wells": phys["design_active_wells"],
            "Design_Built_Wells": phys["design_built_wells"],
            "Assumed_Top_Depth_m": DEFAULT_CONSTANTS["top_depth_ft"] * 0.3048,
            "Estimated_Well_Depth_m": phys["well_depth_m"],
            "Estimated_Plume_Area_km2": phys["plume_area_km2"],
            "Pump_Electricity_kWh_per_tCO2": energy,
            "Pump_Contract_Demand_kW": phys["pump_peak_kw"],
            "Electricity_USD2020_per_kWh": tariff["Electricity_USD2020_per_kWh"],
            "Demand_USD2020_per_kW_month": tariff["Demand_USD2020_per_kW_month"],
            "Grid_Region": tariff["Grid_Region"],
            "Tariff_Year_Months": tariff["Tariff_Year_Months"],
            "Tariff_Assumption_Flags": tariff["Tariff_Assumption_Flags"],
        }
        rows.append(row)
        details.append({"subsite": subsite, "capacity": capacity, "tariff": tariff, "engineering": {
            k: v for k, v in eng.items() if k != "assumptions"}, "site": site,
            "operations": operations, "lifecycle": lifecycle, "costs": row})
        for age in range(1, 31):
            schedules.append({"Subsite_ID": sid, "Operating_Age": age,
                              **{field.removesuffix("_by_age"): operations[key][age - 1] * conversion
                                 for field, key in schedule_fields.items()}})
        for part in operations["components"]:
            operation_components.append({"Subsite_ID": sid, **part})
        for part in lifecycle["components"]:
            lifecycle_components.append({"Subsite_ID": sid, **part})
        groups = [
            ("F_r", "surface", eng["site_surface_components_loaded_usd2008"]),
            ("F_r", "site", site["site_components_usd2008"]),
            ("F_w", "well", engineering_well_capital),
            ("F_w", "ancillary", site["per_well_components_usd2008"]),
        ]
        for parameter, category, items in groups:
            for name, cost in items.items():
                components.append({"Subsite_ID": sid, "Parameter": parameter,
                                   "Component": category + ":" + name,
                                   "Value_USD2008": cost, "Value_USD2020": cost * conversion,
                                   "Unit": "USD/site" if parameter == "F_r" else "USD/well"})
        for parameter, values, unit in (
                ("Site_Pre_OPEX", site["site_pre_opex_by_offset"], "USD/site"),
                ("Well_Pre_OPEX", pre_well, "USD/well")):
            for offset, amount in values.items():
                components.append({"Subsite_ID": sid, "Parameter": parameter,
                                   "Component": f"expense_at_opening_year_offset_{offset}",
                                   "Value_USD2008": amount, "Value_USD2020": amount * conversion,
                                   "Unit": unit})
        components.append({"Subsite_ID": sid, "Parameter": "V_r", "Component": "electricity",
                           "Value_USD2008": "", "Value_USD2020": vr_electricity, "Unit": "USD/tCO2"})
        for label, fee in (("analogue_pore_fee", pore_fee), ("analogue_oversight", oversight_fee),
                           ("analogue_ERR", err_fee), ("analogue_stewardship", stewardship_fee)):
            components.append({"Subsite_ID": sid, "Parameter": "V_r", "Component": label,
                               "Value_USD2008": fee,
                               "Value_USD2020": fee * conversion, "Unit": "USD/tCO2"})
        for label, amount, unit in (("M_r", mr_operating, "USD/site/year"), ("M_w", mw, "USD/well/year")):
            components.append({"Subsite_ID": sid, "Parameter": label,
                               "Component": "annual_fixed_costs_see_operating_components",
                               "Value_USD2008": amount / conversion,
                               "Value_USD2020": amount, "Unit": unit})
        components.append({"Subsite_ID": sid, "Parameter": "M_r",
                           "Component": "electricity_demand_charge",
                           "Value_USD2008": "", "Value_USD2020": mr_demand,
                           "Unit": "USD/site/year"})

    # Verify raw coefficients and independently stored schedule shapes.
    sums = Counter()
    for part in components:
        sums[(part["Subsite_ID"], part["Parameter"])] += part["Value_USD2020"]
    for row in rows:
        for parameter, field in (("F_r", "F_r_USD2020"), ("F_w", "F_w_USD2020_per_well"),
                                 ("V_r", "V_r_USD2020_per_tCO2"),
                                 ("M_r", "M_r_USD2020_per_year"),
                                 ("M_w", "M_w_USD2020_per_well_year")):
            if not math.isclose(sums[(row["Subsite_ID"], parameter)], row[field], rel_tol=1e-12, abs_tol=1e-8):
                raise AssertionError(f"{row['Subsite_ID']}: {parameter} does not reconcile")
        for field in schedule_fields:
            values = json.loads(row[field])
            if len(values) != 30 or any(not math.isfinite(v) or v < 0 for v in values):
                raise AssertionError(f"{row['Subsite_ID']}: invalid {field}")
    if hashes != {str(p.relative_to(ROOT)): source_hash(p) for p in INPUT_FILES.values()}:
        raise AssertionError("A source input changed during the calculation")

    capacity_checks = {}
    for basin_id, basin in basin_capacities.items():
        children = [item["capacity"] for item in details if item["subsite"]["Sink_ID"] == basin_id]
        allocated = math.fsum(child["Subsite_Max_Annual_Injection_t"] for child in children)
        if children:
            area = basin["Parent_Retained_Area_km2"]
            tolerance = basin["Mainland_Max_Annual_Injection_t"] * source_audit["Coverage_Tolerance_km2"] / area
            if abs(allocated - basin["Mainland_Max_Annual_Injection_t"]) > tolerance:
                raise AssertionError(f"{basin_id}: annual allocation does not reconcile within spatial tolerance")
        capacity_checks[basin_id] = {
            "Subsite_Count": len(children), "Allocated_Annual_Injection_t": allocated,
            "Basin_Max_Annual_Injection_t": basin["Basin_Max_Annual_Injection_t"],
            "Mainland_Max_Annual_Injection_t": basin["Mainland_Max_Annual_Injection_t"],
            "Allocation_Difference_t": allocated - basin["Mainland_Max_Annual_Injection_t"] if children else None,
            "Status": "passed" if children else "no_boundary_no_subsite_allocation",
        }

    assumptions = {
        "description_zh": "基于NETL工作簿和本地子场数据的统一工程情景估算，不是各子场实测造价。",
        "price_basis": "constant 2020 USD", "discounting": False, "capital_annualization": False,
        "source_subsite_count": len(subsites),
        "count_note_zh": f"本批覆盖源CSV及空间审计中的全部{len(subsites)}个子场。",
        "design_basis": args.design,
        "design_note_zh": "sink.xlsx中最大年注入量对应原盆地总面积，先按大陆交集/原盆地面积折减，再按子场/大陆交集面积分配，作为费用设计目标；30年项目，不是200年寿命。" if args.design == "quota" else "每子场统一设计年注入量320万吨，寿命30年；此费用敏感性情景不保证覆盖分配的最大注入能力。",
        "scope_excluded": ["water management (workbook OFF)",
                           "external feeder pipeline (network boundary)",
                           "operating eddy startup row255 (unresolved overlap with row281)",
                           "trust/escrow/tax cash flows and endogenous financing"],
        "nonenergy_fees_note_zh": "孔隙使用、运营监管、ERR和长期托管按吨费采用NETL美国类比情景，不代表中国必缴收费。",
        "pore_fee_usd2008_per_t": pore_fee,
        "oversight_fee_usd2008_per_t": oversight_fee, "ERR_fee_usd2008_per_t": err_fee,
        "stewardship_fee_usd2008_per_t": stewardship_fee,
        "storage_cost_schema_version": 3, "cost_schedule_years": 30,
        "lifecycle_notes": lifecycle["assumptions"],
        "closure_basis": "All opened sites and injection wells stop at the common network horizon; PISC follows for the workbook duration.",
        "lifecycle_cashflow_basis": "Pre-injection Expense uses offsets from opening; all terminal CAPEX/OPEX is paid at its event year, without new loans. Operating capital retains CRF and horizon debt settlement.",
        "model_csv_columns": MODEL_COST_COLUMNS,
        "coordinate_note_zh": "Longitude、Latitude继承子场源表，单位为度，坐标系为WGS84（EPSG:4326）；所有子场统一在母盆地局部LAEA投影中使用ArcPy labelPoint，经严格区内检查后转为经纬度，Point_Method均为interior_label_point。",
        "capacity_input_note_zh": "CO2_EWR(tonne)为原完整盆地最大年注入量；大陆内上限=CO2_EWR*大陆交集面积/原盆地面积；子场上限=大陆内上限*子场面积/大陆交集面积。Well_Annual_Capacity_t=365*min(地层日能力,3660)，子场继承母盆地单井能力，Max_Well_Num=floor(子场最大年量/母盆地单井最大年能力)。",
        "capacity_basis": {
            "source": "sink.xlsx, sheet sink; CO2_EWR(tonne) is t/year for the full original basin area, as specified by user",
            "mainland_reduction": "CO2_EWR(tonne) * Parent_Retained_Area_km2 / Parent_Original_Area_km2",
            "allocation": "mainland maximum annual injection * subsite Area_km2 / Parent_Retained_Area_km2; equivalently CO2_EWR(tonne) * subsite Area_km2 / Parent_Original_Area_km2; no child-area-sum normalization",
            "area_definition": "Parent_Retained_Area_km2 is the geodesic area of basin boundary intersected with the largest mainland polygon from China_boundary",
            "area_tolerance_km2": source_audit["Coverage_Tolerance_km2"],
            "well_annual_formula": "365 * min(Law-Bachu formation daily rate, 3660 t/day)",
            "well_limit_formula": "floor((mainland-reduced basin maximum annual injection * subsite area / mainland-intersection area) / parent well maximum annual injection)",
            "availability": "365 days/year at maximum rate; engineering upper bound, not measured injectivity",
            "design_redundancy": "Peak/average=1.25 and spare wells=10% remain cost-design margins only; not reductions to maximum well capacity",
            "integer_rounding": "Floor each subsite's unrounded ratio; fractional-well remainder is not redistributed and is reported in subsite_capacity.csv",
        },
        "basin_capacities": basin_capacities,
        "schedule_fields": schedule_fields,
        "operations_notes": operations["assumptions"],
        "engineering_constants": DEFAULT_CONSTANTS, "constant_sources": SOURCE_CELLS,
        "engineering_notes": eng["assumptions"],
        "pump_operating_case": pump_check,
        "electricity_demand_basis": "Fixed contract demand equals design peak pump kW; 12 monthly charges per operating year, paid regardless of realized flow and stopped at the common injection horizon. Added to site M only, without NETL loading, CRF or post-injection charges. Capacity-based billing and excess-demand penalties are not modeled.",
        "component_note_scope_zh": "engineering_notes和site_notes仅说明初始建设模块的边界；本批整体范围以scope_excluded及operations_notes为准。",
        "site_notes": {k: v for k, v in site.get("assumptions", {}).items()
                       if k not in {"initial_above_seal_monitoring_wells", "initial_dual_monitoring_wells",
                                    "monitoring_counts_by_area", "upfront_monitoring_design_well_count"}},
        "site_monitoring_counts_note_zh": "首批监测井数量随子场规模变化，逐子场数量保存在calculation_details.json。",
        "price_conversion": {"CPI2008": CPI_2008, "CPI2020": cpi2020,
                             "USD2008_to_USD2020": conversion, "CPI2008_source": CPI_2008_SOURCE,
                             "electricity": "CNY_y / FX_y * CPI2020 / CPI_y; each tariff uses its own year"},
        "model_integration_notes_zh": [
            "F_r按每个独立子场开放一次计；F_w按每口新注入井计；V_r按实际吨数计。",
            "F只含资本；M/V、运营年龄及生命周期实际支出均不折现、不年化；运营数组下标0对应投运年龄1。",
            "M为每年固定费用，年龄数组只存M之外的增量；读取后不得将M再次加进原始数组。",
            "场址M含需量费：合同需量假定等于设计峰值泵功率，每年计12个月，按电价年份换算USD2020后直接加入，不重复乘NETL系数；停注后停止计费。",
            "当前父模型以千美元和kt/年计价：F、M和年龄支出均除1000，V美元/吨保持数值。",
            "场址新增资本按sink_lifetime，每井新增资本按well_lifetime，在实际支出年启动CRF及期末余额计算。",
            "运营年龄活动仅计至共同规划期末；终期封井、剩余旧井整治和后续PISC通过单独生命周期现金流计入。",
            "前期Expense相对开场年定位，可能早于规划基准年；它是新开场引致的预先承诺费用，不是已发生沉没成本。",
            "终期矩阵行下标为截至规划期末的场龄减1，列0为规划期末、列1至L为PISC第1至L年；井级向量对每口已建注入井计一次。",
            "PISC沿用截至停注时已建深部监测井数量和设计羽流监测面积，不追建未发生的运营期后续设施。",
            "场址监测计划按固定设计规模执行；逐注入井周期按该井井龄执行，均不随求解流量内生改变。",
            "按源子场完整ID接入父模型，禁止位置或行号映射。",
            "共用泵和汇管按设计规模一次建设；模型实际流量与累计井数不得超出这套设计的设备能力。",
            "Design_Built_Wells含峰值与备用冗余，不等于Max_Well_Num；后者按年注入额度和单井最大能力向下取整。",
            "原表美国产业费用和工程标准作为类比，未作中国造价校准；局部电网代理均保留标记。",
        ],
    }
    audit = {
        "passed": True, "source_rows": len(subsites), "output_rows": len(rows),
        "unique_output_ids": len({r["Subsite_ID"] for r in rows}),
        "basins": len({r["Sink_ID"] for r in rows}), "provinces": len({r["Province"] for r in rows}),
        "component_reconciliation": "all rows passed", "inputs_unchanged": True,
        "design_mass_within_allocated_resource": True, "plume_within_subsite_area": True,
        "capacity_populated_rows": len(rows),
        "mapped_basin_count": len({r["Sink_ID"] for r in rows}),
        "source_basin_count": len(basin_capacities),
        "capacity_checks": capacity_checks,
        "well_capacity_inherited_from_parent": True,
        "reference_checks": reference_check, "pump_operating_case": pump_check,
        "input_sha256": hashes,
        "tariff_flag_counts": dict(Counter(flag for r in rows for flag in r["Tariff_Assumption_Flags"].split(";"))),
        "cost_ranges": {field: {"min": min(r[field] for r in rows), "max": max(r[field] for r in rows)}
                        for field in ("F_r_USD2020", "F_w_USD2020_per_well", "V_r_USD2020_per_tCO2",
                                      "M_r_USD2020_per_year", "M_w_USD2020_per_well_year",
                                      "M_r_Demand_USD2020_per_year")},
    }
    write_csv(OUTPUT_FILES["costs"],
              [{field: row[field] for field in MODEL_COST_COLUMNS} for row in rows])
    write_csv(OUTPUT_FILES["basin_capacity"], list(basin_capacities.values()))
    write_csv(OUTPUT_FILES["subsite_capacity"], [item["capacity"] for item in details])
    write_csv(OUTPUT_FILES["cost_components"], components)
    write_csv(OUTPUT_FILES["schedule"], schedules)
    write_json(OUTPUT_FILES["operating_components"], operation_components)
    write_json(OUTPUT_FILES["lifecycle_components"], lifecycle_components)
    write_json(OUTPUT_FILES["calculation_details"], details)
    write_json(OUTPUT_FILES["assumptions"], assumptions)
    write_json(OUTPUT_FILES["validation"], audit)
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT_DIR), "cost_ranges": audit["cost_ranges"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
