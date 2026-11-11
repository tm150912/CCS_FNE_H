"""Engineering proxies for storage costs, without Excel/VBA or financial factors.

Costs returned here are undiscounted, unannualized USD2008.  The drilling curve
is a US analogue, not a Chinese quotation.  CO2 properties are fixed NETL
reference proxies, not an equation-of-state calculation at each site's P/T.
Only initial construction and electricity intensity are included.
"""

import math


DEFAULT_CONSTANTS = {
    "top_depth_ft": 5500.0,
    "rathole_ft": 50.0,
    "surface_temperature_f": 59.0,
    "temperature_gradient_f_per_100ft": 1.37,
    "hydrostatic_gradient_psi_per_ft": 0.464,
    "lithostatic_gradient_psi_per_ft": 1.0,
    "fracture_pressure_fraction": 0.6,
    "max_injection_fracture_fraction": 0.9,
    "vertical_horizontal_permeability_ratio": 0.3,
    "reservoir_density_kg_m3": 670.7896933974779,
    "reservoir_viscosity_cp": 0.056858132493166763,
    "law_bachu_coefficient": 0.0208,
    "mechanical_daily_t": 3660.0,
    "peak_average_ratio": 1.25,
    "spare_well_fraction": 0.1,
    "minimum_built_wells": 4,
    "injection_years": 30.0,
    "drilling_curve": "US_total_onshore",
    "drilling_localization_factor": 1.0,
    "drilling_2006_to_2008_factor": 1.025625,
    "capital_contingency_fraction": 0.15,
    "pump_density_kg_m3": 923.2193909657408,
    "pump_efficiency": 0.75,
    "pump_delta_mpa": 6.89475729,
    "pump_fixed_usd2008": 87000.0,
    "pump_usd2008_per_kw": 1400.0,
    "wells_per_header": 10,
    "header_usd2008": 200000.0,
    "branch_radius_multiplier": 0.333,
    "building_usd2008": 200000.0,
    "control_usd2008": 120000.0,
    "road_length_mile": 0.25,
    "road_usd2008_per_mile": 1000000.0,
    "custody_gauge_usd2008": 250000.0,
}

# Exact source cells inspected in SalineStorageCostModel.xlsm.  Site inputs
# replace the American formation's porosity, thickness, permeability and E.
SOURCE_CELLS = {
    "top_depth_ft": "Geol Sal!D34 (Woodbine reference, not local depth)",
    "rathole_ft": "Geo-Activity Interaction!E26",
    "surface_temperature_f": "Geol Sal!D185",
    "temperature_gradient_f_per_100ft": "Geol Sal!D186",
    "hydrostatic_gradient_psi_per_ft": "Geol Sal!D188",
    "lithostatic_gradient_psi_per_ft": "Geol Sal!D187",
    "fracture_pressure_fraction": "Geol Sal!D189",
    "max_injection_fracture_fraction": "Geol Sal!D326",
    "vertical_horizontal_permeability_ratio": "Geol Sal!D190",
    "reservoir_density_kg_m3": "Geol Sal!D360 cached reference property",
    "reservoir_viscosity_cp": "Geol Sal!D502 cached reference property",
    "law_bachu_coefficient": "Geol Sal!D505",
    "mechanical_daily_t": "Geol Sal!D324",
    "peak_average_ratio": "Geol Sal!D474",
    "spare_well_fraction": "Geol Sal!D330",
    "minimum_built_wells": "Geol Sal!D329",
    "injection_years": "Geol Sal!D471",
    "drilling_curve": "Drilling Costs!H48:J48; Texas check H12:J12",
    "drilling_localization_factor": "Scenario assumption; no China calibration",
    "drilling_2006_to_2008_factor": "Drilling Costs!BI8/BI4, original legacy factor",
    "capital_contingency_fraction": "Key_Inputs!AA33; Back-End_Cost Items!AQ",
    "pump_density_kg_m3": "Surf Eq Cost!D46 cached reference property",
    "pump_efficiency": "Surf Eq Cost!D47",
    "pump_delta_mpa": "Surf Eq Cost!D44-D42",
    "pump_fixed_usd2008": "Surf Eq Cost!D50",
    "pump_usd2008_per_kw": "Surf Eq Cost!D51",
    "wells_per_header": "Surf Eq Cost!D71 formula",
    "header_usd2008": "Surf Eq Cost!D72",
    "branch_radius_multiplier": "Surf Eq Cost!D87",
    "building_usd2008": "Surf Eq Cost!D102",
    "control_usd2008": "Surf Eq Cost!D113",
    "road_length_mile": "Surf Eq Cost!D124",
    "road_usd2008_per_mile": "Surf Eq Cost!D125",
    "custody_gauge_usd2008": "Surf Eq Cost!D138",
}

WELL_COMPONENT_SOURCES = {
    "permits": "Activity_Inputs!BK28; Back-End_Cost Items!303",
    "drill_complete": "Activity_Inputs!BK37; Back-End_Cost Items!312",
    "site_preparation": "Activity_Inputs!BK40; Back-End_Cost Items!325",
    "wireline_logging": "Activity_Inputs!BK69; Back-End_Cost Items!338",
    "coring": "Activity_Inputs!BK85; Back-End_Cost Items!350",
    "fluid_sampling": "Activity_Inputs!BK97; Back-End_Cost Items!359",
    "well_tests": "Activity_Inputs!BK107; Back-End_Cost Items!367",
    "vsp_characterization": "Activity_Inputs!BK115; Back-End_Cost Items!379",
    "reservoir_analysis": "Activity_Inputs!BK128; Back-End_Cost Items!386",
    "wellhead_equipment": "Activity_Inputs!BK136; Back-End_Cost Items!394 Expense",
    "other_completion": "Activity_Inputs!BK151; Back-End_Cost Items!405 Expense",
    "downhole_equipment": "Activity_Inputs!BK162; Back-End_Cost Items!415",
    "onsite_branch": "Surf Eq Cost!D84:D90; Back-End_Cost Items!473",
}


def drilling_cost_usd2008(depth_ft, constants=None):
    """Reproduce the selected US API depth curve and original price conversion."""
    c = {**DEFAULT_CONSTANTS, **(constants or {})}
    if c["drilling_curve"] == "US_total_onshore":
        a, b, quadratic = -131.45, 0.0528, 0.00003
    elif c["drilling_curve"] == "Texas":
        a, b, quadratic = 162.68, 0.0091, 0.00003
    else:
        raise ValueError("drilling_curve must be US_total_onshore or Texas")
    value = 1000.0 * (a + b * depth_ft + quadratic * depth_ft**2)
    if value <= 0:
        raise ValueError("Depth lies outside the positive-cost part of the API curve")
    return value * c["drilling_2006_to_2008_factor"] * c["drilling_localization_factor"]


def well_construction_components(depth_ft, constants=None):
    """Per-new-well direct and loaded costs for the workbook's selected options.

    This preserves its separate API drilling and CO2-resistant completion
    entries. A future all-inclusive local drilling quote must remove overlap.
    Backend rows 394 and 405 are construction-time Expense, with no 15% uplift.
    Annual O&M, MIT during operations, and plugging are deliberately absent.
    """
    c = {**DEFAULT_CONSTANTS, **(constants or {})}
    direct = {
        "permits": 100.0,
        "drill_complete": drilling_cost_usd2008(depth_ft, c),
        "site_preparation": 0.0,
        # Selected quad-combo, NMR, imaging, casing inspection, cement bond.
        "wireline_logging": (1.25 + 0.75 + 0.75 + 4.15 + 0.25) * depth_ft + 2070.0,
        "coring": 3100.0 + 1500.0 * 10.0,
        "fluid_sampling": 2000.0 + 1.0 * 207.0,
        "well_tests": 2070.0 + 2070.0,
        "vsp_characterization": 300000.0,
        "reservoir_analysis": 0.0,
        "wellhead_equipment": 15000.0,
        # Casing 7.5in, tubing 4in, and two selected CO2-resistant cement items.
        "other_completion": (2.7 * 7.5 + 1.15 * 4.0 + 1.5 + 1.5) * depth_ft,
        "downhole_equipment": 500.0,
    }
    loaded = {
        name: cost * (1.0 if name in {"wellhead_equipment", "other_completion"}
                      else 1.0 + c["capital_contingency_fraction"])
        for name, cost in direct.items()
    }
    return direct, loaded


def compute_engineering(subsite, annual_t, constants=None):
    """Return physical proxies, USD2008 initial costs and kWh/t for one subsite.

    Required subsite fields are Net_Sand_Thickness_m, Porosity_fraction,
    Permeability_mD, and Storage_Coefficient. annual_t is exogenous design
    throughput, not an optimizer's realized throughput. constants overrides
    documented scenario assumptions; the caller converts currency/price year.
    """
    c = {**DEFAULT_CONSTANTS, **(constants or {})}
    thickness_m = float(subsite["Net_Sand_Thickness_m"])
    porosity = float(subsite["Porosity_fraction"])
    kh_md = float(subsite["Permeability_mD"])
    storage_coefficient = float(subsite["Storage_Coefficient"])
    annual_t = float(annual_t)
    for name, value in (("thickness", thickness_m), ("porosity", porosity),
                        ("permeability", kh_md), ("storage_coefficient", storage_coefficient),
                        ("annual_t", annual_t)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if porosity > 1 or storage_coefficient > 1:
        raise ValueError("Porosity and storage coefficient must be fractions")

    thickness_ft = thickness_m / 0.3048
    top_ft = c["top_depth_ft"]
    midpoint_ft = top_ft + thickness_ft / 2.0
    bottom_ft = top_ft + thickness_ft
    well_depth_ft = bottom_ft + c["rathole_ft"]
    ambient_psi = midpoint_ft * c["hydrostatic_gradient_psi_per_ft"]
    fracture_top_psi = (top_ft * c["lithostatic_gradient_psi_per_ft"]
                        * c["fracture_pressure_fraction"])
    maximum_psi = (fracture_top_psi * c["max_injection_fracture_fraction"]
                   + thickness_ft / 2.0 * c["hydrostatic_gradient_psi_per_ft"])
    delta_reservoir_mpa = (maximum_psi - ambient_psi) * 0.0068948
    if delta_reservoir_mpa <= 0:
        raise ValueError("Assumed maximum injection pressure does not exceed ambient pressure")
    temperature_f = (c["surface_temperature_f"] + midpoint_ft / 100.0
                     * c["temperature_gradient_f_per_100ft"])
    kv_md = kh_md * c["vertical_horizontal_permeability_ratio"]
    effective_md = math.sqrt(kh_md * kv_md)
    formation_daily_t = (c["law_bachu_coefficient"] * effective_md
                         / c["reservoir_viscosity_cp"] * thickness_m * delta_reservoir_mpa)
    daily_t = min(formation_daily_t, c["mechanical_daily_t"])
    active_wells = math.ceil(annual_t * c["peak_average_ratio"] / (365.0 * daily_t))
    built_wells = max(int(c["minimum_built_wells"]),
                      math.ceil((1.0 + c["spare_well_fraction"]) * active_wells))
    # Continuous proxy; does not reproduce integer rounding/minimum-four wells.
    effective_annual_t = 365.0 * daily_t / (
        c["peak_average_ratio"] * (1.0 + c["spare_well_fraction"]))

    plume_area_m2 = (annual_t * c["injection_years"] * 1000.0
                     / (thickness_m * porosity * c["reservoir_density_kg_m3"]
                        * storage_coefficient))
    plume_area_mi2 = plume_area_m2 / 2589988.110336
    branch_miles = math.sqrt(plume_area_mi2 / math.pi) * c["branch_radius_multiplier"]
    active_well_mt_year = annual_t / active_wells / 1e6
    # Surf Eq Cost!D89 thresholds use annual average per active well.
    if active_well_mt_year < 0.29:
        branch_cost_per_mile = 360000.0
    elif active_well_mt_year < 0.676:
        branch_cost_per_mile = 540000.0
    elif active_well_mt_year < 1.16:
        branch_cost_per_mile = 720000.0
    else:
        branch_cost_per_mile = 900000.0

    electricity_kwh_per_t = (1000.0 * c["pump_delta_mpa"]
                             / (3.6 * c["pump_density_kg_m3"] * c["pump_efficiency"]))
    pump_kw = annual_t * c["peak_average_ratio"] * electricity_kwh_per_t / 8760.0
    well_direct, well_loaded = well_construction_components(well_depth_ft, c)
    well_direct["onsite_branch"] = branch_miles * branch_cost_per_mile
    well_loaded["onsite_branch"] = (well_direct["onsite_branch"]
                                    * (1.0 + c["capital_contingency_fraction"]))
    surface_direct = {
        "boosting_pump": c["pump_fixed_usd2008"] + c["pump_usd2008_per_kw"] * pump_kw,
        "headers": math.ceil(built_wells / c["wells_per_header"]) * c["header_usd2008"],
        "building": c["building_usd2008"],
        "control": c["control_usd2008"],
        "access_road": c["road_length_mile"] * c["road_usd2008_per_mile"],
        "custody_gauge": c["custody_gauge_usd2008"],
    }
    surface_loaded = {name: value * (1.0 + c["capital_contingency_fraction"])
                      for name, value in surface_direct.items()}
    return {
        "physical": {
            "design_annual_t": annual_t, "top_depth_ft": top_ft,
            "thickness_m": thickness_m, "bottom_depth_ft": bottom_ft,
            "well_depth_ft": well_depth_ft, "well_depth_m": well_depth_ft * 0.3048,
            "reservoir_temperature_c_proxy": (temperature_f - 32.0) * 5.0 / 9.0,
            "reservoir_ambient_mpa_proxy": ambient_psi * 0.0068948,
            "reservoir_maximum_mpa_proxy": maximum_psi * 0.0068948,
            "reservoir_delta_mpa_proxy": delta_reservoir_mpa,
            "horizontal_permeability_md": kh_md, "vertical_permeability_md_proxy": kv_md,
            "effective_permeability_md_proxy": effective_md,
            "formation_daily_t": formation_daily_t, "well_daily_t": daily_t,
            "well_effective_annual_t": effective_annual_t,
            "design_active_wells": active_wells, "design_built_wells": built_wells,
            "design_annual_t_per_built_well": annual_t / built_wells,
            "plume_area_km2": plume_area_m2 / 1e6,
            "plume_area_mi2": plume_area_mi2,
            "branch_length_mile_per_well": branch_miles,
            "branch_unit_cost_usd2008_per_mile": branch_cost_per_mile,
            "annual_mt_per_active_well": active_well_mt_year,
            "pump_peak_kw": pump_kw,
        },
        "well_components_direct_usd2008": well_direct,
        "well_components_loaded_usd2008": well_loaded,
        "site_surface_components_direct_usd2008": surface_direct,
        "site_surface_components_loaded_usd2008": surface_loaded,
        "totals": {
            "well_initial_usd2008": sum(well_loaded.values()),
            "well_initial_excluding_branch_usd2008": sum(well_loaded.values()) - well_loaded["onsite_branch"],
            "site_surface_usd2008": sum(surface_loaded.values()),
            "electricity_kwh_per_t": electricity_kwh_per_t,
        },
        "assumptions": {
            "constants": {key: c[key] for key in DEFAULT_CONSTANTS},
            "overrides": constants or {},
            "source_cells": SOURCE_CELLS.copy(),
            "well_component_sources": WELL_COMPONENT_SOURCES.copy(),
            "thickness": "Local effective sand thickness used as gross formation thickness (gross=net).",
            "depth": "NETL Woodbine top-depth proxy unless explicitly overridden; no local measured depth.",
            "properties": "Fixed NETL reference CO2 density and viscosity; no EOS/PVT recalculation.",
            "unit_conversions": "Exact 0.3048 m/ft; original geological 0.0068948 MPa/psi; exact square-mile area.",
            "well_capacity": "Continuous spare/peak-adjusted annual proxy; integer design wells calculated separately.",
            "allocation": "All shared surface equipment sized at design annual_t and charged to site opening; branch charged per well.",
            "price_basis": "USD2008, undiscounted and unannualized; legacy 1.025625 drill-price factor preserved, not 2-year compounding.",
            "scope": "Initial construction and pump electricity only; excludes fixed/per-well O&M, closure, postclosure, water management and external feeder.",
        },
    }


def validate_reference_case():
    """Check independently recorded NETL reference costs and electricity values."""
    direct, loaded = well_construction_components(6250.0, {"drilling_curve": "Texas"})
    checks = {
        "texas_drilling_usd2008": (direct["drill_complete"], 1427085.39375),
        "direct_well_usd2008": (sum(direct.values()), 1987952.39375),
        "loaded_well_usd2008": (sum(loaded.values()), 2257785.8778125),
        "expense_completion_no_uplift": (loaded["other_completion"], 174062.5),
    }
    ref = compute_engineering({"Net_Sand_Thickness_m": 213.36, "Porosity_fraction": 0.2,
                               "Permeability_mD": 500.0, "Storage_Coefficient": 0.054333333333333324},
                              3200000.0, {"drilling_curve": "Texas"})
    checks.update({
        "pump_peak_kw": (ref["physical"]["pump_peak_kw"], 1263.0082460507733),
        "electricity_usd_per_t": (ref["totals"]["electricity_kwh_per_t"] * 0.1036,
                                    916980.3612703477 / 3200000.0),
        "plume_area_mi2": (ref["physical"]["plume_area_mi2"], 23.83294624827236),
        "branch_direct_usd2008": (ref["well_components_direct_usd2008"]["onsite_branch"],
                                   2641500.2228992712 / 4.0),
        "active_wells": (ref["physical"]["design_active_wells"], 3),
        "built_wells": (ref["physical"]["design_built_wells"], 4),
    })
    for name, (actual, expected) in checks.items():
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
            raise AssertionError(f"{name}: actual={actual}, expected={expected}")
    return {name: {"actual": actual, "expected": expected, "passed": True}
            for name, (actual, expected) in checks.items()}


def validate_pump_operating_case(workbook_cells):
    """Verify the selected workbook pump case without recalculating its EOS.

    Source pressures are psig. Their MPa conversions contain no atmospheric
    offset; matching the cached density is not an absolute-pressure PVT check.
    """
    source = {}
    checks = {}
    surface = "Surf Eq Cost"

    def value(sheet, cell):
        item = workbook_cells.get(sheet, {}).get(cell, {})
        raw = item.get("v", "")
        source[f"{sheet}!{cell}"] = {"value": raw, "formula": item.get("f", "")}
        if raw in (None, ""):
            raise ValueError(f"Pump operating case requires {sheet}!{cell}")
        return raw

    def number(sheet, cell):
        raw = value(sheet, cell)
        try:
            result = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid pump input {sheet}!{cell}: {raw!r}") from error
        if not math.isfinite(result):
            raise ValueError(f"Non-finite pump input {sheet}!{cell}")
        return result

    def same(name, actual, expected):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError(f"Pump operating case {name}: actual={actual}, expected={expected}")
        checks[name] = {"actual": actual, "expected": expected, "passed": True}

    for sheet, cell, expected in (
            ("Key_Inputs", "T5", "yes"), (surface, "D35", "yes"),
            ("Key_Inputs", "S6", "psig"), ("Key_Inputs", "S7", "psig"),
            (surface, "E41", "psig"), (surface, "E43", "psig"),
            (surface, "E42", "mpa"), (surface, "E44", "mpa"),
            (surface, "E40", "kg/sec"), (surface, "E46", "kg/m3"),
            (surface, "E48", "kw")):
        if str(value(sheet, cell)).strip().lower() != expected:
            raise ValueError(f"Pump operating case requires {sheet}!{cell}={expected}")

    # Check these source expressions as well as their caches: adding atmospheric
    # pressure with a stale cached value must not pass a gauge-basis audit.
    for cell, expected in {
            "D41": "Key_Inputs!R6", "D42": "0.00689475729*D41",
            "D43": "Key_Inputs!R7", "D44": "0.00689475729*D43",
            "D45": "(D44+D42)*0.5", "D46": "denPRCO2(0,D45,D39)"}.items():
        value(surface, cell)
        actual = workbook_cells[surface][cell].get("f", "")
        normalize = lambda formula: "".join(formula.replace("$", "").split()).upper()
        if normalize(actual) != normalize(expected):
            raise ValueError(f"Pump operating case source formula changed at {surface}!{cell}")

    values = {cell: number(surface, cell) for cell in (
        "D15", "D36", "D37", "D38", "D39", "D40", "D41", "D42", "D43", "D44",
        "D45", "D46", "D47", "D48", "D50", "D51", "D52", "D57", "D58", "D59", "D60", "D61")}
    inlet = number("Key_Inputs", "R6")
    outlet = number("Key_Inputs", "R7")
    annual_t = number("Key_Inputs", "F48")
    peak_ratio = number("Key_Inputs", "F49")
    same("fixed_inlet_psig", inlet, 1200.0)
    same("fixed_outlet_psig", outlet, 2200.0)
    same("inlet_source_link", values["D41"], inlet)
    same("outlet_source_link", values["D43"], outlet)
    same("inlet_gauge_conversion", values["D42"], inlet * 0.00689475729)
    same("outlet_gauge_conversion", values["D44"], outlet * 0.00689475729)
    delta_mpa = values["D44"] - values["D42"]
    same("default_pressure_lift_mpa", delta_mpa, DEFAULT_CONSTANTS["pump_delta_mpa"])
    same("mean_source_pressure_mpa", values["D45"], (values["D44"] + values["D42"]) / 2.0)
    same("fixed_pump_ground_temperature_f", values["D37"], 53.0)
    same("pump_temperature_c", values["D38"], (values["D37"] - 32.0) * 5.0 / 9.0)
    same("pump_temperature_k", values["D39"], values["D38"] + 273.15)
    same("default_cached_density_kg_m3", values["D46"], DEFAULT_CONSTANTS["pump_density_kg_m3"])
    same("default_pump_efficiency", values["D47"], DEFAULT_CONSTANTS["pump_efficiency"])
    same("default_peak_average_ratio", peak_ratio, DEFAULT_CONSTANTS["peak_average_ratio"])
    same("peak_average_source_link", values["D36"], peak_ratio)
    same("reference_annual_tonnes", values["D15"] * 1e6, annual_t)
    if annual_t <= 0 or delta_mpa <= 0 or values["D46"] <= 0 or not 0 < values["D47"] <= 1:
        raise ValueError("Pump operating case requires positive flow, lift, density and valid efficiency")

    # Work = pressure in Pa * volume per tonne in m3 / efficiency, then J -> kWh.
    energy = delta_mpa * 1e6 * (1000.0 / values["D46"]) / values["D47"] / 3.6e6
    default_energy = (1000.0 * DEFAULT_CONSTANTS["pump_delta_mpa"] /
                      (3.6 * DEFAULT_CONSTANTS["pump_density_kg_m3"] * DEFAULT_CONSTANTS["pump_efficiency"]))
    same("default_electricity_kwh_per_t", default_energy, energy)
    mass_peak = annual_t * 1000.0 * peak_ratio / (365.0 * 24.0 * 3600.0)
    same("reference_peak_mass_flow_kg_s", values["D40"], mass_peak)
    peak_kw = mass_peak * delta_mpa * 1e6 / (values["D46"] * values["D47"]) / 1000.0
    same("reference_peak_power_kw", values["D48"], peak_kw)
    same("annual_energy_peak_power_kw", annual_t * peak_ratio * energy / 8760.0, peak_kw)
    same("default_pump_fixed_cost", values["D50"], DEFAULT_CONSTANTS["pump_fixed_usd2008"])
    same("default_pump_cost_per_kw", values["D51"], DEFAULT_CONSTANTS["pump_usd2008_per_kw"])
    same("reference_pump_direct_capital", values["D52"], values["D50"] + values["D51"] * peak_kw)
    same("reference_pump_maintenance", values["D58"], values["D52"] * values["D57"])
    same("reference_annual_electricity_cost", values["D60"], annual_t * energy * values["D59"])
    same("reference_total_om", values["D61"], values["D58"] + values["D60"])

    return {
        "passed": True, "case": "fixed_NETL_workbook_pump_case", "pump_enabled": True,
        "pressure_basis": "psig inputs converted to MPa without adding atmospheric pressure",
        "inlet_pressure_mpa_gauge": values["D42"], "outlet_pressure_mpa_gauge": values["D44"],
        "pressure_lift_mpa": delta_mpa, "mean_source_pressure_mpa": values["D45"],
        "density_kg_m3": values["D46"], "efficiency": values["D47"],
        "pump_temperature_c": values["D38"], "pump_temperature_k": values["D39"],
        "electricity_kwh_per_t": energy, "reference_annual_tonnes": annual_t,
        "peak_average_ratio": peak_ratio, "reference_peak_power_kw": peak_kw,
        "source_cells": source, "checks": checks,
        "audit_scope": [
            "Verifies source inputs, conversions, cached density and pump cost arithmetic against current constants",
            "The density is the source denPRCO2 cache; VBA/EOS was not rerun and absolute-pressure PVT was not verified",
            "No local delivery-pressure, wellhead-pressure, wellbore-friction or hydraulic verification is claimed",
            "Geological reservoir overpressure is separate from the surface pump pressure lift",
            "The 16-to-27 MPa external reference is a different scenario and is not used",
            "Pump temperature is 53 F; the 59 F geological surface-temperature proxy is a different input",
        ],
    }
