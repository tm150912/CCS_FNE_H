"""Thirty operating years of raw NETL-analogue storage costs, in USD2008.

Age 1 is the first operating year.  A site profile follows site age; a per-well
profile follows the age of each injection-well cohort.  These are cash amounts,
not discounted values or annuities.  Original input workbooks are never changed.
"""

import math


BACKEND = "Back-End_Cost Items"
YEARS = 30


def compute_operations(subsite, engineering, site, workbook_cells):
    """Return basic annual O&M, non-electric $/t and four disjoint age profiles.

    Initial site/well investments and pump electricity remain in the existing
    engineering/site modules.  ``components`` records the underlying row,
    quantity, loading and complete age profile for reconciliation.
    """
    def value(sheet, cell):
        return workbook_cells[sheet].get(cell, {}).get("v", "")

    def number(sheet, cell):
        raw = value(sheet, cell)
        return 0.0 if raw in (None, "") else float(raw)

    def n(cell):
        return number("Activity_Inputs", cell)

    def loading(row):
        return (number(BACKEND, f"AM{row}") * number(BACKEND, f"AN{row}") *
                number(BACKEND, f"AP{row}") *
                (1.0 + number("Key_Inputs", "AA33")
                 if value(BACKEND, f"H{row}") == "Capital" else 1.0))

    def raw_unit(row):
        # Deliberately reject unresolved strings instead of reading AR/AY errors.
        return sum(number(BACKEND, f"{a}{row}") * number(BACKEND, f"{b}{row}")
                   for a, b in (("O", "Q"), ("T", "V"), ("Y", "AA"),
                                ("AD", "AF"), ("AI", "AK")))

    p = engineering["physical"]
    if str(value("Key_Inputs", "L37")).upper() != "OFF":
        raise ValueError("Water management must remain OFF for this storage-cost scenario")
    if int(engineering["assumptions"]["constants"]["injection_years"]) != YEARS:
        raise ValueError("Operating profiles require the agreed 30-year design")
    ages = list(range(1, YEARS + 1))
    profiles = {f"{scope}_extra_{kind}_age_usd2008": [0.0] * YEARS
                for scope in ("site", "well") for kind in ("capital", "opex")}
    components = []
    basic = {"site": 0.0, "well": 0.0}

    def record_basic(row, scope, amount, driver, source):
        basic[scope] += amount
        components.append({"backend_row": row, "item": value(BACKEND, f"D{row}"),
                           "bucket": f"{scope}_fixed_om", "classification": "Expense",
                           "driver": driver, "source": source,
                           "annual_cost_usd2008": amount})

    def add(row, scope, quantities, driver, base=None, factor=None, note=""):
        base = raw_unit(row) if base is None else base
        factor = loading(row) if factor is None else factor
        classification = value(BACKEND, f"H{row}")
        kind = "capital" if classification == "Capital" else "opex"
        key = f"{scope}_extra_{kind}_age_usd2008"
        costs = [base * factor * q for q in quantities]
        if any(not math.isfinite(x) or x < 0 for x in costs):
            raise ValueError(f"Invalid operating cost in backend row {row}")
        profiles[key] = [a + b for a, b in zip(profiles[key], costs)]
        components.append({"backend_row": row, "item": value(BACKEND, f"D{row}"),
                           "bucket": key, "classification": classification,
                           "driver": driver, "base_unit_usd2008": base,
                           "loading_factor": factor, "quantities_age": quantities,
                           "age_costs_usd2008": costs,
                           "source": f"{BACKEND}!O{row}:AQ{row}", "note": note})

    def events(row, first=None):
        period = max(1, int(number(BACKEND, f"AX{row}")))
        if first is None:
            first = period if value(BACKEND, f"J{row}") == "Monitoring" else 1
        return [int(age >= first and (age - first) % period == 0) for age in ages]

    def well_om(col, depth):
        return n(f"{col}183") + n(f"{col}185") * depth

    def mit(col, depth):
        return n(f"{col}203") + n(f"{col}209") * depth

    # Baseline O&M contains maintenance only. Pump electricity is already in V.
    surface = engineering["site_surface_components_direct_usd2008"]
    for row, item, cell in ((470, "boosting_pump", "D57"),
                            (472, "headers", "D78"), (476, "building", "D107"),
                            (478, "control", "D118"), (480, "access_road", "D131")):
        record_basic(row, "site", surface[item] * number("Surf Eq Cost", cell),
                     "Installed shared equipment direct cost times maintenance fraction",
                     f"Surf Eq Cost!{cell}; engineering direct equipment cost")
    for row, col, depth in ((425, "BK", p["well_depth_ft"]),
                            (432, "CA", 500.0), (434, "CC", 10.0)):
        record_basic(row, "well", well_om(col, depth),
                     "One injection well plus one associated well of each shallow type",
                     f"Activity_Inputs!{col}183+{col}185*depth_ft")
    record_basic(474, "well", p["branch_length_mile_per_well"] *
                 number("Surf Eq Cost", "D95"), "One onsite branch length in miles",
                 "Surf Eq Cost!D95; engineering branch_length_mile_per_well")

    # Source well schedules grow with the cumulative design plume. The first
    # batch was already purchased in F_r, as was design for the entire program.
    area_counts = site["assumptions"]["monitoring_counts_by_area"]
    monitoring = {}
    for kind, col, depth, maintenance_row in (
            ("above_seal", "BS", p["top_depth_ft"] - 200.0 + 50.0, 428),
            ("dual", "BW", p["well_depth_ft"], 430)):
        parts = [v for k, v in area_counts.items() if k.startswith(kind)]
        counts = [sum(max(v["initial"], (v["final"] * age + YEARS - 1) // YEARS)
                      for v in parts) for age in ages]
        new = [counts[0]] + [counts[i] - counts[i - 1] for i in range(1, YEARS)]
        monitoring[kind] = {"counts_age": counts, "new_wells_age": new,
                            "initial": counts[0], "depth_ft": depth}
        om = well_om(col, depth)
        record_basic(maintenance_row, "site", counts[0] * om,
                     f"Initial {kind} monitoring wells", f"Activity_Inputs!{col}187 at scenario depth")
        add(maintenance_row, "site", [c - counts[0] for c in counts],
            f"Later {kind} monitoring wells in service", om)
        for item in site["components"]:
            if not item.get("note", "").startswith(f"{kind} initial construction;"):
                continue
            if item["quantity"] == 0:
                continue
            add(item["backend_row"], "site", [0] + new[1:],
                f"Additional {kind} wells, excluding initial batch",
                item["base_unit_usd2008"], item["loading_factor"],
                "Reuse initial-well unit construction; upfront row48 design not repeated")

    all_years = [1] * YEARS
    # Site reporting, operating outreach, atmospheric monitoring and analysis.
    # AX84 is literally a four-year interval in the original backend; preserve
    # that schedule rather than silently interpret it as four events per year.
    for row in (84, 86, 102, 104, 112, 114, 251, 259, 293, 295, 297, 299):
        add(row, "site", events(row), "One site event at the source interval")
    add(94, "site", [1] + [0] * (YEARS - 1), "Operating public outreach, first year only")

    # The fixed design pressure footprint determines one old-well inventory.
    # Integer batches partition it once; they do not multiply the source's
    # fractional wells-per-event reference quantity again.
    old_well_count = site["physical"]["old_well_count"]
    remediation_events = events(88)
    event_count = sum(remediation_events)
    batch, remainder = divmod(old_well_count, event_count)
    remediation_quantities = []
    event_index = 0
    for flag in remediation_events:
        remediation_quantities.append(batch + int(event_index < remainder) if flag else 0)
        event_index += flag
    remediation_unit = sum(n(cell) for cell in ("J148", "L148", "N148"))
    add(88, "site", remediation_quantities,
        "Fixed analogue old deep wells partitioned into integer remediation batches",
        remediation_unit,
        note="One cleanout, logging and re-plug per old well; no repeat remediation")
    remaining_old_wells = []
    remaining = old_well_count
    for quantity in remediation_quantities:
        remaining -= quantity
        remaining_old_wells.append(remaining)

    # These updates have separate site and per-injection-well engineer hours.
    # Initial updates are already represented by permit rows77/78/81. The
    # source moves the first event to construction year6; retain later updates.
    engineer_rate = n("E10")
    for row, activity in ((106, 28), (108, 29), (110, 30)):
        site_base = ((n(f"AL{activity}") + n(f"AR{activity}")) * n("E9") +
                     (n(f"AM{activity}") + n(f"AP{activity}") + n(f"AS{activity}")) * engineer_rate +
                     (n(f"AN{activity}") + n(f"AT{activity}")) * n("E11") +
                     (n(f"AO{activity}") + n(f"AU{activity}")) * n("E12"))
        add(row, "site", events(row, 6), "Site-age plan update excluding initial permit update",
            site_base, note="AoR plan separate from old-well assessment and remediation")
        add(row, "well", events(row, 6), "Injection-well-age plan update",
            n(f"AQ{activity}") * engineer_rate)

    # 3-D acquisition area starts at a fraction of the final uncertain plume,
    # grows linearly over 30 years, and includes depth-dependent edge coverage.
    uncertainty_area = site["physical"]["plume_uncertainty_area_mi2"]
    first_fraction = number("Key_Inputs", "AD18")
    depth_mile = p["bottom_depth_ft"] / 5280.0
    edge = depth_mile * math.tan(math.radians(number("Key_Inputs", "AC15")))
    seismic_area = [math.pi * (math.sqrt(uncertainty_area *
                    (first_fraction + (1 - first_fraction) * (age - 1) / (YEARS - 1)) /
                    math.pi) + edge) ** 2 for age in ages]
    add(190, "site", [q * flag for q, flag in zip(seismic_area, events(190))],
        "Operating 3-D acquisition area in square miles; Plume&Well Schedule rows80/81")

    # Deep-monitoring wells belong to the fixed site program. Their scheduled
    # MIT/VSP follows site age, multiplying all wells in service that year.
    for kind, sample_row, pressure_row, mit_row, vsp_row, col in (
            ("above_seal", 123, 132, 441, 450, "BS"),
            ("dual", 121, 130, 443, 452, "BW")):
        mon = monitoring[kind]
        for row in (sample_row, pressure_row):
            add(row, "site", mon["counts_age"], f"All operating {kind} monitoring wells")
        for row, unit in ((mit_row, mit(col, mon["depth_ft"])),
                          (vsp_row, n(f"{col}227"))):
            event_pattern = events(row)
            quantities = [count * event for count, event in
                          zip(mon["counts_age"], event_pattern)]
            add(row, "site", quantities, f"{kind} wells in service at the site testing date", unit)

    # Costs attached to a new injection well include its shallow monitoring
    # wells. They are charged per cohort, independently of site design counts.
    for row in (125, 134, 288, 289, 291, 292):
        add(row, "well", events(row), "Per injection-well cohort (including associated shallow wells)")
    add(438, "well", events(438), "Per injection well; selected annual MIT",
        mit("BK", p["well_depth_ft"]))
    add(447, "well", events(447), "Per injection well; VSP option", n("BK227"))
    flux = n("AL65") * n("AL64") * (10.0 + n("AO64"))
    add(138, "well", all_years, "20 samples/well/occasion, 4 occasions/year", flux,
        note="Replaces PriPerOccu and SampCntYr text placeholders; no built-well multiplier")
    tracer_daily_t = p["design_annual_t"] / (365.0 * p["design_active_wells"])
    tracer_base = (n("AN54") * n("AL54") + n("AO54") * n("AL54") * n("AM54") +
                   n("AP54") + n("AQ54") * tracer_daily_t)
    add(127, "well", all_years, "One day of average design flow per active injection well",
        tracer_base, note="Activity_Inputs!AI54 says Cost for one day; resolve mCO2Avedywell proxy")

    variable_components = [
        {"backend_row": 89, "item": "Injection fee for lease holders",
         "source": "Key_Inputs!AA35; Activity_Inputs!AL8/AO8",
         "unit_usd2008_per_t": n("AL8") if n("AO8") > 0 else 0.0},
        {"backend_row": 90, "item": "Long-term Stewardship Trust Fund (State)",
         "source": "Key_Inputs!AA36; Activity_Inputs!AL9/AO9",
         "unit_usd2008_per_t": n("AL9") if n("AO9") > 0 else 0.0},
        {"backend_row": 91, "item": "Operational Oversight Fund (State)",
         "source": "Key_Inputs!AA37; Activity_Inputs!AL10/AO10",
         "unit_usd2008_per_t": n("AL10") if n("AO10") > 0 else 0.0},
        {"backend_row": 495, "item": "Emergency and Remedial Response insurance",
         "source": "Fin_Resp_Inputs!K40; FinMod_Main!F272/274",
         "unit_usd2008_per_t": number("Fin_Resp_Inputs", "K40")},
    ]
    # Optional techniques are explicitly off/zero in the selected workbook.
    # Record that boundary and fail if a changed workbook enables an unmodeled
    # option: silently dropping a newly selected technology would underprice it.
    disabled_rows = [142, 150, 154, 158, 162, 166, 174, 178, 182, 186,
                     194, 198, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
                     243, 247, 263, 267, 271, 275, 279, 282, 285, 448,
                     454, 455, 456, 457]
    for row in disabled_rows:
        if loading(row) != 0 and raw_unit(row) != 0:
            raise ValueError(f"Workbook row {row} enables an option outside this scenario")

    # M_r/M_w contain the entire constant annual Expense baseline, including
    # sampling, reports, analyses and yearly MIT. Extra profiles contain only
    # events or growth above this baseline. Split each component once so that
    # the delivered M and four profiles remain mutually exclusive.
    annual_records = []
    for component in components:
        costs = component.get("age_costs_usd2008")
        if costs is None or component["classification"] != "Expense":
            continue
        baseline = min(costs)
        if baseline <= 0:
            continue
        key = component["bucket"]
        scope = key.split("_", 1)[0]
        basic[scope] += baseline
        profiles[key] = [x - baseline for x in profiles[key]]
        component["age_costs_usd2008"] = [x - baseline for x in costs]
        component["annual_baseline_removed_usd2008"] = baseline
        annual_records.append({
            "backend_row": component["backend_row"], "item": component["item"],
            "bucket": f"{scope}_fixed_om", "classification": "Expense",
            "driver": "Constant annual baseline: " + component["driver"],
            "source": component["source"], "annual_cost_usd2008": baseline,
            "note": "Removed exactly once from this row's age profile",
        })
    components.extend(annual_records)
    for key in profiles:
        profiles[key] = [math.fsum(c["age_costs_usd2008"][i] for c in components
                                  if c["bucket"] == key) for i in range(YEARS)]

    return {
        "site_fixed_om_usd2008_per_year": basic["site"],
        "well_fixed_om_usd2008_per_year": basic["well"],
        "nonenergy_variable_usd2008_per_t": sum(c["unit_usd2008_per_t"] for c in variable_components),
        **profiles, "components": components, "variable_components": variable_components,
        "monitoring_schedule": monitoring,
        "old_well_remediation": {
            "count": old_well_count,
            "unit_usd2008": remediation_unit * loading(88),
            "quantities_by_age": remediation_quantities,
            "remaining_by_age": remaining_old_wells,
        },
        "seismic_area_age_mi2": seismic_area,
        "assumptions": {
            "price_basis": "USD2008; READ_ME_FIRST!C64; no discounting or annualization",
            "ages": "1..30; site age for site profiles, injection-well cohort age for well profiles",
            "basic_om_boundary": "All constant annual Expense: equipment/well maintenance, baseline monitoring, reporting, analyses and annual injection MIT. Only increments/events remain in extra profiles.",
            "later_monitoring": "Design plume growth; later construction, maintenance and sampling increments in site extras; deep-monitoring tests follow source site-age schedule",
            "initial_boundary": "Upfront Capital and Expense are separated by the site/engineering modules; initial monitor wells and row48 full design are not repeated in operating profiles",
            "plan_boundary": "Rows106/108/110 initial updates overlap permit rows77/78/81; only ages6/11/16/21/26 added",
            "plan_scaling": "Activity_Inputs!AP24 labels site hours and AQ24 labels per-injection-well hours; AQ28:AQ30 are charged to each well, whereas the legacy backend simply adds those hours once",
            "eddy_boundary": "Row255 remains excluded under the existing overlap assumption with row281; source notes do not prove asset equivalence, so this boundary remains unresolved",
            "excluded_startup_rows": [255],
            "excluded_rows": [116, 146, 170, 459, 467, 468],
            "excluded_scope": ["post-injection and closure handled separately", "optional old-well aerial surveys", "water management", "external feeder pipeline charged to network"],
            "old_well_remediation": "Fixed design-pressure footprint times NETL deep-well density, rounded down; six integer batches at ages1/6/11/16/21/26 remediate every analogue old well once. Any outstanding inventory at early closure is returned for terminal settlement.",
            "post_injection_plan_report": "Row114 occurs during operations every five years; its title does not make it a post-injection cash flow",
            "disabled_or_zero_rows": disabled_rows,
            "zero_quantity_rows": [119, 128, 426, 439, 448],
            "fee_localization": "0.25+0.07+0.01+0.75 USD2008/t is the selected NETL US analogue; not a China statutory fee",
            "tracer_flow_proxy": "One average-flow day per year per built injection well, using design flow / active design wells / 365; text placeholder has no usable cached numeric value",
            "original_classification": "Original Capital/Expense and row-specific process/project loading preserved",
        },
    }
