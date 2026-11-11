"""Initial common storage-site costs from the NETL activity inventory.

All returned costs are undiscounted, unannualized USD2008.  Activities in the
one-year screening, three-year characterization and two-year permitting
stages are summed as actual pre-operation events. Capital and Expense remain
separate; Expense cash flows retain their year relative to first injection.
No operating maintenance,
recurring operating monitoring, future monitoring-well additions or closure
costs are included.  workbook_cells is the extracted sheet/cell/v dictionary.
"""

import math

from storage_engineering import drilling_cost_usd2008


BACKEND = "Back-End_Cost Items"


def compute_site(subsite, engineering, workbook_cells):
    """Return common site costs and additions charged once per injection well."""
    def value(sheet, cell):
        item = workbook_cells[sheet].get(cell, {})
        return item.get("v", "")

    def number(sheet, cell):
        raw = value(sheet, cell)
        # Empty optional component slots are zero; nonnumeric placeholders fail.
        return 0.0 if raw in ("", None) else float(raw)

    p = engineering["physical"]
    constants = engineering["assumptions"]["constants"]
    capital_factor = 1.0 + constants["capital_contingency_fraction"]
    records = []

    characterization_counts = []
    for cell in ("L26", "L27", "L28"):
        count = number("Key_Inputs", cell)
        if not math.isfinite(count) or count < 1 or not count.is_integer():
            raise ValueError(f"Key_Inputs!{cell} must be a positive integer")
        characterization_counts.append(int(count))
    candidate_count, lines_per_candidate, successful_site_strat_wells = characterization_counts
    strat_well_count = candidate_count - 1 + successful_site_strat_wells
    if any(number("Key_Inputs", cell) != 0
           for cell in ("L29", "L31", "M31", "N31", "L32", "M32", "N32")):
        raise ValueError("Stratigraphic test well conversion is outside the selected design")

    uncertainty = number("Plume&Well Schedule", "D28")
    area_unc = p["plume_area_mi2"] * uncertainty
    pressure_multiplier = number("Plume&Well Schedule", "D29")
    pressure_ring_area = area_unc * (pressure_multiplier - 1.0)
    pressure_area = area_unc * pressure_multiplier
    old_well_count = math.floor(pressure_area * number("Key_Inputs", "L41"))
    monitoring_counts = {}
    # Plume&Well Schedule rows 58:67, 132:161: the first operating year has
    # 1 / injection_years of the maximum plume.  Minimums and caps apply to
    # the end-of-injection count before scaling to the first-year count.
    for name, area, col, start_row in (
            ("above_seal_plume", area_unc, "M", 7),
            ("above_seal_pressure", pressure_ring_area, "N", 7),
            ("dual_plume", area_unc, "M", 19),
            ("dual_pressure", pressure_ring_area, "N", 19)):
        spacing = number("Key_Inputs", f"{col}{start_row}")
        minimum_start = number("Key_Inputs", f"{col}{start_row + 1}")
        minimum_end = number("Key_Inputs", f"{col}{start_row + 2}")
        maximum = number("Key_Inputs", f"{col}{start_row + 3}")
        final_count = min(maximum, max(math.ceil(area / spacing), minimum_start, minimum_end))
        initial_count = (0 if final_count == 0 else
                         max(minimum_start, math.ceil(final_count / constants["injection_years"])))
        monitoring_counts[name] = {"initial": int(initial_count), "final": int(final_count)}
    above_count = sum(v["initial"] for k, v in monitoring_counts.items() if k.startswith("above"))
    dual_count = sum(v["initial"] for k, v in monitoring_counts.items() if k.startswith("dual"))
    planned_monitoring_count = sum(v["final"] for v in monitoring_counts.values())

    def add(row, quantity=1.0, base_override=None, bucket="site", note="",
            pre_offsets=None):
        # Rebuild the inspected cost-component arithmetic, never read AR/AY.
        if base_override is None:
            base = sum(number(BACKEND, f"{a}{row}") *
                       number(BACKEND, f"{b}{row}")
                       for a, b in (("O", "Q"), ("T", "V"), ("Y", "AA"),
                                    ("AD", "AF"), ("AI", "AK")))
        else:
            base = base_override
        factor = (number(BACKEND, f"AM{row}") *
                  number(BACKEND, f"AN{row}") *
                  number(BACKEND, f"AP{row}") *
                  (capital_factor if value(BACKEND, f"H{row}") == "Capital" else 1.0))
        amount = base * factor * quantity
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"Invalid initial cost in backend row {row}")
        classification = value(BACKEND, f"H{row}")
        offsets = [0] if pre_offsets is None else pre_offsets
        records.append({
            "bucket": bucket, "backend_row": row,
            "classification": classification,
            "item": value(BACKEND, f"D{row}"),
            "quantity": quantity, "base_unit_usd2008": base,
            "loading_factor": factor, "cost_usd2008": amount,
            "source": f"{BACKEND}!O{row}:AQ{row}", "note": note,
            "pre_opex_by_offset": ({offset: amount / len(offsets) for offset in offsets}
                                   if classification == "Expense" else {}),
        })

    # These are site-level materials, labor, FEED and permit preparation.
    # Rows 51/52/79/80 prepare permit plans, not actual future closure work.
    for row in range(6, 13):
        add(row)
    per_well_rows = {33, 35, 37, 39, 47}
    for row in list(range(13, 20)) + list(range(22, 55)):
        if row in per_well_rows:
            add(row, 3, bucket="per_injection_well",
                note="Three pre-operation design events, charged once on adding a well")
        elif row == 48:
            add(row, 3 * planned_monitoring_count,
                note="Three upfront design years for all planned deep monitoring wells; source row48")
        elif row == 42:
            add(row, 3 * old_well_count,
                note="Three characterization reviews per analogue old deep well")
        elif row != 43:
            add(row, 3)
    for row in (55, 56, 57):
        add(row)
    for row in range(61, 74):
        add(row)
    add(74, bucket="per_injection_well", note="Initial injection-well permit")
    for row in range(76, 84):
        add(row, 2)
    add(92, pre_offsets=[-5], note="Source project year 2; first injection in year 7")
    add(93, pre_offsets=[-2], note="Source project year 5; first injection in year 7")
    for row in (96, 97, 98, 99, 100, 101, 200, 281):
        add(row)

    # Full design plume is used for pre-injection surveys, as in the source's
    # pre-operation Plume&Well Schedule rows 78/81/83, not resource-block area.
    depth_mile = p["bottom_depth_ft"] / 5280.0
    radius_unc = math.sqrt(area_unc / math.pi)
    angle3 = math.radians(number("Key_Inputs", "AC15"))
    angle2 = math.radians(number("Key_Inputs", "AC16"))
    area3 = math.pi * (radius_unc + depth_mile * math.tan(angle3)) ** 2
    length2 = 2.0 * (radius_unc + depth_mile * math.tan(angle2))
    for row in (20, 21):
        add(row, area_unc, number(BACKEND, f"O{row}"),
            note="Resolve ACO2plUnc text to design plume uncertainty area; one planning event")
    for row in (58, 59, 60):
        add(row, area3, note="Initial land/lease amount over seismic acquisition area")
    for row in (140, 144):
        add(row, 3, note="Three pre-operation baseline survey events")
    for row in (164, 168):
        add(row, 3 * area_unc, note="Three baseline surveys times plume uncertainty mi2")
    add(188, area3, note="One pre-operation 3D survey")
    # Each opened subsite funds its own candidate campaign. All candidates use
    # this subsite's single-line length; only the successful site receives 3D.
    add(192, length2, number(BACKEND, "O192") * candidate_count * lines_per_candidate,
        note=f"{candidate_count} candidates, {lines_per_candidate} 2D lines each, one survey")

    # Flux-chamber baseline is proportional to injection-well count.  Each
    # occurrence has 20 samples/well, $10 collection + $25 analysis per sample.
    flux_base = (number("Activity_Inputs", "AL65") *
                 number("Activity_Inputs", "AL64") *
                 (10.0 + number("Activity_Inputs", "AO64")))
    add(136, 3, flux_base, bucket="per_injection_well",
        note="Three pre-operation baseline years; 20 samples per well, 4 events/year",
        pre_offsets=[-5, -4, -3])
    add(458, strat_well_count,
        note=f"All {strat_well_count} stratigraphic test wells abandoned in source project year 4",
        pre_offsets=[-3])

    # The reference case starts with one above-seal and two dual wells; larger
    # projects may need more in their first year.  Later additions are excluded.
    well_specs = [
        ("BI", "strat", p["well_depth_ft"] + 500.0, strat_well_count, "site",
         [302, 311, 324, 337, 349, 358, 366, 378, 385]),
        ("BS", "above_seal", p["top_depth_ft"] - 200.0 + 50.0, above_count, "site",
         [305, 316, 329, 342, 352, 361, 371, 381, 388, 398, 409, 419]),
        ("BW", "dual", p["well_depth_ft"], dual_count, "site",
         [306, 318, 331, 344, 353, 362, 373, 382, 389, 400, 411, 421]),
        ("CA", "groundwater", 500.0, 1, "per_injection_well",
         [307, 320, 333, 346, 354, 363, 375, None, 390, 402]),
        ("CC", "vadose", 10.0, 1, "per_injection_well",
         [308, 321, 334, None, 355, None, None, None, 391]),
    ]
    activity_rows = [28, 37, 40, 69, 85, 97, 107, 115, 128, 136, 151, 162]
    for col, kind, depth, count, bucket, backend_rows in well_specs:
        for source_row, backend_row in zip(activity_rows, backend_rows):
            if backend_row is None:
                continue
            if source_row == 37 and kind not in {"groundwater", "vadose"}:
                base = drilling_cost_usd2008(depth, constants)
            elif source_row == 69:
                base = (number("Activity_Inputs", f"{col}64") * depth +
                        number("Activity_Inputs", f"{col}68"))
            elif source_row == 151 and kind == "dual":
                base = (number("Activity_Inputs", f"{col}139") *
                        number("Activity_Inputs", f"{col}7") +
                        number("Activity_Inputs", f"{col}147")) * depth
            else:
                base = number("Activity_Inputs", f"{col}{source_row}")
            add(backend_row, count, base, bucket,
                f"{kind} initial construction; {depth:g} ft; Activity_Inputs!{col}{source_row}")

    capital_records = [r for r in records if r["classification"] == "Capital"]
    site_total = sum(r["cost_usd2008"] for r in capital_records if r["bucket"] == "site")
    well_total = sum(r["cost_usd2008"] for r in capital_records
                     if r["bucket"] == "per_injection_well")
    pre_opex = {"site": {}, "per_injection_well": {}}
    for record in records:
        for offset, amount in record["pre_opex_by_offset"].items():
            bucket = pre_opex[record["bucket"]]
            bucket[offset] = bucket.get(offset, 0.0) + amount
    return {
        "site_initial_usd2008": site_total,
        "per_injection_well_initial_usd2008": well_total,
        "site_pre_opex_by_offset": pre_opex["site"],
        "well_pre_opex_by_offset": pre_opex["per_injection_well"],
        "site_components_usd2008": {
            f"row_{r['backend_row']}: {r['item']}": r["cost_usd2008"]
            for r in capital_records if r["bucket"] == "site"},
        "per_well_components_usd2008": {
            f"row_{r['backend_row']}: {r['item']}": r["cost_usd2008"]
            for r in capital_records if r["bucket"] == "per_injection_well"},
        "components": records,
        "physical": {"plume_uncertainty_area_mi2": area_unc,
                     "pressure_area_mi2": pressure_area,
                     "old_well_count": old_well_count,
                     "seismic_3d_area_mi2": area3,
                     "seismic_2d_length_mile": length2},
        "assumptions": {
            "screening_candidates": candidate_count,
            "seismic_2d_lines_per_candidate": lines_per_candidate,
            "successful_site_strat_test_wells": successful_site_strat_wells,
            "strat_test_wells": strat_well_count,
            "scenario_choice": "Each opened subsite independently funds all characterization candidates; NETL initial monitoring options",
            "candidate_cost_basis": "Key_Inputs!L26/L27/L28: candidate count, 2D lines per candidate, successful-site strat wells. Each unsuccessful candidate has one strat well. All candidates use this subsite's depth and 2D line length. Costs are future opening commitments, not shared regional or sunk costs.",
            "initial_above_seal_monitoring_wells": above_count,
            "initial_dual_monitoring_wells": dual_count,
            "monitoring_count_basis": "Source spacing/minimum/cap rules, scaled to first injection year",
            "monitoring_counts_by_area": monitoring_counts,
            "upfront_monitoring_design_well_count": planned_monitoring_count,
            "pressure_front_area_multiplier": pressure_multiplier,
            "groundwater_wells_per_injection_well": 1,
            "vadose_wells_per_injection_well": 1,
            "screening_years": 1, "characterization_years": 3,
            "permitting_years": 2, "plume_uncertainty_multiplier": uncertainty,
            "old_well_count_basis": "floor(uncertain design plume mi2 * pressure multiplier * Key_Inputs!L41); NETL density analogue, not a local well inventory",
            "old_well_assessment": "Row42: three characterization reviews per analogue old deep well; row43 shallow old-well activity remains zero in source",
            "old_well_rows_excluded": [43],
            "initial_classification": "Initial totals and component dictionaries contain Capital only; all Expense is returned separately by offset from injection/build year",
            "pre_operation_timing": "Source first injection is year7; upfront Expense keeps source offsets. Initial monitoring-well completion Expense occurs at offset0; later additions remain in operating profiles.",
            "strat_abandonment": f"Row458: all {strat_well_count} stratigraphic test wells, once at offset -3, Expense; no well conversion",
            "area_cost_boundaries": {
                "rows_20_21": "Two separate upfront survey-planning activities, each rate times uncertainty area once",
                "rows_58_60": "Lease/land allowance over 3D acquisition area once; row59 alone is nonzero",
                "rows_164_168": "Aerial and magnetic acquisition over uncertainty area, three pre-operation events",
                "row_188": "One 3D acquisition; distinct from planning and land rights",
            },
            "eddy_initial_facilities": "Row281: Activity_Inputs!AK138*AM138 = 70000*5, plus 40 geologist hours; once",
            "closure_plan_boundary": "Initial permit-plan preparation and pre-operation stratigraphic-test-well abandonment included; injection/monitoring closure handled separately",
            "labor_rate_basis": "Source loaded hourly rates already include G&A",
            "excluded": ["fixed O&M", "per-well O&M", "operating monitoring",
                         "future monitoring-well additions", "injection/monitoring-well plugging",
                         "post-injection monitoring"],
        },
    }
