"""Closure and PISC cash flows at a common project end, in raw USD2008.

Offset zero is the final injection year. Offsets 1..L are post-injection
years. Site rows depend on operating age at shutdown; each injection well
has one groundwater and one vadose monitoring well through the PISC end.
"""

import math


BACKEND = "Back-End_Cost Items"
YEARS = 30


def compute_lifecycle(engineering, site, operations, workbook_cells):
    """Return undiscounted terminal cash flows; never annualize capital."""
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
        return sum(number(BACKEND, f"{a}{row}") * number(BACKEND, f"{b}{row}")
                   for a, b in (("O", "Q"), ("T", "V"), ("Y", "AA"),
                                ("AD", "AF"), ("AI", "AK")))

    if int(engineering["assumptions"]["constants"]["injection_years"]) != YEARS:
        raise ValueError("Terminal profiles require the 30-year operating design")
    duration = number("Key_Inputs", "C44")
    if not math.isfinite(duration) or duration < 0 or duration != int(duration):
        raise ValueError("PISC duration must be a nonnegative integer")
    pisc_years = int(duration)
    source_start = int(number("Key_Inputs", "D44"))
    profiles = {
        "site_terminal_capital_by_age": [[0.0] * (pisc_years + 1) for _ in range(YEARS)],
        "site_terminal_opex_by_age": [[0.0] * (pisc_years + 1) for _ in range(YEARS)],
        "well_terminal_capital": [0.0] * (pisc_years + 1),
        "well_terminal_opex": [0.0] * (pisc_years + 1),
    }
    components = []
    covered = set()

    def events(row):
        period = max(1, int(number(BACKEND, f"AX{row}")))
        begin = int(number(BACKEND, f"AV{row}"))
        end = int(number(BACKEND, f"AW{row}"))
        first = begin + (period - 1 if value(BACKEND, f"J{row}") == "Monitoring" else 0)
        return [0] + [int(begin <= source_start + offset - 1 <= end and
                          (source_start + offset - 1 - first) % period == 0)
                      for offset in range(1, pisc_years + 1)]

    def at_offset(offset):
        return [int(i == offset) for i in range(pisc_years + 1)]

    def add(row, scope, quantities, driver, base=None, flags=None, factor=None):
        base = raw_unit(row) if base is None else base
        factor = loading(row) if factor is None else factor
        flags = events(row) if flags is None else flags
        classification = value(BACKEND, f"H{row}")
        kind = "capital" if classification == "Capital" else "opex"
        key = f"{scope}_terminal_{kind}" + ("_by_age" if scope == "site" else "")
        quantities = list(quantities)
        if len(quantities) != (YEARS if scope == "site" else 1):
            raise ValueError(f"Invalid terminal quantities for row {row}")
        if any(not math.isfinite(x) or x < 0 for x in [base, factor, *quantities]):
            raise ValueError(f"Invalid terminal cost for row {row}")
        for age, quantity in enumerate(quantities):
            target = profiles[key][age] if scope == "site" else profiles[key]
            for offset, flag in enumerate(flags):
                target[offset] += base * factor * quantity * flag
        components.append({
            "backend_row": row, "item": value(BACKEND, f"D{row}"),
            "bucket": key, "classification": classification, "driver": driver,
            "base_unit_usd2008": base, "loading_factor": factor,
            "quantity_by_end_age": quantities, "event_flags_by_offset": flags,
            "source": f"{BACKEND}!O{row}:AX{row}",
        })
        covered.add(row)

    one_site = [1] * YEARS
    p = engineering["physical"]
    # Shutdown reports are per site; injection-well plugging is per built well.
    add(116, "site", one_site, "Injection-well plugging report once at project end",
        flags=at_offset(0))
    add(459, "well", [1], "One injection well plugged at the common project end",
        base=n("BK304") * p["well_depth_ft"] + n("BK312"), flags=at_offset(0))

    # Site reporting, outreach, atmospheric monitoring and reservoir analysis.
    for row in (85, 87, 95, 103, 105, 115, 252, 256, 294, 296, 298, 300):
        add(row, "site", one_site, "One site event on the source PISC schedule")
    for row in (117, 118):
        add(row, "site", one_site, "One site closure report at the PISC end",
            flags=at_offset(pisc_years))
    add(191, "site", operations["seismic_area_age_mi2"],
        "3-D acquisition area frozen at the site's operating age at shutdown")

    # Deep wells are only those built by shutdown; no tail-period expansion.
    for kind, col, depth, sample, pressure, maintenance, mit, vsp, plugging in (
            ("above_seal", "BS", p["top_depth_ft"] - 150.0, 124, 133, 429, 442, 451, 461),
            ("dual", "BW", p["well_depth_ft"], 122, 131, 431, 444, 453, 462)):
        counts = operations["monitoring_schedule"][kind]["counts_age"]
        for row in (sample, pressure):
            add(row, "site", counts, f"Existing {kind} wells at shutdown")
        for row, base in ((maintenance, n(f"{col}183") + n(f"{col}185") * depth),
                          (mit, n(f"{col}203") + n(f"{col}209") * depth),
                          (vsp, n(f"{col}227"))):
            add(row, "site", counts, f"Existing {kind} wells on the source PISC schedule", base)
        add(plugging, "site", counts, f"Existing {kind} wells plugged at PISC end",
            n(f"{col}304") * depth + n(f"{col}312"), flags=at_offset(pisc_years))

    # One of each shallow monitoring type remains for every built injection well.
    for row in (126, 135):
        add(row, "well", [1], "One associated shallow monitoring well")
    for col, depth, maintenance, plugging in (("CA", 500.0, 433, 463),
                                              ("CC", 10.0, 435, 464)):
        add(maintenance, "well", [1], "One associated shallow well during PISC",
            n(f"{col}183") + n(f"{col}185") * depth)
        add(plugging, "well", [1], "One associated shallow well plugged at PISC end",
            n(f"{col}304") * depth + n(f"{col}312"), flags=at_offset(pisc_years))
    add(139, "well", [1], "20 flux samples per occasion, four occasions per year per injection well",
        n("AL65") * n("AL64") * (10.0 + n("AO64")))

    remediation = operations["old_well_remediation"]
    add(88, "site", remediation["remaining_by_age"],
        "Complete outstanding old-well remediation at the common project end",
        remediation["unit_usd2008"], flags=at_offset(0), factor=1.0)

    # In-reservoir-only wells are absent from this selected engineering design.
    zero_quantity_rows = [120, 129, 427, 440, 449, 460]
    if number("Plume&Well Schedule", "D127") != 0:
        raise ValueError("In-reservoir-only PISC wells are outside the selected design")
    disabled_rows = []
    for row in range(6, 501):
        if value(BACKEND, f"B{row}") != "PISC and Site Closure":
            continue
        if row in covered or row in zero_quantity_rows:
            continue
        if loading(row) != 0 and raw_unit(row) != 0:
            raise ValueError(f"Workbook PISC row {row} enables an option outside this scenario")
        disabled_rows.append(row)

    return {
        **profiles, "pisc_years": pisc_years, "closure_basis": "common_project_end",
        "components": components,
        "assumptions": {
            "price_basis": "Raw USD2008 cash amounts; no discounting, annualization or CRF",
            "offsets": "0 = final injection year; 1..pisc_years = post-injection years",
            "end_ages": "Site operating ages 1..30; downstream must reject ages outside this range",
            "monitoring_extent": "Deep monitoring wells and 3-D area freeze at operating age at shutdown",
            "well_basis": "Each built injection well closes at offset0; its groundwater/vadose wells remain until PISC end",
            "pisc_equipment": "Row256 is included as the source PISC first-period equipment allocation; no claim of independently verified replacement need",
            "remediation": "All fixed-design old-well remediation not yet paid in operations is settled at offset0",
            "disabled_or_zero_rows": disabled_rows, "zero_quantity_rows": zero_quantity_rows,
            "source_pisc_start_year": source_start,
        },
    }
