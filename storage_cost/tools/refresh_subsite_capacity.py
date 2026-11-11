"""Refresh annual allocations without repartitioning or changing total resources."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
ANNUAL_BASIS = "original_basin_maximum_with_mainland_area_clipping"
ANNUAL_NOTE = "CO2_EWR(tonne)对应原盆地最大年注入量；先乘大陆交集面积/原盆地面积，再按子场面积/大陆交集面积分配，即CO2_EWR(tonne)*子场面积/原盆地面积。合计对账大陆折减量，差异仅来自已验收的边界面积数值误差。"


def main():
    directory = ROOT / "sink_subsite/subsite_result"
    csv_path = directory / "sink_subsite.csv"
    geo_path = directory / "sink_subsite.geojson"
    audit_path = directory / "sink_subsite_audit.json"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    geojson = json.loads(geo_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    workbook = openpyxl.load_workbook(ROOT / "sink.xlsx", read_only=True, data_only=True)
    try:
        source_rows = workbook["sink"].iter_rows(values_only=True)
        headers = next(source_rows)
        source = {row[0]: dict(zip(headers, row)) for row in source_rows if row[0]}
    finally:
        workbook.close()

    by_id = {row["Subsite_ID"]: row for row in rows}
    features = {feature["properties"]["Subsite_ID"]: feature for feature in geojson["features"]}
    if (len(by_id) != len(rows) or len(features) != len(geojson["features"])
            or set(by_id) != set(features) or len(rows) != audit["Subsite_Count"]):
        raise ValueError("CSV, GeoJSON and audit must cover identical unique subsites")
    groups = defaultdict(list)
    for row in rows:
        sid = row["Sink_ID"]
        area = float(row["Area_km2"])
        retained = float(row["Parent_Retained_Area_km2"])
        original = float(row["Parent_Original_Area_km2"])
        annual = float(source[sid]["CO2_EWR(tonne)"])
        if any(not math.isfinite(value) or value <= 0 for value in (area, retained, original, annual)):
            raise ValueError(f"{row['Subsite_ID']}: invalid area or parent annual maximum")
        properties = features[row["Subsite_ID"]]["properties"]
        if (float(properties["Area_km2"]) != area
                or float(properties["Parent_Retained_Area_km2"]) != retained
                or float(properties["Parent_Original_Area_km2"]) != original):
            raise ValueError(f"{row['Subsite_ID']}: CSV and GeoJSON areas differ")
        updates = {
            "Allocated_Annual_Quota_t": annual * area / original,
            "Annual_Allocation_Area_Share": area / retained,
            "Annual_Allocation_Basis": ANNUAL_BASIS,
        }
        row.update(updates)
        properties.update(updates)
        groups[sid].append(row)

    summaries = {basin["Sink_ID"]: basin for basin in audit["Basins"]}
    if set(summaries) != set(groups):
        raise ValueError("Basin IDs differ between subsite data and audit")
    for sid, members in groups.items():
        retained = float(summaries[sid]["Retained_Area_km2"])
        original = float(summaries[sid]["Original_Area_km2"])
        if any(float(row["Parent_Retained_Area_km2"]) != retained for row in members):
            raise ValueError(f"{sid}: inconsistent parent intersection area")
        if any(float(row["Parent_Original_Area_km2"]) != original for row in members):
            raise ValueError(f"{sid}: inconsistent original basin area")
        area_difference = sum(float(row["Area_km2"]) for row in members) - retained
        if abs(area_difference) > audit["Coverage_Tolerance_km2"]:
            raise ValueError(f"{sid}: subsite areas do not cover the parent intersection")
        parent_annual = float(source[sid]["CO2_EWR(tonne)"])
        mainland_annual = parent_annual * retained / original
        allocated = sum(row["Allocated_Annual_Quota_t"] for row in members)
        summaries[sid].update({
            "Allocated_Annual_Quota_t": allocated,
            "Parent_Max_Annual_Injection_t": parent_annual,
            "Mainland_Max_Annual_Injection_t": mainland_annual,
            "Annual_Quota_Sum_Difference_t": allocated - mainland_annual,
        })
    audit["Annual_Allocation_Basis"] = ANNUAL_BASIS
    audit["Annual_Allocation_Note"] = ANNUAL_NOTE
    for field in ("Annual_Allocation_Area_Share", "Annual_Allocation_Basis"):
        if field not in fields:
            fields.append(field)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    geo_path.write_text(json.dumps(geojson, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Refreshed annual allocations for {len(rows)} subsites in {len(groups)} basins.")


if __name__ == "__main__":
    main()
