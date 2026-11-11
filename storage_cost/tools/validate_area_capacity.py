"""Reconcile saved capacities, areas and coordinates with source data."""

import csv
import json
import math
from pathlib import Path
import sys

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
import read


def load_rows(path, key):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row[key]: row for row in rows}
    assert len(rows) == len(result)
    return result


def main():
    output = ROOT / "storage_cost_results"
    workbook = openpyxl.load_workbook(ROOT / "sink.xlsx", read_only=True, data_only=True)
    source_rows = workbook["sink"].iter_rows(values_only=True)
    headers = next(source_rows)
    sources = {row[0]: dict(zip(headers, row)) for row in source_rows if row[0]}
    workbook.close()
    subsites = load_rows(ROOT / "sink_subsite/subsite_result/sink_subsite.csv", "Subsite_ID")
    basins = load_rows(output / "basin_capacity.csv", "Sink_ID")
    capacities = load_rows(output / "subsite_capacity.csv", "Subsite_ID")
    costs = load_rows(output / "sink_subsite_cost.csv", "Subsite_ID")
    assert set(sources) == set(basins)
    assert set(subsites) == set(capacities) == set(costs)
    columns = list(next(iter(costs.values())))
    assumptions = json.loads((output / "assumptions.json").read_text(encoding="utf-8"))
    assert columns == assumptions["model_csv_columns"]
    area_index = columns.index("Area_km2")
    assert columns[area_index + 1:area_index + 3] == ["Longitude", "Latitude"]
    largest_error = 0.0
    for sid, subsite in subsites.items():
        assert subsite["Point_Method"] == "interior_label_point"
        for field in ("Area_km2", "Longitude", "Latitude"):
            value = float(costs[sid][field])
            assert math.isfinite(value) and value == float(subsite[field])
        assert -180 <= float(costs[sid]["Longitude"]) <= 180
        assert -90 <= float(costs[sid]["Latitude"]) <= 90
        basin_id = subsite["Sink_ID"]
        parent = sources[basin_id]
        annual = parent["CO2_EWR(tonne)"] * float(subsite["Area_km2"]) / float(subsite["Parent_Original_Area_km2"])
        capacity = capacities[sid]
        actual = float(capacity["Subsite_Max_Annual_Injection_t"])
        largest_error = max(largest_error, abs(actual - annual))
        assert math.isclose(actual, annual, rel_tol=1e-12)
        assert math.isclose(float(subsite["Allocated_Annual_Quota_t"]), annual, rel_tol=1e-12)
        well = float(capacity["Well_Annual_Capacity_t"])
        assert well == float(basins[basin_id]["Well_Annual_Capacity_t"])
        assert well == float(costs[sid]["Well_Annual_Capacity_t"])
        count = int(capacity["Max_Well_Num"])
        assert count == int(costs[sid]["Max_Well_Num"]) == math.floor(annual / well)
        assert count * well <= annual < (count + 1) * well
    for basin_id, basin in basins.items():
        assert float(basin["Basin_Max_Annual_Injection_t"]) == sources[basin_id]["CO2_EWR(tonne)"]
        if int(basin["Subsite_Count"]) == 0:
            assert basin["Mainland_Max_Annual_Injection_t"] == ""
            assert basin["Parent_Original_Area_km2"] == ""
        else:
            expected = float(basin["Basin_Max_Annual_Injection_t"]) * float(basin["Parent_Retained_Area_km2"]) / float(basin["Parent_Original_Area_km2"])
            assert math.isclose(float(basin["Mainland_Max_Annual_Injection_t"]), expected, rel_tol=1e-12)
    frame = read._read_sink_subsite_cost(data_dir=output, sink_nodes=list(subsites))
    assert len(frame) == len(subsites)
    assert frame.attrs["storage_cost_schema"] == "lifecycle_costs"
    report = {
        "passed": True, "source_basins": len(sources), "subsites": len(subsites),
        "annual_formula": "CO2_EWR(tonne) * Area_km2 / Parent_Original_Area_km2",
        "maximum_annual_difference_t": largest_error,
        "parent_well_capacity_inheritance": "passed",
        "integer_capacity_bounds": "passed", "model_csv_reader": "lifecycle costs passed",
        "area_and_coordinates_match_source": "passed",
        "point_method": "interior_label_point",
        "coordinate_columns_after_area": ["Longitude", "Latitude"],
        "unmapped_basins": [sid for sid, basin in basins.items() if int(basin["Subsite_Count"]) == 0],
    }
    (output / "capacity_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
