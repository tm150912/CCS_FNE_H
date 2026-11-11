"""Validate saved subsite geometry and regenerate its two current audit reports.

Run with the ArcGIS Pro Python environment used by sink_subsite.py. The source
boundaries and saved CSV/GeoJSON are read-only; this does not rerun partitioning.
"""
from collections import defaultdict
import csv
import datetime as audit_datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sink_subsite"))
import sink_subsite as spatial


def main():
    inputs = [spatial.OUTPUT_CSV, spatial.OUTPUT_GEOJSON, spatial.SINK_WORKBOOK,
              Path(spatial.__file__), Path(__file__)]
    for base in (spatial.CHINA_BOUNDARY, spatial.BASIN_BOUNDARY):
        inputs.extend(base.with_suffix(suffix) for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"))
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in inputs}
    sinks = spatial.read_inputs()
    _, basins, unmapped = spatial.load_geography(sinks)
    basin_map = {b["Sink_ID"]: b for b in basins}
    saved = json.loads(spatial.OUTPUT_GEOJSON.read_text(encoding="utf-8"))
    with spatial.OUTPUT_CSV.open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_by_id = {row["Subsite_ID"]: row for row in csv_rows}
    feature_ids = [feature["id"] for feature in saved["features"]]
    if len(csv_by_id) != len(csv_rows) or len(set(feature_ids)) != len(feature_ids) or set(feature_ids) != set(csv_by_id):
        raise ValueError("CSV and GeoJSON must have identical unique subsite IDs")
    grouped = defaultdict(list)
    area_errors, compactness_errors, geometries = [], [], []
    for feature in saved["features"]:
        cell = dict(feature["properties"])
        serialized = {key: "" if value is None else str(value) for key, value in cell.items()}
        if serialized != csv_by_id[feature["id"]]:
            raise ValueError("CSV and GeoJSON properties differ: " + feature["id"])
        geo = spatial.arcpy.AsShape(feature["geometry"], False)
        area = spatial.area_km2(geo)
        error = abs(area - cell["Area_km2"])
        if error >= 1e-5 or area > spatial.MAX_AREA_KM2:
            raise ValueError("Saved geometry area differs: " + feature["id"])
        area_errors.append(error)
        basin = basin_map[cell["Sink_ID"]]
        if cell["Point_Method"] != "interior_label_point":
            raise ValueError("Unexpected representative-point method: " + feature["id"])
        for field in ("Original_Area_km2", "Retained_Area_km2"):
            if not math.isclose(cell["Parent_" + field], basin[field],
                                rel_tol=0, abs_tol=spatial.COVERAGE_TOL_KM2):
                raise ValueError("Saved parent area differs: " + feature["id"] + "/" + field)
        local = geo.projectAs(basin["local_geometry"].spatialReference)
        compactness = 4 * math.pi * local.area / local.length ** 2
        compactness_errors.append(abs(compactness - cell["Compactness_4piA_P2"]))
        if not math.isclose(compactness, cell["Compactness_4piA_P2"], abs_tol=1e-8):
            raise ValueError("Saved compactness differs: " + feature["id"])
        cell.update(geometry=geo, local_geometry=local)
        grouped[cell["Sink_ID"]].append(cell)
        geometries.append(geo)

    reports = []
    for basin in basins:
        # Partition search diagnostics are historical, not rerun by this check.
        basin["Partition_Diagnostics"] = []
        result = spatial.validate_basin(basin, grouped[basin["Sink_ID"]])
        result.pop("Partition_Diagnostics")
        reports.append(result)
        print(f"{basin['Sink_ID']}: {result['Subsite_Count']} subsites passed", flush=True)

    spatial.arcpy.management.CopyFeatures(geometries, "memory/alignment_saved_cells")
    spatial.arcpy.management.CheckGeometry("memory/alignment_saved_cells", "memory/alignment_check", "OGC")
    issues = int(spatial.arcpy.management.GetCount("memory/alignment_check")[0])
    if issues:
        raise ValueError(f"Saved geometry has {issues} OGC issues")
    if hashes != {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in inputs}:
        raise ValueError("Input files changed during validation; rerun against a stable snapshot")
    common = {
        "passed": True, "date": audit_datetime.datetime.now(
            audit_datetime.timezone(audit_datetime.timedelta(hours=8))).date().isoformat(),
        "subsites": len(geometries), "basins": len(basins), "input_sha256": hashes,
        "method": "ArcGIS saved-GeoJSON validation and source-boundary clipping in production projections",
        "source_data_written": False, "partition_search_rerun": False,
    }
    representative = {
        **common, "point_method": "interior_label_point", "strictly_inside": len(geometries),
        "single_part_subsites": len(geometries), "ogc_geometry_issues": issues,
        "max_saved_area_difference_km2": max(area_errors),
    }
    geography = {
        **common, "unmapped_basins": unmapped,
        "annual_formula": "CO2_EWR(tonne) * Area_km2 / Parent_Original_Area_km2",
        "coverage_tolerance_km2": spatial.COVERAGE_TOL_KM2,
        "max_compactness_difference": max(compactness_errors),
        "max_absolute_annual_sum_difference_t": max(abs(r["Annual_Quota_Sum_Difference_t"]) for r in reports),
        "basin_checks": reports,
    }
    output = ROOT / "storage_cost_results"
    for name, report in (("representative_point_validation.json", representative),
                         ("geography_capacity_validation.json", geography)):
        (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "subsites": len(geometries), "basins": len(basins),
                      "ogc_geometry_issues": issues, "output": str(output)}))


if __name__ == "__main__":
    main()
