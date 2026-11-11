"""Evaluate the supplied linear plans and combine the three curve comparisons.

All source workbooks are read only. Exponential and logarithmic results are
reused only after their recorded workbook and model-input hashes are checked.
No solver is run and no existing comparison output is overwritten.
"""

import csv
import json
from pathlib import Path

import numpy as np
import openpyxl

import prepare_e2e6_comparison as common
from prepare_l2e6_comparison import provenance


DATA_DIR = common.ROOT / "experiment_graph"
OUTPUT = DATA_DIR / "linear2e6_comparison_data.json"
COMBINED_OUTPUT = DATA_DIR / "three_curves_comparison_data.json"
TARGET = np.round(2e6 * np.arange(1, 16) / 15, 3)
SOURCES = {
    "GRB": DATA_DIR / "CCS_FNE_gurobi_2e6" / "CCS_FNE_gurob_2e6.xlsx",
    "HEU": DATA_DIR / "CCS_FNE_Heuristic_2e6" / "CCS_FNE_Heuristic_2e6.xlsx",
}


def verify_sources(data):
    records = data["provenance"]["inputs"] + [r["source"] for r in data["methods"].values()]
    for record in records:
        assert common.fingerprint(Path(record["path"])) == record, record["path"]


def combine_comparisons():
    files = {"Linear": OUTPUT, "Logarithmic": DATA_DIR / "l2e6_comparison_data.json",
             "Exponential": DATA_DIR / "e2e6_comparison_data.json"}
    combined = {"curve_order": list(files), "cases": {}, "summary": {}}
    for curve, path in files.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        verify_sources(data)
        combined["cases"][curve] = data
        combined["summary"][curve] = {
            "source_cache": common.fingerprint(path),
            "total_capture_gt": data["provenance"]["total_capture_gt"],
            "cost_reduction_percent": data["cost_reduction_percent"],
            "methods": {label: {
                "cost_billion_usd": row["cost_billion_usd"],
                "terminal_utilization_percent": {
                    facility: {metric: values[-1] for metric, values in stats.items()}
                    for facility, stats in row["utilization_percent"].items()},
                "solver_summary": row["solver_summary"],
            } for label, row in data["methods"].items()},
        }
    COMBINED_OUTPUT.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(COMBINED_OUTPUT, flush=True)


def main():
    network = common.load_network_data(common.MODEL_DIR)
    with (common.MODEL_DIR / "arcs.csv").open(encoding="utf-8-sig", newline="") as source:
        distance = {(r["i"], r["j"]): float(r["distance"]) for r in csv.DictReader(source)}
    metadata = provenance()
    metadata.update(observed_curve="linear", coefficient=None,
                    capture_target_kt_per_year=TARGET.tolist(),
                    total_capture_gt=float(2 * TARGET.sum() / 1e6))
    data = {"provenance": metadata, "methods": {}}
    for label, path in SOURCES.items():
        solution, audit = common.read_solution(path, network)
        result = common.evaluate(solution, network, distance, target=TARGET)
        result.update(source=common.fingerprint(path), row_key_audit=audit)
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        result["solver_summary"] = (dict(list(workbook["solve_summary"].values)[1:])
                                    if "solve_summary" in workbook.sheetnames else None)
        workbook.close()
        assert result["solver_summary"] is None, "Recheck newly added solver metadata before comparing bounds"
        flow = np.array(result["period_totals"]["Capture"]["flow_kt_per_year"])
        result["curve_max_absolute_error_kt_per_year"] = {
            "linear_common_terminal": float(np.max(np.abs(flow - TARGET)))}
        data["methods"][label] = result
        print(label, result["cost_billion_usd"], result["constraint_audit"], flush=True)
    grb, heu = (data["methods"][m] for m in ("GRB", "HEU"))
    data["cost_reduction_percent"] = {k: 100 * (v - heu["cost_billion_usd"][k]) / v
                                       for k, v in grb["cost_billion_usd"].items()}
    data["common_bound_comparison"] = None
    verify_sources(data)
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(OUTPUT, flush=True)
    combine_comparisons()


if __name__ == "__main__":
    main()
