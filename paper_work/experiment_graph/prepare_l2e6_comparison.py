"""Recompute the L_2e6 comparison; read inputs only and never run a solver."""

import csv
import json

import numpy as np
import openpyxl

import prepare_e2e6_comparison as common


OUTPUT = common.ROOT / "experiment_graph" / "l2e6_comparison_data.json"
TARGET = np.round(2e6 * np.log1p(np.arange(1, 16)) / np.log(16), 3)


def read_result(label, network, distance):
    method = {"GRB": "gurobi", "HEU": "Heuristic"}[label]
    path = common.ROOT / "experiment_graph" / f"CCS_FNE_{method}_L_2e6" / f"CCS_FNE_{method}_L_2e6.xlsx"
    solution, audit = common.read_solution(path, network)
    result = common.evaluate(solution, network, distance, target=TARGET)
    result.update(source=common.fingerprint(path), row_key_audit=audit)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    result["solver_summary"] = (dict(list(workbook["solve_summary"].values)[1:])
                                if "solve_summary" in workbook.sheetnames else None)
    workbook.close()
    if result["solver_summary"]:
        result["objective_recomputation_difference_thousand_usd"] = (
            result["cost_billion_usd"]["Total"] * 1e6 - result["solver_summary"]["objective"])
    flow = np.array(result["period_totals"]["Capture"]["flow_kt_per_year"])
    fraction = np.arange(1, 16) / 15
    candidates = {"linear_common_terminal": np.round(2e6 * fraction, 3)}
    for coefficient in (2, 2.5, 15, 500):
        candidates[f"logarithmic_{coefficient}_common_terminal"] = np.round(
            2e6 * np.log1p(coefficient * fraction) / np.log1p(coefficient), 3)
        candidates[f"logarithmic_{coefficient}_linear_total_normalized"] = np.array(list(
            common.generate_capture_targets(15, 2e6, "logarithmic", coefficient,
                                            normalize_capture_target=True).values()))
    result["curve_max_absolute_error_kt_per_year"] = {
        name: float(np.max(np.abs(flow - values))) for name, values in candidates.items()}
    print(label, result["cost_billion_usd"], result["constraint_audit"], flush=True)
    return result


def provenance():
    inputs = [common.MODEL_DIR / name for name in (
        "nodes.csv", "arcs.csv", "hub_unit_location_cost.csv", "sink_subsite_cost.csv",
        "pipeline_capacity.csv", "pipeline_cost.csv", "read.py", "experiment_setup.py",
        "cost_amortize.py", "capture_target_sequence.py", "run_gurobi.py")]
    return {
        "inputs": [common.fingerprint(p) for p in inputs], "periods": 15, "years_per_period": 2,
        "pipeline_install_limit": 3, "diameters": common.D_LIST[-1],
        "observed_curve": "logarithmic", "coefficient": 15,
        "normalize_capture_target": False, "capture_target_kt_per_year": TARGET.tolist(),
        "total_capture_gt": float(2 * TARGET.sum() / 1e6),
        "discount_rate": common.DISCOUNT_RATE, "interest_rate": common.INTEREST_RATE,
        "technical_lifetime_years": common.TECHNICAL_LIFETIME,
        "cost_conversion": "Model thousand USD divided by 1e6 gives billion USD.",
        "weighted_utilization": "100 * sum(flow) / sum(cumulative count * annual capacity)",
        "mean_utilization": common.MEAN_DEFINITION, "transport_distance_weighting": False,
        "length_definition": "Sum of input distances over directed arcs with at least one installed pipe; each directed arc is counted once across diameters and periods.",
    }


def complete_comparison(data):
    grb, heu = (data["methods"][m] for m in ("GRB", "HEU"))
    data["cost_reduction_percent"] = {k: 100 * (v - heu["cost_billion_usd"][k]) / v
                                       for k, v in grb["cost_billion_usd"].items()}
    if grb["solver_summary"] and grb["solver_summary"].get("lower_bound") is not None:
        lower_bound = grb["solver_summary"]["lower_bound"] / 1e6
        data["common_bound_comparison"] = {
            "lower_bound_billion_usd": lower_bound,
            "origin": "Global ObjBound recorded by GRB for the same instance; not an HEU-produced bound.",
            "gap_definition": "100 * (objective - GRB lower bound) / objective",
            "gap_percent": {label: 100 * (row["cost_billion_usd"]["Total"] - lower_bound)
                            / row["cost_billion_usd"]["Total"]
                            for label, row in data["methods"].items()},
        }
    else:
        data["common_bound_comparison"] = None
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(OUTPUT, flush=True)


def main():
    network = common.load_network_data(common.MODEL_DIR)
    with (common.MODEL_DIR / "arcs.csv").open(encoding="utf-8-sig", newline="") as source:
        distance = {(r["i"], r["j"]): float(r["distance"]) for r in csv.DictReader(source)}
    data = {"provenance": provenance(), "methods": {
        label: read_result(label, network, distance) for label in ("GRB", "HEU")}}
    complete_comparison(data)


if __name__ == "__main__":
    main()
