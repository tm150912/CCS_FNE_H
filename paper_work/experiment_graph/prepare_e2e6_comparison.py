"""Audit and summarize the supplied E_2e6 plans without running a solver.

The saved annual flows identify the unnormalized exponential curve (2.5),
which the user confirmed after the initial curve discrepancy was checked.
Source files are read only; current model coefficients are recomputed.
"""

import csv
import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
import openpyxl

from prepare_section53_data import (
    ROOT, MODEL_DIR, MEAN_DEFINITION, compute_cost_data, fingerprint,
    load_network_data, period_schedule, read_solution, storage_sink_mean,
)
from cost_amortize import DISCOUNT_RATE, INTEREST_RATE, TECHNICAL_LIFETIME
from experiment_setup import D_LIST
from capture_target_sequence import generate_capture_targets


OUTPUT = ROOT / "experiment_graph" / "e2e6_comparison_data.json"
PERIODS, DURATIONS = period_schedule(15)
TARGET = np.round(2e6 * np.expm1(2.5 * np.arange(1, 16) / 15) / np.expm1(2.5), 3)


def evaluate(solution, network, distance, target=TARGET):
    active_pipes = sorted(set(solution["x"]) | set(solution["y"]))
    costs = compute_cost_data(replace(network, ijd=active_pipes), PERIODS, DURATIONS,
                              apply_shared_capture_target=False)
    fields = dict(s="source_fixed", a="source_variable", r="sink_fixed",
                  w="well_fixed", b="sink_variable", y="pipe_fixed", x="pipe_variable")
    components = {
        name: sum(getattr(costs, field)[(*key, h)] * value
                  for key, values in solution[name].items()
                  for h, value in zip(PERIODS, values)) / 1e6
        for name, field in fields.items()
    }
    totals = dict(Capture=components["s"] + components["a"],
                  Storage=components["r"] + components["w"] + components["b"],
                  Transport=components["y"] + components["x"])
    totals["Total"] = sum(totals.values())
    utilization, details, violations = {}, {}, {}
    zero = np.zeros(15)
    for label, build, flow, capacity in (
        ("Capture", "s", "a", lambda k: network.source_capacity[k]),
        ("Storage", "w", "b", lambda k: network.well_capacity[k[0]]),
        ("Transport", "y", "x", lambda k: network.pipe_capacity[k[2]]),
    ):
        keys = sorted(set(solution[build]) | set(solution[flow]))
        counts = np.asarray([np.cumsum(solution[build].get(k, zero)) for k in keys])
        flows = np.asarray([solution[flow].get(k, zero) for k in keys])
        capacities = np.asarray([capacity(k) for k in keys])[:, None]
        installed = counts * capacities
        total_count, total_capacity = counts.sum(axis=0), installed.sum(axis=0)
        total_flow = flows.sum(axis=0)
        utilization[label] = {
            "weighted": (100 * total_flow / total_capacity).tolist(),
            "mean": (100 * (flows / capacities).sum(axis=0) / total_count).tolist(),
        }
        details[label] = {"flow_kt_per_year": total_flow.tolist(),
                          "capacity_kt_per_year": total_capacity.tolist(),
                          "installed_count": total_count.astype(int).tolist()}
        violations[f"{label.lower()}_capacity"] = float(max(0, np.max(flows - installed)))
    sink_mean, sink_count = storage_sink_mean(solution, network)
    utilization["Storage"]["mean"] = sink_mean
    details["Storage"]["installed_sink_count"] = sink_count

    hubs, pipes, balance = (defaultdict(lambda: np.zeros(15)) for _ in range(3))
    for (i, _), values in solution["s"].items():
        hubs[i] += np.cumsum(values)
    for (i, j, _), values in solution["y"].items():
        pipes[i, j] += np.cumsum(values)
    for (i, _), values in solution["a"].items():
        balance[i] -= values
    for (j,), values in solution["b"].items():
        balance[j] += values
    for (i, j, _), values in solution["x"].items():
        balance[i] += values
        balance[j] -= values
    violations["node_balance"] = float(max(np.abs(v).max() for v in balance.values()))
    violations["source_once"] = float(max(0, max(v.sum() - 1 for v in solution["s"].values())))
    violations["sink_once"] = float(max(0, max(v.sum() - 1 for v in solution["r"].values())))
    violations["pipe_full_horizon_limit"] = float(max(0, max(v[-1] - 3 for v in pipes.values())))
    violations["well_activation_and_maximum"] = float(max(0, max(
        np.max(np.cumsum(values) - network.max_wells[j] * np.cumsum(solution["r"].get((j,), zero)))
        for (j,), values in solution["w"].items())))
    violations["capture_target"] = float(max(0, np.max(target - details["Capture"]["flow_kt_per_year"])))
    for label in ("Capture", "Storage"):
        np.testing.assert_allclose(details[label]["flow_kt_per_year"], target, rtol=0, atol=1e-3)
    details["Capture"]["installed_hub_count"] = np.sum(
        [v > 0 for v in hubs.values()], axis=0).astype(int).tolist()
    details["Storage"]["opened_sink_count"] = np.sum(
        [np.cumsum(v) > 0 for v in solution["r"].values()], axis=0).astype(int).tolist()
    details["Transport"]["installed_length_km"] = sum(
        distance[edge] * (counts > 0) for edge, counts in pipes.items()).tolist()
    details["Transport"]["directed_corridor_count"] = np.sum(
        [v > 0 for v in pipes.values()], axis=0).astype(int).tolist()
    details["Transport"]["repeat_installation_count"] = (
        np.asarray(details["Transport"]["installed_count"])
        - details["Transport"]["directed_corridor_count"]).tolist()
    details["Transport"]["installed_by_diameter"] = {
        d: np.sum([np.cumsum(v) for (*_, diameter), v in solution["y"].items()
                   if diameter == d], axis=0).astype(int).tolist()
        if any(k[2] == d for k in solution["y"]) else [0] * 15
        for d in D_LIST[-1]
    }
    maximum = max(violations.values())
    return {"cost_billion_usd": totals, "cost_components_billion_usd": components,
            "utilization_percent": utilization, "period_totals": details,
            "constraint_audit": {
                "method": "Independent check of all constraints in current run_gurobi.solve_original_model; no optimizer call.",
                "storage_scope": "Well capacity and maximum wells; current original model has no additional cumulative reservoir-volume constraint.",
                "maximum_violation_by_family": violations,
                "max_violation": maximum, "passes_1e_5_tolerance": maximum <= 1e-5,
                "passes_1e_6_tolerance": maximum <= 1e-6,
                "input_checks": "read_solution checks exact row keys, period headers, nonnegativity, finite values and integrality before integer rounding.",
            }}


def main():
    network = load_network_data(MODEL_DIR)
    with (MODEL_DIR / "arcs.csv").open(encoding="utf-8-sig", newline="") as source:
        distance = {(r["i"], r["j"]): float(r["distance"]) for r in csv.DictReader(source)}
    inputs = [MODEL_DIR / name for name in (
        "nodes.csv", "arcs.csv", "hub_unit_location_cost.csv", "sink_subsite_cost.csv",
        "pipeline_capacity.csv", "pipeline_cost.csv", "read.py", "experiment_setup.py",
        "cost_amortize.py", "capture_target_sequence.py", "run_gurobi.py")]
    data = {"provenance": {
        "inputs": [fingerprint(p) for p in inputs], "periods": 15, "years_per_period": 2,
        "pipeline_install_limit": 3, "diameters": D_LIST[-1],
        "confirmed_curve": "exponential", "observed_curve": "exponential",
        "coefficient": 2.5, "normalize_capture_target": False,
        "capture_target_kt_per_year": TARGET.tolist(),
        "total_capture_gt": float(2 * TARGET.sum() / 1e6),
        "discount_rate": DISCOUNT_RATE, "interest_rate": INTEREST_RATE,
        "technical_lifetime_years": TECHNICAL_LIFETIME,
        "cost_conversion": "Model thousand USD divided by 1e6 gives billion USD.",
        "weighted_utilization": "100 * sum(flow) / sum(cumulative count * annual capacity)",
        "mean_utilization": MEAN_DEFINITION, "transport_distance_weighting": False,
        "length_definition": "Sum of input distances over directed arcs with at least one installed pipe; each directed arc is counted once across diameters and periods.",
    }, "methods": {}}
    for label, method in (("GRB", "gurobi"), ("HEU", "Heuristic")):
        path = ROOT / "experiment_graph" / f"CCS_FNE_{method}_E_2e6" / f"CCS_FNE_{method}_E_2e6.xlsx"
        solution, audit = read_solution(path, network)
        result = evaluate(solution, network, distance)
        result.update(source=fingerprint(path), row_key_audit=audit)
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        result["solver_summary"] = (dict(list(workbook["solve_summary"].values)[1:])
                                    if "solve_summary" in workbook.sheetnames else None)
        workbook.close()
        if result["solver_summary"]:
            result["objective_recomputation_difference_thousand_usd"] = (
                result["cost_billion_usd"]["Total"] * 1e6 - result["solver_summary"]["objective"])
        flow = np.array(result["period_totals"]["Capture"]["flow_kt_per_year"])
        result["curve_max_absolute_error_kt_per_year"] = {
            "exponential_common_terminal": float(np.max(np.abs(flow - TARGET))),
            "logarithmic_common_terminal": float(np.max(np.abs(
                flow - np.round(2e6 * np.log1p(2.5 * np.arange(1, 16) / 15) / np.log1p(2.5), 3)))),
            **{f"{curve}_linear_total_normalized": float(np.max(np.abs(flow - np.array(list(
                generate_capture_targets(15, 2e6, curve, 2.5,
                                         normalize_capture_target=True).values())))))
               for curve in ("exponential", "logarithmic")},
        }
        data["methods"][label] = result
        print(label, result["cost_billion_usd"], result["constraint_audit"], flush=True)
    grb, heu = (data["methods"][m] for m in ("GRB", "HEU"))
    data["cost_reduction_percent"] = {k: 100 * (v - heu["cost_billion_usd"][k]) / v
                                       for k, v in grb["cost_billion_usd"].items()}
    lower_bound = grb["solver_summary"]["lower_bound"] / 1e6
    data["common_bound_comparison"] = {
        "lower_bound_billion_usd": lower_bound,
        "origin": "Global ObjBound recorded by GRB for the same instance; not an HEU-produced bound.",
        "gap_definition": "100 * (objective - GRB lower bound) / objective",
        "gap_percent": {label: 100 * (row["cost_billion_usd"]["Total"] - lower_bound)
                        / row["cost_billion_usd"]["Total"]
                        for label, row in data["methods"].items()},
    }
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
