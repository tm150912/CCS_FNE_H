"""Recompute Section 5.3 plot data from the two supplied solution workbooks.

The 5.3_graph workbooks match the current CCS_FNE_H network row keys.
No solver is run and no source workbook or manuscript is changed.
"""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT.parent
NETWORK_DIR = MODEL_DIR
DATA_DIR = ROOT / "5.3_graph"
OUTPUT = DATA_DIR / "section53_plot_data.json"
MEAN_DEFINITION = (
    "Capture and transport: 100 * sum(flow / annual per-asset capacity) / sum(cumulative count). "
    "Storage: equal-weight mean of 100 * sink flow / sink cumulative installed annual injection capacity; "
    "each sink is a subsite; include sinks with positive installed capacity, including zero-flow sinks."
)
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_DIR))

from experiment_setup import compute_cost_data, load_network_data, period_schedule
from cost_amortize import DISCOUNT_RATE, INTEREST_RATE, TECHNICAL_LIFETIME


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_solution(path, network):
    expected = {"s": set(network.ig), "a": set(network.ig),
                "r": {(j,) for j in network.sinks},
                "w": {(j,) for j in network.sinks},
                "b": {(j,) for j in network.sinks},
                "x": set(network.ijd), "y": set(network.ijd)}
    result, audit = {}, {}
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for name, keys in expected.items():
        sheet = workbook[f"variable_{name}"]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        offset = len(next(iter(keys)))
        assert list(header[offset:]) == [f"period{h}" for h in range(1, 16)]
        seen, nonzero = set(), {}
        for row in rows:
            key = tuple(row[:offset])
            assert key in keys and key not in seen, (sheet.title, key)
            seen.add(key)
            values = np.asarray(row[offset:], dtype=float)
            assert np.all(np.isfinite(values)) and np.min(values) >= -1e-5
            if name in ("s", "r", "w", "y"):
                assert np.max(np.abs(values - np.rint(values))) < 1e-5
                values = np.rint(values)
            values = np.maximum(values, 0)
            if np.any(values):
                nonzero[key] = values
        assert seen == keys, (sheet.title, len(seen), len(keys))
        result[name] = nonzero
        audit[name] = {"rows": len(seen), "nonzero_rows": len(nonzero)}
        print(f"{path.name}: {name}, {len(nonzero)} active rows", flush=True)
    workbook.close()
    return result, audit


def storage_sink_mean(solution, network):
    keys = sorted(set(solution["w"]) | set(solution["b"]))
    zero = np.zeros(15)
    flows = np.asarray([solution["b"].get(key, zero) for key in keys])
    capacities = np.asarray([
        np.cumsum(solution["w"].get(key, zero)) * network.well_capacity[key[0]]
        for key in keys])
    installed = capacities > 0
    assert np.all(installed.sum(axis=0) > 0)
    assert np.max(flows - capacities) < 1e-3
    ratios = np.divide(flows, capacities, out=np.zeros_like(flows), where=installed)
    return ((100 * ratios.sum(axis=0) / installed.sum(axis=0)).tolist(),
            installed.sum(axis=0).tolist())


def evaluate(solution, network):
    periods, durations = period_schedule(15)
    # Zero rows contribute no costs. Restrict coefficient construction to the
    # nonzero pipe rows, using exactly the shared model's amortization routine.
    active_pipes = sorted(set(solution["x"]) | set(solution["y"]))
    cost_network = replace(network, ijd=active_pipes)
    costs = compute_cost_data(cost_network, periods, durations,
                              apply_shared_capture_target=False)
    coefficients = {"s": costs.source_fixed, "a": costs.source_variable,
                    "r": costs.sink_fixed, "w": costs.well_fixed,
                    "b": costs.sink_variable, "y": costs.pipe_fixed,
                    "x": costs.pipe_variable}
    components = {
        name: sum(coefficients[name][(*key, h)] * value
                  for key, values in solution[name].items()
                  for h, value in zip(periods, values)) / 1e6
        for name in coefficients}
    totals = {"Capture": components["s"] + components["a"],
              "Storage": components["r"] + components["w"] + components["b"],
              "Transport": components["y"] + components["x"]}
    totals["Total"] = sum(totals.values())
    utilization, details = {}, {}
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
        assert np.max(flows - installed) < 1e-3, label
        total_count, total_capacity = counts.sum(axis=0), installed.sum(axis=0)
        total_flow = flows.sum(axis=0)
        assert np.all(total_count > 0) and np.all(total_capacity > 0)
        utilization[label] = {
            "weighted": (100 * total_flow / total_capacity).tolist(),
            # Each installed unit/well/pipe receives equal weight, including
            # idle assets and repeat installations on a directed arc.
            "mean": (100 * (flows / capacities).sum(axis=0) / total_count).tolist()}
        details[label] = {"flow_kt_per_year": total_flow.tolist(),
                          "capacity_kt_per_year": total_capacity.tolist(),
                          "installed_count": total_count.astype(int).tolist(),
                          "max_capacity_violation_kt_per_year":
                              float(max(0, np.max(flows - installed)))}
    sink_mean, sink_count = storage_sink_mean(solution, network)
    utilization["Storage"]["mean"] = sink_mean
    details["Storage"]["installed_sink_count"] = sink_count
    target = np.round(2e6 * np.arange(1, 16) / 15, 3)
    for label in ("Capture", "Storage"):
        np.testing.assert_allclose(details[label]["flow_kt_per_year"], target,
                                   rtol=0, atol=1e-3)
    return {"cost_billion_usd": totals, "cost_components_billion_usd": components,
            "utilization_percent": utilization, "period_totals": details}


def main():
    network = load_network_data(NETWORK_DIR)
    inputs = [NETWORK_DIR / name for name in
              ("nodes.csv", "arcs.csv", "hub_unit_location_cost.csv",
               "sink_subsite_cost.csv", "pipeline_capacity.csv", "pipeline_cost.csv")]
    inputs += [MODEL_DIR / name for name in
               ("read.py", "experiment_setup.py", "cost_amortize.py")]
    data = {"provenance": {
        "network_reason": "Exact solution row-key match against the current CCS_FNE_H network.",
        "inputs": [fingerprint(p) for p in inputs],
        "periods": 15, "years_per_period": 2,
        "discount_rate": DISCOUNT_RATE, "interest_rate": INTEREST_RATE,
        "technical_lifetime_years": TECHNICAL_LIFETIME,
        "cost_conversion": "Model thousand USD divided by 1e6 gives billion USD.",
        "weighted_utilization": "100 * sum(flow) / sum(cumulative count * annual capacity)",
        "mean_utilization": MEAN_DEFINITION,
        "transport_distance_weighting": False,
        "solver_status_runtime_gap": "Not recorded in the supplied variable workbooks."},
        "methods": {}}
    for label, method in (("GRB", "gurobi"), ("LRH", "Heuristic")):
        path = DATA_DIR / f"CCS_FNE_{method}_variables.xlsx"
        solution, audit = read_solution(path, network)
        data["methods"][label] = evaluate(solution, network)
        data["methods"][label].update(source=fingerprint(path), row_key_audit=audit)
        print(label, data["methods"][label]["cost_billion_usd"], flush=True)
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
