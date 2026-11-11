"""Batch persistence and resume tests; no optimization or national data loading."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heuristic_stages.common import Plan
from LRforFNE import CCS_FNE_LR as batch
from LRforFNE.lr_types import LRResult
from LRforFNE.result_io import write_result


FILES = {
    "result.json", "solution.json", "history.jsonl",
    "initialization_history.jsonl", "initial_multipliers.json", "best_multipliers.json",
}
CASE = (5, 500000.0, 3, ("linear", None), ["p4", "p6", "p20", "p48"])


def result(status="FEASIBLE_TIME_LIMIT"):
    return LRResult(
        method="bundle", status=status, termination_reason="test_stop",
        initial_objective=12.0, objective=10.0, lower_bound=9.0, gap=0.1,
        solution=Plan(s={("H1", "G1", 1): 1}),
        history=[{"evaluation_id": 1, "ell": 9.0}],
        warm_start_runtime=2.0, lr_runtime=3.0, total_runtime=5.0,
        best_multipliers=np.array([[0.25]]), multiplier_bound=9.0,
        multiplier_evaluation=1, lower_bound_origin="LR_oracle_ell",
        audit={"max_violation": 0.0},
        configuration={"multiplier_initialization": {
            "status": "OPTIMAL", "objective": 8.0, "runtime": 0.5}},
        instance={"nodes": ["H1"], "periods": [1], "fingerprint": "test"},
        timings={"shared_warm_start_wall": 2.0},
        initial_multipliers=np.array([[0.1]]),
        initialization_history=[{"stage": "multiplier_LP"}])


def prepare(monkeypatch, cases=(CASE,)):
    monkeypatch.setattr(batch, "_case_grid", lambda: iter(cases))
    monkeypatch.setattr(batch, "load_network_data", lambda path: object())
    monkeypatch.setattr(batch, "compute_cost_data", lambda *a, **k: object())
    monkeypatch.setattr(batch, "with_capture_target", lambda costs, *a, **k: costs)
    calls = []

    def solve(*args, **kwargs):
        calls.append(kwargs)
        return result()

    monkeypatch.setattr(batch, "solve_lr", solve)
    return calls


def run(tmp_path, **kwargs):
    return batch.run_batch(result_dir=tmp_path, normalize_capture_target=False,
                          config=batch.build_config(output_flag=False), **kwargs)


def saved_row(status):
    return dict(period_num=5, D=4, Np=3, max_capture_amount=500000.0,
                curve="linear", coef=None, status=status, method="bundle")


def test_editable_defaults_and_output_paths():
    assert batch.METHOD == "bundle"
    assert batch.build_config().lr_time_limit == 1200
    assert batch.build_config().max_iterations == 100
    options = batch.build_heuristic_options()
    assert options.time_limit == 2400
    assert options.lp_time_limit is None
    assert batch.result_file("bundle").parts[-2:] == (
        "batch", "CCS_FNE_LR_bundle_30Y_result.csv")
    assert batch.result_file("bundle", normalize_capture_target=True).parts[-3:] == (
        "batch", "normalized", "CCS_FNE_LR_bundle_30Y_result.csv")


@pytest.mark.parametrize("normalized", [False, True])
def test_json_precedes_atomic_csv(monkeypatch, tmp_path, normalized):
    calls = prepare(monkeypatch)
    original_write = batch._atomic_write_csv
    csv_writes = []

    def check_details(frame, path):
        detail = Path(frame.iloc[-1]["result_dir"])
        assert {p.name for p in detail.iterdir()} == FILES
        assert json.loads((detail / "result.json").read_text())["objective"] == 10.0
        solution = json.loads((detail / "solution.json").read_text())
        assert solution["s"] == [{"key": ["H1", "G1", 1], "value": 1}]
        assert set(solution) == {"s", "r", "w", "y", "a", "b", "x"}
        assert not list(tmp_path.rglob("*.xlsx"))
        csv_writes.append(path)
        original_write(frame, path)

    monkeypatch.setattr(batch, "_atomic_write_csv", check_details)
    rows = batch.run_batch(result_dir=tmp_path, normalize_capture_target=normalized,
                          config=batch.build_config(output_flag=False))
    path = batch.result_file("bundle", result_dir=tmp_path,
                             normalize_capture_target=normalized)
    assert csv_writes == [path]
    assert len(calls) == 1
    assert Path(rows["bundle"][0]["result_dir"]).relative_to(path.parent).parts[0] == "cases"
    assert list(pd.read_csv(path).columns) == batch.RESULT_COLUMNS


def test_json_failure_never_commits_csv(monkeypatch, tmp_path):
    prepare(monkeypatch)

    def failed_export(*args, **kwargs):
        raise OSError("test export failure")

    monkeypatch.setattr(batch, "write_result", failed_export)
    with pytest.raises(OSError, match="test export failure"):
        run(tmp_path)
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.parametrize("status", batch.COMPLETED_STATUSES)
def test_resume_skips_completed_before_loading_data(monkeypatch, tmp_path, status):
    calls = prepare(monkeypatch)
    batch.save_results([saved_row(status)], "bundle", result_dir=tmp_path,
                       normalize_capture_target=False)

    def unexpected_load(path):
        raise AssertionError("completed cases must not load the network")

    monkeypatch.setattr(batch, "load_network_data", unexpected_load)
    assert run(tmp_path)["bundle"][0]["status"] == status
    assert calls == []


@pytest.mark.parametrize("status", [
    "NO_INITIAL_FEASIBLE_SOLUTION", "FEASIBLE_SOLVER_ERROR", "FEASIBLE_INTERRUPTED",
    "NUMERICAL_ERROR", "MULTIPLIER_INITIALIZATION_INCOMPLETE", "ORACLE_INCOMPLETE",
])
def test_resume_retries_failed_case_and_replaces_row(monkeypatch, tmp_path, status):
    calls = prepare(monkeypatch)
    batch.save_results([saved_row(status)], "bundle", result_dir=tmp_path,
                       normalize_capture_target=False)
    rows = run(tmp_path)["bundle"]
    assert len(calls) == len(rows) == 1
    assert rows[0]["status"] == "FEASIBLE_TIME_LIMIT"
    assert {p.name for p in Path(rows[0]["result_dir"]).iterdir()} == FILES


def test_method_rejected_before_loading(monkeypatch, tmp_path):
    calls = prepare(monkeypatch)
    with pytest.raises(ValueError, match="bundle"):
        run(tmp_path, method="subgradient")
    assert calls == []


def test_explicit_output_root_remains_confined(tmp_path):
    with pytest.raises(ValueError, match="selected output root"):
        write_result(result(), "../outside", output_root=tmp_path)


def test_infeasible_records_preserve_absent_bounds(tmp_path):
    infeasible = replace(result("PROVEN_INFEASIBLE"), objective=None,
                         initial_objective=None, lower_bound=None, gap=None,
                         audit=None, solution=Plan(), best_multipliers=None,
                         multiplier_bound=None, multiplier_evaluation=None,
                         initial_multipliers=None)
    directory = write_result(infeasible, "infeasible", output_root=tmp_path)
    data = json.loads((directory / "result.json").read_text())
    assert data["status"] == "PROVEN_INFEASIBLE"
    assert data["objective"] is None and data["gap"] is None
    assert all(not records for records in json.loads((directory / "solution.json").read_text()).values())
