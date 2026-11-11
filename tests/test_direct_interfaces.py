"""Output and entry-point checks for Data_Suplementary.tex."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import CCS_FNE as batch
import CCS_FNE_PH as batch_ph
import run_gurobi as gurobi
import run_PH as ph


class DirectInterfaceTests(unittest.TestCase):
    def test_documented_single_case_settings(self):
        for entry in (gurobi, ph):
            self.assertEqual((entry.PERIOD_NUM, entry.NP, len(entry.DIAMETERS),
                              entry.CURVE, entry.MAX_CAPTURE_AMOUNT),
                             (15, 3, 12, "linear", 1e6))
            self.assertFalse(entry.NORMALIZE_CAPTURE_TARGET)
        self.assertEqual(gurobi.TIME_LIMIT, 15000)
        self.assertEqual(batch.TIME_LIMIT, 11000)
        for entry in (ph, batch_ph):
            self.assertEqual((entry.LP_TIME_LIMIT, entry.TIME_LIMIT, entry.MAX_ROUNDS),
                             (None, 3600, 200))

    def test_legacy_snapshot_columns_hold_actual_threshold_values(self):
        values = {batch.gp.GRB.Callback.MIP_OBJBST: 120,
                  batch.gp.GRB.Callback.MIP_OBJBND: 80}
        model = SimpleNamespace(Params=SimpleNamespace(), Status=batch.gp.GRB.TIME_LIMIT,
                                SolCount=1, ObjVal=100, ObjBound=90, MIPGap=0.1,
                                cbGet=values.__getitem__, dispose=lambda: None)

        def optimize(callback):
            for runtime, ub, lb in ((3599, 130, 70), (3600, 120, 80),
                                    (3601, 110, 85), (10800, 105, 88)):
                values.update({batch.gp.GRB.Callback.RUNTIME: runtime,
                               batch.gp.GRB.Callback.MIP_OBJBST: ub,
                               batch.gp.GRB.Callback.MIP_OBJBND: lb})
                callback(model, batch.gp.GRB.Callback.MIP)

        model.optimize = optimize
        with patch.object(batch, "build_milp", return_value=(model, {})):
            result = batch.solve_milp(None, None, [], pipeline_install_limit=3)
        row = dict(zip(batch.METRIC_COLUMNS, result.metrics()))
        self.assertEqual((row["2hUB"], row["2hLB"]), (120, 80))
        self.assertEqual((row["5hUB"], row["5hLB"]), (105, 88))
        self.assertEqual((row["UB"], row["LB"]), (100, 90))

    def test_single_case_workbooks_preserve_logs_without_an_incumbent(self):
        network = SimpleNamespace(ig=[("H1", "u1")], sinks={"P1": ()}, ij=[("H1", "P1")])
        costs = SimpleNamespace(periods=[1])
        metadata_sheets = {"model_parameters", "algorithm_parameters", "solve_summary", "solve_log"}
        variable_sheets = {"variable_" + name for name in ("s", "r", "w", "y", "a", "b", "x")}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            for objective in (None, 100.0):
                result = gurobi.GurobiResult(
                    "TIME_LIMIT", objective, 80.0, None if objective is None else 0.2,
                    {}, {}, {}, {}, {}, {}, {}, solver_log=["saved solver message"])
                target = Path(directory) / "gurobi.xlsx"
                with patch.object(gurobi, "RESULT_FILE", target):
                    gurobi.save_gurobi_results(result, network, costs, ["p4"])
                expected = metadata_sheets | (variable_sheets if objective is not None else set())
                with pd.ExcelFile(target) as workbook:
                    self.assertEqual(set(workbook.sheet_names), expected)
                    self.assertEqual(pd.read_excel(workbook, "solve_log").iloc[0]["message"],
                                     "saved solver message")

                heuristic = SimpleNamespace(
                    status="FEASIBLE_TIME_LIMIT", objective=objective, lower_bound=80, gap=None,
                    runtime=2, s={}, r={}, w={}, y={}, a={}, b={}, x={},
                    lower_bound_origin="LP", lp_status="OPTIMAL", lp_objective=80,
                    lp_runtime=1, configuration={}, history=[{"stage": "stop"}])
                target = Path(directory) / "ph.xlsx"
                ph.save_heuristic_results(heuristic, network, costs, ["p4"], target)
                with pd.ExcelFile(target) as workbook:
                    self.assertEqual(set(workbook.sheet_names), expected)

    def test_ph_reads_the_documented_historical_checkpoint(self):
        self.assertEqual(batch_ph.CHECKPOINT_FILE.name, "CCS_FNE_H_30Y_checkpoint.csv")
        case = next(batch_ph._case_grid())
        periods, maximum, limit, (curve, coefficient), diameters = case
        row = dict(period_num=periods, D=len(diameters), Np=limit,
                   max_capture_amount=maximum, curve=curve, coef=coefficient,
                   status="FEASIBLE_HEURISTIC", UB=100, LB=80, gap=0.2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            checkpoint = root / batch_ph.CHECKPOINT_FILE.name
            result = root / batch_ph.RESULT_FILE.name
            pd.DataFrame([row]).to_csv(checkpoint, index=False)
            with patch.object(batch_ph, "RESULT_DIR", root), \
                    patch.object(batch_ph, "RESULT_FILE", result), \
                    patch.object(batch_ph, "CHECKPOINT_FILE", checkpoint), \
                    patch.object(batch_ph, "_case_grid", return_value=iter([case])), \
                    patch.object(batch_ph, "OUTPUT_FLAG", False), \
                    patch.object(batch_ph, "load_network_data") as load:
                batch_ph.main()
                load.assert_not_called()
            self.assertEqual(pd.read_csv(result).iloc[0]["UB"], 100)


if __name__ == "__main__":
    unittest.main()
