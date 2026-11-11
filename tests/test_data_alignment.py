"""Check the inputs and target conventions specified in Data_Suplementary.tex."""
import importlib
import math
import unittest
from collections import Counter
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import CCS_FNE
import experiment_setup as setup
from capture_target_sequence import generate_capture_targets
from ESM.esm_common import build_capture_target as esm_targets
from heuristic_stages.data_io import build_capture_target as heuristic_targets
from run_gurobi import build_capture_target as gurobi_targets


class DataAlignmentTests(unittest.TestCase):
    def test_all_batch_grids_have_the_paper_counts(self):
        modules = ["CCS_FNE", "CCS_FNE_LP", "CCS_FNE_LP_IM", "CCS_FNE_RLP",
                   "CCS_FNE_RLP_IM", "CCS_FNE_MH", "CCS_FNE_RH", "CCS_FNE_PH",
                   "LRforFNE.CCS_FNE_LR"]
        for name in modules:
            with self.subTest(module=name):
                cases = list(importlib.import_module(name)._case_grid())
                self.assertEqual(len(cases), 224)
                self.assertEqual(len({(t, c, n, curve, tuple(d))
                                      for t, c, n, curve, d in cases}), 224)
                self.assertEqual(Counter(c[3][0] for c in cases),
                                 {"linear": 32, "exponential": 96, "logarithmic": 96})
                self.assertEqual(Counter(len(c[4]) for c in cases), {4: 112, 12: 112})
                self.assertEqual(Counter(c[0] for c in cases),
                                 {5: 56, 10: 56, 15: 56, 30: 56})

    def test_single_case_defaults_are_independent_of_batch_grid(self):
        for name, periods in (("run_gurobi", 15), ("run_PH", 15),
                              ("LRforFNE.run_lrh", 5)):
            module = importlib.import_module(name)
            with self.subTest(module=name):
                self.assertEqual(module.MAX_CAPTURE_AMOUNT, 1e6)
                self.assertEqual(module.PERIOD_NUM, periods)
                self.assertEqual(module.NP, 3)
                self.assertEqual(module.DIAMETERS, setup.D_LIST[-1])
                self.assertFalse(module.NORMALIZE_CAPTURE_TARGET)

    def test_targets_match_equations_and_all_entry_points(self):
        for count, maximum, (curve, coefficient) in product(
                setup.PERIOD_NUM_LIST, setup.MAX_CAPTURE_AMOUNT_LIST, setup.CURVE_SPECS):
            with self.subTest(count=count, maximum=maximum, curve=curve, coef=coefficient):
                def fraction(t):
                    if curve == "exponential":
                        return math.expm1(coefficient * t / count) / math.expm1(coefficient)
                    if curve == "logarithmic":
                        return math.log1p(coefficient * t / count) / math.log1p(coefficient)
                    return t / count

                expected = {t: round(maximum * fraction(t), 3) for t in range(1, count + 1)}
                self.assertEqual(generate_capture_targets(count, maximum, curve, coefficient), expected)
                periods, durations = setup.period_schedule(count)
                costs = setup.CostData(periods, durations, [], [], [], {}, {}, {}, {}, {}, {}, {}, {})
                self.assertEqual(setup.with_capture_target(
                    costs, curve, coefficient, maximum_capture_amount=maximum).capture_target, expected)
                for builder in (gurobi_targets, heuristic_targets, esm_targets):
                    self.assertEqual(builder(count, curve, coefficient, maximum), expected)
                self.assertEqual(expected[count], maximum)
                self.assertEqual(sum(durations.values()), 30)
                linear_total = sum(round(maximum * t / count, 3) for t in periods)
                if curve == "exponential":
                    self.assertLess(sum(expected.values()), linear_total)
                elif curve == "logarithmic":
                    self.assertGreater(sum(expected.values()), linear_total)

    def test_explicit_total_normalization_is_preserved(self):
        for curve, coefficient in setup.CURVE_SPECS[1:]:
            expected = generate_capture_targets(
                15, 2e6, curve, coefficient, normalize_capture_target=True)
            self.assertAlmostEqual(sum(expected.values()), 16e6, places=6)
            self.assertNotEqual(expected[15], 2e6)
            periods, durations = setup.period_schedule(15)
            costs = setup.CostData(periods, durations, [], [], [], {}, {}, {}, {}, {}, {}, {}, {})
            self.assertEqual(setup.with_capture_target(
                costs, curve, coefficient, maximum_capture_amount=2e6,
                normalize_capture_target=True).capture_target, expected)
            for builder in (gurobi_targets, heuristic_targets, esm_targets):
                self.assertEqual(builder(15, curve, coefficient, 2e6,
                                         normalize_capture_target=True), expected)

    def test_reported_cumulative_targets(self):
        for curve, coefficient, gt in (("exponential", 2.5, 20.690), ("logarithmic", 15, 44.250)):
            target = generate_capture_targets(15, 2e6, curve, coefficient)
            self.assertEqual(round(2 * sum(target.values()) / 1e6, 3), gt)

    def test_real_input_counts_and_capacity_units(self):
        network = setup.load_network_data()
        self.assertEqual((len(network.sources), len(network.sinks), len(network.intersections),
                          len(network.ig), len(network.ij)), (580, 271, 1666, 3069, 14372))
        self.assertEqual((min(network.max_wells.values()), max(network.max_wells.values()),
                          sum(network.max_wells.values())), (5, 49, 7573))
        self.assertEqual(network.pipe_capacity["p16"], 6750)
        arcs = set(network.ij)
        self.assertEqual(len(arcs), 14372)
        self.assertTrue(all(u != v and (v, u) in arcs for u, v in arcs))

    def test_solve_result_uses_actual_snapshot_labels(self):
        model = SimpleNamespace(Params=SimpleNamespace(), Status=CCS_FNE.gp.GRB.OPTIMAL,
                                SolCount=1, ObjVal=100.0, MIPGap=0.0, ObjBound=100.0,
                                optimize=lambda callback: None, dispose=lambda: None)
        with patch.object(CCS_FNE, "build_milp", return_value=(model, {})):
            result = CCS_FNE.solve_milp(None, None, [], pipeline_install_limit=3)
        metrics = dict(zip(CCS_FNE.METRIC_COLUMNS, result.metrics()))
        self.assertEqual(metrics, {"UB": 100, "LB": 100, "gap": 0,
                                   "2hUB": 100, "2hLB": 100, "2hgap": 0,
                                   "5hUB": 100, "5hLB": 100, "5hgap": 0})
        self.assertEqual(CCS_FNE.RECORD_TIMES, (("1h", 3600), ("3h", 10800)))


if __name__ == "__main__":
    unittest.main()
