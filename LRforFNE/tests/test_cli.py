"""CLI routing, editable settings, and a small mathematical integration check."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from CCS_FNE import build_milp
from LRforFNE import CCS_FNE_LR, run_lr, run_lrh
from LRforFNE.solver import solve_lr
from LRforFNE.tiny_instance import build_tiny_instance


def test_editable_single_defaults():
    assert (run_lrh.PERIOD_NUM, run_lrh.NP, len(run_lrh.DIAMETERS)) == (5, 3, 12)
    assert run_lrh.MAX_CAPTURE_AMOUNT == 1e6
    assert run_lrh.METHOD == 'bundle'
    assert run_lrh.RESULT_DIR.name == 'run_lrh'
    assert run_lrh.build_config().lr_time_limit == 1200
    assert run_lrh.build_config().max_iterations == 100
    options = run_lrh.build_heuristic_options()
    assert options.time_limit == 2400
    assert options.lp_time_limit is None


def test_only_bundle_is_accepted():
    with pytest.raises(SystemExit):
        run_lr.build_parser().parse_args(['--method', 'subgradient'])
    with pytest.raises(ValueError, match='method must be bundle'):
        solve_lr(None, None, [], pipeline_install_limit=3, method='subgradient')


def test_cli_batch_uses_shared_runner(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(CCS_FNE_LR, 'run_batch', lambda **kwargs: calls.append(kwargs))
    run_lr.main(['--batch', '--quiet', '--output-dir', str(tmp_path),
                 '--max-iterations', '12', '--lr-time-limit', '20',
                 '--warm-start-time-limit', '10', '--lp-time-limit', '5'])
    call, = calls
    assert call['method'] == 'bundle'
    assert call['result_dir'] == tmp_path
    assert call['config'].max_iterations == 12
    assert call['config'].lr_time_limit == 20
    assert not call['config'].output_flag
    assert call['heuristic_options'].time_limit == 10
    assert call['heuristic_options'].lp_time_limit == 5


def test_cli_tiny_writes_timestamped_json_without_loading_csv(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError('tiny mode must not read the national CSV files')
    monkeypatch.setattr(run_lr, 'load_network_data', unexpected)
    result = SimpleNamespace(configuration={}, timings={}, status='TEST',
                             objective=10, lower_bound=9, gap=.1)
    calls = []
    monkeypatch.setattr(run_lr, 'solve_lr', lambda *args, **kwargs: result)
    def write(value, case_name, *, output_root):
        calls.append((value, case_name, output_root))
        return output_root / case_name / 'bundle'
    monkeypatch.setattr(run_lr, 'write_result', write)
    assert run_lr.main(['--tiny', '--quiet', '--output-dir', str(tmp_path)]) is result
    value, name, root = calls[0]
    assert value is result and root == tmp_path
    assert name.parts[0] == 'single'
    assert len(name.parts) == 2 and name.name[:8].isdigit()
    assert result.configuration['invocation']['tiny'] is True


def test_editable_entry_exports_expected_workbook_without_plotting(monkeypatch, tmp_path):
    network, costs, diameters, _ = build_tiny_instance()
    monkeypatch.setattr(run_lrh, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(run_lrh, 'RESULT_DIR', tmp_path / 'run_lrh')
    monkeypatch.setattr(run_lrh, 'load_network_data', lambda path: network)
    monkeypatch.setattr(run_lrh, 'compute_cost_data', lambda *args, **kwargs: costs)
    monkeypatch.setattr(run_lrh, 'with_local_capture_target', lambda *args, **kwargs: costs)
    result = SimpleNamespace(configuration={}, timings={}, status='TEST',
        termination_reason='test', objective=10, lower_bound=9, gap=.1, total_runtime=0)
    monkeypatch.setattr(run_lrh, 'solve_lr', lambda *args, **kwargs: result)
    monkeypatch.setattr(run_lrh, 'write_result',
                        lambda result, name: tmp_path / name / 'bundle')
    targets = []
    def write_workbook(result, network, costs, diameters, target):
        targets.append(target)
        return target
    monkeypatch.setattr(run_lrh, 'write_workbook', write_workbook)
    run_lrh.main()
    assert targets == [tmp_path / 'run_lrh' / 'bundle' / 'variables.xlsx']
    assert not hasattr(run_lrh, 'plot_results')


@pytest.mark.parametrize('args', [
    ['--tiny', '--batch'], ['--tiny', '--period-num', '5'],
    ['--batch', '--np', '1'], ['--tiny', '--normalize-capture-target'],
    ['--lp-time-limit', '0'],
])
def test_cli_rejects_conflicting_or_invalid_options(args):
    with pytest.raises(SystemExit):
        run_lr.main(args)


def test_tiny_original_model_and_bundle_from_external_feasible_plan():
    """Check the real LP and Bundle; PH search is deliberately outside this test."""
    network, costs, diameters, limit = build_tiny_instance()
    model, variables = build_milp(network, costs, diameters, limit)
    try:
        model.Params.OutputFlag = 0
        model.optimize()
        assert model.Status == 2
        optimum = model.ObjVal
        initial = SimpleNamespace(objective=optimum, runtime=model.Runtime,
            status='OPTIMAL', history=[],
            **{name: {key: var.X for key, var in group.items()}
               for name, group in variables.items()})
        relaxed = model.relax()
        try:
            relaxed.Params.OutputFlag = 0
            relaxed.optimize()
            assert relaxed.ObjVal < optimum - 1e-4
        finally:
            relaxed.dispose()
    finally:
        model.dispose()
    options = replace(run_lrh.build_heuristic_options(), lp_time_limit=5,
                      output_flag=False, gurobi_output_flag=False)
    config = run_lrh.build_config(lr_time_limit=5, max_iterations=12,
        neighborhoods_enabled=False, output_flag=False, gurobi_output_flag=False)
    result = solve_lr(network, costs, diameters, pipeline_install_limit=limit,
        config=config, heuristic_options=options, initial_result=initial)
    assert result.objective == pytest.approx(optimum)
    assert result.lower_bound <= optimum + 1e-5
    assert result.initial_multipliers is not None
    assert result.history
    assert result.configuration['multiplier_initialization']['status'] == 'OPTIMAL'
    assert result.audit['max_violation'] <= config.feasibility_tolerance
