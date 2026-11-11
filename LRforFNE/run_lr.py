"""Command-line Bundle entry; editable single-case settings live in run_lrh.py."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'LRforFNE'

from experiment_setup import D_LIST, compute_cost_data, load_network_data, period_schedule
from heuristic_stages.data_io import with_local_capture_target
from . import run_lrh
from .result_io import OUTPUT_ROOT, write_result
from .solver import solve_lr
from .tiny_instance import build_tiny_instance


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--batch', action='store_true', help='Run the shared experiment grid.')
    mode.add_argument('--tiny', action='store_true', help='Run the built-in two-period fixture.')
    parser.add_argument('--method', choices=['bundle'], default='bundle')
    parser.add_argument('--data-dir', type=Path, help='Directory containing the six input CSVs.')
    parser.add_argument('--output-dir', type=Path, help='Result root; single runs add a timestamp.')
    parser.add_argument('--period-num', type=int, help='Single case: number of equal periods.')
    parser.add_argument('--np', type=int, help='Single case: full-horizon directed-arc pipe limit.')
    parser.add_argument('--diameters', type=int, choices=[4, 12], help='Single case: catalogue size.')
    parser.add_argument('--curve', choices=['linear', 'exponential', 'logarithmic'])
    parser.add_argument('--coefficient', type=float)
    parser.add_argument('--max-capture-amount', type=float, help='Single case: annual target in kt/year.')
    parser.add_argument('--capture-amount-sequence', type=float, nargs='+',
                        help='Explicit annual targets, overriding the generated single-case curve.')
    parser.add_argument('--normalize-capture-target', action='store_true', default=None,
                        help='Normalize nonlinear targets to the linear cumulative quantity.')
    parser.add_argument('--max-iterations', type=int, help='Maximum oracle evaluations.')
    parser.add_argument('--lr-time-limit', type=float, help='Bundle and subsequent primal-search seconds.')
    parser.add_argument('--warm-start-time-limit', type=float, help='Feasible-initialization seconds.')
    parser.add_argument('--lp-time-limit', type=float, help='Multiplier-LP seconds; default unlimited.')
    parser.add_argument('--quiet', action='store_true', help='Suppress iteration and solver output.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    model_options = ('period_num', 'np', 'diameters', 'curve', 'coefficient',
                     'max_capture_amount', 'capture_amount_sequence')
    if (args.batch or args.tiny) and any(getattr(args, name) is not None for name in model_options):
        parser.error('--batch and --tiny supply their own model settings')
    if args.tiny and args.normalize_capture_target:
        parser.error('--tiny uses its fixed capture targets')
    if args.batch:
        from . import CCS_FNE_LR as entry
    else:
        entry = run_lrh
    overrides = {name: getattr(args, name) for name in ('max_iterations', 'lr_time_limit')
                 if getattr(args, name) is not None}
    if args.quiet:
        overrides.update(output_flag=False, gurobi_output_flag=False)
    config = entry.build_config(**overrides)
    options = entry.build_heuristic_options()
    warm_overrides = {}
    if args.warm_start_time_limit is not None:
        warm_overrides['time_limit'] = args.warm_start_time_limit
    if args.lp_time_limit is not None:
        if args.lp_time_limit <= 0:
            parser.error('--lp-time-limit must be positive')
        warm_overrides['lp_time_limit'] = args.lp_time_limit
    if args.quiet:
        warm_overrides.update(output_flag=False, gurobi_output_flag=False)
    options = replace(options, **warm_overrides)
    options.validate()
    if args.batch:
        return entry.run_batch(config=config, heuristic_options=options,
            data_dir=args.data_dir, result_dir=args.output_dir,
            normalize_capture_target=args.normalize_capture_target, method=args.method)

    started = perf_counter()
    normalize = (run_lrh.NORMALIZE_CAPTURE_TARGET if args.normalize_capture_target is None
                 else args.normalize_capture_target)
    if args.tiny:
        normalize = False
        network, costs, diameters, pipeline_limit = build_tiny_instance()
        invocation = dict(entrypoint='LRforFNE.run_lr', method=args.method, tiny=True,
                          normalize_capture_target=False)
    else:
        data_dir = args.data_dir or run_lrh.DATA_DIR
        network = load_network_data(data_dir)
        period_num = run_lrh.PERIOD_NUM if args.period_num is None else args.period_num
        periods, durations = period_schedule(period_num)
        diameters = (list(run_lrh.DIAMETERS) if args.diameters is None else
                     list(next(catalogue for catalogue in D_LIST if len(catalogue) == args.diameters)))
        pipeline_limit = run_lrh.NP if args.np is None else args.np
        curve = run_lrh.CURVE if args.curve is None else args.curve
        coefficient = run_lrh.COEFFICIENT if args.coefficient is None else args.coefficient
        maximum = run_lrh.MAX_CAPTURE_AMOUNT if args.max_capture_amount is None else args.max_capture_amount
        sequence = (run_lrh.CAPTURE_AMOUNT_SEQUENCE if args.capture_amount_sequence is None
                    else args.capture_amount_sequence)
        costs = with_local_capture_target(
            compute_cost_data(network, periods, durations, apply_shared_capture_target=False),
            curve, coefficient, maximum, sequence, normalize_capture_target=normalize)
        invocation = dict(entrypoint='LRforFNE.run_lr', method=args.method, tiny=False,
            data_dir=str(Path(data_dir).resolve()), period_num=period_num,
            pipeline_install_limit=pipeline_limit, diameters=diameters, curve=curve,
            coefficient=coefficient, max_capture_amount=maximum,
            capture_amount_sequence=sequence, normalize_capture_target=normalize)
    preparation_runtime = perf_counter() - started
    result = solve_lr(network, costs, diameters, pipeline_install_limit=pipeline_limit,
                      method=args.method, config=config, heuristic_options=options)
    result.configuration['invocation'] = invocation
    result.configuration['normalize_capture_target'] = normalize
    result.timings['data_and_cost_preparation_wall'] = preparation_runtime
    case_name = Path('single') / ('normalized' if normalize else '')
    case_name /= datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = write_result(result, case_name, output_root=args.output_dir or OUTPUT_ROOT)
    print(f'status={result.status}, UB={result.objective}, LB={result.lower_bound}, '
          f'gap={result.gap}, results={destination}')
    return result


if __name__ == '__main__':
    main()
