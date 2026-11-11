# Lagrangian reproduction workflow

This directory implements the standard proximal Bundle workflow described in
`Data_Suplementary.tex`. It reads the six processed CSV inputs in the project
root; spatial preprocessing is not required for a normal optimization run.

## Entry points

Run these commands from the project root:

```powershell
python -B LRforFNE/run_lrh.py
python -B LRforFNE/CCS_FNE_LR.py
python -B -m LRforFNE.run_lr --help
```

- `run_lrh.py` is the editable single-case entry. Its defaults are 5 periods,
  NP = 3, 12 pipeline classes, a linear profile and 1,000,000 kt/year.
- `CCS_FNE_LR.py` is the editable batch entry. Its model grid comes from
  `experiment_setup.py`, independently of the single-case settings.
- `run_lr.py` is the command-line entry; `--batch` calls the same batch runner.
  Single-case CLI results use timestamped directories.

The previous editable `run_LR.py` has been moved to `run_lrh.py`. On Windows,
`run_LR.py` and `run_lr.py` denote the same filename, so editable configuration
and CLI code use the two distinct names above.

`METHOD`, CLI `--method`, and the Python `method` argument accept only `bundle`.
Both editable LR entries use a 2,400-second feasible-initialization budget,
an unlimited multiplier LP, a 1,200-second LR budget and at most 100 oracle
evaluations. The multiplier LP and initialization are outside the LR deadline.
Reaching the evaluation cap may leave time for further feasible-plan search.

## Inputs and units

The input files are `nodes.csv`, `arcs.csv`, `hub_unit_location_cost.csv`,
`sink_subsite_cost.csv`, `pipeline_capacity.csv` and `pipeline_cost.csv`.
`read.py` validates identifiers, coverage and arc-class completeness.
The supplied network contains 580 source hubs, 3,069 capture units, 271 storage
subsites, 1,666 transshipment nodes and 14,372 directed arcs.

Model flows and targets are annual kt/year. The objective is in thousands of
dollars. Loading divides tonne-based capacities and dollar-based fixed costs
by 1,000; per-tonne operating rates keep their numerical values. NP counts
all pipelines installed on one directed arc across all classes and periods.

The 224-case batch grid crosses 4/12 classes, 5/10/15/30 periods, NP = 1/3,
0.5/2 Gt/year terminal targets and seven target profiles. Generated targets
have a common terminal value and are rounded to three decimal places.
`NORMALIZE_CAPTURE_TARGET=True` explicitly requests equal cumulative targets
instead. An explicit single-case `CAPTURE_AMOUNT_SEQUENCE` takes precedence.

## Initialization, bounds and results

The primal heuristic first constructs and audits a feasible plan. Its separate
bound-only LP is skipped in this initialization. One full original-model LP
then uses barrier without crossover to initialize nodal multipliers.
Capture and storage oracles use Gurobi; the base transportation oracle is
analytical and separates by directed arc. Bundle iterations use certified
oracle bounds. Only plans audited against the original model supply upper
bounds. A neighborhood objective or bound is not a global lower bound.

Editable single-case output is under `output/run_lrh/bundle/`, or
`output/run_lrh/normalized/bundle/` with normalization enabled:

| File | Contents |
| --- | --- |
| `result.json` | Bounds, status, termination reason, settings, fingerprint, timing and audit |
| `solution.json` | Sparse records of s, r, w, y, a, b and x with array-valued keys |
| `history.jsonl` | Oracle and Bundle iteration records |
| `initialization_history.jsonl` | Feasible initialization and multiplier-LP records |
| `initial_multipliers.json` | Initial LP multipliers and metadata |
| `best_multipliers.json` | Multipliers associated with the best certified oracle bound |
| `variables.xlsx` | Variable sheets, settings, summary and structured logs |

Without a feasible plan, the workbook omits variable sheets. LR entries do not
automatically generate maps. Use the shared `plot_results.py` separately.

Batch output is under `output/batch/`, with a separate `normalized/` subtree
when requested. It contains `CCS_FNE_LR_bundle_30Y_result.csv` and detailed
per-case JSON directories under `cases/`. JSON is written before the summary
CSV is atomically replaced. Batch runs do not produce variable workbooks.

Resume keys are period count, catalogue size, NP, maximum target, curve and
coefficient. Normally completed runs are skipped; failed initialization,
solver errors and interruptions remain eligible for retry. Use a new result
directory after changing data, same-size catalogue contents or algorithm
settings. Resume does not restore internal Bundle iterations. A feasible
time-limit or iteration-limit status is not a proof of optimality.

## Verification

```powershell
python -B -m LRforFNE.run_lr --tiny --method bundle `
  --max-iterations 12 --lr-time-limit 20 `
  --warm-start-time-limit 10 --lp-time-limit 5 --quiet
New-Item -ItemType Directory -Force tmp | Out-Null
python -B -m pytest -q -p no:cacheprovider `
  --basetemp=tmp/lr_tests LRforFNE/tests
python -B -m unittest discover -s tests -v
```

The tiny fixture uses an artificial small network, not the national data.
A valid Gurobi license is required for solver-based checks. The supplement's
reference environment is Python 3.10 and Gurobi 12.0.3. The available local
validation environment is Python 3.9.12 and Gurobi 11.0.0; matching the reference
environment is necessary for a controlled runtime comparison.
The optimization workflow also needs NumPy, pandas, NetworkX, openpyxl,
Matplotlib and SciPy; `pytest` runs the tests. SciPy is used by Gurobi's
`Model.getA()` in the RSS initialization diagnostics. These solver dependencies
are listed in the root `requirements.txt`; ArcPy/GDAL are only needed for the
separate spatial preprocessing stages described in their local guides.

See `STRENGTHENING.md` for the boundary between standard certified bounds,
primal improvements and additional algorithm extensions.
