# Strengthening scope

The standard implementation uses the full original-model LP and the proximal
Bundle oracle as its global lower-bound sources. The standard entries are
`run_lrh.py`, `CCS_FNE_LR.py` and `run_lr.py`; none enables an additional
cut-generation or branching algorithm.

## Implemented improvement and diagnostic routines

| Component | Code | Meaning of its result |
| --- | --- | --- |
| Oracle refinement | `solver.py`, `subproblems.py` | Intersects certified intervals at the same multiplier vector |
| Bundle diagnostic | `dual_diagnostic.py` | A dual-model diagnostic, not a replacement global primal lower bound |
| Construction delay, time windows and joint route/facility search | `neighborhoods.py` | An audited improving feasible plan can update the upper bound |
| Corridor expansion | `corridor_expansion.py` | Enlarges a primal search neighborhood |
| Pipeline polishing | `pipeline_polish.py` | Improves an audited feasible plan within its search domain |

The neighborhood and polishing controls are in the LR neighborhood section
of each editable entry. Some of these primal routines are enabled in the
standard workflow. They do not establish new global lower bounds from their
restricted subproblems.

## Additional lower-bound extensions

The supplied source tree contains no implementation of extra joint capacity
cuts, dynamic Lagrangianization or a branching framework. The supplementary
description refers to optional strengthening routines but does not define
their algorithms or provide their source. This file records that limitation;
it does not expose an unimplemented switch or claim those extensions were run.

Any future such extension must retain validity for the original instance,
record the source of each certified bound, and keep restricted-neighborhood
bounds separate from global bounds. Modified multiplier dimensions must stay
consistent across the oracle, Bundle model and saved multiplier records.
Compare an extension with the standard workflow using separate result
directories, since the six-field resume key does not identify algorithm changes.
