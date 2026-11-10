"""Small, shared parameter snapshots for single-case result files."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd


def parameter_frame(parameters):
    """Keep scalar cells readable and encode structured values without losing None."""
    rows = []
    for name, value in parameters.items():
        if value is None or isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, Path):
            value = str(value)
        rows.append((name, value))
    return pd.DataFrame(rows, columns=["parameter", "value"])


def build_model_parameters(network, costs, diameters, pipeline_install_limit):
    """Snapshot the supplied instance; never infer costs from current defaults."""
    parameters = deepcopy(getattr(costs, "model_parameters", {}))
    parameters.update(
        period_num=len(costs.periods), periods=list(costs.periods),
        durations=dict(costs.durations), total_years=sum(costs.durations.values()),
        capture_target=dict(costs.capture_target), diameters=list(diameters),
        D=len(diameters), pipeline_install_limit=pipeline_install_limit,
        construction_limits="full_horizon_v1",
        source_count=len(network.sources), source_unit_count=len(network.ig),
        sink_count=len(network.sinks), arc_count=len(network.ij),
        flow_unit="kt_CO2/year", cost_unit="thousand_USD")
    return parameters
