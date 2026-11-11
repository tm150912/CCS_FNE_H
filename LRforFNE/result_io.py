"""JSON with typed key arrays and single-case workbooks inside this package."""
from pathlib import Path
from dataclasses import fields
import json
import numpy as np
import pandas as pd
from heuristic_stages.common import Plan, VARIABLES
from heuristic_stages.data_io import _variable_frame
from result_metadata import parameter_frame

OUTPUT_ROOT = Path(__file__).resolve().parent / 'output'


def plan_to_records(plan):
    return {name:[dict(key=list(key), value=value) for key,value in getattr(plan,name).items() if value != 0]
            for name in VARIABLES}


def plan_from_records(data):
    if set(data) != set(VARIABLES):
        raise ValueError('Expected all seven Plan variable groups')
    return Plan(**{name:{tuple(row['key']):row['value'] for row in data[name]} for name in VARIABLES})


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        # Diagnostic tuple keys are records, never executable tuple strings.
        if any(not isinstance(key,(str,int,float,bool)) and key is not None for key in value):
            return [dict(key=_json_value(key), value=_json_value(v)) for key,v in value.items()]
        return {key:_json_value(v) for key,v in value.items()}
    if isinstance(value,(list,tuple,set)):
        return [_json_value(v) for v in value]
    return value


def _write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_result(result, case_name, *, output_root=None):
    root = Path(OUTPUT_ROOT if output_root is None else output_root).resolve()
    destination = (root / case_name / result.method).resolve()
    if root not in destination.parents:
        raise ValueError('Output must remain inside the selected output root')
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {item.name:getattr(result,item.name) for item in fields(result)
                if item.name not in ('solution','history','best_multipliers','initial_multipliers',
                                     'initialization_history')}
    metadata['logs'] = dict(initialization='initialization_history.jsonl',
        iterations='history.jsonl', primal_search_and_diagnostics='result.json:diagnostics',
        format='structured stage and iteration records')
    _write(destination/'result.json', metadata)
    _write(destination/'solution.json', plan_to_records(result.solution))
    for name, history in (('history.jsonl', result.history),
                          ('initialization_history.jsonl', getattr(result, 'initialization_history', []))):
        history_file = destination/name
        temporary = history_file.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            for row in history:
                stream.write(json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False)+'\n')
        temporary.replace(history_file)
    values = []
    if result.best_multipliers is not None:
        values = [dict(node=node, period=h, value=float(result.best_multipliers[i,j]))
                  for i,node in enumerate(result.instance['nodes'])
                  for j,h in enumerate(result.instance['periods'])]
    _write(destination/'best_multipliers.json', dict(instance=result.instance,
           evaluation_id=result.multiplier_evaluation, lower_bound=result.multiplier_bound,
           source='LR_oracle_ell' if values else 'unavailable', values=values))
    initial_values = []
    if result.initial_multipliers is not None:
        initial_values = [dict(node=node, period=h, value=float(result.initial_multipliers[i,j]))
                          for i,node in enumerate(result.instance['nodes'])
                          for j,h in enumerate(result.instance['periods'])]
    _write(destination/'initial_multipliers.json', dict(instance=result.instance,
        initialization=result.configuration.get('multiplier_initialization'), values=initial_values))
    return destination


def write_workbook(result, network, costs, diameters, output_file):
    """保存原可行方案、实际参数及分开的初始化/对偶/原可行解搜索日志。"""
    target = Path(output_file).resolve()
    if OUTPUT_ROOT.resolve() not in target.parents:
        raise ValueError('Output must remain inside LRforFNE/output')
    frames = {}
    if result.objective is not None:
        unit_rows = (["source", "unit"], list(network.ig))
        sink_rows = (["sink"], [(j,) for j in network.sinks])
        pipe_rows = (["start", "end", "diameter"],
                     [(u, v, d) for u, v in network.ij for d in diameters])
        layouts = dict(s=unit_rows, a=unit_rows, r=sink_rows, w=sink_rows,
                       b=sink_rows, x=pipe_rows, y=pipe_rows)
        frames.update({f"variable_{name}": _variable_frame(
            columns, rows, costs.periods, getattr(result.solution, name))
            for name, (columns, rows) in layouts.items()})
    metrics = ("method", "status", "termination_reason", "initial_objective",
               "objective", "lower_bound", "gap", "lower_bound_origin",
               "warm_start_runtime", "lr_runtime", "total_runtime")
    frames["solve_summary"] = pd.DataFrame(
        [(name, getattr(result, name)) for name in metrics], columns=["metric", "value"])
    frames["model_parameters"] = parameter_frame(result.configuration.get("model_parameters", {}))
    frames["instance"] = parameter_frame(result.instance)
    frames["algorithm_parameters"] = parameter_frame(result.configuration.get("algorithm", {}))
    frames["initialization_parameters"] = parameter_frame({name: result.configuration.get(name)
        for name in ("initialization_options", "warm_start", "multiplier_initialization",
                     "initial_lower_bound", "initial_lower_bound_origin")})
    frames["primal_parameters"] = parameter_frame(result.configuration.get("primal", {}))
    frames["invocation"] = parameter_frame(result.configuration.get("invocation", {}))
    frames["audit"] = parameter_frame(result.audit or {})
    frames["timings"] = parameter_frame(result.timings)
    for sheet, history in (("initialization_log", result.initialization_history),
                           ("iteration_log", result.history),
                           ("primal_log", result.diagnostics)):
        records = [{name: json.dumps(_json_value(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, set, np.ndarray)) else _json_value(value)
                    for name, value in row.items()} for row in history]
        frames[sheet] = pd.DataFrame(records) if records else pd.DataFrame(columns=["stage"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            for sheet, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet, index=False)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
