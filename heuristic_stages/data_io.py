"""捕集目标构造与七类变量工作簿输出；路径由主文件传入。"""
from __future__ import annotations

import os
from dataclasses import replace
from math import expm1, isfinite, log1p
from pathlib import Path
from typing import Optional

import pandas as pd

from experiment_setup import CostData, NetworkData
from capture_target_sequence import generate_capture_targets
from result_metadata import parameter_frame
from .common import HeuristicResult


def _capture_fraction(period, period_count, curve, coefficient):
    """单案例曲线的归一化值，最后一期为 1。"""
    if curve == "linear":
        if coefficient is not None:
            raise ValueError("The linear curve does not use a coefficient")
        return period / period_count
    if curve == "exponential":
        if coefficient is None or not isfinite(coefficient) or coefficient == 0:
            raise ValueError("The exponential coefficient must be finite and non-zero")
        return expm1(coefficient * period / period_count) / expm1(coefficient)
    if curve == "logarithmic":
        if (coefficient is None or not isfinite(coefficient)
                or coefficient <= -1 or coefficient == 0):
            raise ValueError("The logarithmic coefficient must be finite, greater than -1, and non-zero")
        return log1p(coefficient * period / period_count) / log1p(coefficient)
    raise ValueError(f"Unknown capture-target curve: {curve}")


def build_capture_target(
    period_count: int, curve: str, coefficient: Optional[float],
    maximum_capture_amount: Optional[float],
    capture_amount_sequence: Optional[list[float] | tuple[float, ...]] = None,
    *, normalize_capture_target: bool = False,
) -> dict[int, float]:
    """可选按线性基准等总量归一化非线性曲线；显式逐期目标优先。"""
    if period_count <= 0:
        raise ValueError("period_count must be positive")
    if capture_amount_sequence is not None:
        if len(capture_amount_sequence) != period_count:
            raise ValueError(f"CAPTURE_AMOUNT_SEQUENCE must have {period_count} values, got {len(capture_amount_sequence)}")
        if any(not isfinite(value) or value < 0 for value in capture_amount_sequence):
            raise ValueError("CAPTURE_AMOUNT_SEQUENCE must contain finite, non-negative values")
        return {t: float(value) for t, value in enumerate(capture_amount_sequence, start=1)}
    if (maximum_capture_amount is None or not isfinite(maximum_capture_amount)
            or maximum_capture_amount < 0):
        raise ValueError("maximum_capture_amount must be finite and non-negative")
    if normalize_capture_target and curve != "linear":
        return generate_capture_targets(period_count, maximum_capture_amount, curve, coefficient)
    return {t: round(maximum_capture_amount * _capture_fraction(t, period_count, curve, coefficient), 3)
            for t in range(1, period_count + 1)}


def with_local_capture_target(
    costs: CostData, curve: str, coefficient: Optional[float],
    maximum_capture_amount: Optional[float],
    capture_amount_sequence: Optional[list[float] | tuple[float, ...]] = None,
    *, normalize_capture_target: bool = False,
) -> CostData:
    """仅替换捕集目标，保留传入的全部期间成本与索引。"""
    count = len(costs.periods)
    if costs.periods != list(range(1, count + 1)):
        raise ValueError("cost periods must be consecutive and start at 1")
    target = build_capture_target(count, curve, coefficient, maximum_capture_amount,
                                  capture_amount_sequence,
                                  normalize_capture_target=normalize_capture_target)
    parameters = dict(costs.model_parameters)
    parameters.update(curve=curve, coefficient=coefficient,
                      max_capture_amount=maximum_capture_amount,
                      normalize_capture_target=normalize_capture_target,
                      capture_amount_sequence=None if capture_amount_sequence is None else list(capture_amount_sequence),
                      target_source="generated_curve" if capture_amount_sequence is None else "explicit_sequence")
    return replace(costs, capture_target=target, model_parameters=parameters)


def _variable_frame(index_columns: list[str], index_rows: list[tuple],
                    periods: list[int], values: dict) -> pd.DataFrame:
    """按设施索引和 period1/period2/... 列生成变量宽表。"""
    rows = [[*indices, *(values.get((*indices, t), 0) for t in periods)] for indices in index_rows]
    return pd.DataFrame(rows, columns=[*index_columns, *(f"period{t}" for t in periods)])


def save_heuristic_results(result: HeuristicResult, network: NetworkData,
                           costs: CostData, diameters: list[str],
                           output_file: Path, *, include_logs: bool = False) -> Path:
    """输出七张变量宽表；可附加参数、求解摘要及阶段日志，无解时仍保存这些记录。"""
    target = Path(output_file)
    unit_rows = (["source", "unit"], list(network.ig))
    sink_rows = (["sink"], [(j,) for j in network.sinks])
    pipe_rows = (["start", "end", "diameter"], [(u, v, d) for u, v in network.ij for d in diameters])
    layouts = dict(s=unit_rows, a=unit_rows, r=sink_rows, w=sink_rows,
                   b=sink_rows, x=pipe_rows, y=pipe_rows)
    frames = {f"variable_{name}": _variable_frame(columns, rows, costs.periods, getattr(result, name))
              for name, (columns, rows) in layouts.items()} if result.objective is not None or not include_logs else {}
    if include_logs:
        metrics = ("status", "objective", "lower_bound", "gap", "runtime",
                   "lower_bound_origin", "lp_status", "lp_objective", "lp_runtime")
        frames["solve_summary"] = pd.DataFrame(
            [(name, getattr(result, name)) for name in metrics], columns=["metric", "value"])
        frames["model_parameters"] = parameter_frame(result.configuration.get("model_parameters", {}))
        algorithm_parameters = dict(result.configuration.get("options", {}))
        for name in ("initialization", "time_limit_scope"):
            if name in result.configuration:
                algorithm_parameters[name] = result.configuration[name]
        frames["algorithm_parameters"] = parameter_frame(algorithm_parameters)
        frames["solve_log"] = (pd.DataFrame(result.history) if result.history else
                               pd.DataFrame(columns=["stage", "elapsed", "best_objective"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            for sheet, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
