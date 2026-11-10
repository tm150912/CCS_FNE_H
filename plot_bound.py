"""Plot E-Select upper and lower bound updates without network data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


DPI = 150
BOUND_PLOT_TITLE = "E-Select | Bound convergence"
BOUND_FIGURE_SIZE = (9, 5.4)


def _number(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number")
    return result


def plot_bound_history(history, *, output_file: str | Path,
                           title=BOUND_PLOT_TITLE, dpi=DPI,
                           objective_label="E-Select objective value") -> Path:
    """Save E-Select ``bound_history`` as a step chart.

    The zero-based x coordinate is the snapshot's position, not a local-search
    iteration or elapsed time. No iteration/timestamp fields are required.
    Only upper_bound/lower_bound are read, in their original order. Missing,
    nonnumeric or nonfinite bounds leave gaps; nothing is filled, clipped or
    recomputed. The caller remains responsible for the validity of the bounds.
    Empty histories produce an annotated chart; no network or Excel is needed.
    """
    if history is None or isinstance(history, (Mapping, str, bytes)):
        raise TypeError("history must be an iterable of bound snapshot mappings")
    x_values, upper_values, lower_values = [], [], []
    for index, snapshot in enumerate(history):
        if not isinstance(snapshot, Mapping):
            raise TypeError(f"history[{index}] must be a mapping")
        x_values.append(index)
        for key, values in (("upper_bound", upper_values), ("lower_bound", lower_values)):
            try:
                value = float(snapshot.get(key))
            except (TypeError, ValueError, OverflowError):
                value = math.nan
            values.append(value if math.isfinite(value) else math.nan)
    dpi = _number(dpi, "dpi")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    path = Path(output_file)
    if path.suffix.lower() not in (".png", ".svg"):
        raise ValueError("output_file must end in .png or .svg")
    figure = None
    try:
        figure = Figure(figsize=BOUND_FIGURE_SIZE, dpi=dpi)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        axis.set_title(title)
        axis.set_xlabel("Bound-update index (0-based)")
        axis.set_ylabel(objective_label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 5), useMathText=True)
        axis.grid(True, alpha=0.25)
        has_bounds = False
        for values, label, linestyle, marker in (
            (upper_values, "Upper bound (UB)", "-", "o"),
            (lower_values, "Lower bound (LB)", "--", "s"),
        ):
            if any(math.isfinite(value) for value in values):
                axis.step(x_values, values, where="post", label=label,
                          linestyle=linestyle, marker=marker, markersize=4)
                has_bounds = True
        # Include all observed updates, but never extend a bound backwards.
        axis.set_xlim(-0.1, max(1.0, (x_values[-1] if x_values else 0) + 0.1))
        if has_bounds:
            axis.legend(frameon=False)
        else:
            axis.text(0.5, 0.5, "No valid bounds", ha="center", va="center",
                      transform=axis.transAxes)
            axis.set_yticks([])
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        if figure is not None:
            figure.clear()
            figure.set_canvas(None)
    return path


