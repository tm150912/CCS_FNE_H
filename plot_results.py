"""Render solved CCS construction plans without importing or running a solver.

Call :func:`plot_results` with the in-memory result and network. Shared rendering
settings live below; each main script keeps its output paths. Index shapes identify static
or time-indexed construction; an explicit period sequence preserves idle periods
when a sparse result cannot reveal its complete horizon. Pipeline counts are nonnegative integers,
so an arc's installed capacity is the sum of ``pipe_capacity[d] * y[i,j,d,h]``.
Only ``s``, ``r``, ``w`` and ``y`` are read. Use :func:`plot_results_from_excel`
to read these construction decisions from an exported variable workbook.
Coordinates and the optional supplied China boundary use WGS84.
Use :func:`plot_bound_history` for solver-bound convergence, without map inputs.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from read import read_well_injection_capacities

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# Shared controls for every plotting entry point.
# Output directories and image names remain in each solver's single-case block.
SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ENABLED = True
PLOT_BOUNDARY_FILE = (SCRIPT_DIR / "triangle network" / "China_boundary"
                      / "\u4e2d\u56fd\u8f6e\u5ed3\u7ebf.shp")  # None disables the basemap.
PLOT_VALUE_TOLERANCE = 1e-6  # Construction-value rounding tolerance, in [0, 0.5).

PLOT_STYLE = {
    "figure_size": (14, 9.2), "dpi": 150,  # Inches and pixels per inch.
    "page_color": "#FFFFFF", "map_color": "#FFFFFF", "land_color": "#FFFFFF",
    "boundary_color": "#B7C0C7", "candidate_color": "#DDE3E8", "dash_color": "#66777D",
    # Single-period maps use facility-type colors; evolution maps use status colors.
    "single_period_source_color": "#E69F45",  # Orange circles.
    "single_period_sink_color": "#87CEEB",  # Sky-blue subsite squares.
    "single_period_pipeline_color": "#969696",  # Gray arcs.
    "source_color": "#1A7CBB", "sink_color": "#1A7CBB",  # New.
    "expanded_source_color": "#D55E00", "expanded_sink_color": "#D55E00",
    "existing_source_color": "#2CA02C", "existing_sink_color": "#2CA02C",
    # This outline applies to ALL node states and legend point symbols, never pipes.
    "new_facility_edge_color": "#000000", "facility_edge_width": 1.15,
    "pipeline_new_color": "#1A7CBB", "pipeline_expanded_color": "#D55E00",
    "pipeline_existing_color": "#2CA02C",
    "pipeline_width_min": 1.5, "pipeline_width_max": 5.0, "pipeline_zorder": 10.0,
    "facility_area_min": 20.0, "facility_area_max": 120.0, "facility_zorder": 13.0,
    # Widths use points; node areas use squared points and scale with installed capacity.
    "mainland_extent": (72.0, 136.0, 17.5, 54.8),  # West, east, south, north (degrees).
    "inset_extent": (107.5, 123.5, 2.5, 25.5),
    "show_candidate_network": True, "show_inset": True, "show_legend": True,
    "font_family": ["DejaVu Sans", "Arial", "sans-serif"],
    "legend_fontsize": 12.0, "legend_marker_size": 10.5,
    "boundary_linewidth": 0.45, "candidate_linewidth": 0.28, "candidate_alpha": 0.18,
    "dash_linewidth": 0.75, "inset_bounds": (0.858, 0.025, 0.127, 0.283),
    "map_reference_latitude": 35.0, "inset_reference_latitude": 14.0,
    "figure_margins": {"left": 0.015, "right": 0.985, "bottom": 0.025, "top": 0.985},
    "legend_location": "lower left", "legend_anchor": (0.015, 0.025),
    "legend_type_color": "#A6AEB8", "legend_status_anchor": (0.19, 0.025),
    "legend_status_handlelength": 5.2,
    "legend_handlelength": 3.0, "legend_handletextpad": 1.0, "legend_labelspacing": 0.9,
    "inset_boundary_linewidth": 0.25, "inset_dash_linewidth": 0.9, "inset_frame_linewidth": 0.75,
}

# Bound-chart controls are independent of geographic construction-plan styling.
# Solver entry points own the output path and choose iteration or elapsed time.
BOUND_PLOT_STYLE = {
    "figure_size": (8.2, 4.8), "dpi": 180,
    "page_color": "#FFFFFF", "axes_color": "#FFFFFF",
    "upper_bound_color": "#D55E00", "lower_bound_color": "#0072B2",
    "upper_bound_linestyle": "-", "lower_bound_linestyle": "--",
    "upper_bound_marker": "o", "lower_bound_marker": "s",
    "line_width": 1.8, "marker_size": 2.8,
    "font_family": ["DejaVu Sans", "Arial", "sans-serif"],
    "font_size": 10.5, "title_fontsize": 12.0, "legend_fontsize": 10.0,
    "legend_location": "best", "show_legend": True,
    "grid_color": "#DDE3E8", "grid_linewidth": 0.6, "grid_alpha": 0.8,
    "axes_color_line": "#6B7280", "axes_linewidth": 0.7,
    "y_label": "Objective value",
}

_DEFAULT_BOUNDARY = object()


@dataclass(frozen=True)
class ChinaBoundary:
    outline_parts: tuple[np.ndarray, ...]
    mainland_ring: np.ndarray
    island_rings: tuple[np.ndarray, ...]
    ten_dash_parts: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PreparedPlan:
    """Validated incremental construction decisions, independent of solvers."""

    periods: tuple[int, ...]
    coordinates: dict
    candidate_arcs: tuple
    source_builds: dict
    storage_builds: dict
    well_builds: dict
    pipe_builds: dict
    source_capacities: dict
    well_capacity: dict
    pipe_capacities: dict
    source_capacity_bounds: tuple[float, float]
    sink_capacity_bounds: tuple[float, float]
    pipeline_capacity_bounds: tuple[float, float]


@dataclass(frozen=True)
class PipelineDrawRecord:
    start: object
    end: object
    state: str
    cumulative_capacity: float
    linewidth: float
    color: str
    zorder: float


@dataclass(frozen=True)
class FacilityDrawRecord:
    node: object
    state: str
    cumulative_capacity: float
    area: float


def _number(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _nonnegative(value, label: str) -> float:
    number = _number(value, label)
    if number < 0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _decision(value, label: str, tolerance: float, binary: bool) -> int:
    number = _number(value, label)
    if number < -tolerance or abs(number - round(number)) > tolerance:
        raise ValueError(f"{label} must be a nonnegative integer within tolerance")
    count = max(0, round(number))
    if binary and count > 1:
        raise ValueError(f"{label} must be binary")
    return count


def _normalize_decisions(result, name: str, single_period: bool,
                         periods: tuple, tolerance: float) -> dict:
    values = result[name] if isinstance(result, Mapping) else getattr(result, name)
    if not isinstance(values, Mapping):
        raise TypeError(f"result.{name} must be a mapping of indices to numbers")
    dimensions = {"s": 2, "r": 1, "w": 1, "y": 3}
    normalized = {}
    for key, value in values.items():
        if single_period and name in ("r", "w"):
            index = (key, periods[0])
        else:
            expected = dimensions[name] + int(not single_period)
            if not isinstance(key, tuple) or len(key) != expected:
                raise ValueError(f"Invalid {name} index {key!r}; expected {expected} components")
            index = (*key, periods[0]) if single_period else key
        if index[-1] not in periods:
            raise ValueError(f"Unknown period in {name}{key!r}: {index[-1]!r}")
        count = _decision(value, f"{name}{key!r}", tolerance, name in ("s", "r"))
        if count:
            normalized[index] = count
    return normalized


def _positive_bounds(values) -> tuple[float, float]:
    positive = [value for value in values if value > 0]
    return (min(positive), max(positive)) if positive else (0.0, 0.0)


def _infer_index_format(result, network) -> tuple[bool, set]:
    """Use index structure, including zero entries, without inspecting class names."""
    formats, observed_periods = set(), set()
    for name, dimensions in (("s", 2), ("r", 1), ("w", 1), ("y", 3)):
        values = result[name] if isinstance(result, Mapping) else getattr(result, name)
        if not isinstance(values, Mapping):
            raise TypeError(f"result.{name} must be a mapping of indices to numbers")
        for key in values:
            if name in ("r", "w"):
                # Tuple-valued sink IDs are valid static indices as well.
                temporal = key not in network.sinks and isinstance(key, tuple) and len(key) == 2
            else:
                if not isinstance(key, tuple) or len(key) not in (dimensions, dimensions + 1):
                    raise ValueError(f"Invalid {name} construction index: {key!r}")
                temporal = len(key) == dimensions + 1
            formats.add("temporal" if temporal else "static")
            if temporal:
                h = key[-1]
                if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1:
                    raise ValueError(f"Period in {name}{key!r} must be a positive integer")
                observed_periods.add(int(h))
    if len(formats) > 1:
        raise ValueError("Construction variables mix static and time-indexed formats")
    return "static" in formats, observed_periods


def _prepare_plan(result, network, *, periods=None, diameters,
                  value_tolerance) -> PreparedPlan:
    """Normalize single/multi-period indices and retain integer multiplicities."""
    tolerance = _nonnegative(value_tolerance, "value_tolerance")
    if tolerance >= 0.5:
        raise ValueError("value_tolerance must be less than 0.5 for integer decisions")
    single_period, observed_periods = _infer_index_format(result, network)
    if periods is not None:
        ordered_periods = tuple(periods)
    elif observed_periods:
        # This project's model periods are 1..H; include idle interior periods.
        ordered_periods = tuple(range(1, max(observed_periods) + 1))
    else:
        ordered_periods = (1,)
    if not ordered_periods:
        raise ValueError("periods must not be empty")
    if any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1
           for h in ordered_periods):
        raise ValueError("periods must be positive integers")
    if any(a >= b for a, b in zip(ordered_periods, ordered_periods[1:])):
        raise ValueError("periods must be strictly increasing")
    if single_period and len(ordered_periods) != 1:
        raise ValueError("Static construction indices cannot describe a multi-period horizon")
    chosen_diameters = tuple(diameters)
    if not chosen_diameters or len(set(chosen_diameters)) != len(chosen_diameters):
        raise ValueError("diameters must be nonempty and must not contain duplicates")
    pipe_capacities = {}
    for diameter in chosen_diameters:
        if diameter not in network.pipe_capacity:
            raise ValueError(f"Missing pipeline capacity: {diameter!r}")
        capacity = _nonnegative(network.pipe_capacity[diameter], f"Pipeline capacity {diameter!r}")
        if capacity == 0:
            raise ValueError(f"Pipeline capacity {diameter!r} must be positive")
        pipe_capacities[diameter] = capacity

    coordinates = {}
    for node, point in {**network.intersections, **network.sources, **network.sinks}.items():
        if len(point) != 2:
            raise ValueError(f"Coordinates for {node!r} must be (longitude, latitude)")
        coordinates[node] = tuple(_number(value, f"Coordinates for {node!r}") for value in point)
    builds = {name: _normalize_decisions(result, name, single_period, ordered_periods, tolerance)
              for name in ("s", "r", "w", "y")}
    source_capacities = {key: _nonnegative(value, f"Source capacity {key!r}")
                         for key, value in network.source_capacity.items()}
    source_totals = {}
    for (source, _), capacity in source_capacities.items():
        source_totals[source] = source_totals.get(source, 0.0) + capacity
    well_capacity = read_well_injection_capacities(network.well_capacity, network.sinks)
    max_wells = {sink: _nonnegative(value, f"Maximum wells {sink!r}")
                 for sink, value in network.max_wells.items()}
    for source, unit, _ in builds["s"]:
        if source not in network.sources or (source, unit) not in source_capacities:
            raise ValueError(f"Unknown selected source unit: {(source, unit)!r}")
        if source_capacities[source, unit] <= 0:
            raise ValueError(f"Selected source unit {(source, unit)!r} must have positive capacity")
    for name in ("r", "w"):
        for sink, _ in builds[name]:
            if sink not in network.sinks or sink not in max_wells:
                raise ValueError(f"Unknown selected sink: {sink!r}")
    if any(well_capacity[sink] == 0 for sink, _ in builds["w"]):
        raise ValueError("Selected wells must have positive well_capacity")
    for sink, h in builds["w"]:
        if not any(j == sink and t <= h for j, t in builds["r"]):
            raise ValueError(f"Wells built at unopened storage site: {(sink, h)!r}")

    candidate_arcs = tuple(network.ij)
    arc_set = set(candidate_arcs)
    for start, end, diameter, _ in builds["y"]:
        if diameter not in pipe_capacities:
            raise ValueError(f"Unknown selected diameter: {diameter!r}")
        if start not in coordinates or end not in coordinates:
            raise ValueError(f"Pipeline endpoints have no coordinates: {(start, end)!r}")
        if (start, end) not in arc_set:
            raise ValueError(f"Selected pipeline is outside the candidate network: {(start, end)!r}")
    cumulative_by_arc = {}
    observed_capacities = []
    for h in ordered_periods:
        for (start, end, diameter, period), count in builds["y"].items():
            if period == h:
                arc = (start, end)
                cumulative_by_arc[arc] = cumulative_by_arc.get(arc, 0.0) + pipe_capacities[diameter] * count
        observed_capacities.extend(cumulative_by_arc.values())
    positive_sources = [capacity for capacity in source_capacities.values() if capacity > 0]
    source_bounds = ((min(positive_sources), max(source_totals.values()))
                     if positive_sources else (0.0, 0.0))
    sink_maximum = max((well_capacity[j] * max_wells[j] for j in network.sinks), default=0.0)
    positive_wells = [well_capacity[j] for j in network.sinks
                      if well_capacity[j] > 0 and max_wells[j] > 0]
    sink_bounds = ((min(positive_wells), sink_maximum) if positive_wells else (0.0, 0.0))
    return PreparedPlan(
        periods=ordered_periods, coordinates=coordinates, candidate_arcs=candidate_arcs,
        source_builds=builds["s"], storage_builds=builds["r"], well_builds=builds["w"],
        pipe_builds=builds["y"], source_capacities=source_capacities,
        well_capacity=well_capacity, pipe_capacities=pipe_capacities,
        source_capacity_bounds=source_bounds, sink_capacity_bounds=sink_bounds,
        pipeline_capacity_bounds=_positive_bounds(observed_capacities),
    )


def _scaled_size(value: float, bounds: tuple, minimum: float, maximum: float) -> float:
    low, high = bounds
    if value <= 0 or high <= 0:
        return minimum
    if math.isclose(low, high):
        return 0.5 * (minimum + maximum)
    position = min(1.0, max(0.0, (value - low) / (high - low)))
    return minimum + (maximum - minimum) * position


def _construction_state(previous: bool, current: bool) -> str:
    return "expanded-current" if previous and current else "new-current" if current else "existing"


def _pipeline_draw_records(plan: PreparedPlan, through_period: int,
                            style: Mapping) -> tuple[PipelineDrawRecord, ...]:
    """One directed-arc record, summing every diameter and parallel pipe."""
    if through_period not in plan.periods:
        raise ValueError(f"Unknown period: {through_period!r}")
    previous, current = {}, {}
    for (start, end, diameter, h), count in plan.pipe_builds.items():
        if h <= through_period:
            target = previous if h < through_period else current
            arc = (start, end)
            target[arc] = target.get(arc, 0.0) + plan.pipe_capacities[diameter] * count
    records = []
    color_keys = {"new-current": "pipeline_new_color", "expanded-current": "pipeline_expanded_color",
                  "existing": "pipeline_existing_color"}
    levels = {"new-current": 0.1, "expanded-current": 0.2, "existing": 0.0}
    for start, end in sorted(previous.keys() | current.keys(), key=lambda arc: tuple(map(str, arc))):
        arc = (start, end)
        capacity = previous.get(arc, 0.0) + current.get(arc, 0.0)
        state = _construction_state(arc in previous, arc in current)
        records.append(PipelineDrawRecord(
            start, end, state, capacity,
            _scaled_size(capacity, plan.pipeline_capacity_bounds,
                         style["pipeline_width_min"], style["pipeline_width_max"]),
            style[color_keys[state]], style["pipeline_zorder"] + levels[state],
        ))
    return tuple(records)


def _facility_records(plan: PreparedPlan, through_period: int,
                      style: Mapping) -> tuple[tuple[FacilityDrawRecord, ...], tuple[FacilityDrawRecord, ...]]:
    """Installed source-unit and well capacities determine marker areas."""
    if through_period not in plan.periods:
        raise ValueError(f"Unknown period: {through_period!r}")
    previous_sources, current_sources, source_capacity = set(), set(), {}
    for (source, unit, h), count in plan.source_builds.items():
        if h <= through_period:
            (previous_sources if h < through_period else current_sources).add(source)
            source_capacity[source] = source_capacity.get(source, 0.0) + plan.source_capacities[source, unit] * count
    previous_sinks = {sink for sink, h in plan.storage_builds if h < through_period}
    current_sinks = {sink for sink, h in plan.storage_builds if h == through_period}
    current_wells = {sink for sink, h in plan.well_builds if h == through_period}
    sink_capacity = {sink: 0.0 for sink in previous_sinks | current_sinks}
    for (sink, h), count in plan.well_builds.items():
        if h <= through_period:
            sink_capacity[sink] += plan.well_capacity[sink] * count
    source_records = tuple(FacilityDrawRecord(
        source, _construction_state(source in previous_sources, source in current_sources), capacity,
        _scaled_size(capacity, plan.source_capacity_bounds, style["facility_area_min"], style["facility_area_max"]),
    ) for source, capacity in sorted(source_capacity.items(), key=lambda pair: str(pair[0])))
    sink_records = tuple(FacilityDrawRecord(
        sink, _construction_state(sink in previous_sinks, sink in current_sinks or sink in current_wells), capacity,
        _scaled_size(capacity, plan.sink_capacity_bounds, style["facility_area_min"], style["facility_area_max"]),
    ) for sink, capacity in sorted(sink_capacity.items(), key=lambda pair: str(pair[0])))
    return source_records, sink_records


def _read_polyline_records(shapefile_path: Path) -> tuple[tuple[np.ndarray, ...], ...]:
    """Read ESRI PolyLine/Polygon geometry without requiring a GIS package."""
    shapefile_path = Path(shapefile_path)
    if not shapefile_path.is_file():
        raise FileNotFoundError(f"China boundary shapefile not found: {shapefile_path}")
    records = []
    with shapefile_path.open("rb") as stream:
        header = stream.read(100)
        if len(header) != 100 or struct.unpack(">i", header[:4])[0] != 9994:
            raise ValueError(f"Invalid shapefile header: {shapefile_path}")
        if struct.unpack("<i", header[32:36])[0] not in (3, 5):
            raise ValueError("China boundary must use PolyLine or Polygon geometry")
        while True:
            record_header = stream.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError("Incomplete shapefile record header")
            _, content_words = struct.unpack(">2i", record_header)
            if content_words < 2:
                raise ValueError("Invalid shapefile record length")
            content = stream.read(content_words * 2)
            if len(content) != content_words * 2:
                raise ValueError("Incomplete shapefile record")
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                records.append(tuple())
                continue
            if shape_type not in (3, 5) or len(content) < 44:
                raise ValueError(f"Unsupported boundary geometry type: {shape_type}")
            part_count, point_count = struct.unpack("<2i", content[36:44])
            starts_end = 44 + 4 * part_count
            point_data_end = starts_end + 16 * point_count
            if part_count < 1 or point_count < 2 or point_data_end > len(content):
                raise ValueError("Invalid boundary part or point count")
            starts = list(struct.unpack(f"<{part_count}i", content[44:starts_end]))
            if starts[0] != 0 or any(a >= b for a, b in zip(starts, starts[1:])) or starts[-1] >= point_count:
                raise ValueError("Invalid boundary part offsets")
            points = np.frombuffer(content, dtype="<f8", count=point_count * 2,
                                   offset=starts_end).reshape(point_count, 2).copy()
            if not np.isfinite(points).all():
                raise ValueError("Boundary coordinates must be finite")
            starts.append(point_count)
            records.append(tuple(points[start:end] for start, end in zip(starts[:-1], starts[1:])
                                 if end - start >= 2))
    if not records:
        raise ValueError("China boundary shapefile contains no geometry")
    return tuple(records)


def _build_mainland_ring(outline_parts: tuple[np.ndarray, ...]) -> np.ndarray:
    """Join the supplied boundary's two open mainland coastlines."""
    open_parts = [part for part in outline_parts if not np.allclose(part[0], part[-1], atol=1e-8)]
    if len(open_parts) < 2:
        raise ValueError("Mainland outline must contain two open parts")
    first = max(open_parts, key=len)
    second = min((part for part in open_parts if part is not first), key=lambda part: min(
        np.linalg.norm(first[-1] - part[0]) + np.linalg.norm(part[-1] - first[0]),
        np.linalg.norm(first[-1] - part[-1]) + np.linalg.norm(part[0] - first[0]),
    ))
    direct = np.linalg.norm(first[-1] - second[0]) + np.linalg.norm(second[-1] - first[0])
    reverse = np.linalg.norm(first[-1] - second[-1]) + np.linalg.norm(second[0] - first[0])
    if reverse < direct:
        second = second[::-1]
    if not (np.allclose(first[-1], second[0], atol=1e-6)
            and np.allclose(second[-1], first[0], atol=1e-6)):
        raise ValueError("The two mainland outline parts do not form a ring")
    return np.vstack((first, second[1:]))


def _load_china_boundary(shapefile_path: Path) -> ChinaBoundary:
    records = _read_polyline_records(Path(shapefile_path))
    if len(records) < 2:
        raise ValueError("China boundary must contain outline and dash-line records")
    outline_parts, dash_parts = records[:2]
    if len(dash_parts) != 10:
        raise ValueError(f"Expected 10 supplied dash-line parts, found {len(dash_parts)}")
    return ChinaBoundary(outline_parts, _build_mainland_ring(outline_parts),
                         tuple(part for part in outline_parts if np.allclose(part[0], part[-1], atol=1e-8)),
                         dash_parts)


def _validate_style(style: Mapping) -> None:
    required = (
        "figure_size dpi page_color map_color land_color boundary_color candidate_color dash_color "
        "source_color sink_color new_facility_edge_color "
        "facility_edge_width pipeline_new_color pipeline_expanded_color pipeline_existing_color "
        "pipeline_width_min pipeline_width_max pipeline_zorder facility_area_min facility_area_max "
        "facility_zorder mainland_extent inset_extent show_candidate_network show_inset show_legend "
        "font_family legend_fontsize legend_marker_size boundary_linewidth candidate_linewidth "
        "candidate_alpha dash_linewidth inset_bounds map_reference_latitude inset_reference_latitude "
        "figure_margins legend_location legend_anchor legend_handlelength legend_handletextpad "
        "legend_labelspacing inset_boundary_linewidth inset_dash_linewidth inset_frame_linewidth"
    ).split()
    missing = [key for key in required if key not in style]
    if missing:
        raise ValueError(f"Missing plot style settings: {', '.join(missing)}")
    for kind in ("source", "sink"):
        if f"expanded_{kind}_color" not in style and f"expanded_{kind}_edge_color" not in style:
            raise ValueError(f"Missing plot style settings: expanded_{kind}_color")
    for name in ("dpi", "legend_fontsize", "legend_marker_size", "pipeline_width_min",
                 "pipeline_width_max", "facility_area_min", "facility_area_max"):
        if _number(style[name], name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("facility_edge_width", "boundary_linewidth", "candidate_linewidth", "dash_linewidth",
                 "inset_boundary_linewidth", "inset_dash_linewidth", "inset_frame_linewidth",
                 "legend_handlelength", "legend_handletextpad", "legend_labelspacing"):
        _nonnegative(style[name], name)
    for prefix in ("pipeline_width", "facility_area"):
        if style[f"{prefix}_max"] < style[f"{prefix}_min"]:
            raise ValueError(f"{prefix}_max must not be less than {prefix}_min")
    if not 0 <= _number(style["candidate_alpha"], "candidate_alpha") <= 1:
        raise ValueError("candidate_alpha must be in [0, 1]")
    for name in ("mainland_extent", "inset_extent"):
        bounds = tuple(style[name])
        if len(bounds) != 4 or not all(math.isfinite(float(v)) for v in bounds):
            raise ValueError(f"{name} must have four finite coordinates")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError(f"{name} must be (west, east, south, north) with increasing limits")
    for name in ("map_reference_latitude", "inset_reference_latitude"):
        if abs(_number(style[name], name)) >= 90:
            raise ValueError(f"{name} must lie strictly between -90 and 90")


def _format_axis(axis, extent, latitude, style, inset: bool) -> None:
    west, east, south, north = extent
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_aspect(1.0 / math.cos(math.radians(latitude)), adjustable="box")
    axis.set_facecolor(style["map_color"])
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(inset)
        if inset:
            spine.set_color(style["dash_color"])
            spine.set_linewidth(style["inset_frame_linewidth"])


def _draw_background(axis, plan: PreparedPlan, boundary: ChinaBoundary | None,
                     style: Mapping, *, inset: bool) -> None:
    if boundary is not None:
        land = PolyCollection([boundary.mainland_ring, *boundary.island_rings], closed=True,
                              facecolors=style["land_color"], edgecolors=style["boundary_color"],
                              linewidths=style["inset_boundary_linewidth" if inset else "boundary_linewidth"],
                              antialiaseds=True, zorder=0)
        land.set_gid("china:land")
        axis.add_collection(land)
        dashes = LineCollection(boundary.ten_dash_parts, colors=style["dash_color"],
                                linewidths=style["inset_dash_linewidth" if inset else "dash_linewidth"],
                                capstyle="round", zorder=3 if inset else 2)
        dashes.set_gid("china:dash-lines")
        axis.add_collection(dashes)
    if style["show_candidate_network"] and not inset:
        seen, segments = set(), []
        for start, end in plan.candidate_arcs:
            corridor = frozenset((start, end))
            if corridor in seen or start not in plan.coordinates or end not in plan.coordinates:
                continue
            seen.add(corridor)
            segments.append([plan.coordinates[start], plan.coordinates[end]])
        if segments:
            candidates = LineCollection(segments, colors=style["candidate_color"],
                                        linewidths=style["candidate_linewidth"], alpha=style["candidate_alpha"],
                                        zorder=1, rasterized=True)
            candidates.set_gid("candidate-network")
            axis.add_collection(candidates)


def _facility_appearance(kind: str, state: str, style: Mapping):
    """Use state-colored fills and the same visible outline for every node."""
    if state == "expanded-current":
        # Accept legacy caller palettes whose expanded fills were outline colors.
        face = style.get(f"expanded_{kind}_color", style.get(f"expanded_{kind}_edge_color"))
    elif state == "new-current":
        face = style[f"{kind}_color"]
    else:
        face = style.get(f"existing_{kind}_color", style["pipeline_existing_color"])
    return face, style["new_facility_edge_color"], style["facility_edge_width"]


def _draw_construction(axis, plan: PreparedPlan, pipeline_records, source_records,
                       sink_records, style: Mapping) -> None:
    for record in sorted(pipeline_records, key=lambda item: (item.zorder, -item.linewidth)):
        start, end = plan.coordinates[record.start], plan.coordinates[record.end]
        line, = axis.plot([start[0], end[0]], [start[1], end[1]], color=record.color,
                          linewidth=record.linewidth, solid_capstyle="round", solid_joinstyle="round",
                          marker="None", path_effects=[], zorder=record.zorder)
        line.set_gid(f"pipeline:{record.state}:{record.start}:{record.end}:capacity={record.cumulative_capacity:g}")
    for kind, marker, records in (("source", "o", source_records), ("sink", "s", sink_records)):
        for state in ("existing", "new-current", "expanded-current"):
            selected = [record for record in records if record.state == state]
            if not selected:
                continue
            face, edge, edge_width = _facility_appearance(kind, state, style)
            points = [plan.coordinates[record.node] for record in selected]
            nodes = axis.scatter([point[0] for point in points], [point[1] for point in points],
                                  marker=marker, s=[record.area for record in selected],
                                  facecolors=face, edgecolors=edge,
                                  linewidths=edge_width,
                                  zorder=style["facility_zorder"])
            nodes.set_gid(f"{kind}:{state}")


def _add_legend(axis, style: Mapping, *, evolution: bool = False) -> None:
    """Show facility types, adding construction status only for evolution maps."""
    type_color = style.get("legend_type_color", "#A6AEB8")
    handles = [Line2D([0], [0], marker=marker, linestyle="",
                      markerfacecolor=type_color if evolution else style[f"{kind}_color"],
                      markeredgecolor=style["new_facility_edge_color"],
                      markeredgewidth=style["facility_edge_width"],
                      markersize=style["legend_marker_size"],
                      label="Sink subsite" if kind == "sink" and not evolution else kind.title())
               for kind, marker in (("source", "o"), ("sink", "s"))]
    width = 0.5 * (style["pipeline_width_min"] + style["pipeline_width_max"])
    handles.append(Line2D([0], [0], color=type_color if evolution else style["pipeline_new_color"],
                          linewidth=width,
                          marker="None", path_effects=[], solid_capstyle="round", label="Pipe"))
    legend = axis.legend(handles=handles, loc=style["legend_location"], bbox_to_anchor=style["legend_anchor"],
                         ncol=1, title="Facility type", title_fontsize=style["legend_fontsize"],
                         fontsize=style["legend_fontsize"], frameon=False, borderaxespad=0.0,
                         handlelength=style["legend_handlelength"], handletextpad=style["legend_handletextpad"],
                         labelspacing=style["legend_labelspacing"])
    legend.set_gid("legend:facility-type")
    legend.get_title().set_fontweight("bold")
    legend.set_zorder(max(style["pipeline_zorder"], style["facility_zorder"]) + 20)
    if not evolution:
        return
    status_handles, status_labels = [], []
    for state, label, pipe_key in (
            ("new-current", "New", "pipeline_new_color"),
            ("expanded-current", "Expanded", "pipeline_expanded_color"),
            ("existing", "Existing", "pipeline_existing_color")):
        symbols = []
        for kind, marker in (("source", "o"), ("sink", "s")):
            face, edge, edge_width = _facility_appearance(kind, state, style)
            symbols.append(Line2D([0], [0], marker=marker, linestyle="", markerfacecolor=face,
                                  markeredgecolor=edge, markeredgewidth=edge_width,
                                  markersize=style["legend_marker_size"]))
        symbols.append(Line2D([0], [0], color=style[pipe_key], linewidth=width,
                              marker="None", path_effects=[], solid_capstyle="round"))
        status_handles.append(tuple(symbols))
        status_labels.append(label)
    anchor = style["legend_anchor"]
    status_legend = Legend(
        axis, status_handles, status_labels, loc=style["legend_location"],
        bbox_to_anchor=style.get("legend_status_anchor", (anchor[0] + 0.175, anchor[1])),
        bbox_transform=axis.transAxes, title="Construction status", title_fontsize=style["legend_fontsize"],
        fontsize=style["legend_fontsize"], frameon=False, borderaxespad=0.0,
        handlelength=style.get("legend_status_handlelength", 5.2),
        handletextpad=style["legend_handletextpad"], labelspacing=style["legend_labelspacing"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    status_legend.set_gid("legend:construction-status")
    axis.add_artist(status_legend)
    status_legend.get_title().set_fontweight("bold")
    status_legend.set_zorder(max(style["pipeline_zorder"], style["facility_zorder"]) + 20)


def _create_cumulative_plan_figure(plan: PreparedPlan, *, through_period: int,
                                   boundary: ChinaBoundary | None, style: Mapping,
                                   evolution: bool | None = None):
    """Create an unregistered Figure; the caller owns and must clear it."""
    if evolution is None:
        evolution = len(plan.periods) > 1
    if not evolution:
        style = dict(style)
        style["source_color"] = style.get("single_period_source_color", style["source_color"])
        style["sink_color"] = style.get("single_period_sink_color", style["sink_color"])
        style["pipeline_new_color"] = style.get("single_period_pipeline_color", style["pipeline_new_color"])
    pipeline_records = _pipeline_draw_records(plan, through_period, style)
    source_records, sink_records = _facility_records(plan, through_period, style)
    figure = Figure(figsize=style["figure_size"], dpi=style["dpi"], facecolor=style["page_color"])
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    figure.subplots_adjust(**style["figure_margins"])
    _draw_background(axis, plan, boundary, style, inset=False)
    _format_axis(axis, style["mainland_extent"], style["map_reference_latitude"], style, False)
    _draw_construction(axis, plan, pipeline_records, source_records, sink_records, style)
    if style["show_inset"]:
        inset = axis.inset_axes(style["inset_bounds"], zorder=max(style["pipeline_zorder"], style["facility_zorder"]) + 10)
        # The inset supplies geographic context;
        # construction and the candidate network are drawn on the main map only.
        _draw_background(inset, plan, boundary, style, inset=True)
        _format_axis(inset, style["inset_extent"], style["inset_reference_latitude"], style, True)
    if style["show_legend"]:
        _add_legend(axis, style, evolution=evolution)
    if evolution:
        axis.text(0.02, 0.97, f"Period {through_period}", transform=axis.transAxes,
                  va="top", fontsize=style["legend_fontsize"] + 2)
    return figure, axis


def plot_results_from_excel(workbook, network, *, diameters, output_file,
                            **plot_options) -> list[Path]:
    """Render variable_s/r/w/y sheets, preserving every exported period column."""
    if not PLOT_ENABLED:
        return []
    index_columns = {
        "s": ["source", "unit"], "r": ["sink"], "w": ["sink"],
        "y": ["start", "end", "diameter"],
    }
    result, periods = {}, None
    with pd.ExcelFile(workbook, engine="openpyxl") as excel:
        for name, indices in index_columns.items():
            sheet = f"variable_{name}"
            frame = pd.read_excel(excel, sheet_name=sheet, dtype=object,
                                  keep_default_na=False)
            if list(frame.columns[:len(indices)]) != indices:
                raise ValueError(f"{sheet}: expected index columns {indices}")
            columns = list(frame.columns[len(indices):])
            if not columns or any(
                not isinstance(column, str) or not column.startswith("period")
                or not column[6:].isdigit() for column in columns
            ):
                raise ValueError(f"{sheet}: expected period columns such as period1")
            sheet_periods = tuple(int(column[6:]) for column in columns)
            if any(h < 1 for h in sheet_periods) or any(
                a >= b for a, b in zip(sheet_periods, sheet_periods[1:])
            ):
                raise ValueError(f"{sheet}: periods must be positive and increasing")
            if periods is not None and sheet_periods != periods:
                raise ValueError(f"{sheet}: period columns differ between sheets")
            periods = sheet_periods
            if frame.duplicated(subset=indices).any():
                raise ValueError(f"{sheet}: duplicate variable indices")
            result[name] = {
                (*row[:len(indices)], h): value
                for row in frame.itertuples(index=False, name=None)
                for h, value in zip(periods, row[len(indices):])
            }
    return plot_results(result, network, periods=periods, diameters=diameters,
                        output_file=output_file, **plot_options)


def plot_results(result, network, *, periods=None, diameters, output_file,
                 period_output_dir=None, boundary_file=_DEFAULT_BOUNDARY,
                 style=None, value_tolerance=None,
                 evolution_only=False) -> list[Path]:
    """Automatically render one static map or a multi-period evolution series.

    ``result`` is either an object exposing numeric ``s/r/w/y`` mappings or a
    dictionary containing them. Index shapes identify the model format. Pass
    the complete ``periods`` sequence when sparse values omit idle final periods
    or when a multi-period solution is entirely empty. With multiple periods,
    only the cumulative period panels are saved. A missing period output
    directory is derived from the final image path. A one-period model always
    produces only the final image, even if its variables carry a period index.
    With ``evolution_only=True``, save only period panels (also for one period)
    in the period directory and omit the additional final image.
    ``boundary_file=None`` disables the boundary layer.
    Existing unrelated files are never deleted. Omitted rendering arguments use
    this module's current configuration at call time. Explicit overrides remain
    available for isolated renders and tests; main scripts only supply output
    paths and model data. ``PLOT_ENABLED=False`` disables every caller's plots.
    """
    if not PLOT_ENABLED:
        return []
    style = PLOT_STYLE if style is None else style
    value_tolerance = PLOT_VALUE_TOLERANCE if value_tolerance is None else value_tolerance
    if boundary_file is _DEFAULT_BOUNDARY:
        boundary_file = PLOT_BOUNDARY_FILE
    _validate_style(style)
    plan = _prepare_plan(result, network, periods=periods, diameters=diameters,
                         value_tolerance=value_tolerance)
    boundary = _load_china_boundary(Path(boundary_file)) if boundary_file is not None else None
    output_file = Path(output_file)
    targets = []
    if len(plan.periods) > 1 or evolution_only:
        period_dir = (Path(period_output_dir) if period_output_dir is not None
                      else output_file.parent / f"{output_file.stem}_periods")
        targets.extend((h, period_dir / f"construction_cumulative_period_{h:02d}.png") for h in plan.periods)
    if len(plan.periods) == 1 and not evolution_only:
        targets.append((plan.periods[-1], output_file))
    if len({path.resolve() for _, path in targets}) != len(targets):
        raise ValueError("The final image and period images must use distinct output paths")
    written = []
    with matplotlib.rc_context({"font.family": style["font_family"], "axes.unicode_minus": False}):
        for period, path in targets:
            figure = None
            try:
                figure, _ = _create_cumulative_plan_figure(
                    plan, through_period=period, boundary=boundary, style=style,
                    evolution=evolution_only or len(plan.periods) > 1)
                path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(path, dpi=style["dpi"], bbox_inches="tight", facecolor=style["page_color"])
                written.append(path)
            finally:
                if figure is not None:
                    # Direct Figure/Agg objects never enter pyplot's global
                    # manager registry.  Clear artists and detach the canvas to
                    # release their references without touching callers' figures.
                    figure.clear()
                    figure.set_canvas(None)
    return written


def plot_bound_history(history, *, output_file, x_axis="iteration",
                       title="BD bound convergence", style=None) -> list[Path]:
    """Save a step chart of bound snapshots, without solving or loading a map.

    Each snapshot is a mapping with ``iteration`` (bound-update count, starting
    at zero), ``elapsed`` (seconds), and optional ``upper_bound``/``lower_bound``.
    The caller is responsible for supplying audited feasible upper bounds and
    valid full-problem lower bounds. The renderer neither recomputes bounds nor
    fills missing values: absent, invalid, and nonfinite bounds leave gaps.
    Snapshots remain in their supplied order, and the chosen x coordinate must
    be nonnegative and nondecreasing; repeated coordinates are allowed, e.g.
    when a final elapsed-time snapshot repeats the last update iteration.

    A single valid bound is visible as a marker. Empty or wholly unknown bounds
    produce an annotated chart. ``style`` partially overrides the current
    ``BOUND_PLOT_STYLE``; ``PLOT_ENABLED=False`` returns before inspecting input.
    """
    if not PLOT_ENABLED:
        return []
    if x_axis not in ("iteration", "elapsed"):
        raise ValueError("x_axis must be 'iteration' or 'elapsed'")
    chart_style = dict(BOUND_PLOT_STYLE)
    if style is not None:
        chart_style.update(style)

    x_values, upper_values, lower_values = [], [], []
    for index, snapshot in enumerate(history):
        if not isinstance(snapshot, Mapping):
            raise TypeError(f"history[{index}] must be a mapping")
        x_value = _nonnegative(snapshot.get(x_axis), f"history[{index}].{x_axis}")
        if x_axis == "iteration" and not x_value.is_integer():
            raise ValueError("iteration coordinates must be nonnegative integers")
        if x_values and x_value < x_values[-1]:
            raise ValueError(f"history {x_axis} coordinates must be nondecreasing")
        x_values.append(x_value)
        for key, values in (("upper_bound", upper_values), ("lower_bound", lower_values)):
            try:
                value = float(snapshot.get(key))
            except (TypeError, ValueError, OverflowError):
                value = math.nan
            values.append(value if math.isfinite(value) else math.nan)

    output_file = Path(output_file)
    figure = None
    with matplotlib.rc_context({"font.family": chart_style["font_family"],
                                "font.size": chart_style["font_size"],
                                "axes.unicode_minus": False}):
        try:
            figure = Figure(figsize=chart_style["figure_size"], dpi=chart_style["dpi"],
                            facecolor=chart_style["page_color"], layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.add_subplot(111, facecolor=chart_style["axes_color"])
            axes.set_title(title, fontsize=chart_style["title_fontsize"], pad=12)
            axes.set_xlabel("Bound update iteration" if x_axis == "iteration" else "Elapsed time (s)")
            axes.set_ylabel(chart_style["y_label"])
            axes.set_axisbelow(True)
            axes.grid(True, color=chart_style["grid_color"], linewidth=chart_style["grid_linewidth"],
                      alpha=chart_style["grid_alpha"])
            for side, spine in axes.spines.items():
                spine.set_visible(side in ("left", "bottom"))
                spine.set_color(chart_style["axes_color_line"])
                spine.set_linewidth(chart_style["axes_linewidth"])
            if x_axis == "iteration":
                axes.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
            axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 5), useMathText=True)

            has_bounds = False
            for key, values, label in (("upper_bound", upper_values, "Upper bound (UB)"),
                                       ("lower_bound", lower_values, "Lower bound (LB)")):
                if any(math.isfinite(value) for value in values):
                    axes.step(x_values, values, where="post", label=label,
                              color=chart_style[f"{key}_color"],
                              linestyle=chart_style[f"{key}_linestyle"],
                              marker=chart_style[f"{key}_marker"],
                              markersize=chart_style["marker_size"],
                              linewidth=chart_style["line_width"])
                    has_bounds = True
            # Keep the complete observed horizon, including time before either
            # bound exists. No bound is extended backward to the origin.
            x_end = x_values[-1] if x_values else 0.0
            axes.set_xlim(0.0, x_end * 1.02 if x_end > 0 else 1.0)
            if has_bounds and chart_style["show_legend"]:
                axes.legend(loc=chart_style["legend_location"], frameon=False,
                            fontsize=chart_style["legend_fontsize"])
            if not has_bounds:
                axes.text(0.5, 0.5, "No valid bounds", ha="center", va="center",
                          transform=axes.transAxes, color=chart_style["axes_color_line"])
                axes.set_yticks([])

            output_file.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_file, dpi=chart_style["dpi"], bbox_inches="tight",
                           facecolor=chart_style["page_color"])
        finally:
            if figure is not None:
                # Direct Agg figures never enter pyplot's manager registry.
                figure.clear()
                figure.set_canvas(None)
    return [output_file]
