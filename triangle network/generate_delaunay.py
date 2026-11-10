"""Generate adaptive intersection nodes and a mainland-constrained Delaunay network.

The two input tables are de-duplicated to one node per hub/subsite.  Interior
Delaunay circumcentres and mainland probes locate coverage gaps. Boundary probes
are validation locations, not mandatory network nodes; new nodes preferentially
move inland. A polygon coverage check and redundancy pruning keep the entire
mainland within MAX_EMPTY_RADIUS_KM of a node, including outside the fixed-node
convex hull. This is distance coverage, not an exact coastal triangulation.
After network generation, terminal auxiliary nodes and their incident arcs are
deleted recursively. This second stage preserves all remaining coordinates and
arcs; it does not restore coverage or retriangulate. Final coverage is reported
separately from the verified coverage of the initially generated network.

Example
-------
Edit the ``USER CONFIGURATION`` section below, then run:

python generate_delaunay.py

Command-line arguments can still temporarily override those defaults.
Only nodes.csv, arcs.csv and network.png are written to the output directory;
generation statistics and coverage diagnostics are printed to the terminal.

Dependencies: numpy, pandas, matplotlib, shapely>=2.0. openpyxl is only needed for reference
schema validation when --reference-nodes/--reference-arcs are supplied.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep


# =============================================================================
# USER CONFIGURATION / 用户配置
# 修改这一节即可运行；相对路径均以本脚本所在目录为基准。
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 输入文件
HUB_INPUT_FILE = SCRIPT_DIR.parent / "source_cluster" / "hub_unit_location_cost.csv"
SINK_INPUT_FILE = SCRIPT_DIR.parent / "storage_cost" / "sink_subsite" / "subsite_result" / "sink_subsite.csv"
BOUNDARY_FILE = SCRIPT_DIR / "China_boundary" / "中国轮廓线.shp"

# 参考表只用于校验输出字段结构；设为 None 可跳过相应校验。
REFERENCE_NODES_FILE: Path | None = None
REFERENCE_ARCS_FILE: Path | None = None

# 输出目录和文件名
OUTPUT_DIR = SCRIPT_DIR / "output"
NODES_OUTPUT_FILENAME = "nodes.csv"
ARCS_OUTPUT_FILENAME = "arcs.csv"
NETWORK_PNG_FILENAME = "network.png"

# 网络图参数
OUTPUT_NETWORK_PNG = True
# 与 construction_cumulative_period_01.png 保持相同画布和版式。
NETWORK_FIGURE_SIZE_INCHES = (14.0, 9.2)
NETWORK_PNG_DPI = 600
MAINLAND_MAP_EXTENT = (72.0, 136.0, 17.5, 54.8)
SOUTH_CHINA_SEA_EXTENT = (107.5, 123.5, 2.5, 25.5)
MAP_LAND_COLOR = "#FFFFFF"
NETWORK_ARC_COLOR = "#66777D"
NETWORK_ARC_LINE_WIDTH = 0.30
NETWORK_ARC_ALPHA = 0.38
INTERSECTION_NODE_COLOR = "#374151"
INTERSECTION_NODE_ALPHA = 0.88
INTERSECTION_MARKER_SIZE = 2.0
HUB_MARKER_SIZE = 16.0
SINK_MARKER_SIZE = 19.0

# 自适应加密参数
# 最大空洞半径越小，intersection nodes 越多、网络越密。
MAX_EMPTY_RADIUS_KM = 55
# 防止异常参数导致无限加点的安全上限。
MAX_INTERSECTION_NODES = 10_000
# 每轮最多加入的候选点数；可能影响布点顺序，但不放宽最终覆盖条件。
INSERTION_BATCH_SIZE = 256
# 内部探针间距 = 最大空洞半径 × 此比例；探针不会直接全部输出为节点。
COVERAGE_GRID_SPACING_RATIO = 0.5
# 批量插点的最小间距比例，越大越抑制局部过密；覆盖修补不受此软限制。
INSERTION_MIN_SEPARATION_RATIO = 0.75
# 精确覆盖检查使用每象限32段的内接圆多边形，不依赖探针碰巧命中空洞。
COVERAGE_BUFFER_QUAD_SEGS = 32
# 删除可被其他节点覆盖范围完全替代的辅助节点，固定源汇节点始终保留。
PRUNE_REDUNDANT_INTERSECTIONS = True

# 边界与弧过滤参数
# 沿边界按累计弧长采样；仅用于发现覆盖缺口，不强制成为网络节点。
BOUNDARY_SAMPLE_SPACING_KM = 20.0
# 新增辅助节点优先距边界至少此距离；狭长地区允许减小，避免覆盖缺口。
INTERSECTION_BOUNDARY_SETBACK_KM = 20.0
# 允许真实沿海节点因坐标取整而位于海岸线外的最大距离。
COASTAL_NODE_TOLERANCE_KM = 10.0
# 弧的大陆边界容差；设为0执行严格大陆约束，不能通过连通修复绕过。
ARC_BOUNDARY_TOLERANCE_KM = 2.0
# 兼容旧接口的采样间隔；现已改为整段精确检查，此参数不再影响筛选结果。
ARC_CONTAINMENT_SAMPLE_STEP_KM = 10.0
# 0表示不限长；正值限制所有弧（包括连通性补边）的投影长度。
MAX_ARC_LENGTH_KM = 0.0
# 第二步：递归删除无向度数<=1的辅助节点及其连接弧，固定源汇不删。
# 直接删点，不移动其余节点、不重新三角剖分、不补边或补覆盖。
REMOVE_UNUSED_INTERSECTION_BRANCHES = True


# =============================================================================
# FIXED GEOMETRY CONSTANTS / 固定几何常数（通常不需要修改）
# =============================================================================

EARTH_RADIUS_KM = 6371.0088
ALBERS_LON_0 = 105.0
ALBERS_LAT_0 = 0.0
ALBERS_LAT_1 = 25.0
ALBERS_LAT_2 = 47.0


def albers_constants() -> tuple[float, float, float, float]:
    lon0 = math.radians(ALBERS_LON_0)
    lat0 = math.radians(ALBERS_LAT_0)
    lat1 = math.radians(ALBERS_LAT_1)
    lat2 = math.radians(ALBERS_LAT_2)
    n = (math.sin(lat1) + math.sin(lat2)) / 2.0
    c = math.cos(lat1) ** 2 + 2.0 * n * math.sin(lat1)
    rho0 = EARTH_RADIUS_KM / n * math.sqrt(c - 2.0 * n * math.sin(lat0))
    return lon0, n, c, rho0


def project(lon_lat: np.ndarray) -> np.ndarray:
    lon0, n, c, rho0 = albers_constants()
    lon = np.radians(lon_lat[:, 0])
    lat = np.radians(lon_lat[:, 1])
    rho = EARTH_RADIUS_KM / n * np.sqrt(c - 2.0 * n * np.sin(lat))
    theta = n * (lon - lon0)
    return np.column_stack((rho * np.sin(theta), rho0 - rho * np.cos(theta)))


def unproject(xy: np.ndarray) -> np.ndarray:
    lon0, n, c, rho0 = albers_constants()
    x, y = xy[:, 0], xy[:, 1]
    rho = np.hypot(x, rho0 - y)
    theta = np.arctan2(x, rho0 - y)
    lat = np.arcsin(np.clip((c - (rho * n / EARTH_RADIUS_KM) ** 2) / (2 * n), -1, 1))
    lon = lon0 + theta / n
    return np.column_stack((np.degrees(lon), np.degrees(lat)))


def read_shape_records(shp: Path) -> list[list[np.ndarray]]:
    """Read PolyLine/Polygon records while preserving their part grouping."""
    records: list[list[np.ndarray]] = []
    with shp.open("rb") as stream:
        header = stream.read(100)
        if len(header) != 100:
            raise ValueError(f"Invalid shapefile header: {shp}")
        while True:
            record_header = stream.read(8)
            if not record_header:
                break
            _, words = struct.unpack(">2i", record_header)
            content = stream.read(words * 2)
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in (3, 5):
                raise ValueError("Boundary shapefile must contain PolyLine/Polygon geometry")
            part_count, point_count = struct.unpack("<2i", content[36:44])
            starts = list(struct.unpack(f"<{part_count}i", content[44:44 + 4 * part_count]))
            offset = 44 + 4 * part_count
            points = np.frombuffer(content, dtype="<f8", count=point_count * 2, offset=offset).reshape(-1, 2).copy()
            starts.append(point_count)
            parts = [points[a:b] for a, b in zip(starts[:-1], starts[1:]) if b - a >= 2]
            if parts:
                records.append(parts)
    if not records:
        raise ValueError(f"No usable boundary geometry in {shp}")
    return records


def read_polyline_parts(shp: Path) -> list[np.ndarray]:
    """Read and flatten all PolyLine/Polygon parts."""
    return [part for record in read_shape_records(shp) for part in record]


def mainland_ring(parts: list[np.ndarray]) -> np.ndarray:
    """Select/build the large mainland ring, excluding island rings."""
    # This dataset stores the mainland as two very long open polylines; the many
    # closed records are islands.  Prefer the longest open part and its endpoint
    # mate instead of accidentally selecting an island polygon.
    open_parts = [p for p in parts if not np.allclose(p[0], p[-1], atol=1e-8)]
    if len(open_parts) < 2:
        raise ValueError("Mainland boundary is not closed")
    first = max(open_parts, key=len)
    second = min(
        (p for p in open_parts if p is not first),
        key=lambda p: min(
            np.linalg.norm(first[-1] - p[0]) + np.linalg.norm(p[-1] - first[0]),
            np.linalg.norm(first[-1] - p[-1]) + np.linalg.norm(p[0] - first[0]),
        ),
    )
    direct = np.linalg.norm(first[-1] - second[0]) + np.linalg.norm(second[-1] - first[0])
    reverse = np.linalg.norm(first[-1] - second[-1]) + np.linalg.norm(second[0] - first[0])
    if reverse < direct:
        second = second[::-1]
    ring = np.vstack((first, second[1:]))
    if not np.allclose(ring[0], ring[-1], atol=1e-6):
        ring = np.vstack((ring, ring[0]))
    return ring


def load_fixed_nodes(hub_csv: Path, sink_csv: Path) -> pd.DataFrame:
    hubs_raw = pd.read_csv(hub_csv)
    sinks_raw = pd.read_csv(sink_csv)
    required_h = {"hub", "longitude", "latitude"}
    sink_schemas = (["Subsite_ID", "Longitude", "Latitude"],
                    ["ju", "Subsite_Longitude", "Subsite_Latitude"])
    sink_columns = next((cols for cols in sink_schemas if set(cols).issubset(sinks_raw.columns)), None)
    if not required_h.issubset(hubs_raw.columns):
        raise ValueError(f"Hub CSV missing columns: {sorted(required_h - set(hubs_raw.columns))}")
    if sink_columns is None:
        raise ValueError("Sink CSV requires Subsite_ID/Longitude/Latitude "
                         "or ju/Subsite_Longitude/Subsite_Latitude")

    hubs = hubs_raw[["hub", "longitude", "latitude"]].copy()
    hubs.columns = ["I", "Longitude", "Latitude"]
    sinks = sinks_raw[sink_columns].copy()
    sinks.columns = ["I", "Longitude", "Latitude"]
    for frame, label in ((hubs, "hub"), (sinks, "sink")):
        frame["I"] = frame["I"].astype(str).str.strip()
        frame[["Longitude", "Latitude"]] = frame[["Longitude", "Latitude"]].apply(pd.to_numeric, errors="raise")
        spread = frame.groupby("I")[["Longitude", "Latitude"]].agg(lambda s: s.max() - s.min())
        if (spread > 1e-9).any(axis=None):
            raise ValueError(f"One {label} ID has inconsistent coordinates")
        frame.drop_duplicates("I", inplace=True)
    fixed = pd.concat((hubs, sinks), ignore_index=True)
    if fixed["I"].duplicated().any():
        raise ValueError("Hub and sink node IDs overlap")
    if not np.isfinite(fixed[["Longitude", "Latitude"]].to_numpy()).all():
        raise ValueError("Coordinates contain NaN or infinity")
    return fixed


def resample_ring(ring_xy: np.ndarray, spacing_km: float) -> np.ndarray:
    """Sample cumulative arc length, independently of shapefile vertex density."""
    lengths = np.linalg.norm(np.diff(ring_xy, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    count = max(3, int(math.ceil(cumulative[-1] / spacing_km)))
    distance = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    return np.column_stack([
        np.interp(distance, cumulative, ring_xy[:, axis]) for axis in range(2)
    ])


def circumcenters(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = points[triangles]
    ax, ay = p[:, 0, 0], p[:, 0, 1]
    bx, by = p[:, 1, 0], p[:, 1, 1]
    cx, cy = p[:, 2, 0], p[:, 2, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    valid = np.abs(d) > 1e-10
    ux = np.full(len(p), np.nan)
    uy = np.full(len(p), np.nan)
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux[valid] = (aa[valid] * (by[valid] - cy[valid]) + bb[valid] * (cy[valid] - ay[valid]) + cc[valid] * (ay[valid] - by[valid])) / d[valid]
    uy[valid] = (aa[valid] * (cx[valid] - bx[valid]) + bb[valid] * (ax[valid] - cx[valid]) + cc[valid] * (bx[valid] - ax[valid])) / d[valid]
    centres = np.column_stack((ux, uy))
    radii = np.linalg.norm(centres - p[:, 0], axis=1)
    return centres, radii


def nearest_distances(samples: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bound temporary memory without requiring a spatial-index dependency."""
    result = np.empty(len(samples))
    for start in range(0, len(samples), 256):
        delta = samples[start:start + 256, None, :] - points[None, :, :]
        result[start:start + 256] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return result


def coverage_geometry(xy: np.ndarray, radius_km: float, quad_segs: int = 32):
    """Union of INSCRIBED disks: empty mainland difference proves radius coverage.

    Keep a small relative numerical/export margin. Distances are Albers projected kilometres,
    not geodesic pipeline lengths. No uncovered-area tolerance is used.
    """
    radius = radius_km * (1.0 - 1e-7)
    return unary_union([Point(point).buffer(radius, quad_segs=quad_segs) for point in xy])


def prune_intersections(
    fixed_xy: np.ndarray,
    intersection_xy: np.ndarray,
    domain: Polygon,
    radius_km: float,
    quad_segs: int = 32,
) -> tuple[np.ndarray, int]:
    """Remove a node only if its entire mainland footprint is already covered."""
    points = np.vstack((fixed_xy, intersection_xy))
    disks = [Point(p).buffer(radius_km * (1.0 - 1e-7), quad_segs=quad_segs) for p in points]
    active = np.ones(len(points), dtype=bool)
    # Prefer deleting boundary-near nodes; fixed facilities are never candidates.
    order = sorted(range(len(fixed_xy), len(points)), key=lambda i: domain.boundary.distance(Point(points[i])))
    for index in order:
        distance2 = np.sum((points - points[index]) ** 2, axis=1)
        nearby = np.flatnonzero(active & (distance2 <= (2.0 * radius_km) ** 2))
        neighbors = [disks[j] for j in nearby if j != index]
        if neighbors and domain.intersection(disks[index]).difference(unary_union(neighbors)).is_empty:
            active[index] = False
    kept = intersection_xy[active[len(fixed_xy):]]
    return kept, len(intersection_xy) - len(kept)


def adaptive_intersections(
    fixed_xy: np.ndarray,
    ring_xy: np.ndarray,
    radius_km: float,
    max_intersections: int,
    batch_size: int,
    *,
    grid_spacing_ratio: float = COVERAGE_GRID_SPACING_RATIO,
    boundary_spacing_km: float = BOUNDARY_SAMPLE_SPACING_KM,
    setback_km: float = INTERSECTION_BOUNDARY_SETBACK_KM,
    min_separation_ratio: float = INSERTION_MIN_SEPARATION_RATIO,
    prune_redundant: bool = PRUNE_REDUNDANT_INTERSECTIONS,
    buffer_quad_segs: int = COVERAGE_BUFFER_QUAD_SEGS,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Refine only gaps, move auxiliary points inland, then prove domain coverage."""
    numeric = (radius_km, max_intersections, batch_size, grid_spacing_ratio,
               boundary_spacing_km, setback_km, min_separation_ratio, buffer_quad_segs)
    if (not np.isfinite(numeric).all() or radius_km <= 0 or max_intersections < 0 or batch_size <= 0
            or grid_spacing_ratio <= 0 or boundary_spacing_km <= 0 or setback_km < 0
            or not 0 < min_separation_ratio <= 1 or buffer_quad_segs < 4):
        raise ValueError("Invalid refinement parameters")
    domain = Polygon(ring_xy)
    if not domain.is_valid or domain.is_empty:
        raise ValueError("Mainland polygon must be nonempty and valid; boundary was not altered")
    if len(fixed_xy) == 0 or not np.isfinite(fixed_xy).all():
        raise ValueError("At least one finite fixed node is required")
    prepared = prep(domain)
    path = MplPath(ring_xy, closed=True)
    step = radius_km * grid_spacing_ratio
    xmin, ymin, xmax, ymax = domain.bounds
    gx, gy = np.meshgrid(np.arange(xmin, xmax + step, step), np.arange(ymin, ymax + step, step))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    grid = grid[path.contains_points(grid)]
    probes = np.vstack((grid, resample_ring(ring_xy, boundary_spacing_km)))
    cores = [domain.buffer(-distance) for distance in (setback_km, setback_km / 2, setback_km / 4)] if setback_km else []

    def place_inland(witness: np.ndarray) -> np.ndarray:
        point = Point(witness)
        for core in cores:
            if core.is_empty:
                continue
            moved = nearest_points(point, core)[1]
            # Limit displacement so the witness remains comfortably covered.
            if point.distance(moved) <= radius_km * 0.5 and prepared.covers(moved):
                return np.asarray(moved.coords[0])
        return witness.copy()  # Narrow peninsulas can require boundary-near nodes.

    intersections: list[np.ndarray] = []
    iterations = 0
    repair_rounds = 0
    while True:
        points = np.vstack((fixed_xy, np.asarray(intersections).reshape(-1, 2)))
        candidates = probes
        if len(points) >= 3 and np.linalg.matrix_rank(points - points[0]) == 2:
            tri = mtri.Triangulation(points[:, 0], points[:, 1])
            centres, radii = circumcenters(points, tri.triangles)
            valid = np.isfinite(radii) & path.contains_points(centres)
            candidates = np.vstack((candidates, centres[valid]))
        distances = nearest_distances(candidates, points)
        order = np.argsort(-distances, kind="stable")
        candidates = candidates[order[distances[order] > radius_km * 0.995]]
        if not len(candidates):
            uncovered = domain.difference(coverage_geometry(points, radius_km, buffer_quad_segs))
            if uncovered.is_empty:
                break
            # A probe grid alone can miss thin slivers or regions outside the hull.
            parts = list(uncovered.geoms) if hasattr(uncovered, "geoms") else [uncovered]
            candidates = np.asarray([part.representative_point().coords[0] for part in parts if not part.is_empty])
            repair_rounds += 1
        room = max_intersections - len(intersections)
        if room <= 0:
            raise RuntimeError(f"Mainland coverage not achieved at max_intersections={max_intersections}; increase the cap or radius")
        selected: list[np.ndarray] = []
        for witness in candidates:
            if selected and np.min(np.linalg.norm(np.asarray(selected) - witness, axis=1)) <= radius_km * 0.995:
                continue
            candidate = place_inland(witness)
            if not prepared.covers(Point(candidate)):
                continue
            separation = float(np.min(np.linalg.norm(points - candidate, axis=1)))
            if selected:
                separation = min(separation, float(np.min(np.linalg.norm(np.asarray(selected) - candidate, axis=1))))
            if separation < radius_km * min_separation_ratio:
                continue
            selected.append(candidate)
            if len(selected) >= min(batch_size, room):
                break
        if not selected:
            # The spacing and setback are preferences; coverage is mandatory.
            candidate = place_inland(candidates[0])
            if np.min(np.linalg.norm(points - candidate, axis=1)) < radius_km * 1e-6:
                candidate = candidates[0]
            if not prepared.covers(Point(candidate)):
                candidate = np.asarray(nearest_points(Point(candidate), domain)[1].coords[0])
            if np.min(np.linalg.norm(points - candidate, axis=1)) < radius_km * 1e-9:
                raise RuntimeError("Coverage repair stalled at numerical precision")
            selected = [candidate]
        intersections.extend(selected)
        iterations += 1
        print(f"  refinement {iterations}: {len(intersections)} intersection nodes", flush=True)

    result = np.asarray(intersections).reshape(-1, 2)
    pruned = 0
    if prune_redundant:
        print("  pruning redundant intersection nodes...", flush=True)
        result, pruned = prune_intersections(fixed_xy, result, domain, radius_km, buffer_quad_segs)
    final_xy = np.vstack((fixed_xy, result))
    uncovered = domain.difference(coverage_geometry(final_xy, radius_km, buffer_quad_segs))
    if not uncovered.is_empty:
        raise RuntimeError("Final mainland coverage validation failed")
    clearance = np.asarray([domain.boundary.distance(Point(p)) for p in result])
    return result, {
        "iterations": iterations,
        "converged": True,
        "coverage_verified": True,
        "coverage_definition": "Every mainland point is within the configured Albers projected radius of a network node",
        "coverage_radius_upper_bound_km": radius_km,
        "probe_max_nearest_node_distance_km": float(nearest_distances(probes, final_xy).max()),
        "uncovered_mainland_area_km2": float(uncovered.area),
        "coverage_repair_rounds": repair_rounds,
        "pruned_intersection_nodes": pruned,
        "boundary_intersection_nodes": int(np.sum(clearance <= 1e-4)),
        "intersection_nodes_within_setback": int(np.sum(clearance < setback_km - 1e-4)),
    }


def segment_inside(
    path: MplPath,
    a: np.ndarray,
    b: np.ndarray,
    sample_step_km: float = ARC_CONTAINMENT_SAMPLE_STEP_KM,
    *,
    boundary_tolerance_km: float = ARC_BOUNDARY_TOLERANCE_KM,
    coastal_tolerance_km: float = 0.0,
    allow_coastal_a: bool = False,
    allow_coastal_b: bool = False,
    mainland_polygon=None,
    buffered_polygon=None,
    prepared_buffer=None,
) -> bool:
    """Check the complete segment, including short arcs and both endpoints.

    ``sample_step_km`` is retained for compatibility; containment is now an exact
    polygon operation and cannot miss a narrow bay between sampling positions.
    A real offshore facility may use a coastal exception only over the first or
    last ``coastal_tolerance_km`` of its incident segment. The rest must be inside
    the ordinary mainland buffer. A wholly offshore segment is not an exception.
    """
    from shapely.geometry import LineString, Point, Polygon

    if not np.isfinite(sample_step_km) or sample_step_km <= 0:
        raise ValueError("sample_step_km must be positive and finite")
    if any(not np.isfinite(value) or value < 0 for value in
           (boundary_tolerance_km, coastal_tolerance_km)):
        raise ValueError("Boundary and coastal tolerances must be nonnegative and finite")
    if mainland_polygon is None:
        mainland_polygon = Polygon(path.vertices)
        if not mainland_polygon.is_valid or mainland_polygon.is_empty:
            raise ValueError("Mainland boundary must define a valid, nonempty polygon")
    if buffered_polygon is None:
        buffered_polygon = (mainland_polygon.buffer(boundary_tolerance_km)
                            if boundary_tolerance_km else mainland_polygon)
    ordinary_domain = prepared_buffer if prepared_buffer is not None else buffered_polygon
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    segment = LineString((a, b)) if length else Point(a)
    if ordinary_domain.covers(segment):
        return True
    if not length or coastal_tolerance_km == 0:
        return False

    def coastal_endpoint(point: np.ndarray, allowed: bool) -> bool:
        if not allowed:
            return False
        geometry = Point(point)
        return (not mainland_polygon.covers(geometry)
                and mainland_polygon.distance(geometry) <= coastal_tolerance_km)

    start = min(length, coastal_tolerance_km) if coastal_endpoint(a, allow_coastal_a) else 0.0
    end = min(length, coastal_tolerance_km) if coastal_endpoint(b, allow_coastal_b) else 0.0
    if start == 0.0 and end == 0.0:
        return False
    if ((start == 0.0 and not ordinary_domain.covers(Point(a)))
            or (end == 0.0 and not ordinary_domain.covers(Point(b)))):
        return False
    if start + end >= length:
        # Overlapping endpoint neighborhoods must still meet the ordinary domain.
        return bool(buffered_polygon.intersects(segment))
    delta = (b - a) / length
    core = LineString((a + start * delta, b - end * delta))
    return bool(ordinary_domain.covers(core))


def build_arcs(
    nodes: pd.DataFrame,
    xy: np.ndarray,
    ring_xy: np.ndarray,
    *,
    fixed_count: int | None = None,
    sample_step_km: float = ARC_CONTAINMENT_SAMPLE_STEP_KM,
    boundary_tolerance_km: float = ARC_BOUNDARY_TOLERANCE_KM,
    coastal_tolerance_km: float = COASTAL_NODE_TOLERANCE_KM,
    max_arc_length_km: float = MAX_ARC_LENGTH_KM,
) -> tuple[pd.DataFrame, int]:
    """Filter Delaunay edges and connect components using only admissible arcs.

    Repair searches existing node pairs in increasing distance, without reviving
    a rejected edge or relaxing any boundary/length setting. Zero maximum length
    disables the length cap. All retained undirected edges are exported both ways.
    """
    from shapely.geometry import Polygon
    from shapely.prepared import prep

    xy = np.asarray(xy, dtype=float)
    if xy.shape != (len(nodes), 2) or not np.isfinite(xy).all() or len(nodes) < 2:
        raise ValueError("At least two nodes with finite two-dimensional coordinates are required")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError(
            "Distinct nodes share identical coordinates; positive-distance arc output "
            "does not support coincident nodes. Resolve duplicate coordinates in the input."
        )
    if not np.isfinite(sample_step_km) or sample_step_km <= 0:
        raise ValueError("sample_step_km must be positive and finite")
    if any(not np.isfinite(value) or value < 0 for value in
           (boundary_tolerance_km, coastal_tolerance_km, max_arc_length_km)):
        raise ValueError("Arc tolerances and maximum length must be nonnegative and finite")
    if fixed_count is None:
        fixed_mask = ~nodes["I"].astype(str).str.startswith("I").to_numpy()
    else:
        if not isinstance(fixed_count, (int, np.integer)) or not 0 <= fixed_count <= len(nodes):
            raise ValueError("fixed_count must be an integer between zero and the number of nodes")
        fixed_mask = np.arange(len(nodes)) < fixed_count
    polygon = Polygon(ring_xy)
    if not polygon.is_valid:
        raise ValueError("Mainland boundary must define a valid polygon")
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("Mainland boundary does not define a usable polygon")
    buffered_polygon = polygon.buffer(boundary_tolerance_km) if boundary_tolerance_km else polygon
    prepared_buffer = prep(buffered_polygon)
    path = MplPath(ring_xy, closed=True)

    edges: set[tuple[int, int]] = set()
    triangle_count = 0
    if len(np.unique(xy, axis=0)) >= 3 and np.linalg.matrix_rank(xy - xy[0]) == 2:
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
        triangle_count = len(tri.triangles)
        for triangle in tri.triangles:
            for u, v in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                         (triangle[2], triangle[0])):
                edges.add((min(int(u), int(v)), max(int(u), int(v))))
    # Collinear/two-node inputs use the same component repair below instead of
    # requiring a two-dimensional triangulation.
    checked: dict[tuple[int, int], bool] = {}
    rejected_length = 0
    rejected_boundary = 0

    def admissible(u: int, v: int) -> bool:
        nonlocal rejected_length, rejected_boundary
        edge = (min(u, v), max(u, v))
        if edge in checked:
            return checked[edge]
        distance = float(np.linalg.norm(xy[u] - xy[v]))
        if max_arc_length_km and distance > max_arc_length_km:
            rejected_length += 1
            checked[edge] = False
            return False
        accepted = segment_inside(
            path, xy[u], xy[v], sample_step_km,
            boundary_tolerance_km=boundary_tolerance_km,
            coastal_tolerance_km=coastal_tolerance_km,
            allow_coastal_a=bool(fixed_mask[u]), allow_coastal_b=bool(fixed_mask[v]),
            mainland_polygon=polygon, buffered_polygon=buffered_polygon,
            prepared_buffer=prepared_buffer,
        )
        if not accepted:
            rejected_boundary += 1
        checked[edge] = accepted
        return accepted

    retained = [(u, v) for u, v in sorted(edges) if admissible(u, v)]
    initial_retained_count = len(retained)
    parent = list(range(len(nodes)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    for u, v in retained:
        union(u, v)
    component_count = len({find(index) for index in range(len(nodes))})
    initial_component_count = component_count
    while component_count > 1:
        labels = np.asarray([find(index) for index in range(len(nodes))])
        roots, counts = np.unique(labels, return_counts=True)
        smallest_root = roots[np.argmin(counts)]
        inside = np.flatnonzero(labels == smallest_root)
        outside = np.flatnonzero(labels != smallest_root)
        best_edge: tuple[int, int] | None = None
        best_distance = float("inf")
        # Search the smallest component first: isolated coastal points are cheap
        # to repair, and no N-by-N distance matrix or all-pairs list is needed.
        for u in inside:
            distances = np.linalg.norm(xy[outside] - xy[u], axis=1)
            order = np.argsort(distances, kind="stable")
            for position in order:
                distance = float(distances[position])
                if distance >= best_distance or (max_arc_length_km and distance > max_arc_length_km):
                    break
                v = int(outside[position])
                if admissible(int(u), v):
                    best_edge = (min(int(u), v), max(int(u), v))
                    best_distance = distance
                    break
        if best_edge is None:
            sample_ids = nodes.iloc[inside[:8]]["I"].astype(str).tolist()
            raise RuntimeError(
                f"Arc filtering leaves {component_count} disconnected components; "
                f"no admissible connection for component of {len(inside)} nodes "
                f"(examples: {sample_ids}). boundary_tolerance_km={boundary_tolerance_km}, "
                f"coastal_tolerance_km={coastal_tolerance_km}, "
                f"max_arc_length_km={max_arc_length_km}. "
                "Add interior support nodes or explicitly adjust these limits."
            )
        retained.append(best_edge)
        union(*best_edge)
        component_count -= 1

    rows = []
    ids = nodes["I"].to_numpy()
    for u, v in sorted(retained):
        distance = float(np.linalg.norm(xy[u] - xy[v]))
        rows.append((ids[u], ids[v], distance))
        rows.append((ids[v], ids[u], distance))
    arcs = pd.DataFrame(rows, columns=["i", "j", "distance"])
    arcs.attrs.update({
        "delaunay_candidate_edges": len(edges),
        "retained_delaunay_edges": initial_retained_count,
        "rejected_boundary_edges": rejected_boundary,
        "rejected_length_edges": rejected_length,
        "components_before_repair": initial_component_count,
        "connectivity_repair_edges": len(retained) - initial_retained_count,
        "connected_components": component_count,
        "maximum_arc_length_km": float(arcs["distance"].max()),
        "containment_method": "exact_polygon_with_local_coastal_endpoint_exceptions",
    })
    return arcs, triangle_count


def prune_terminal_intersections(
    nodes: pd.DataFrame, xy: np.ndarray, arcs: pd.DataFrame, *, fixed_count: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Delete non-facility leaves and incident arcs until no such leaves remain.

    Degrees count distinct undirected neighbors, not incoming/outgoing CSV rows.
    Fixed facilities are the first fixed_count nodes. Surviving rows, coordinates
    and distances are unchanged; coverage is never a condition for deletion.
    """
    if not 0 <= fixed_count <= len(nodes) or np.shape(xy) != (len(nodes), 2):
        raise ValueError("Invalid fixed count or coordinate shape")
    ids = nodes["I"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Node IDs must be unique")
    fixed_ids = set(ids[:fixed_count])
    adjacency = {node_id: set() for node_id in ids}
    for u, v in zip(arcs["i"], arcs["j"]):
        if u not in adjacency or v not in adjacency:
            raise ValueError("Arc endpoint missing from nodes")
        adjacency[u].add(v)
        adjacency[v].add(u)
    queue = deque(node_id for node_id in ids[fixed_count:] if len(adjacency[node_id]) <= 1)
    initial_count = len(queue)
    deleted: list[str] = []
    deleted_set = set()
    while queue:
        node_id = queue.popleft()
        if node_id in deleted_set or node_id in fixed_ids or len(adjacency[node_id]) > 1:
            continue
        deleted.append(node_id)
        deleted_set.add(node_id)
        for neighbor in sorted(adjacency[node_id]):
            adjacency[neighbor].discard(node_id)
            if neighbor not in fixed_ids and len(adjacency[neighbor]) <= 1:
                queue.append(neighbor)
        adjacency[node_id].clear()
    keep_nodes = ~nodes["I"].isin(deleted_set).to_numpy()
    keep_arcs = ~arcs["i"].isin(deleted_set) & ~arcs["j"].isin(deleted_set)
    result_nodes = nodes.loc[keep_nodes].reset_index(drop=True)
    result_arcs = arcs.loc[keep_arcs].reset_index(drop=True)
    result_arcs.attrs = {}  # Generation statistics describe the graph before deletion.
    return result_nodes, xy[keep_nodes].copy(), result_arcs, {
        "terminal_intersection_nodes_before": initial_count,
        "terminal_intersection_nodes_after": sum(
            len(adjacency[node_id]) <= 1
            for node_id in ids[fixed_count:] if node_id not in deleted_set
        ),
        "deleted_terminal_intersection_nodes": len(deleted),
        "deleted_terminal_node_ids": deleted,
        "deleted_incident_directed_arcs": int((~keep_arcs).sum()),
        "cleanup_method": "recursive_degree_le_one_deletion_only",
    }


def validate_reference_schema(nodes_xlsx: Path | None, arcs_xlsx: Path | None) -> None:
    if nodes_xlsx:
        columns = list(pd.read_excel(nodes_xlsx, nrows=0).columns)
        if columns != ["I", "Longitude", "Latitude"]:
            raise ValueError(f"Reference nodes schema differs: {columns}")
    if arcs_xlsx:
        columns = list(pd.read_excel(arcs_xlsx, nrows=0).columns)
        if columns != ["i", "j", "distance"]:
            raise ValueError(f"Reference arcs schema differs: {columns}")


def plot_network(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    outline_parts: list[np.ndarray],
    ten_dash_parts: list[np.ndarray],
    output_png: Path,
) -> None:
    """Render the network with the reference map layout and South China Sea inset."""
    coordinates = nodes.set_index("I")[["Longitude", "Latitude"]]
    node_order = {node_id: index for index, node_id in enumerate(nodes["I"])}
    undirected = arcs[
        [node_order[u] < node_order[v] for u, v in zip(arcs["i"], arcs["j"])]
    ]
    segments = np.asarray(
        [
            [coordinates.loc[u].to_numpy(float), coordinates.loc[v].to_numpy(float)]
            for u, v in zip(undirected["i"], undirected["j"])
        ]
    )

    figure, axis = plt.subplots(
        figsize=NETWORK_FIGURE_SIZE_INCHES,
        dpi=NETWORK_PNG_DPI,
        facecolor="#FFFFFF",
    )
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.985)
    axis.set_facecolor("#FFFFFF")
    mainland = mainland_ring(outline_parts)
    island_rings = [
        part for part in outline_parts
        if np.allclose(part[0], part[-1], atol=1e-8)
    ]
    axis.add_collection(
        PolyCollection(
            [mainland, *island_rings],
            closed=True,
            facecolors=MAP_LAND_COLOR,
            edgecolors="#C7CECB",
            linewidths=0.45,
            antialiaseds=True,
            zorder=0,
        )
    )

    axis.add_collection(
        LineCollection(
            segments,
            colors=NETWORK_ARC_COLOR,
            linewidths=NETWORK_ARC_LINE_WIDTH,
            alpha=NETWORK_ARC_ALPHA,
            rasterized=True,
            zorder=1,
        )
    )
    axis.add_collection(
        LineCollection(
            outline_parts,
            colors="#C7CECB",
            linewidths=0.45,
            zorder=2,
        )
    )
    axis.add_collection(
        LineCollection(
            ten_dash_parts,
            colors="#66777D",
            linewidths=0.75,
            linestyles="solid",
            capstyle="round",
            zorder=2,
        )
    )

    ids = nodes["I"].astype(str)
    hub_mask = ids.str.startswith("H")
    sink_mask = ~hub_mask & ~ids.str.startswith("I")
    intersection_mask = ids.str.startswith("I")
    axis.scatter(
        nodes.loc[intersection_mask, "Longitude"],
        nodes.loc[intersection_mask, "Latitude"],
        s=INTERSECTION_MARKER_SIZE,
        c=INTERSECTION_NODE_COLOR,
        alpha=INTERSECTION_NODE_ALPHA,
        linewidths=0,
        zorder=3,
    )
    axis.scatter(
        nodes.loc[hub_mask, "Longitude"],
        nodes.loc[hub_mask, "Latitude"],
        s=HUB_MARKER_SIZE,
        c="#4B9FD8",
        alpha=0.95,
        linewidths=0.55,
        edgecolors="#111111",
        zorder=4,
    )
    axis.scatter(
        nodes.loc[sink_mask, "Longitude"],
        nodes.loc[sink_mask, "Latitude"],
        s=SINK_MARKER_SIZE,
        c="#54B98B",
        marker="s",
        alpha=0.95,
        linewidths=0.55,
        edgecolors="#111111",
        zorder=5,
    )

    west, east, south, north = MAINLAND_MAP_EXTENT
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_aspect(1.0 / math.cos(math.radians(35.0)), adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor="#4B9FD8",
            markeredgecolor="#4B9FD8", markersize=10.5, label="Source",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="", markerfacecolor="#54B98B",
            markeredgecolor="#54B98B", markersize=10.5, label="Sink",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=INTERSECTION_NODE_COLOR,
            markeredgecolor=INTERSECTION_NODE_COLOR,
            markersize=5.0, label="Intersection node",
        ),
        Line2D(
            [0], [0], color=NETWORK_ARC_COLOR, linewidth=2.4,
            solid_capstyle="round", label="Arc",
        ),
    ]
    legend = axis.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.025),
        fontsize=12.0,
        frameon=False,
        borderaxespad=0.0,
        handlelength=3.0,
        handletextpad=1.0,
        labelspacing=0.9,
    )
    legend.set_zorder(30)
    # Halve only the Arc swatch, preserving all legend marker/text positions.
    arc_handle = legend.get_lines()[-1]
    arc_x = np.asarray(arc_handle.get_xdata(), dtype=float)
    arc_handle.set_xdata(arc_x.mean() + 0.5 * (arc_x - arc_x.mean()))

    inset_scale = math.sqrt(0.5)
    inset_width = 0.18 * inset_scale
    inset_height = 0.40 * inset_scale
    inset_right = 0.985
    inset_bottom = 0.025
    inset = axis.inset_axes(
        [inset_right - inset_width, inset_bottom, inset_width, inset_height],
        zorder=20,
    )
    inset.set_facecolor("#FFFFFF")
    inset.add_collection(
        PolyCollection(
            [mainland, *island_rings],
            closed=True,
            facecolors=MAP_LAND_COLOR,
            edgecolors="#C7CECB",
            linewidths=0.25,
            antialiaseds=True,
            zorder=0,
        )
    )
    inset.add_collection(
        LineCollection(
            ten_dash_parts,
            colors="#66777D",
            linewidths=0.9,
            linestyles="solid",
            capstyle="round",
            zorder=3,
        )
    )
    inset_west, inset_east, inset_south, inset_north = SOUTH_CHINA_SEA_EXTENT
    inset.set_xlim(inset_west, inset_east)
    inset.set_ylim(inset_south, inset_north)
    inset.set_aspect(1.0 / math.cos(math.radians(14.0)), adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_color("#66777D")
        spine.set_linewidth(0.75)

    figure.savefig(output_png, dpi=NETWORK_PNG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate(args: argparse.Namespace) -> dict[str, object]:
    validate_reference_schema(args.reference_nodes, args.reference_arcs)
    fixed = load_fixed_nodes(args.hub_csv, args.sink_csv)
    boundary_records = read_shape_records(args.boundary)
    if len(boundary_records) < 2:
        raise ValueError("China boundary shapefile must contain outline and ten-dash records")
    outline_parts = boundary_records[0]
    ten_dash_parts = boundary_records[1]
    if len(ten_dash_parts) != 10:
        raise ValueError(f"Expected 10 ten-dash-line parts, found {len(ten_dash_parts)}")
    ring_lon_lat = mainland_ring(outline_parts)
    ring_xy = project(ring_lon_lat)
    fixed_xy = project(fixed[["Longitude", "Latitude"]].to_numpy(float))
    domain = Polygon(ring_xy)
    if not domain.is_valid:
        raise ValueError("Invalid mainland polygon; refusing to alter the coverage domain")
    coastal_distance = np.asarray([domain.distance(Point(point)) for point in fixed_xy])
    accepted = coastal_distance <= args.coastal_node_tolerance_km
    if not accepted.all():
        bad = fixed.loc[~accepted, "I"].tolist()[:10]
        raise ValueError(f"Fixed nodes outside mainland boundary: {bad}")

    intersection_xy, refinement = adaptive_intersections(
        fixed_xy, ring_xy, args.max_empty_radius_km, args.max_intersections, args.batch_size,
        grid_spacing_ratio=args.coverage_grid_spacing_ratio,
        boundary_spacing_km=args.boundary_sample_spacing_km,
        setback_km=args.intersection_boundary_setback_km,
        min_separation_ratio=args.insertion_min_separation_ratio,
        prune_redundant=args.prune_redundant_intersections,
        buffer_quad_segs=args.coverage_buffer_quad_segs,
    )
    lon_lat = unproject(intersection_xy)
    width = max(5, len(str(len(intersection_xy))))
    intersections = pd.DataFrame({
        "I": [f"I{k:0{width}d}" for k in range(1, len(intersection_xy) + 1)],
        "Longitude": lon_lat[:, 0],
        "Latitude": lon_lat[:, 1],
    })
    nodes = pd.concat((fixed, intersections), ignore_index=True)
    # Validate exactly the coordinates that will be exported (8 decimal places).
    # Match the original CSV formatter exactly, including half-way rounding of
    # fixed facilities; DataFrame.round uses a different floating-point path.
    nodes[["Longitude", "Latitude"]] = nodes[["Longitude", "Latitude"]].applymap(
        lambda value: float(format(value, ".8f"))
    )
    xy = project(nodes[["Longitude", "Latitude"]].to_numpy(float))
    if not domain.difference(coverage_geometry(xy, args.max_empty_radius_km, args.coverage_buffer_quad_segs)).is_empty:
        raise RuntimeError("Rounded output coordinates fail mainland coverage")
    print("  filtering arcs and checking connectivity...", flush=True)
    arcs, triangle_count = build_arcs(
        nodes, xy, ring_xy, fixed_count=len(fixed),
        boundary_tolerance_km=args.arc_boundary_tolerance_km,
        coastal_tolerance_km=args.coastal_node_tolerance_km,
        max_arc_length_km=args.max_arc_length_km,
    )
    generation_arc_statistics = dict(arcs.attrs)
    nodes_before_cleanup, arcs_before_cleanup = len(nodes), len(arcs)
    cleanup = {}
    if args.remove_unused_intersection_branches:
        print("  deleting terminal intersection nodes and incident arcs...", flush=True)
        nodes, xy, arcs, cleanup = prune_terminal_intersections(
            nodes, xy, arcs, fixed_count=len(fixed),
        )

    # The requested deletion is unconditional. Measure its coverage effect rather
    # than relocating nodes, adding replacement points, or claiming old coverage.
    uncovered = domain.difference(coverage_geometry(xy, args.max_empty_radius_km, args.coverage_buffer_quad_segs))
    clearance = np.asarray([domain.boundary.distance(Point(p)) for p in xy[len(fixed):]])
    for key in ("coverage_verified", "coverage_radius_upper_bound_km", "uncovered_mainland_area_km2", "probe_max_nearest_node_distance_km"):
        refinement[key + "_before_cleanup"] = refinement.pop(key)
    refinement.update({
        "coverage_radius_km": args.max_empty_radius_km,
        "coverage_verified": bool(uncovered.is_empty),
        "uncovered_mainland_area_km2": float(uncovered.area),
        "mainland_coverage_area_fraction": float(1.0 - uncovered.area / domain.area),
        "boundary_intersection_nodes": int(np.sum(clearance <= 1e-4)),
        "intersection_nodes_within_setback": int(np.sum(clearance < args.intersection_boundary_setback_km - 1e-4)),
    })

    # Output invariants: exact schemas, unique IDs, symmetric arcs, positive distances.
    assert list(nodes.columns) == ["I", "Longitude", "Latitude"]
    assert list(arcs.columns) == ["i", "j", "distance"]
    assert nodes["I"].is_unique and (arcs["distance"].round(6) > 0).all()
    pairs = set(zip(arcs["i"], arcs["j"]))
    assert len(pairs) == len(arcs) and all(u != v for u, v in pairs)
    assert all((v, u) in pairs for u, v in pairs)
    adjacency = {node_id: set() for node_id in nodes["I"]}
    for u, v in pairs:
        adjacency[u].add(v)
    reached = set()
    stack = [nodes.iloc[0]["I"]]
    while stack:
        current = stack.pop()
        if current in reached:
            continue
        reached.add(current)
        stack.extend(adjacency[current] - reached)
    assert len(reached) == len(nodes)
    # Final graph checks remain mandatory; post-deletion coverage is diagnostic.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.output_dir / NODES_OUTPUT_FILENAME
    arcs_path = args.output_dir / ARCS_OUTPUT_FILENAME
    network_png_path = args.output_dir / NETWORK_PNG_FILENAME
    nodes.to_csv(nodes_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    arcs.to_csv(arcs_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    if OUTPUT_NETWORK_PNG:
        plot_network(nodes, arcs, outline_parts, ten_dash_parts, network_png_path)
    result = {
        "hub_nodes": int(fixed["I"].str.startswith("H").sum()),
        "sink_nodes": int(len(fixed) - fixed["I"].str.startswith("H").sum()),
        "intersection_nodes": len(nodes) - len(fixed),
        "total_nodes": len(nodes),
        "directed_arcs": len(arcs),
        "total_nodes_before_cleanup": nodes_before_cleanup,
        "directed_arcs_before_cleanup": arcs_before_cleanup,
        "triangles_before_cleanup": triangle_count,
        "arc_generation_statistics": generation_arc_statistics,
        "connected_components": 1,
        "maximum_arc_length_km": float(arcs["distance"].max()),
        **refinement,
        **cleanup,
        "parameters": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "nodes_csv": str(nodes_path.resolve()),
        "arcs_csv": str(arcs_path.resolve()),
        "network_png": str(network_png_path.resolve()) if OUTPUT_NETWORK_PNG else "disabled",
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hub_csv", nargs="?", type=Path, default=HUB_INPUT_FILE)
    parser.add_argument("sink_csv", nargs="?", type=Path, default=SINK_INPUT_FILE)
    parser.add_argument("--boundary", type=Path, default=BOUNDARY_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-nodes", type=Path, default=REFERENCE_NODES_FILE)
    parser.add_argument("--reference-arcs", type=Path, default=REFERENCE_ARCS_FILE)
    parser.add_argument(
        "--max-empty-radius-km",
        type=float,
        default=MAX_EMPTY_RADIUS_KM,
        help=f"Largest permitted empty-circle radius (default: {MAX_EMPTY_RADIUS_KM:g} km)",
    )
    parser.add_argument("--max-intersections", type=int, default=MAX_INTERSECTION_NODES)
    parser.add_argument("--batch-size", type=int, default=INSERTION_BATCH_SIZE)
    parser.add_argument("--coverage-grid-spacing-ratio", type=float, default=COVERAGE_GRID_SPACING_RATIO)
    parser.add_argument("--boundary-sample-spacing-km", type=float, default=BOUNDARY_SAMPLE_SPACING_KM)
    parser.add_argument("--intersection-boundary-setback-km", type=float, default=INTERSECTION_BOUNDARY_SETBACK_KM)
    parser.add_argument("--insertion-min-separation-ratio", type=float, default=INSERTION_MIN_SEPARATION_RATIO)
    parser.add_argument("--coverage-buffer-quad-segs", type=int, default=COVERAGE_BUFFER_QUAD_SEGS)
    parser.add_argument("--prune-redundant-intersections", action=argparse.BooleanOptionalAction, default=PRUNE_REDUNDANT_INTERSECTIONS)
    parser.add_argument("--coastal-node-tolerance-km", type=float, default=COASTAL_NODE_TOLERANCE_KM)
    parser.add_argument("--arc-boundary-tolerance-km", type=float, default=ARC_BOUNDARY_TOLERANCE_KM)
    parser.add_argument("--max-arc-length-km", type=float, default=MAX_ARC_LENGTH_KM)
    parser.add_argument("--remove-unused-intersection-branches", action=argparse.BooleanOptionalAction, default=REMOVE_UNUSED_INTERSECTION_BRANCHES)
    args = parser.parse_args(argv)
    numeric = [value for value in vars(args).values() if isinstance(value, (float, int))]
    if not np.isfinite(numeric).all():
        parser.error("numeric parameters must be finite")
    if args.max_empty_radius_km <= 0 or args.max_intersections < 0 or args.batch_size <= 0:
        parser.error("radius and batch-size must be positive; max-intersections must be nonnegative")
    if args.coverage_grid_spacing_ratio <= 0 or args.boundary_sample_spacing_km <= 0:
        parser.error("coverage grid ratio and boundary spacing must be positive")
    if not 0 < args.insertion_min_separation_ratio <= 1 or args.coverage_buffer_quad_segs < 4:
        parser.error("separation ratio must be in (0, 1]; buffer quad segments must be >= 4")
    if min(args.intersection_boundary_setback_km, args.coastal_node_tolerance_km,
           args.arc_boundary_tolerance_km, args.max_arc_length_km) < 0:
        parser.error("setback, tolerances and maximum arc length must be nonnegative")
    return args


if __name__ == "__main__":
    result = generate(parse_args())
    print("Generation complete")
    for key, value in result.items():
        print(f"  {key}: {value}")
