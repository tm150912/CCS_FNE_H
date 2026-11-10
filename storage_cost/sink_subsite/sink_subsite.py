# -*- coding: utf-8 -*-
"""盆地连续资源分区：严格测地面积上限、区内代表点、CSV 与地图。

运行：使用 ArcGIS Pro 的 Python 3（不支持 Python 2；需要 arcpy、numpy、GDAL/OGR、openpyxl、matplotlib）。
直接运行本文件即可；所有输入、输出与假设均在下面配置，不依赖工作目录。
最新划分结果统一写入 subsite_result，不保留旧版对比输出。
算法：大陆裁剪 -> 组件数量协调 -> 等面积加权 Voronoi 与质心迭代 -> 空间验收。
加权单元由 ||x-s_i||²-w_i 最小确定；调权重平衡面积，移动中心改善形状。
多起点后先改善最差单元，再以其最低紧凑度为底线提高全部子场的平均紧凑度。
两阶段均只保留完整、连通、通过面积验收的候选，分区数量保持不变。
真实盆地边界保持不变，因此边界单元不会与内部单元完全同形。
不移动原始边界、不虚构井数或封存成本。
"""

from pathlib import Path
import csv
import json
import math
import sys
import os
import tempfile
import time
os.environ.setdefault("GDAL_PAM_PROXY_DIR", tempfile.gettempdir())
from osgeo import ogr

import arcpy
import numpy as np
import openpyxl


# ========================= 用户配置 =========================
BASE_DIR = Path(__file__).resolve().parent
CHINA_BOUNDARY = BASE_DIR / "China_boundary/China_boundary.shp"
BASIN_BOUNDARY = BASE_DIR / "sinks_boundary/sinks_boundary.shp"
SINK_WORKBOOK = BASE_DIR.parent / "sink.xlsx"
RESULT_DIR = BASE_DIR / "subsite_result"  # 最新版划分数据、地图与验收报告统一输出到此处。
OUTPUT_CSV = RESULT_DIR / "sink_subsite.csv"
OUTPUT_GEOJSON = RESULT_DIR / "sink_subsite.geojson"  # 保存真实分区边界，供 GIS 复用。
OUTPUT_MAP = RESULT_DIR / "sink_subsite_map.png"
OUTPUT_DETAIL_MAP = RESULT_DIR / "sink_subsite_detail.png"
OUTPUT_AUDIT = RESULT_DIR / "sink_subsite_audit.json"

MAX_AREA_KM2 = 8000.0             # 硬上限；最终逐块测地验收，不增加百分比容差。
CONSTRUCTION_MARGIN_KM2 = 2.0      # 向下留面积平衡/投影数值裕量，绝不放宽硬上限。
BASIN_AREA_RATIO_LIMIT = 1.03      # 同盆地最大/最小子场面积比上限，包括分离组件。
MAX_COMPONENT_COUNT_MULTIPLIER = 2 # 协调组件面积时，子场总数最多为初始最少数量的两倍。
POWER_AREA_REL_TOL = 1e-4          # 每个连通组件内部的等面积相对误差目标。
LLOYD_MAX_ITERATIONS = 30          # 中心移动迭代上限；保留通过面积/连通检查的最好形状。
PARTITION_RESTARTS = 3             # 确定性多起点，减少单次初始点造成的局部形状偏差。
PARTITION_SEED = 47               # 固定种子，可复现。
SHAPE_POLISH_ROUNDS = 8            # 对最差紧凑度子场及邻居做有限方向搜索，每步重新平衡面积。
MEAN_POLISH_ROUNDS = 4             # 随后遍历全部生成点，提高算术平均紧凑度；0 可关闭此阶段。
MIN_COMPACTNESS_SLACK = 1e-6        # 相对均值优化开始时的最低值，仅留固定数值裕量，不逐轮累减。
BASIN_MIN_SUBSITE_COUNTS = {"P16": 10}  # P16 的 7–9 块仍细长；10 块形状改善明显。
VERTEX_SPACING_M = 1000.0          # 投影与输出前加密边界，减少长边重投影误差。
COVERAGE_TOL_KM2 = 0.02            # 每盆地覆盖/重叠的数值验收阈值，非子场面积容差。
SHOW_PROGRESS = True

# 本次按大陆陆上封存口径：仅轮廓线转面中的最大陆地面；不使用十段线。
# 总地质资源默认按原始盆地面积均匀分布；裁去海域/境外部分时同步扣除资源份额。
# 若确认 sink.xlsx 已是裁剪后大陆总地质资源量，可改为 "already_mainland"。
RESOURCE_BASIS = "uniform_original_area"
# 年最大注入量对应原盆地：先按大陆交集/原面积折减，再按子场/大陆交集面积分配。
ANNUAL_ALLOCATION_BASIS = "original_basin_maximum_with_mainland_area_clipping"

# 显式映射：USGS 编码、记录顺序、中文别名均不等同于 Sink_ID。
USGS_TO_SINK = {
    22: "P1", 24: "P2", 20: "P3", 18: "P4", 12: "P6", 11: "P7",
    9: "P8", 6: "P9", 7: "P10", 3: "P11", 2: "P13", 1: "P14",
    4: "P15", 5: "P16",
}
WGS84 = arcpy.SpatialReference(4326)


def progress(message):
    if SHOW_PROGRESS:
        print(message, flush=True)


def area_km2(geometry):
    return geometry.getArea("GEODESIC", "SQUAREKILOMETERS")


def parts(geometry):
    """一个 ArcPy part 是一个外环及其内洞；不把洞拆成独立子场。"""
    result = []
    for part in geometry:
        if not any(point is None for point in part):
            if len(part) > 0:
                result.append(arcpy.Polygon(part, geometry.spatialReference))
            continue
        # 直接把含 None 的 part 传给 Polygon 会丢失内洞，必须显式构造各环。
        rings, ring = [], arcpy.Array()
        for point in part:
            if point is None:
                if len(ring) > 0:
                    rings.append(ring)
                    ring = arcpy.Array()
            else:
                ring.add(point)
        if len(ring) > 0:
            rings.append(ring)
        result.append(arcpy.Polygon(arcpy.Array(rings), geometry.spatialReference))
    return result


def points(geometry):
    return [(p.X, p.Y) for part in geometry for p in part if p is not None]


def local_projection(longitude, latitude):
    sr = arcpy.SpatialReference()
    sr.loadFromString(
        'PROJCS["Basin_LAEA",GEOGCS["GCS_WGS_1984",'
        'DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
        'PROJECTION["Lambert_Azimuthal_Equal_Area"],'
        'PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],'
        f'PARAMETER["Central_Meridian",{longitude}],'
        f'PARAMETER["Latitude_Of_Origin",{latitude}],UNIT["Meter",1]]'
    )
    return sr


def geographic(geometry):
    return geometry.densify("DISTANCE", VERTEX_SPACING_M, 0.1).projectAs(WGS84)


def read_inputs():
    workbook = openpyxl.load_workbook(SINK_WORKBOOK, read_only=True, data_only=True)
    rows = list(workbook["sink"].iter_rows(values_only=True))
    headers = rows[0]
    sinks = {r[0]: dict(zip(headers, r)) for r in rows[1:] if r[0]}
    workbook.close()
    return sinks


def load_geography(sinks):
    """原数据只读；所有转面和几何检查均写入 ArcGIS 内存工作区。"""
    arcpy.env.overwriteOutput = True
    text_fields = [f.name for f in arcpy.ListFields(str(CHINA_BOUNDARY))
                   if f.type == "String"]
    outline = []
    with arcpy.da.SearchCursor(str(CHINA_BOUNDARY), ["SHAPE@"] + text_fields) as cur:
        for row in cur:
            if any("轮廓" in str(value) for value in row[1:]):
                outline.append(row[0])
    if not outline:
        raise ValueError("中国边界未找到‘轮廓线’记录；不能用十段线或记录顺序代替。")
    arcpy.management.FeatureToPolygon(outline, "memory/ss_land")
    land = [g for (g,) in arcpy.da.SearchCursor("memory/ss_land", ["SHAPE@"])]
    mainland = max(land, key=area_km2).projectAs(WGS84)
    basins = []
    seen = set()
    with arcpy.da.SearchCursor(str(BASIN_BOUNDARY), ["USGS_BASIN", "SHAPE@"]) as cur:
        for code, original in cur:
            sid = USGS_TO_SINK[int(code)]
            if sid in seen:
                raise ValueError("盆地映射重复：" + sid)
            seen.add(sid)
            # 分母与裁剪后分子使用相同的边界加密口径，避免资源份额因长边解释不同而超过 1。
            original = original.densify("DISTANCE", VERTEX_SPACING_M, 0.1)
            original_area = area_km2(original)
            clipped = original.intersect(mainland.projectAs(original.spatialReference), 4)
            if clipped.pointCount == 0:
                raise ValueError("盆地与大陆无有效交集：" + sid)
            # 在源投影中先加密，保留裁剪边界转换为局部等积投影时的形状。
            clipped = clipped.densify("DISTANCE", VERTEX_SPACING_M, 0.1)
            retained_area = area_km2(clipped)
            if retained_area > original_area + COVERAGE_TOL_KM2:
                raise ValueError("裁剪后面积大于原盆地面积：" + sid)
            center = clipped.projectAs(WGS84).trueCentroid
            sr = local_projection(center.X, center.Y)
            projected = clipped.projectAs(sr)
            basins.append({"Sink_ID": sid, "Sink_Name": sinks[sid]["Sink_Name"],
                           "USGS_BASIN": int(code), "source": sinks[sid],
                           "Original_Area_km2": original_area,
                           "Retained_Area_km2": retained_area,
                           "local_geometry": projected, "geometry": clipped.projectAs(WGS84)})
    if seen != set(USGS_TO_SINK.values()):
        raise ValueError("盆地边界未完整覆盖配置的 14 个盆地。")
    basins.sort(key=lambda b: int(b["Sink_ID"][1:]))
    arcpy.management.CopyFeatures([b["local_geometry"].projectAs(WGS84) for b in basins],
                                  "memory/ss_targets")
    arcpy.management.CheckGeometry("memory/ss_targets", "memory/ss_target_check", "OGC")
    issues = int(arcpy.management.GetCount("memory/ss_target_check")[0])
    if issues:
        raise ValueError(f"大陆裁剪后的盆地存在 {issues} 项几何问题，停止划分。")
    return mainland, basins, sorted(set(sinks) - seen, key=lambda s: int(s[1:]))


def _transform_in_place(geometry, origin, scale, inverse=False):
    if geometry.GetGeometryCount():
        for index in range(geometry.GetGeometryCount()):
            _transform_in_place(geometry.GetGeometryRef(index), origin, scale, inverse)
    else:
        for index in range(geometry.GetPointCount()):
            x, y, *_ = geometry.GetPoint(index)
            if inverse:
                geometry.SetPoint_2D(index, x * scale + origin[0], y * scale + origin[1])
            else:
                geometry.SetPoint_2D(index, (x - origin[0]) / scale, (y - origin[1]) / scale)


def _polygons(geometry):
    if geometry is None or geometry.IsEmpty():
        return []
    if geometry.GetGeometryName() == 'POLYGON':
        return [geometry]
    result = []
    for index in range(geometry.GetGeometryCount()):
        child = geometry.GetGeometryRef(index)
        if child.GetGeometryName() in ('POLYGON', 'MULTIPOLYGON', 'GEOMETRYCOLLECTION'):
            result.extend(_polygons(child))
    return result


def _polygon(vertices):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in vertices:
        ring.AddPoint_2D(float(x), float(y))
    ring.AddPoint_2D(float(vertices[0, 0]), float(vertices[0, 1]))
    result = ogr.Geometry(ogr.wkbPolygon)
    result.AddGeometry(ring)
    return result


def _clip(vertices, normal, bound):
    if len(vertices) < 3:
        return np.empty((0, 2), dtype=float)
    distances = vertices @ normal - bound
    inside = distances <= 1e-12
    if inside.all():
        return vertices
    if not inside.any():
        return np.empty((0, 2), dtype=float)
    result = []
    previous = vertices[-1]
    previous_distance = distances[-1]
    previous_inside = inside[-1]
    for current, distance, current_inside in zip(vertices, distances, inside):
        if current_inside != previous_inside:
            ratio = previous_distance / (previous_distance - distance)
            result.append(previous + ratio * (current - previous))
        if current_inside:
            result.append(current)
        previous, previous_distance, previous_inside = current, distance, current_inside
    if len(result) < 3:
        return np.empty((0, 2), dtype=float)
    result = np.asarray(result)
    keep = np.linalg.norm(result - np.roll(result, 1, axis=0), axis=1) > 1e-11
    return result[keep]


class _PowerProblem:
    def __init__(self, domain, count):
        self.domain = domain
        self.count = count
        self.target = domain.GetArea() / count
        xmin, xmax, ymin, ymax = domain.GetEnvelope()
        padding = max(xmax-xmin, ymax-ymin) * 0.02 + 1.0
        self.box = np.array([(xmin-padding,ymin-padding),(xmax+padding,ymin-padding),
                             (xmax+padding,ymax+padding),(xmin-padding,ymax+padding)])
        self.evaluations = 0

    def evaluate(self, sites, weights):
        count = len(sites)
        squared = np.sum(sites*sites, axis=1)
        areas = np.zeros(count)
        cells, convex_cells = [], []
        for i in range(count):
            vertices = self.box.copy()
            for k in range(count):
                if i == k:
                    continue
                normal = 2.0*(sites[k]-sites[i])
                bound = squared[k]-squared[i]+weights[i]-weights[k]
                vertices = _clip(vertices, normal, bound)
                if len(vertices) < 3:
                    break
            convex_cells.append(vertices)
            cell = None if len(vertices) < 3 else self.domain.Intersection(_polygon(vertices))
            cells.append(cell)
            areas[i] = 0.0 if cell is None else cell.GetArea()
        self.evaluations += 1
        return areas, cells, convex_cells

    def jacobian(self, sites, weights, convex_cells):
        count = len(sites)
        squared = np.sum(sites*sites, axis=1)
        matrix = np.zeros((count,count))
        for i, vertices in enumerate(convex_cells):
            normals = 2.0*(sites-sites[i])
            norms = np.linalg.norm(normals,axis=1)
            bounds = squared-squared[i]+weights[i]-weights
            for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
                if np.linalg.norm(b-a) < 1e-10:
                    continue
                middle = (a+b)*0.5
                distance = np.full(count,np.inf)
                valid = norms > 1e-12
                distance[valid] = np.abs(normals[valid] @ middle-bounds[valid])/norms[valid]
                k = int(np.argmin(distance))
                if k <= i or distance[k] > 1e-7:
                    continue
                line = ogr.Geometry(ogr.wkbLineString)
                line.AddPoint_2D(float(a[0]),float(a[1]))
                line.AddPoint_2D(float(b[0]),float(b[1]))
                segment = self.domain.Intersection(line)
                length = 0.0 if segment is None else segment.Length()
                coefficient = length/norms[k]
                matrix[i,i] += coefficient
                matrix[k,k] += coefficient
                matrix[i,k] -= coefficient
                matrix[k,i] -= coefficient
        return matrix

    def balance(self, sites, weights, tolerance, max_iterations=55):
        weights = weights.copy()-np.mean(weights)
        areas,cells,convex = self.evaluate(sites,weights)
        for iteration in range(max_iterations):
            error = self.target-areas
            relative_error = np.max(np.abs(error))/self.target
            if relative_error <= tolerance:
                return weights,areas,cells,True,iteration
            matrix = self.jacobian(sites,weights,convex)
            augmented = np.zeros((self.count+1,self.count+1))
            augmented[:-1,:-1] = matrix
            augmented[-1,:-1] = 1.0
            augmented[:-1,-1] = 1.0
            rhs = np.r_[error,0.0]
            try:
                direction = np.linalg.solve(augmented,rhs)[:-1]
            except np.linalg.LinAlgError:
                direction = np.linalg.lstsq(augmented,rhs,rcond=1e-10)[0][:-1]
            if not np.isfinite(direction).all() or np.max(np.abs(direction)) > 1e6:
                direction = error.copy()
            old_loss = float(error @ error)
            accepted = False
            for proposed_direction in (direction,error):
                step = 1.0
                for _ in range(18):
                    trial = weights+step*proposed_direction
                    trial -= np.mean(trial)
                    new_areas,new_cells,new_convex = self.evaluate(sites,trial)
                    new_error = self.target-new_areas
                    if (np.min(new_areas)>self.target*1e-7
                        and new_error@new_error < old_loss*(1-1e-5*step)):
                        weights,areas,cells,convex = trial,new_areas,new_cells,new_convex
                        accepted = True
                        break
                    step *= 0.5
                if accepted:
                    break
            if not accepted:
                return weights,areas,cells,False,iteration+1
        return weights,areas,cells,False,max_iterations


def _sample_sites(domain, count, rng):
    xmin,xmax,ymin,ymax = domain.GetEnvelope()
    samples = []
    desired = max(600,count*70)
    point = ogr.Geometry(ogr.wkbPoint)
    for _ in range(80):
        candidates = rng.uniform((xmin,ymin),(xmax,ymax),size=(desired,2))
        for coordinate in candidates:
            point.SetPoint_2D(0,float(coordinate[0]),float(coordinate[1]))
            if domain.Contains(point):
                samples.append(coordinate)
                if len(samples) >= desired:
                    break
        if len(samples) >= desired:
            break
    candidates = np.asarray(samples)
    if len(candidates)<count:
        raise RuntimeError('Unable to generate enough interior seed candidates')
    chosen = [int(rng.integers(len(candidates)))]
    nearest = np.sum((candidates-candidates[chosen[0]])**2,axis=1)
    for _ in range(1,count):
        chosen.append(int(np.argmax(nearest)))
        nearest = np.minimum(nearest,np.sum((candidates-candidates[chosen[-1]])**2,axis=1))
    return candidates[chosen].copy()


def _quality(cells, target):
    if any(cell is None or len(_polygons(cell)) != 1 or not cell.IsValid() for cell in cells):
        return None
    compactness = np.asarray([4*math.pi*cell.GetArea()/cell.Boundary().Length()**2 for cell in cells])
    return (float(np.min(compactness)),float(np.mean(compactness)))


def balanced_partition(arcpy_geometry, count, seed=47, restarts=3, max_outer=30,
                       area_tolerance=1e-4, progress=None, polish_rounds=4, mean_rounds=4):
    """Return (single-part ArcPy polygons, metadata); raises if no valid candidate.

    Input geometry must be one connected projected polygon, with any holes retained.
    Weights use an analytic shared-edge area Jacobian; no numerical Jacobian.
    """
    import arcpy
    started = time.perf_counter()
    if count < 1:
        raise ValueError('count must be positive')
    domain = ogr.CreateGeometryFromWkb(bytes(arcpy_geometry.WKB))
    if len(_polygons(domain)) != 1:
        raise ValueError('Input must have exactly one polygon component')
    center = domain.Centroid()
    origin = np.array([center.GetX(),center.GetY()])
    scale = math.sqrt(domain.GetArea()/count)
    _transform_in_place(domain,origin,scale)
    problem = _PowerProblem(domain,count)
    if count == 1:
        quality=_quality([domain],problem.target)
        return [arcpy_geometry], {'count':1,'max_relative_area_error':0.0,
                                 'min_compactness':quality[0],'mean_compactness':quality[1],
                                 'chosen_restart':0,'chosen_outer':0,'evaluations':0,
                                 'restarts':[],'polish_accepted':0,
                                 'seconds':time.perf_counter()-started}
    balance_tolerance=min(area_tolerance*0.15,1e-6)
    best = None
    restart_records = []
    for restart in range(restarts):
        rng = np.random.default_rng(seed+restart*1009)
        sites = _sample_sites(domain,count,rng)
        weights = np.zeros(count)
        valid_iterations = 0
        total_inner = 0
        best_restart = None
        for outer in range(max_outer+1):
            weights,areas,cells,balanced,inner = problem.balance(sites,weights,balance_tolerance)
            total_inner += inner
            quality = _quality(cells,problem.target) if balanced else None
            if quality is not None:
                valid_iterations += 1
                record = {'quality':quality,'cells':[cell.Clone() for cell in cells],
                          'sites':sites.copy(),'weights':weights.copy(),
                          'restart':restart,'outer':outer,
                          'max_relative_area_error':float(np.max(np.abs(areas/problem.target-1)))}
                if best_restart is None or _better_quality(quality,best_restart['quality']):
                    best_restart=record
                if best is None or _better_quality(quality,best['quality']):
                    best=record
            if progress and (outer%10==0 or outer==max_outer):
                progress('  起点 %d/%d，迭代 %d：相对面积误差 %.3g，连续有效=%s，最差紧凑度=%s'
                         %(restart+1,restarts,outer,np.max(np.abs(areas/problem.target-1)),
                           quality is not None,None if quality is None else round(quality[0],4)))
            if not balanced or outer==max_outer:
                break
            centroids = np.array([[cell.Centroid().GetX(),cell.Centroid().GetY()] for cell in cells])
            # At occasional invalid iterations, move seeds towards their largest piece;
            # this changes generators only. No geometry or resource fragment is discarded.
            if quality is None and outer%4==3:
                for i,cell in enumerate(cells):
                    polygons = _polygons(cell)
                    if len(polygons)>1:
                        centroid=max(polygons,key=lambda polygon:polygon.GetArea()).Centroid()
                        centroids[i]=[centroid.GetX(),centroid.GetY()]
            displacement = float(np.max(np.linalg.norm(centroids-sites,axis=1)))
            if displacement<2e-4 and quality is not None:
                break
            sites += 0.8*(centroids-sites)
        restart_records.append({'restart':restart,'valid_iterations':valid_iterations,
                                'inner_iterations':total_inner,
                                'best_min_compactness':None if best_restart is None else best_restart['quality'][0]})
    if best is None:
        raise RuntimeError('No connected valid balanced power candidate: '+str(restart_records))
    best = _polish_worst(problem, best, balance_tolerance, polish_rounds, progress)
    best, mean_diagnostics = _polish_mean(problem, best, balance_tolerance, mean_rounds, progress)
    output=[]
    for geometry in best['cells']:
        restored=geometry.Clone()
        _transform_in_place(restored,origin,scale,inverse=True)
        cell=arcpy.FromWKT(restored.ExportToWkt(),arcpy_geometry.spatialReference)
        if cell.partCount != 1:
            raise RuntimeError('ArcPy conversion produced a disconnected cell')
        output.append(cell)
    target=arcpy_geometry.area/count
    final_error=max(abs(cell.area/target-1) for cell in output)
    if final_error>area_tolerance:
        raise RuntimeError('Final ArcPy area balance exceeds tolerance: '+str(final_error))
    metadata={'count':count,'max_relative_area_error':final_error,
              'min_compactness':best['quality'][0],'mean_compactness':best['quality'][1],
              'chosen_restart':best['restart'],'chosen_outer':best['outer'],
              'evaluations':problem.evaluations,'restarts':restart_records,
              'polish_accepted':best.get('polish_accepted',0),
              'mean_optimization':mean_diagnostics,
              'seconds':time.perf_counter()-started}
    return output,metadata


def _polish_worst(problem, initial, tolerance, rounds, progress):
    """Bounded pattern search explicitly improving the worst compactness."""
    best=initial
    accepted=0
    step=0.35
    directions=np.array([(math.cos(a),math.sin(a)) for a in np.arange(8)*math.pi/4])
    for iteration in range(rounds):
        compactness=np.asarray([4*math.pi*cell.GetArea()/cell.Boundary().Length()**2 for cell in best['cells']])
        worst=int(np.argmin(compactness))
        sites=best['sites']
        neighbours=np.argsort(np.sum((sites-sites[worst])**2,axis=1))[:min(5,len(sites))]
        iteration_best=best
        for index in neighbours:
            for direction in directions:
                candidate_sites=sites.copy()
                candidate_sites[index]+=step*direction
                weights,areas,cells,balanced,_=problem.balance(candidate_sites,best['weights'],tolerance,max_iterations=20)
                quality=_quality(cells,problem.target) if balanced else None
                if quality is None:
                    continue
                # Ignore sub-microscopic numerical differences in the limiting cell.
                improvement=_better_quality(quality,iteration_best['quality'])
                if improvement:
                    iteration_best=dict(best,quality=quality,cells=[cell.Clone() for cell in cells],
                                        sites=candidate_sites,weights=weights,
                                        max_relative_area_error=float(np.max(np.abs(areas/problem.target-1))))
        if iteration_best is best:
            step*=0.5
        else:
            best=iteration_best
            accepted+=1
        if progress:
            progress('  形状微调 %d/%d：最差/平均紧凑度 %.5f/%.5f'
                     %(iteration+1,rounds,best['quality'][0],best['quality'][1]))
    best=dict(best,polish_accepted=accepted)
    return best


def _polish_mean(problem, initial, tolerance, rounds, progress):
    """提高算术平均紧凑度；最低值底线固定在本阶段开始时，不随迭代下移。"""
    started=time.perf_counter()
    evaluations_before=problem.evaluations
    floor=initial['quality'][0]-MIN_COMPACTNESS_SLACK
    current=initial
    accepted=0
    trials=0
    history=[]

    def trial(sites, reference):
        nonlocal trials
        trials+=1
        weights,areas,cells,balanced,_=problem.balance(sites,reference['weights'],tolerance,max_iterations=20)
        quality=_quality(cells,problem.target) if balanced else None
        if quality is None or quality[0]<floor or quality[1]<=reference['quality'][1]+1e-6:
            return None
        return dict(reference,quality=quality,sites=sites.copy(),weights=weights,
                    cells=[cell.Clone() for cell in cells],
                    max_relative_area_error=float(np.max(np.abs(areas/problem.target-1))))

    for iteration in range(rounds):
        accepted_this_round=0
        centroids=np.array([[cell.Centroid().GetX(),cell.Centroid().GetY()] for cell in current['cells']])
        directions=centroids-current['sites']
        for step in (0.8,0.4,0.2,0.1,0.05,0.025):
            candidate=trial(current['sites']+step*directions,current)
            if candidate is not None:
                current=candidate
                accepted+=1
                accepted_this_round+=1
                break
        values=np.array([4*math.pi*cell.GetArea()/cell.Boundary().Length()**2 for cell in current['cells']])
        worst=int(values.argmin())
        # 先调整离最低紧凑度单元较远的生成点，再处理其邻域。
        order=np.argsort(-np.sum((current['sites']-current['sites'][worst])**2,axis=1))
        for index in order:
            cell=current['cells'][index]
            centroid=np.array([cell.Centroid().GetX(),cell.Centroid().GetY()])
            direction=centroid-current['sites'][index]
            accepted_site=False
            if np.linalg.norm(direction)>=1e-4:
                for step in (1.0,0.5,0.25,0.125):
                    sites=current['sites'].copy()
                    sites[index]+=step*direction
                    candidate=trial(sites,current)
                    if candidate is not None:
                        current=candidate
                        accepted+=1
                        accepted_this_round+=1
                        accepted_site=True
                        break
            if not accepted_site:
                step=0.10*(0.5**iteration)
                best_candidate=None
                for direction in ((step,0),(-step,0),(0,step),(0,-step)):
                    sites=current['sites'].copy()
                    sites[index]+=direction
                    candidate=trial(sites,current)
                    if candidate is not None and (best_candidate is None or candidate['quality'][1]>best_candidate['quality'][1]):
                        best_candidate=candidate
                if best_candidate is not None:
                    current=best_candidate
                    accepted+=1
                    accepted_this_round+=1
        history.append({'round':iteration+1,'accepted':accepted_this_round,
                        'min_compactness':current['quality'][0],
                        'mean_compactness':current['quality'][1]})
        if progress:
            progress('  均值优化 %d/%d：接受 %d 步，最差/平均紧凑度 %.6f/%.6f'
                     %(iteration+1,rounds,accepted_this_round,current['quality'][0],current['quality'][1]))
        # 大步长未改善时，下一轮继续尝试更小步长，不能提前判定没有改善空间。
    metadata={'min_before':initial['quality'][0],'mean_before':initial['quality'][1],
              'fixed_min_floor':floor,'min_after':current['quality'][0],
              'mean_after':current['quality'][1],'mean_gain':current['quality'][1]-initial['quality'][1],
              'accepted':accepted,'trials':trials,'evaluations':problem.evaluations-evaluations_before,
              'seconds':time.perf_counter()-started,'history':history}
    return current,metadata


def _better_quality(candidate, incumbent):
    # A difference below 1e-4 in worst compactness is a tie; prefer the mean.
    difference=candidate[0]-incumbent[0]
    if difference>1e-4:
        return True
    return abs(difference)<=1e-4 and candidate[1]>incumbent[1]+1e-5



def representative(geometry):
    point = arcpy.PointGeometry(geometry.labelPoint, geometry.spatialReference)
    if not geometry.contains(point):
        raise ValueError("无法生成严格位于子场内部的代表点。")
    return point.projectAs(WGS84).firstPoint, "interior_label_point"


def great_circle_km(lon, lat, longitudes, latitudes):
    lon1, lat1 = np.radians([lon, lat])
    lon2, lat2 = np.radians(longitudes), np.radians(latitudes)
    hav = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))


def build_cells(basin):
    sid = basin["Sink_ID"]
    components = sorted(parts(basin["local_geometry"]),
                        key=lambda g: (-g.area, g.trueCentroid.X, g.trueCentroid.Y))
    leaves = []
    counts = component_counts(components, BASIN_MIN_SUBSITE_COUNTS.get(sid, 0))
    basin["Partition_Diagnostics"] = []
    for component_id, (component, count) in enumerate(zip(components, counts), 1):
        result, diagnostics = balanced_partition(
            component, count, seed=PARTITION_SEED + int(sid[1:]) * 101 + component_id,
            restarts=PARTITION_RESTARTS, max_outer=LLOYD_MAX_ITERATIONS,
            area_tolerance=POWER_AREA_REL_TOL, progress=progress,
            polish_rounds=SHAPE_POLISH_ROUNDS, mean_rounds=MEAN_POLISH_ROUNDS)
        basin["Partition_Diagnostics"].append(diagnostics)
        result.sort(key=lambda g: (round(g.trueCentroid.X, 3), round(g.trueCentroid.Y, 3)))
        leaves.extend((component_id, g) for g in result)
    exported = [(part_id, g, geographic(g)) for part_id, g in leaves]
    total_area = sum(area_km2(g) for _, _, g in exported)
    # 裁剪不能增加资源；通过面积异常检查后，仅截去 1 附近的浮点舍入误差。
    ratio = min(1.0, basin["Retained_Area_km2"] / basin["Original_Area_km2"])
    allocation_ratio = ratio if RESOURCE_BASIS == "uniform_original_area" else 1.0
    source = basin["source"]
    cells = []
    for index, (part_id, local, geo) in enumerate(exported, 1):
        point, method = representative(local)
        area = area_km2(geo)
        share = area / total_area
        annual_share = area / basin["Retained_Area_km2"]
        boundary = points(geo)
        radii = great_circle_km(point.X, point.Y, [p[0] for p in boundary], [p[1] for p in boundary])
        cells.append({
            "Subsite_ID": f"{sid}-S{index:03d}", "Sink_ID": sid,
            "Sink_Name": basin["Sink_Name"], "USGS_BASIN": basin["USGS_BASIN"],
            "Component_ID": part_id, "Longitude": point.X, "Latitude": point.Y,
            "Point_Method": method, "Area_km2": area, "Area_Upper_Limit_km2": MAX_AREA_KM2,
            "Compactness_4piA_P2": 4 * math.pi * local.area / local.length ** 2,
            "Max_Vertex_GreatCircle_km": float(np.max(radii)),
            "Parent_Original_Area_km2": basin["Original_Area_km2"],
            "Parent_Retained_Area_km2": basin["Retained_Area_km2"],
            "Retained_Area_Ratio": ratio, "Subsite_Area_Share": share,
            "Allocated_Total_CO2_t": float(source["Total_CO2_EWR(Gt)"]) * 1e9 * allocation_ratio * share,
            "Allocated_Annual_Quota_t": float(source["CO2_EWR(tonne)"]) * area / basin["Original_Area_km2"],
            "Annual_Allocation_Area_Share": annual_share,
            "Annual_Allocation_Basis": ANNUAL_ALLOCATION_BASIS,
            "Storage_Coefficient": float(source["Storage Coefficient"]),
            "Net_Sand_Thickness_m": float(source["Average Net Sand Thickness (meters)"]),
            "Porosity_fraction": float(source["Average Porosity (%)"]),
            "Permeability_mD": float(source["Representative Permeability (mD)"]),
            "Resource_Allocation_Basis": RESOURCE_BASIS,
            "geometry": geo, "local_geometry": local,
        })
    return cells


def component_counts(components, minimum_count=0):
    """增配平均面积最大的组件，使分离陆块也达到同盆地面积一致性要求。"""
    areas = np.asarray([area_km2(geographic(g)) for g in components])
    counts = np.maximum(1, np.ceil(areas / (MAX_AREA_KM2 - CONSTRUCTION_MARGIN_KM2))).astype(int)
    count_limit = int(sum(counts) * MAX_COMPONENT_COUNT_MULTIPLIER)
    while (sum(counts) < minimum_count or
           max(areas / counts) / min(areas / counts) > BASIN_AREA_RATIO_LIMIT / (1 + 2 * POWER_AREA_REL_TOL)):
        if sum(counts) >= count_limit:
            raise ValueError("分离组件的面积/数量要求不兼容，请核对面积比与数量增加上限。")
        counts[int(np.argmax(areas / counts))] += 1
    return counts.tolist()


def validate_basin(basin, cells):
    collection = ogr.Geometry(ogr.wkbMultiPolygon)
    for cell in cells:
        geometry = ogr.CreateGeometryFromWkb(bytes(cell["local_geometry"].WKB))
        for polygon in _polygons(geometry):
            collection.AddGeometry(polygon)
    union = collection.UnionCascaded()
    target = ogr.CreateGeometryFromWkb(bytes(basin["local_geometry"].WKB))
    missing = target.Difference(union).GetArea() / 1e6
    outside = union.Difference(target).GetArea() / 1e6
    overlap = max(0.0, collection.GetArea() - union.GetArea()) / 1e6
    maximum = max(c["Area_km2"] for c in cells)
    if maximum > MAX_AREA_KM2:
        raise ValueError(f"{basin['Sink_ID']} 面积超限：{maximum:.12f} km²")
    if max(missing, outside, overlap) > COVERAGE_TOL_KM2:
        raise ValueError(f"{basin['Sink_ID']} 覆盖/重叠异常：{missing}, {outside}, {overlap}")
    for cell in cells:
        geo = cell["geometry"]
        point = arcpy.PointGeometry(arcpy.Point(cell["Longitude"], cell["Latitude"]), WGS84)
        if geo.partCount != 1 or not geo.contains(point):
            raise ValueError(f"{cell['Subsite_ID']} 非连通或代表点不在区内")
    ratio = min(1.0, basin["Retained_Area_km2"] / basin["Original_Area_km2"])
    factor = ratio if RESOURCE_BASIS == "uniform_original_area" else 1.0
    total_resource = float(basin["source"]["Total_CO2_EWR(Gt)"]) * 1e9
    if not math.isclose(sum(c["Allocated_Total_CO2_t"] for c in cells),
                        total_resource * factor, rel_tol=1e-12, abs_tol=1e-3):
        raise ValueError("资源分配不守恒：Allocated_Total_CO2_t")
    parent_annual = float(basin["source"]["CO2_EWR(tonne)"])
    mainland_annual = parent_annual * basin["Retained_Area_km2"] / basin["Original_Area_km2"]
    for cell in cells:
        expected = parent_annual * cell["Area_km2"] / basin["Original_Area_km2"]
        if not math.isclose(cell["Allocated_Annual_Quota_t"], expected, rel_tol=1e-12, abs_tol=1e-3):
            raise ValueError("年度最大注入量分配不符合大陆交集面积比例：" + cell["Subsite_ID"])
    areas = [c["Area_km2"] for c in cells]
    compactness = [c["Compactness_4piA_P2"] for c in cells]
    if max(areas) / min(areas) > BASIN_AREA_RATIO_LIMIT:
        raise ValueError("同盆地子场面积极差比超限：" + basin["Sink_ID"])
    if abs(sum(areas) - basin["Retained_Area_km2"]) > COVERAGE_TOL_KM2:
        raise ValueError("保存边界的测地面积合计与盆地面积不一致：" + basin["Sink_ID"])
    return {"Sink_ID": basin["Sink_ID"], "Sink_Name": basin["Sink_Name"], "Subsite_Count": len(cells),
            "Original_Area_km2": basin["Original_Area_km2"],
            "Retained_Area_km2": basin["Retained_Area_km2"],
            "Retained_Area_Ratio": ratio,
            "Allocated_Total_CO2_t": sum(c["Allocated_Total_CO2_t"] for c in cells),
            "Allocated_Annual_Quota_t": sum(c["Allocated_Annual_Quota_t"] for c in cells),
            "Parent_Max_Annual_Injection_t": parent_annual,
            "Mainland_Max_Annual_Injection_t": mainland_annual,
            "Annual_Quota_Sum_Difference_t": sum(c["Allocated_Annual_Quota_t"] for c in cells) - mainland_annual,
            "Min_Area_km2": min(areas), "Max_Area_km2": maximum,
            "Area_CV": float(np.std(areas) / np.mean(areas)),
            "Area_Max_Min_Ratio": max(areas) / min(areas),
            "Compactness_Min": min(compactness), "Compactness_Mean": float(np.mean(compactness)),
            "Compactness_Median": float(np.median(compactness)),
            "Compactness_Below_0_5_Count": sum(c < 0.5 for c in compactness),
            "Partition_Diagnostics": basin["Partition_Diagnostics"],
            "Missing_Area_km2": missing, "Outside_Area_km2": outside, "Overlap_Area_km2": overlap,
            "Area_Sum_Difference_km2": sum(areas) - basin["Retained_Area_km2"]}


def export_results(cells, audit):
    fields = [key for key in cells[0] if key not in {"geometry", "local_geometry"}]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: cell[key] for key in fields} for cell in cells)
    features = [{"type": "Feature", "id": cell["Subsite_ID"],
                 "properties": {key: cell[key] for key in fields},
                 "geometry": cell["geometry"].__geo_interface__} for cell in cells]
    OUTPUT_GEOJSON.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                       ensure_ascii=False), encoding="utf-8")
    # 对保存后的边界再次读取验收，CSV 中的面积必须与这一最终几何一致。
    saved = json.loads(OUTPUT_GEOJSON.read_text(encoding="utf-8"))
    saved_geometries = []
    for feature in saved["features"]:
        geometry = arcpy.AsShape(feature["geometry"], False)
        area = area_km2(geometry)
        if area > MAX_AREA_KM2 or not math.isclose(area, feature["properties"]["Area_km2"], abs_tol=1e-5):
            raise ValueError("保存后面积验收失败：" + feature["id"])
        point = arcpy.PointGeometry(arcpy.Point(feature["properties"]["Longitude"],
                                              feature["properties"]["Latitude"]), WGS84)
        if geometry.partCount != 1 or not geometry.contains(point):
            raise ValueError("保存后连通性/代表点验收失败：" + feature["id"])
        saved_geometries.append(geometry)
    arcpy.management.CopyFeatures(saved_geometries, "memory/ss_final_cells")
    arcpy.management.CheckGeometry("memory/ss_final_cells", "memory/ss_final_check", "OGC")
    issues = int(arcpy.management.GetCount("memory/ss_final_check")[0])
    if issues:
        raise ValueError(f"输出子场存在 {issues} 项 OGC 几何问题。")
    audit["OGC_Geometry_Issue_Count"] = issues
    audit["Saved_Geometry_Area_Check"] = "passed"
    OUTPUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def layout_statistics(rows):
    areas = np.asarray([float(r["Area_km2"]) for r in rows])
    compactness = np.asarray([float(r["Compactness_4piA_P2"]) for r in rows])
    return {"Area_Min_km2": float(min(areas)), "Area_Max_km2": float(max(areas)),
            "Compactness_Mean": float(np.mean(compactness))}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not (0 < CONSTRUCTION_MARGIN_KM2 < MAX_AREA_KM2) or BASIN_AREA_RATIO_LIMIT <= 1:
        raise ValueError("面积裕量或同盆地面积极差比配置无效。")
    if RESOURCE_BASIS not in {"uniform_original_area", "already_mainland"}:
        raise ValueError("未知资源分配口径。")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sinks = read_inputs()
    progress("正在由轮廓线构建大陆面，并检查 14 个盆地边界……")
    mainland, basins, excluded = load_geography(sinks)
    cells, summaries = [], []
    for index, basin in enumerate(basins, 1):
        progress(f"[{index}/{len(basins)}] {basin['Sink_ID']} {basin['Sink_Name']}：开始等面积形状优化")
        result = build_cells(basin)
        summary = validate_basin(basin, result)
        cells.extend(result)
        summaries.append(summary)
        progress(f"  {len(result)} 个子场；面积 {summary['Min_Area_km2']:.2f}–{summary['Max_Area_km2']:.2f} km²；最小/平均紧凑度 {summary['Compactness_Min']:.3f}/{summary['Compactness_Mean']:.3f}；验收通过")
    audit = {"Method": "capacity_constrained_power_voronoi_lloyd",
             "Scope": "largest_mainland_polygon_from_outline_only",
             "Max_Area_km2": MAX_AREA_KM2, "Subsite_Count": len(cells),
             "Point_Method": "interior_label_point", "Representative_Points_Strictly_Inside": True,
             "Construction_Margin_km2": CONSTRUCTION_MARGIN_KM2,
             "Coverage_Tolerance_km2": COVERAGE_TOL_KM2,
             "Vertex_Spacing_m": VERTEX_SPACING_M,
             "Basin_Area_Ratio_Limit": BASIN_AREA_RATIO_LIMIT,
             "Power_Area_Relative_Tolerance": POWER_AREA_REL_TOL,
             "Partition_Restarts": PARTITION_RESTARTS,
             "Lloyd_Max_Iterations": LLOYD_MAX_ITERATIONS,
             "Partition_Seed": PARTITION_SEED,
             "Shape_Polish_Rounds": SHAPE_POLISH_ROUNDS,
             "Mean_Polish_Rounds": MEAN_POLISH_ROUNDS,
             "Minimum_Compactness_Slack": MIN_COMPACTNESS_SLACK,
             "Compactness_Objective": "arithmetic_mean_4piA_P2_in_each_basin_local_LAEA",
             "Basin_Min_Subsite_Counts": BASIN_MIN_SUBSITE_COUNTS,
             "Resource_Allocation_Basis": RESOURCE_BASIS,
             "Annual_Allocation_Basis": ANNUAL_ALLOCATION_BASIS,
             "Annual_Allocation_Note": "CO2_EWR(tonne)对应原盆地最大年注入量；先乘大陆交集面积/原盆地面积，再按子场面积/大陆交集面积分配，即CO2_EWR(tonne)*子场面积/原盆地面积。合计对账大陆折减量，差异仅来自已验收的边界面积数值误差。",
             "Resource_Note": "盆地平均参数和面积比例分配；不是实测子场储量或经验证的注入能力。",
             "Engineering_Note": "资源分区不是羽流；本次不估计 Max_Well_Num 或 F/V 成本。",
             "Excluded_Missing_Boundary": excluded,
             "Basins": summaries}
    export_results(cells, audit)
    progress("CSV 和真实分区边界已通过验收，正在绘制总图与逐盆地详图……")
    plot_maps(mainland, basins, cells, OUTPUT_MAP, OUTPUT_DETAIL_MAP)
    progress(f"完成：{len(basins)} 个盆地、{len(cells)} 个子场，最大面积 {max(c['Area_km2'] for c in cells):.6f} km²。")


def plot_maps(mainland, basins, cells, overview_path, detail_path):
    """Draw WGS84 ArcPy polygon inputs in a China Albers equal-area projection."""
    import os
    import collections
    import arcpy
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from matplotlib import font_manager
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Patch
    from matplotlib.lines import Line2D

    cjk_font = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyh.ttc")
    font_manager.fontManager.addfont(cjk_font)
    font_name = font_manager.FontProperties(fname=cjk_font).get_name()
    wgs84 = arcpy.SpatialReference(4326)
    albers = arcpy.SpatialReference()
    albers.loadFromString('PROJCS["China_Albers_WGS84",GEOGCS["GCS_WGS_1984",'
                         'DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],'
                         'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
                         'PROJECTION["Albers"],PARAMETER["False_Easting",0],'
                         'PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",105],'
                         'PARAMETER["Standard_Parallel_1",25],PARAMETER["Standard_Parallel_2",47],'
                         'PARAMETER["Latitude_Of_Origin",0],UNIT["Meter",1]]')
    colors = ["#4477AA", "#66A5AD", "#D9A441", "#B86878", "#79A66B", "#8876AE",
              "#C78555", "#67A9CF", "#A4A64F", "#CC879F", "#668E89", "#9891BA",
              "#BDAD7A", "#A17C66"]
    by_sink = collections.defaultdict(list)
    for cell in cells:
        by_sink[str(cell["Sink_ID"])].append(cell)
    palette = {str(b["Sink_ID"]): colors[i % len(colors)] for i, b in enumerate(basins)}

    def point_xy(longitude, latitude):
        point = arcpy.PointGeometry(arcpy.Point(float(longitude), float(latitude)), wgs84)
        point = point.projectAs(albers).firstPoint
        return point.X, point.Y

    def polygon_path(geometry):
        projected = geometry.projectAs(albers)
        vertices, codes = [], []
        for part in projected:
            rings, ring = [], []
            for point in part:
                if point is None:
                    if ring:
                        rings.append(ring)
                        ring = []
                else:
                    ring.append((point.X, point.Y))
            if ring:
                rings.append(ring)
            for ring_index, ring in enumerate(rings):
                xy = np.asarray(ring, dtype=float)
                if np.array_equal(xy[0], xy[-1]):
                    xy = xy[:-1]
                signed_area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                                     - np.roll(xy[:, 0], -1) * xy[:, 1])
                # Nonzero fill: exterior and interior rings must wind oppositely.
                if (signed_area > 0) != (ring_index == 0):
                    xy = xy[::-1]
                vertices.extend(xy.tolist() + [xy[0].tolist()])
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 1) + [Path.CLOSEPOLY])
        return Path(vertices, codes), projected

    mainland_path, mainland_projected = polygon_path(mainland)
    basin_shapes = {str(b["Sink_ID"]): polygon_path(b["geometry"]) for b in basins}
    cell_shapes = {c["Subsite_ID"]: polygon_path(c["geometry"])[0] for c in cells}
    centers = {c["Subsite_ID"]: point_xy(c["Longitude"], c["Latitude"]) for c in cells}

    def frame(ax, geometry, margin=0.045):
        e = geometry.extent
        dx, dy = e.XMax - e.XMin, e.YMax - e.YMin
        ax.set_xlim(e.XMin - dx * margin, e.XMax + dx * margin)
        ax.set_ylim(e.YMin - dy * margin, e.YMax + dy * margin)
        ax.set_aspect("equal")
        ax.set_axis_off()

    legend = [Patch(facecolor="#CFDCE5", edgecolor="#56616B", label="盆地资源分区"),
              Line2D([], [], color="#52606B", linewidth=0.7, label="子场边界"),
              Line2D([], [], color="#C74B38", marker="o", linestyle="None", markersize=5,
                     label="子场代表点")]
    area_limit_label = f"{MAX_AREA_KM2:,g}"
    note = f"每个子场面积 ≤ {area_limit_label} km²；资源分区不代表实际 CO2 羽流。"
    style = {"font.family": font_name, "axes.unicode_minus": False, "figure.facecolor": "white",
             "savefig.facecolor": "white", "font.size": 11}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(18, 15))
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.105, top=0.92)
        ax.add_patch(PathPatch(mainland_path, facecolor="#F0F1F2", edgecolor="#AAB1B7", lw=0.7))
        for b in basins:
            sid = str(b["Sink_ID"])
            ax.add_patch(PathPatch(basin_shapes[sid][0], facecolor=palette[sid],
                                   edgecolor="#4C5964", lw=0.65, alpha=0.62, zorder=2))
        for c in cells:
            ax.add_patch(PathPatch(cell_shapes[c["Subsite_ID"]], facecolor="none",
                                   edgecolor="#546473", lw=0.32, alpha=0.7, zorder=3))
        xy = np.asarray(list(centers.values()), dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c="#C74B38", edgecolors="white", linewidths=0.2, zorder=5)
        for b in basins:
            sid = str(b["Sink_ID"])
            p = basin_shapes[sid][1].labelPoint
            ax.annotate("{} · {} 个子场".format(sid, len(by_sink[sid])), (p.X, p.Y),
                        xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10,
                        color="#24333D", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.86),
                        zorder=6)
        frame(ax, mainland_projected)
        fig.suptitle("中国大陆封存盆地与子场分区", fontsize=23, color="#21323E", y=0.965)
        fig.text(0.5, 0.935, "{} 个盆地  /  {} 个子场".format(len(basins), len(cells)),
                 ha="center", fontsize=12, color="#64717A")
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.048), ncol=3, frameon=False)
        fig.text(0.5, 0.029, note, ha="center", fontsize=10, color="#64717A")
        fig.text(0.5, 0.012, "Albers 等积投影 · 中央经线 105°E · 标准纬线 25°N / 47°N", ha="center", fontsize=9, color="#7B858D")
        fig.savefig(overview_path, dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(24, 23))
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.035, top=0.915, hspace=0.22, wspace=0.14)
        for ax, b in zip(axes.flat, basins):
            sid = str(b["Sink_ID"])
            members = by_sink[sid]
            ax.add_patch(PathPatch(basin_shapes[sid][0], facecolor="#F7F8F9", edgecolor="#475965", lw=1.0))
            for c in members:
                ax.add_patch(PathPatch(cell_shapes[c["Subsite_ID"]], facecolor=palette[sid],
                                       edgecolor="#465964", alpha=0.35, lw=0.65, zorder=2))
            for c in members:
                x, y = centers[c["Subsite_ID"]]
                ax.scatter([x], [y], s=13, c="#C74B38", edgecolors="white", linewidths=0.3, zorder=5)
                label = str(c["Subsite_ID"]).rsplit("-", 1)[-1]
                text = ax.annotate(label, (x, y), xytext=(2, 2), textcoords="offset points",
                                   fontsize=6.0 if len(members) > 40 else 7.5, color="#1C2C37", zorder=6)
                text.set_path_effects([effects.withStroke(linewidth=1.8, foreground="white", alpha=0.9)])
            frame(ax, basin_shapes[sid][1], margin=0.07)
            stats = layout_statistics(members)
            ax.set_title(f"{sid}  {b['Sink_Name']} · {len(members)} 个子场\n"
                         f"{stats['Area_Min_km2']:,.0f}–{stats['Area_Max_km2']:,.0f} km² · 平均 C={stats['Compactness_Mean']:.2f}",
                         fontsize=11, fontweight="bold", color="#21323E", pad=9)
        for ax in list(axes.flat)[len(basins):]:
            ax.set_axis_off()
        axes.flat[-2].legend(handles=legend, loc="center", frameon=False, fontsize=13, labelspacing=1.5)
        axes.flat[-1].text(0.06, 0.64, f"图示说明\n\n每个子场面积 ≤ {area_limit_label} km²\n同盆地最大/最小面积 ≤ {BASIN_AREA_RATIO_LIMIT:g}\n紧凑度 C=4πA/P²，越大越紧凑\n资源分区不代表实际 CO2 羽流\n各盆地独立缩放，图幅比例尺不同", transform=axes.flat[-1].transAxes,
                          va="top", fontsize=12, color="#596975", linespacing=1.7)
        fig.suptitle("封存盆地子场分区详图", fontsize=27, color="#21323E", y=0.977)
        fig.text(0.5, 0.949, "Sxxx 为盆地内子场编号；红点为代表点", ha="center", fontsize=13, color="#64717A")
        fig.savefig(detail_path, dpi=190)
        plt.close(fig)

if __name__ == "__main__":
    main()
