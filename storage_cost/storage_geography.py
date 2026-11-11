"""Assign local electricity snapshots to storage subsites in constant 2020 USD.

Province boundaries use GCJ-02, whereas subsite representative points use WGS84.
The approximate coordinate conversion is suitable for this province-level join,
not a surveyed boundary or confirmation of the site's electricity service area.
Energy and monthly demand rates are converted separately. Downstream demand
charges assume fixed contract demand equal to the design peak pump power.
"""

import csv
import json
import math
from collections import defaultdict


PROVINCE_GRID = {
    "北京市": "Beijing", "天津市": "Tianjin", "河北省": "Hebei",
    "山西省": "Shanxi", "内蒙古自治区": "InnerMongoliaWest",
    "辽宁省": "Liaoning", "吉林省": "Jilin", "黑龙江省": "Heilongjiang",
    "江苏省": "Jiangsu", "安徽省": "Anhui", "山东省": "Shandong", "河南省": "Henan",
    "湖北省": "Hubei", "重庆市": "Chongqing", "四川省": "Sichuan",
    "贵州省": "Guizhou", "陕西省": "Shaanxi", "甘肃省": "Gansu",
    "青海省": "Qinghai", "宁夏回族自治区": "Ningxia",
    "新疆维吾尔自治区": "Xinjiang",
}
PERIODS = ("Sharp", "Peak", "Flat", "Valley", "Deep")
REGION_FLAGS = {
    "Anhui": "industrial_two_part_superpeak_eligibility_assumed",
    "InnerMongoliaWest": "west_tariff_province_proxy_east_tariff_missing",
    "Shaanxi": "non_yulin_tariff_province_proxy_city_unverified",
    "Hebei": "hebei_tariff_province_proxy_jibei_not_distinguished",
    "Sichuan": "state_grid_selected_local_grid_not_verified",
    "Shanxi": "state_grid_selected_local_grid_not_verified",
    "Xinjiang": "regional_government_fee_proxy_prefecture_unverified",
}


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _wgs84_to_gcj02(lon, lat):
    x, y = lon - 105.0, lat - 35.0
    dlat = -100 + 2*x + 3*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
    dlon = 300 + x + 2*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))
    shared = (20*math.sin(6*x*math.pi) + 20*math.sin(2*x*math.pi))*2/3
    dlat += shared + (20*math.sin(y*math.pi) + 40*math.sin(y*math.pi/3))*2/3
    dlat += (160*math.sin(y*math.pi/12) + 320*math.sin(y*math.pi/30))*2/3
    dlon += shared + (20*math.sin(x*math.pi) + 40*math.sin(x*math.pi/3))*2/3
    dlon += (150*math.sin(x*math.pi/12) + 300*math.sin(x*math.pi/30))*2/3
    a, eccentricity = 6378245.0, 0.00669342162296594323
    radians = math.radians(lat)
    magic = 1 - eccentricity*math.sin(radians)**2
    root_magic = math.sqrt(magic)
    dlat *= 180 / (a*(1-eccentricity)/(magic*root_magic)*math.pi)
    dlon *= 180 / (a/root_magic*math.cos(radians)*math.pi)
    return lon + dlon, lat + dlat


def _in_ring(x, y, ring):
    """Ray crossing, treating exact boundary points as belonging to the ring."""
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
        if abs(cross) < 1e-12 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1) + x1:
            inside = not inside
    return inside


def _in_geometry(x, y, geometry):
    polygons = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        polygons = [polygons]
    elif geometry["type"] != "MultiPolygon":
        raise ValueError("Province boundary must be a Polygon or MultiPolygon")
    return any(_in_ring(x, y, p[0]) and not any(_in_ring(x, y, hole) for hole in p[1:])
               for p in polygons)


def build_tariff_assignments(subsites, reference_path, tariffs_path):
    """Return one CSV-ready dict per Subsite_ID; reject ambiguous/missing matches.

    Jilin explicitly uses the equal mean of its February and December 2025
    snapshots. All other mapped grids require exactly one input record. Every
    snapshot is converted using its own year's FX and CPI before averaging.
    Returned energy and demand rates are snapshot proxies, not annual tariffs
    or verified grid contracts. Downstream demand uses design peak pump kW.
    """
    provinces, scalars = [], {}
    for row in _read_csv(reference_path):
        kind, key = row["Source_ID"], row["Record_Key"]
        if kind == "GEOGRAPHY_SOURCE" and key.isdigit() and len(key) == 6:
            feature = json.loads(row["Feature_JSON"])
            provinces.append((key, row["Name"], feature["geometry"]))
        elif kind in ("FX_SOURCE", "CPI_SOURCE"):
            scalar_key = (kind, int(key))
            if scalar_key in scalars:
                raise ValueError(f"Duplicate FX/CPI reference: {scalar_key}")
            value = float(row["Value"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid FX/CPI reference: {scalar_key}")
            if kind == "FX_SOURCE" and row["Unit"] != "CNY/USD":
                raise ValueError("Exchange rates must be CNY/USD")
            scalars[scalar_key] = value

    grouped = defaultdict(list)
    for row in _read_csv(tariffs_path):
        grouped[row["Grid_Region"]].append(row)

    rates = {}
    results, seen = [], set()
    for subsite in subsites:
        sid = subsite["Subsite_ID"]
        if sid in seen:
            raise ValueError(f"Duplicate Subsite_ID: {sid}")
        seen.add(sid)
        point = _wgs84_to_gcj02(float(subsite["Longitude"]), float(subsite["Latitude"]))
        matches = [(code, name) for code, name, geometry in provinces if _in_geometry(*point, geometry)]
        if len(matches) != 1:
            raise ValueError(f"{sid}: expected one province, found {matches}")
        adcode, province = matches[0]
        if province not in PROVINCE_GRID:
            raise ValueError(f"{sid}: no explicit grid rule for {province}")
        region = PROVINCE_GRID[province]
        if region not in rates:
            rows = sorted(grouped[region], key=lambda r: (int(r["Year"]), int(r["Month"])))
            dates = [(int(r["Year"]), int(r["Month"])) for r in rows]
            if (region == "Jilin" and dates != [(2025, 2), (2025, 12)]) or (region != "Jilin" and len(rows) != 1):
                raise ValueError(f"{region}: unexpected tariff snapshots {dates}; explicit selection required")
            cny_rates, usd_rates = [], []
            demand_cny_rates, demand_usd_rates = [], []
            flags = ["monthly_snapshot_not_annual_mean", "uniform_load_assumed",
                     "connection_voltage_and_service_grid_unverified",
                     "contract_demand_equals_design_peak_pump_kw_assumed"]
            if region in REGION_FLAGS:
                flags.append(REGION_FLAGS[region])
            if region == "Jilin":
                flags.append("two_snapshots_equal_mean_not_annual_mean")
            for row in rows:
                try:
                    demand_cny = float(row["Demand_CNY_per_kW_month"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{region}: Demand_CNY_per_kW_month must be present and numeric") from error
                if not math.isfinite(demand_cny) or demand_cny < 0:
                    raise ValueError(
                        f"{region}: Demand_CNY_per_kW_month must be finite and nonnegative")
                hours = [float(row[p + "_Hours"]) for p in PERIODS]
                prices = [float(row[p + "_CNY_per_kWh"]) for p in PERIODS]
                if any(not math.isfinite(v) or v < 0 for v in hours + prices) or not math.isclose(sum(hours), 24.0):
                    raise ValueError(f"{region}: invalid TOU hours or prices")
                method = row["Weighting_Method"]
                if method not in ("time_of_use", "flat_period_proxy"):
                    raise ValueError(f"{region}: unknown weighting method {method}")
                if method == "flat_period_proxy":
                    if hours != [0.0, 0.0, 24.0, 0.0, 0.0]:
                        raise ValueError(f"{region}: flat-period proxy must use 24 flat hours")
                    flags.append("flat_period_proxy")
                year = int(row["Year"])
                if year == 2024:
                    flags.append("2024_snapshot_with_2024_fx_and_cpi")
                cny = sum(h*p for h, p in zip(hours, prices))/24.0
                usd = cny/scalars[("FX_SOURCE", year)]*scalars[("CPI_SOURCE", 2020)]/scalars[("CPI_SOURCE", year)]
                cny_rates.append(cny)
                usd_rates.append(usd)
                demand_cny_rates.append(demand_cny)
                demand_usd_rates.append(
                    demand_cny/scalars[("FX_SOURCE", year)]*
                    scalars[("CPI_SOURCE", 2020)]/scalars[("CPI_SOURCE", year)])
            rates[region] = {
                "Grid_Region": region,
                "Tariff_Year_Months": ";".join(f"{year:04d}-{month:02d}" for year, month in dates),
                "Tariff_Voltage_kV": ";".join(dict.fromkeys(r["Voltage_kV"] for r in rows)),
                "Tariff_Weighting_Method": ";".join(dict.fromkeys(r["Weighting_Method"] for r in rows)),
                "Electricity_CNY_per_kWh": sum(cny_rates)/len(cny_rates),
                "Electricity_USD2020_per_kWh": sum(usd_rates)/len(usd_rates),
                "Demand_CNY_per_kW_month": sum(demand_cny_rates)/len(demand_cny_rates),
                "Demand_USD2020_per_kW_month": sum(demand_usd_rates)/len(demand_usd_rates),
                "Tariff_Source_URLs": ";".join(dict.fromkeys(r["Source_URL"] for r in rows)),
                "Tariff_Evidence_Status": ";".join(dict.fromkeys(r["Evidence_Status"] for r in rows)),
                "Tariff_Assumption_Flags": ";".join(dict.fromkeys(flags)),
            }
        results.append({"Subsite_ID": sid, "Province": province, "Province_Adcode": adcode, **rates[region]})
    return results
