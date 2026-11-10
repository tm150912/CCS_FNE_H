"""Basin well-capacity proxies and mainland-area annual injection allocation.

CO2_EWR(tonne) is the complete original basin's maximum annual injection. First
reduce it by mainland intersection area / original basin area, then allocate
that mainland limit by each subsite's share of the intersection. Maximum well
capacity excludes peak and spare-well design margins.
Local thickness and permeability use the existing NETL engineering scenario;
these estimates are not measured well capacities.
"""

from collections import Counter
import math
import xml.etree.ElementTree as ET
import zipfile

from storage_engineering import DEFAULT_CONSTANTS, compute_engineering


NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
AREA_TOLERANCE_KM2 = 0.02
GEOLOGY_COLUMNS = {
    "Net_Sand_Thickness_m": "Average Net Sand Thickness (meters)",
    "Porosity_fraction": "Average Porosity (%)",
    "Permeability_mD": "Representative Permeability (mD)",
    "Storage_Coefficient": "Storage Coefficient",
}


def _read_sink_rows(path):
    """Read the sink sheet without Excel, macros, or optional GIS packages."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in item.findall(".//s:t", NS))
                       for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in
                ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(s for s in book.find("s:sheets", NS) if s.attrib["name"] == "sink")
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rels[rid]
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        rows = ET.fromstring(archive.read(target)).findall(".//s:sheetData/s:row", NS)
        headers, result = {}, []
        for row in rows:
            cells = {}
            for cell in row.findall("s:c", NS):
                value = cell.findtext("s:v", default="", namespaces=NS)
                if cell.attrib.get("t") == "s":
                    value = strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                cells[column] = value
            if not headers:
                headers = cells
            else:
                record = {name: cells.get(column, "") for column, name in headers.items()}
                if record.get("Sink_ID"):
                    result.append(record)
    return result


def load_basin_capacities(path, subsites):
    """Return all source basins, including basins without subsite boundaries."""
    sources = _read_sink_rows(path)
    counts = Counter(row["Sink_ID"] for row in subsites)
    result = {}
    for source in sources:
        sink_id = source["Sink_ID"]
        if sink_id in result:
            raise ValueError(f"Duplicate source basin: {sink_id}")
        geology = {key: float(source[column]) for key, column in GEOLOGY_COLUMNS.items()}
        annual = float(source["CO2_EWR(tonne)"])
        physical = compute_engineering(geology, annual)["physical"]
        children = [row for row in subsites if row["Sink_ID"] == sink_id]
        original_area = retained_area = retained_ratio = mainland_annual = None
        if children:
            original_area = float(children[0]["Parent_Original_Area_km2"])
            retained_area = float(children[0]["Parent_Retained_Area_km2"])
            if not math.isfinite(original_area) or original_area <= 0:
                raise ValueError(f"{sink_id}: invalid original basin area")
            if not math.isfinite(retained_area) or retained_area <= 0:
                raise ValueError(f"{sink_id}: invalid mainland intersection area")
            if retained_area > original_area + AREA_TOLERANCE_KM2:
                raise ValueError(f"{sink_id}: mainland intersection exceeds original basin area")
            retained_ratio = retained_area / original_area
            mainland_annual = annual * retained_ratio
            for child in children:
                child_original = float(child["Parent_Original_Area_km2"])
                child_retained = float(child["Parent_Retained_Area_km2"])
                if not math.isclose(child_original, original_area, rel_tol=0,
                                    abs_tol=AREA_TOLERANCE_KM2):
                    raise ValueError(f"{sink_id}: inconsistent original basin area")
                if not math.isclose(child_retained, retained_area, rel_tol=0,
                                    abs_tol=AREA_TOLERANCE_KM2):
                    raise ValueError(f"{sink_id}: inconsistent mainland intersection area")
                area = float(child["Area_km2"])
                if not math.isfinite(area) or area <= 0:
                    raise ValueError(f"{child['Subsite_ID']}: invalid subsite area")
                for key, value in geology.items():
                    if not math.isclose(float(child[key]), value, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(f"{child['Subsite_ID']}: {key} differs from parent basin")
            area_sum = math.fsum(float(row["Area_km2"]) for row in children)
            if abs(area_sum - retained_area) > AREA_TOLERANCE_KM2:
                raise ValueError(f"{sink_id}: subsite areas do not cover mainland intersection")
        result[sink_id] = {
            "Sink_ID": sink_id,
            "Sink_Name": source["Sink_Name"],
            "Basin_Max_Annual_Injection_t": annual,
            "Mainland_Max_Annual_Injection_t": mainland_annual,
            "Well_Annual_Capacity_t": 365.0 * physical["well_daily_t"],
            "Formation_Daily_Capacity_t": physical["formation_daily_t"],
            "Well_Daily_Capacity_t": physical["well_daily_t"],
            "Mechanical_Daily_Capacity_t": DEFAULT_CONSTANTS["mechanical_daily_t"],
            "Reservoir_Delta_Pressure_MPa": physical["reservoir_delta_mpa_proxy"],
            **geology,
            "Subsite_Count": counts[sink_id],
            "Parent_Original_Area_km2": original_area,
            "Parent_Retained_Area_km2": retained_area,
            "Retained_Area_Ratio": retained_ratio,
        }
    unknown = counts.keys() - result.keys()
    if unknown:
        raise ValueError(f"Subsites reference unknown source basins: {sorted(unknown)}")
    return result


def allocate_subsite_capacity(subsite, basin):
    """Allocate the reduced mainland limit by the subsite's mainland fraction."""
    if subsite["Sink_ID"] != basin["Sink_ID"]:
        raise ValueError("Subsite and parent basin IDs do not match")
    area = float(subsite["Area_km2"])
    retained_area = float(subsite["Parent_Retained_Area_km2"])
    annual_limit = float(basin["Mainland_Max_Annual_Injection_t"])
    well_capacity = float(basin["Well_Annual_Capacity_t"])
    if any(not math.isfinite(value) or value <= 0
           for value in (area, retained_area, annual_limit, well_capacity)):
        raise ValueError("Areas and annual capacities must be finite and positive")
    share = area / retained_area
    annual = annual_limit * share
    unrounded = annual / well_capacity
    # Do not round the ratio or add epsilon before flooring an annual limit.
    max_wells = math.floor(unrounded)
    return {
        **basin,
        "Subsite_ID": subsite["Subsite_ID"],
        "Area_km2": area,
        "Subsite_Area_Share": share,
        "Subsite_Max_Annual_Injection_t": annual,
        "Max_Well_Num": max_wells,
        "Max_Well_Num_Unrounded": unrounded,
        "Unused_Annual_Injection_t": annual - max_wells * well_capacity,
    }
