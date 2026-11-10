"""Generate pipeline construction and operating costs for every network arc.

The script reads ``output/arcs.csv`` and ``output/nodes.csv`` relative to this
file's directory by default. A sink-subsite node is identified by the project's ``P`` node-ID
prefix (for example, ``P1-S01``). An arc receives the
additional reservoir-connection distance only when its end node is a sink
subsite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas import DataFrame


BASE_DIR = Path(__file__).resolve().parent
ARCS_FILE = BASE_DIR / "output/arcs.csv"
NODES_FILE = BASE_DIR / "output/nodes.csv"
OUTPUT_FILE = BASE_DIR / "pipeline_cost.csv"

DISCOUNT_RATE = 0.08
PIPELINE_LIFETIME_YEARS = 30
CRF = (DISCOUNT_RATE * (1 + DISCOUNT_RATE) ** PIPELINE_LIFETIME_YEARS
    / ((1 + DISCOUNT_RATE) ** PIPELINE_LIFETIME_YEARS - 1))

PIPE_TYPES = (
    "p4", "p6", "p10", "p12", "p16", "p20",
    "p24a", "p24b", "p30a", "p30b", "p36", "p48")

# All cost calculations use annual-average CO2 throughput.
# These kg/s values are converted from Cases!F56:F67 (Mt/year) in
# CO2_T_COM_ver4_final_20230726.xlsm and already include its 90% capacity factor.
FLOW_RATES_KG_PER_S = (
    2.853881279,
    28.53881279,
    71.34703196,
    142.6940639,
    214.0410959,
    285.3881279,
    428.0821918,
    570.7762557,
    856.1643836,
    1141.552511,
    1426.940639,
    2853.881279,
)

OUTPUT_COLUMNS = (
    "node1_ID",
    "node2_ID",
    "CO2 flow rate(kg/s)",
    "Operation_cost($/tCO2)",
    "Investment_cost($)",
)

# 成本列名称，用于统一处理
COST_COLUMNS = [
    "Operation_cost($/tCO2)",
    "Investment_cost($)",
]


def transportation_cost(
    distance_km: float,
    flow_rate_kg_per_s: float,
    to_reservoir: bool,
) -> tuple[float, float]:
    """Return annualized investment and operating costs for one pipeline."""

    if distance_km <= 0:
        raise ValueError("Pipeline distance must be positive")
    if flow_rate_kg_per_s <= 0:
        raise ValueError("Pipeline flow rate must be positive")

    dollar_exchange_rate = 17.7263/14.7966 * 21.06/19.38
    # dollar_exchange_rate = 17.7263 / 14.7966 * 21.06 / 19.38 * 6.8976
    regional_factor = 0.9
    terrain_factor = 1.1
    cost_parameter = 9970
    operation_and_maintenance_factor = 0.025
    routing_factor = 0.12

    routed_distance_km = distance_km * (1 + routing_factor)
    if to_reservoir:
        routed_distance_km += 15 * 1.609344

    # Convert annual-average flow from kg/s to tonnes/day for investment costs.
    flow_rate_tonne_per_day = flow_rate_kg_per_s * 3600 * 24 / 1000
    investment_cost_per_km = (
        cost_parameter
        * flow_rate_tonne_per_day**0.35
        * routed_distance_km**0.13
        * dollar_exchange_rate
    )
    investment_cost = (
        routed_distance_km
        * regional_factor
        * terrain_factor
        * investment_cost_per_km
    )
    annualized_investment_cost = investment_cost * CRF
    # Divide annual O&M costs by annual-average throughput in tonnes/year.
    operation_cost = (
        investment_cost
        * operation_and_maintenance_factor
        / (flow_rate_tonne_per_day * 365)
    )

    return annualized_investment_cost, operation_cost


def _read_network(
    arcs_file: Path,
    nodes_file: Path) -> tuple[DataFrame, set[str]]:
    """Read and validate arcs, returning arcs and known sink-subsite IDs."""

    if not arcs_file.is_file():
        raise FileNotFoundError(f"Arc file does not exist: {arcs_file}")
    if not nodes_file.is_file():
        raise FileNotFoundError(f"Node file does not exist: {nodes_file}")

    arcs = pd.read_csv(arcs_file, dtype={"i": "string", "j": "string"})
    missing_arc_columns = [
        column for column in ("i", "j", "distance") if column not in arcs.columns
    ]
    if missing_arc_columns:
        raise ValueError(
            f"{arcs_file.name} is missing columns: {missing_arc_columns}"
        )
    arcs = arcs.loc[:, ["i", "j", "distance"]].copy()
    arcs["i"] = arcs["i"].str.strip()
    arcs["j"] = arcs["j"].str.strip()
    arcs["distance"] = pd.to_numeric(arcs["distance"], errors="coerce")
    if arcs.empty:
        raise ValueError(f"{arcs_file.name} contains no arcs")
    if arcs.isna().any().any() or (arcs["i"] == "").any() or (arcs["j"] == "").any():
        raise ValueError(f"{arcs_file.name} contains missing or invalid arc data")
    if (arcs["distance"] <= 0).any():
        raise ValueError(f"{arcs_file.name} contains non-positive distances")
    duplicate_arcs = arcs.duplicated(["i", "j"], keep=False)
    if duplicate_arcs.any():
        examples = arcs.loc[duplicate_arcs, ["i", "j"]].head(5).values.tolist()
        raise ValueError(f"{arcs_file.name} contains duplicate directed arcs: {examples}")

    nodes = pd.read_csv(nodes_file, dtype="string")
    if nodes.empty or len(nodes.columns) == 0:
        raise ValueError(f"{nodes_file.name} contains no nodes")
    node_id_column = nodes.columns[0]
    node_ids = nodes[node_id_column].str.strip()
    if node_ids.isna().any() or (node_ids == "").any():
        raise ValueError(f"{nodes_file.name} contains missing node IDs")
    if node_ids.duplicated().any():
        examples = node_ids[node_ids.duplicated(keep=False)].head(5).tolist()
        raise ValueError(f"{nodes_file.name} contains duplicate node IDs: {examples}")

    node_id_set = set(node_ids)
    arc_node_ids = set(arcs["i"]) | set(arcs["j"])
    unknown_node_ids = sorted(arc_node_ids - node_id_set)
    if unknown_node_ids:
        raise ValueError(
            f"{arcs_file.name} references nodes absent from {nodes_file.name}: "
            f"{unknown_node_ids[:10]}"
        )

    sink_subsite_ids = {
        node_id for node_id in node_id_set if node_id.upper().startswith("P")
    }
    if not sink_subsite_ids:
        raise ValueError(
            f"{nodes_file.name} contains no sink-subsite IDs beginning with P"
        )

    return arcs, sink_subsite_ids


def build_pipeline_cost(
    arcs_file: Path = ARCS_FILE,
    nodes_file: Path = NODES_FILE,
) -> DataFrame:
    """Calculate every pipe-type cost for every directed arc."""

    arcs, sink_subsite_ids = _read_network(Path(arcs_file), Path(nodes_file))
    if len(PIPE_TYPES) != len(FLOW_RATES_KG_PER_S):
        raise ValueError("PIPE_TYPES and FLOW_RATES_KG_PER_S must have equal lengths")
    rows: list[tuple[str, str, str, float, float]] = []

    for node1, node2, distance in arcs.itertuples(index=False, name=None):
        to_reservoir = node2 in sink_subsite_ids
        for pipe_type, flow_rate in zip(PIPE_TYPES, FLOW_RATES_KG_PER_S):
            annualized_cost, operation_cost = transportation_cost(
                float(distance), flow_rate, to_reservoir
            )
            rows.append(
                (
                    node1,
                    node2,
                    pipe_type,
                    operation_cost,
                    annualized_cost / CRF,
                )
            )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # 将所有成本列保留两位小数
    df[COST_COLUMNS] = df[COST_COLUMNS].round(2)
    return df


def generate_pipeline_cost_csv(
    arcs_file: Path = ARCS_FILE,
    nodes_file: Path = NODES_FILE,
    output_file: Path = OUTPUT_FILE,
) -> DataFrame:
    """Calculate the cost table, write it as CSV, and return it."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result = build_pipeline_cost(Path(arcs_file), Path(nodes_file))
    # 使用 %.2f 确保 CSV 中数值文本保留两位小数
    result.to_csv(output_file, index=False, float_format="%.2f")
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate all candidate pipeline costs for every directed arc."
    )
    parser.add_argument("--arcs", type=Path, default=ARCS_FILE)
    parser.add_argument("--nodes", type=Path, default=NODES_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    result = generate_pipeline_cost_csv(
        arcs_file=arguments.arcs,
        nodes_file=arguments.nodes,
        output_file=arguments.output,
    )
    arc_count = len(result) // len(PIPE_TYPES)
    sink_arc_count = result.loc[
        result["node2_ID"].str.upper().str.startswith("P"),
        ["node1_ID", "node2_ID"],
    ].drop_duplicates().shape[0]
    print(
        f"Wrote {len(result):,} rows for {arc_count:,} arcs "
        f"({sink_arc_count:,} arcs ending at sink subsites) to {arguments.output}"
    )


if __name__ == "__main__":
    main()
