"""将 source-unit 成本表按 20 km 空间邻接关系合并为 hub-unit 成本表。

均值流程：hub 坐标直接取所属 unit 的原始经纬度均值，不做大陆边界内移。

用法：
    1. 修改下方 INPUT_FILE、OUTPUT_FILE、PLOT_FILE 和 HUB_PLOT_FILE；仅生成 CSV 时将 OUTPUT_PLOTS 设为 False。
    2. 运行 python cluster.py，生成 CSV、聚类结果图和 hub 位置图（绘图需要 matplotlib）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_KM = 6371.0088
CLUSTER_RADIUS_KM = 20

# 直接在这里修改输入、输出文件名，也可以填写完整的绝对路径。
INPUT_FILE = "source_unit_location_cost.csv"
OUTPUT_FILE = "hub_unit_location_cost.csv"
OUTPUT_PLOTS = True
PLOT_FILE = "output/source_hubs.png"
HUB_PLOT_FILE = "output/hub_locations.png"
# 边界仅用于绘图底图；文件不存在时仍绘制 source 和 hub，不修正任何坐标。
BOUNDARY_FILE = "China_boundary/中国轮廓线.shp"

INPUT_COLUMNS = [
    "source", "unit", "longitude", "latitude",
    "Operation_cost($/tCO2)", "Total_Investment_cost($)", "cap",
]
OUTPUT_COLUMNS = [
    "hub", "unit", "longitude", "latitude",
    "Operation_cost($/tCO2)", "Total_Investment_cost($)", "cap",
]
NUMERIC_COLUMNS = [
    "longitude", "latitude", "Operation_cost($/tCO2)", "Total_Investment_cost($)", "cap",
]


def read_sources(input_path: Path) -> pd.DataFrame:
    """读取并校验 source_unit_location_cost.csv 格式的数据。"""
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    data = pd.read_csv(input_path)
    missing = [column for column in INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"输入文件缺少列：{', '.join(missing)}")

    data = data[INPUT_COLUMNS].copy()
    if data.empty:
        raise ValueError("输入文件没有数据行")
    if data.isna().any().any():
        columns = data.columns[data.isna().any()].tolist()
        raise ValueError(f"以下列存在空值：{', '.join(columns)}")

    data["source"] = data["source"].astype(str).str.strip()
    data["unit"] = data["unit"].astype(str).str.strip()
    if (data[["source", "unit"]] == "").any().any():
        raise ValueError("source 或 unit 存在空字符串")
    if not data["unit"].is_unique:
        duplicates = data.loc[data["unit"].duplicated(keep=False), "unit"].unique()
        raise ValueError(f"unit 必须唯一，重复示例：{', '.join(duplicates[:5])}")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not data["longitude"].between(-180, 180).all():
        raise ValueError("longitude 必须位于 [-180, 180]")
    if not data["latitude"].between(-90, 90).all():
        raise ValueError("latitude 必须位于 [-90, 90]")

    coordinate_counts = data.groupby("source")[["longitude", "latitude"]].nunique()
    inconsistent = coordinate_counts.gt(1).any(axis=1)
    if inconsistent.any():
        examples = coordinate_counts.index[inconsistent].tolist()[:5]
        raise ValueError(f"同一 source 存在多组经纬度，示例：{', '.join(examples)}")

    return data


def cluster_sources(data: pd.DataFrame) -> pd.DataFrame:
    """按 20 km 单链接关系聚类，保留未经边界修正的 hub 均值坐标。"""
    source_points = (
        data[["source", "longitude", "latitude"]]
        .drop_duplicates(subset="source")
        .sort_values("source", kind="stable")
        .reset_index(drop=True)
    )
    coordinates_radians = np.radians(
        source_points[["latitude", "longitude"]].to_numpy(dtype=float)
    )
    labels = DBSCAN(
        eps=CLUSTER_RADIUS_KM / EARTH_RADIUS_KM,
        min_samples=1,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coordinates_radians)
    source_points["cluster"] = labels

    cluster_order = (
        source_points.groupby("cluster", sort=False)["source"]
        .min()
        .sort_values(kind="stable")
    )
    hub_by_cluster = {
        int(cluster): f"H{number}"
        for number, cluster in enumerate(cluster_order.index, start=1)
    }
    source_points["hub"] = source_points["cluster"].map(hub_by_cluster)
    source_to_hub = source_points.set_index("source")["hub"]

    output = data.copy()
    output["hub"] = output["source"].map(source_to_hub)
    output[["longitude", "latitude"]] = output.groupby("hub", sort=False)[
        ["longitude", "latitude"]
    ].transform("mean")
    return (
        output[OUTPUT_COLUMNS]
        .sort_values(["hub", "unit"], kind="stable")
        .reset_index(drop=True)
    )


def write_hubs(data: pd.DataFrame, output_path: Path) -> None:
    """写出 hub-unit CSV，保留聚类得到的原始均值坐标。"""
    if output_path.suffix.lower() != ".csv":
        raise ValueError("输出文件必须使用 .csv 扩展名")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def load_plot_boundary(boundary_file: str | Path | None = None) -> list:
    """读取仅供绘图使用的边界线段；文件不存在时返回空列表。"""
    import struct

    # 仅读取现有二维 SHP 的线段，不构造约束域或执行内移。
    boundary_path = Path(__file__).resolve().parent / (
        BOUNDARY_FILE if boundary_file is None else boundary_file
    )
    boundary_lines = []
    if boundary_path.is_file():
        with boundary_path.open("rb") as stream:
            stream.read(100)
            while record_header := stream.read(8):
                _, words = struct.unpack(">2i", record_header)
                content = stream.read(words * 2)
                shape_type = struct.unpack("<i", content[:4])[0]
                if shape_type == 0:
                    continue
                if shape_type not in (3, 5):
                    raise ValueError("绘图边界 SHP 必须为二维 PolyLine 或 Polygon")
                part_count, point_count = struct.unpack("<2i", content[36:44])
                starts = list(struct.unpack(f"<{part_count}i", content[44:44 + 4 * part_count]))
                starts.append(point_count)
                coordinates = np.frombuffer(
                    content, dtype="<f8", count=point_count * 2, offset=44 + 4 * part_count
                ).reshape(-1, 2)
                boundary_lines.extend(coordinates[a:b] for a, b in zip(starts[:-1], starts[1:]))
    return boundary_lines


def plot_clusters(
    sources: pd.DataFrame, hubs: pd.DataFrame, output_path: Path, *,
    cluster_radius_km: float | None = None, boundary_file: str | Path | None = None,
    mainland_distance_km: float | None = None,
) -> None:
    """绘制传入的 source、hub 及成员关系；两种坐标处理方式共用此绘图函数。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    points = sources[["source", "unit", "longitude", "latitude"]].merge(
        hubs[["unit", "hub"]], on="unit", validate="one_to_one"
    ).drop_duplicates("source")
    centers = hubs.drop_duplicates("hub").set_index("hub")[["longitude", "latitude"]]
    counts = points.groupby("hub").size().sort_values(ascending=False, kind="stable")
    boundary_lines = load_plot_boundary(boundary_file)
    if cluster_radius_km is None:
        cluster_radius_km = CLUSTER_RADIUS_KM
    coordinate_label = "原始坐标" if mainland_distance_km is None else "大陆边界修正"
    hub_label = "hub（unit 坐标均值）" if mainland_distance_km is None else "hub（边界修正后）"
    coordinate_note = (
        "所有坐标保持原始聚类结果。" if mainland_distance_km is None
        else f"hub 采用修正后坐标，距大陆边界 > {mainland_distance_km:g} km。"
    )

    blue, orange, ink = "#456D91", "#D35B32", "#233B49"
    with plt.rc_context({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10}):
        fig = plt.figure(figsize=(15, 10), facecolor="white")
        grid = fig.add_gridspec(2, 3, left=0.07, right=0.96, bottom=0.15, top=0.83,
                               wspace=0.5, hspace=0.5)
        overview = fig.add_subplot(grid[:, :2])
        details = [fig.add_subplot(grid[row, 2]) for row in range(2)]

        def draw_panel(ax, members, selected_hubs, title, detail=False):
            source_xy = members[["longitude", "latitude"]].to_numpy()
            hub_xy = centers.loc[members["hub"]].to_numpy()
            all_xy = np.vstack([source_xy, selected_hubs.to_numpy()])
            padding = np.maximum(np.ptp(all_xy, axis=0) * 0.1, 0.06 if detail else 1.0)
            lower, upper = all_xy.min(axis=0) - padding, all_xy.max(axis=0) + padding
            ax.add_collection(LineCollection(boundary_lines, colors="#A5B1AD",
                                             linewidths=0.55, zorder=1))
            ax.add_collection(LineCollection(np.stack([source_xy, hub_xy], axis=1),
                                             colors="#8197A7", linewidths=0.8 if detail else 0.4,
                                             alpha=0.75, zorder=2))
            ax.scatter(*source_xy.T, s=35 if detail else 10, color=blue, zorder=3)
            ax.scatter(selected_hubs.longitude, selected_hubs.latitude,
                       s=90 if detail else 22, marker="D", facecolor="white" if detail else "none",
                       edgecolor=orange, linewidths=1.4 if detail else 0.8, zorder=4)
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            ax.set_aspect(1 / max(np.cos(np.radians(all_xy[:, 1].mean())), 0.01))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if detail else 6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3 if detail else 6))
            ax.set_xlabel("经度 (°)", color=ink)
            ax.set_ylabel("纬度 (°)", color=ink)
            ax.set_title(title, loc="left", color=ink, fontsize=12, pad=12)
            ax.set_facecolor("#F7F9FA")
            ax.grid(color="#DCE4E8", linewidth=0.5)
            ax.tick_params(labelsize=9, colors="#5A6C76")
            for spine in ax.spines.values():
                spine.set_color("#CCD6DA")

        draw_panel(overview, points, centers, "全部 source 与 hub")
        for ax, hub in zip(details, counts.index[:2]):
            draw_panel(ax, points.loc[points["hub"] == hub], centers.loc[[hub]],
                       f"{hub}  |  {counts[hub]} 个 source", detail=True)
        for ax in details[min(2, len(counts)):]:
            ax.set_axis_off()
        fig.suptitle(f"Source 聚类结果 · {coordinate_label}", x=0.07, y=0.96, ha="left",
                     fontsize=24, color=ink, weight="bold")
        fig.text(0.07, 0.905,
                 f"{len(points):,} 个 source  →  {len(centers):,} 个 hub  |  "
                 f"{len(hubs):,} 个 unit  |  邻接阈值 {cluster_radius_km:g} km",
                 fontsize=12, color="#5A6C76")
        fig.legend(handles=[
            Line2D([], [], marker="o", color=blue, lw=0, label="原始 source"),
            Line2D([], [], marker="D", color=orange, markerfacecolor="white", lw=0,
                   label=hub_label),
            Line2D([], [], color="#8197A7", lw=1, label="source → hub 成员关系"),
        ], loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=3, frameon=False)
        boundary_note = "本地边界仅作底图" if boundary_lines else "未提供边界底图"
        fig.text(0.07, 0.035,
                 f"{boundary_note}；{coordinate_note}连线表示归属，"
                 f"不表示距离小于 {cluster_radius_km:g} km。",
                 fontsize=9, color="#687B85")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white")
        plt.close(fig)


def plot_hubs(
    hubs: pd.DataFrame, output_path: Path, *, boundary_file: str | Path | None = None,
    mainland_distance_km: float | None = None,
) -> None:
    """仅绘制传入结果中去重后的 hub 位置，不修改坐标。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator

    centers = hubs.drop_duplicates("hub")[["longitude", "latitude"]]
    coordinates = centers.to_numpy()
    padding = np.maximum(np.ptp(coordinates, axis=0) * 0.12, 1.0)
    lower, upper = coordinates.min(axis=0) - padding, coordinates.max(axis=0) + padding
    boundary_lines = load_plot_boundary(boundary_file)
    coordinate_label = "原始坐标" if mainland_distance_km is None else "大陆边界修正"
    coordinate_note = (
        "hub 坐标保留原始聚类均值。" if mainland_distance_km is None
        else f"hub 采用修正后坐标，距大陆边界严格超过 {mainland_distance_km:g} km。"
    )
    ink = "#233B49"

    with plt.rc_context({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10}):
        fig, ax = plt.subplots(figsize=(14, 10.5), facecolor="white")
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.85)
        ax.add_collection(LineCollection(boundary_lines, colors="#A5B1AD",
                                         linewidths=0.6, zorder=1))
        ax.scatter(centers.longitude, centers.latitude, s=20, marker="D",
                   color="#D35B32", edgecolors="white", linewidths=0.35, zorder=2)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect(1 / max(np.cos(np.radians(coordinates[:, 1].mean())), 0.01))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel("经度 (°)", color=ink)
        ax.set_ylabel("纬度 (°)", color=ink)
        ax.set_facecolor("#F7F9FA")
        ax.grid(color="#DCE4E8", linewidth=0.5)
        ax.tick_params(colors="#5A6C76")
        for spine in ax.spines.values():
            spine.set_color("#CCD6DA")
        fig.suptitle(f"Hub 位置分布 · {coordinate_label}", x=0.08, y=0.96, ha="left",
                     fontsize=24, color=ink, weight="bold")
        fig.text(0.08, 0.905, f"{len(centers):,} 个 hub  |  每个菱形代表一个 hub",
                 fontsize=12, color="#5A6C76")
        boundary_note = "本地边界仅作底图" if boundary_lines else "未提供边界底图"
        fig.text(0.08, 0.04, f"{boundary_note}；{coordinate_note}",
                 fontsize=10, color="#687B85")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white")
        plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    if input_path.resolve() == output_path.resolve():
        raise ValueError("输出文件不能覆盖输入文件")

    sources = read_sources(input_path)
    hubs = cluster_sources(sources)
    write_hubs(hubs, output_path)
    print(
        f"已生成 {output_path}："
        f"{sources['source'].nunique()} 个 source，"
        f"{hubs['hub'].nunique()} 个 hub，"
        f"{len(hubs)} 个 unit。"
    )
    if not OUTPUT_PLOTS:
        return
    plot_path = base_dir / PLOT_FILE
    plot_clusters(sources, hubs, plot_path)
    print(f"已生成聚类结果图：{plot_path}")
    hub_plot_path = base_dir / HUB_PLOT_FILE
    plot_hubs(hubs, hub_plot_path)
    print(f"已生成 hub 位置图：{hub_plot_path}")


if __name__ == "__main__":
    main()
