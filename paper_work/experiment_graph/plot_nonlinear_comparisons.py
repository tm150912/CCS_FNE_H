"""Compare logarithmic and exponential capture with common figure scales."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "graph"
CURVES = ("Logarithmic", "Exponential")
METHODS = ("GRB", "HEU")
METHOD_LABELS = {"GRB": "GRB", "HEU": "LRH"}


def plot_costs(cases):
    figure, axis = plt.subplots(figsize=(7.2, 2.9))
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.88)
    centers = np.arange(len(CURVES)) * 1.25
    x = np.asarray([center + offset for center in centers for offset in (-0.20, 0.20)])
    costs = [cases[curve]["methods"][method]["cost_billion_usd"]
             for curve in CURVES for method in METHODS]
    bottom = np.zeros(len(costs))
    for category, color, hatch in (("Capture", "white", ""),
                                    ("Storage", "#BDBDBD", ""),
                                    ("Transport", "white", "////")):
        values = np.asarray([cost[category] for cost in costs])
        axis.bar(x, values, bottom=bottom, width=0.34, facecolor=color,
                 edgecolor="black", linewidth=0.7, hatch=hatch, label=category)
        bottom += values
    totals = [cost["Total"] for cost in costs]
    np.testing.assert_allclose(bottom, totals, rtol=1e-10, atol=1e-8)
    for location, total in zip(x, totals):
        axis.annotate(f"{total:.2f}", (location, total), xytext=(0, 4),
                      textcoords="offset points", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS] * len(CURVES))
    for location, curve in zip(centers, CURVES):
        axis.text(location, -0.19, curve, transform=axis.get_xaxis_transform(),
                  ha="center", va="top", fontsize=9)
    ceiling = np.ceil(max(totals) * 1.12 / 100) * 100
    axis.set_ylim(0, ceiling)
    axis.set_yticks(np.arange(0, ceiling + 1, 200))
    axis.set_ylabel("Cost (billion USD)")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#CCCCCC", linewidth=0.35)
    axis.spines[["top", "right"]].set_visible(False)
    figure.legend(*axis.get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.54, 1.0), ncol=3, frameon=False)
    figure.savefig(OUTPUT / "nonlinear_curves_cost_comparison.pdf",
                   metadata={"Title": "Logarithmic and exponential capture: GRB and LRH costs"})
    plt.close(figure)


def plot_utilization(cases):
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 2.95), sharex=True, sharey="col")
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.83,
                           wspace=0.20, hspace=0.42)
    periods = np.arange(1, 16)
    for col, facility in enumerate(("Capture", "Storage", "Transport")):
        minimum = min(min(cases[curve]["methods"][method]["utilization_percent"]
                          [facility]["weighted"]) for curve in CURVES for method in METHODS)
        step = 2 if minimum >= 90 else (5 if minimum >= 75 else 20)
        bottom = max(0, np.floor((minimum - 0.2) / step) * step)
        if facility == "Transport":
            bottom, step = 0, 20
        for row, curve in enumerate(CURVES):
            axis = axes[row, col]
            for method, marker, linestyle, fill in (("GRB", "o", "-", "black"),
                                                    ("HEU", "s", "--", "white")):
                values = cases[curve]["methods"][method]["utilization_percent"][facility]["weighted"]
                axis.plot(periods, values, color="black", marker=marker,
                          linestyle=linestyle, markerfacecolor=fill, linewidth=1,
                          markersize=3, markeredgewidth=0.7, label=METHOD_LABELS[method])
            axis.set_title(f"({chr(97 + 3 * row + col)}) {curve}: {facility.lower()}",
                           loc="left", pad=5)
            axis.set_xticks(np.arange(1, 16, 2))
            axis.set_xlim(0.7, 15.3)
            axis.set_ylim(bottom, 100 + step * 0.3)
            axis.set_yticks(np.arange(bottom, 100.1, step))
            axis.grid(axis="y", color="#CCCCCC", linewidth=0.35)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7.5)
            if row == 1:
                axis.set_xlabel("Period", labelpad=2)
            if col == 0:
                axis.set_ylabel("Utilization (%)", labelpad=3)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.53, 1.0), ncol=2, frameon=False)
    figure.savefig(OUTPUT / "nonlinear_curves_utilization_comparison.pdf",
                   metadata={"Title": "Logarithmic and exponential capture: GRB and LRH facility utilization"})
    plt.close(figure)


def main():
    data_path = ROOT / "experiment_graph" / "three_curves_comparison_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.linewidth": 0.65, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "hatch.linewidth": 0.65})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_costs(data["cases"])
    plot_utilization(data["cases"])
    for filename in ("nonlinear_curves_cost_comparison.pdf",
                     "nonlinear_curves_utilization_comparison.pdf"):
        print(OUTPUT / filename)


if __name__ == "__main__":
    main()
