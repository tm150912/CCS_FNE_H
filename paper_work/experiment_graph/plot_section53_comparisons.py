"""Render the two Section 5.3 comparison figures from audited plot data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "graph"
METHOD_LABELS = {"GRB": "GRB", "HEU": "LRH"}


def main():
    data = json.loads((ROOT / "experiment_graph" / "linear2e6_comparison_data.json").read_text())
    methods = data["methods"]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.linewidth": 0.65, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "hatch.linewidth": 0.65})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.04, 2.1))
    figure.subplots_adjust(left=0.14, right=0.985, bottom=0.21, top=0.94)
    categories = ("Capture", "Storage", "Transport", "Total")
    x = np.arange(4)
    for offset, (label, hatch) in zip((-0.16, 0.16), (("GRB", ""), ("HEU", "//"))):
        values = [methods[label]["cost_billion_usd"][k] for k in categories]
        bars = axis.bar(x + offset, values, width=0.32, facecolor="white",
                        edgecolor="black", linewidth=0.7, hatch=hatch, label=METHOD_LABELS[label])
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3,
                       fontsize=7.3)
    axis.set_xticks(x, categories)
    axis.set_ylabel("Cost (billion USD)")
    axis.set_ylim(0, 560)
    axis.set_yticks(np.arange(0, 501, 100))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#CCCCCC", linewidth=0.35)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", bbox_to_anchor=(0.25, 0.98), ncol=2, frameon=False)
    figure.savefig(OUTPUT / "linear_cost_comparison.pdf",
                   metadata={"Title": "Linear capture: GRB and LRH costs"})
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 2.67))
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.81,
                           wspace=0.20, hspace=0.42)
    periods = np.arange(1, 16)
    for row, metric in enumerate(("weighted", "mean")):
        for col, facility in enumerate(("Capture", "Storage", "Transport")):
            axis = axes[row, col]
            for label, marker, linestyle, fill in (("GRB", "o", "-", "black"),
                                                   ("HEU", "s", "--", "white")):
                values = methods[label]["utilization_percent"][facility][metric]
                axis.plot(periods, values, color="black", marker=marker,
                          linestyle=linestyle, markerfacecolor=fill,
                          linewidth=1, markersize=3, markeredgewidth=0.7, label=METHOD_LABELS[label])
            measure = "capacity-weighted" if row == 0 else ("per unit", "per subsite", "per pipe")[col]
            axis.set_title(f"({chr(97 + 3 * row + col)}) {facility}: {measure}", loc="left", pad=5)
            axis.set_xticks(np.arange(1, 16, 2))
            axis.set_xlim(0.7, 15.3)
            if facility == "Transport":
                axis.set_ylim(0, 100)
                axis.set_yticks(np.arange(0, 101, 20))
            elif facility == "Capture":
                axis.set_ylim(98, 100.5)
                axis.set_yticks(np.arange(98, 101, 1))
            else:
                axis.set_ylim(40, 103)
                axis.set_yticks((40, 60, 80, 100))
            axis.grid(axis="y", color="#CCCCCC", linewidth=0.35)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7.5)
            if row == 1:
                axis.set_xlabel("Period", labelpad=2)
            else:
                axis.tick_params(labelbottom=False)
            if col == 0:
                axis.set_ylabel("Utilization (%)", labelpad=3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.51, 1.0))
    figure.savefig(OUTPUT / "linear_utilization_comparison.pdf",
                   metadata={"Title": "Linear capture: GRB and LRH facility utilization"})
    plt.close(figure)


if __name__ == "__main__":
    main()
