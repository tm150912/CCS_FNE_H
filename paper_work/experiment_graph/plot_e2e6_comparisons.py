"""Plot the supplied E/L_2e6 plans using the Section 5.3 figure style."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas

from plot_network_evolution import (
    COLORS, PAGE_HEIGHT, PAGE_WIDTH, PANEL_HEIGHT, PANEL_WIDTH, SCALE, symbols,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "graph"
METHODS = ("GRB", "HEU")
METHOD_LABELS = {"GRB": "GRB", "HEU": "LRH"}


def plot_costs(methods, prefix):
    figure, axis = plt.subplots(figsize=(5.04, 2.1))
    figure.subplots_adjust(left=0.14, right=0.985, bottom=0.21, top=0.94)
    categories = ("Capture", "Storage", "Transport", "Total")
    x = np.arange(len(categories))
    maximum = 0
    for offset, label, hatch in ((-0.16, "GRB", ""), (0.16, "HEU", "//")):
        values = [methods[label]["cost_billion_usd"][k] for k in categories]
        maximum = max(maximum, max(values))
        bars = axis.bar(x + offset, values, width=0.32, facecolor="white",
                        edgecolor="black", linewidth=0.7, hatch=hatch, label=METHOD_LABELS[label])
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                       padding=3, fontsize=7.3)
    axis.set_xticks(x, categories)
    axis.set_ylabel("Cost (billion USD)")
    ceiling = np.ceil(maximum * 1.15 / 50) * 50
    axis.set_ylim(0, ceiling)
    axis.set_yticks(np.arange(0, ceiling + 1, 100 if ceiling > 300 else 50))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#CCCCCC", linewidth=0.35)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", bbox_to_anchor=(0.25, 0.98), ncol=2, frameon=False)
    figure.savefig(OUTPUT / f"{prefix}_cost_comparison.pdf",
                   metadata={"Title": f"{prefix}: GRB and LRH costs"})
    plt.close(figure)


def plot_utilization(methods, prefix):
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 2.67))
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.81,
                           wspace=0.20, hspace=0.42)
    periods = np.arange(1, 16)
    for col, facility in enumerate(("Capture", "Storage", "Transport")):
        minimum = min(min(methods[label]["utilization_percent"][facility][metric])
                      for label in METHODS for metric in ("weighted", "mean"))
        step = 2 if minimum >= 90 else (5 if minimum >= 75 else 20)
        bottom = max(0, np.floor(minimum / step) * step - step)
        if facility == "Transport":
            bottom, step = 0, 20
        for row, metric in enumerate(("weighted", "mean")):
            axis = axes[row, col]
            for label, marker, linestyle, fill in (("GRB", "o", "-", "black"),
                                                   ("HEU", "s", "--", "white")):
                values = methods[label]["utilization_percent"][facility][metric]
                axis.plot(periods, values, color="black", marker=marker,
                          linestyle=linestyle, markerfacecolor=fill,
                          linewidth=1, markersize=3, markeredgewidth=0.7, label=METHOD_LABELS[label])
            measure = "capacity-weighted" if row == 0 else (
                "per unit", "per storage site", "per pipe")[col]
            axis.set_title(f"({chr(97 + 3 * row + col)}) {facility}: {measure}",
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
            else:
                axis.tick_params(labelbottom=False)
            if col == 0:
                axis.set_ylabel("Utilization (%)", labelpad=3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.51, 1.0))
    figure.savefig(OUTPUT / f"{prefix}_utilization_comparison.pdf",
                   metadata={"Title": f"{prefix}: GRB and LRH facility utilization"})
    plt.close(figure)


def plot_network(case, curve):
    path = OUTPUT / f"{case.lower()}2e6_network_evolution_periods_5_10_15.pdf"
    source_suffix = "_2e6" if case == "linear" else f"_{case}_2e6"
    pdf = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    pdf.setTitle(f"{curve} capture: GRB and LRH, periods 5, 10, and 15")
    for row, period in enumerate((5, 10, 15)):
        for col, method in enumerate(("gurobi", "Heuristic")):
            source = (ROOT / "experiment_graph" / f"CCS_FNE_{method}{source_suffix}"
                      / f"construction_cumulative_period_{period:02d}.png")
            with Image.open(source) as image:
                assert image.size == (1892, 1354), (source, image.size)
            x = 14 + col * 494
            top = PAGE_HEIGHT - 40 - row * PANEL_HEIGHT
            y = top - PANEL_HEIGHT
            pdf.drawImage(str(source), x, y, width=PANEL_WIDTH, height=PANEL_HEIGHT)
            # Cover only the old title and the legends in the blank southwest
            # margin, as in Section 5.3. The map and network remain unchanged.
            pdf.setFillColor(white)
            pdf.rect(x + 30 * SCALE, top - 100 * SCALE,
                     250 * SCALE, 70 * SCALE, fill=1, stroke=0)
            pdf.rect(x + 30 * SCALE, top - 1325 * SCALE,
                     670 * SCALE, 215 * SCALE, fill=1, stroke=0)
            pdf.setStrokeColor(HexColor("#E1E6EA"))
            pdf.setLineWidth(0.65)
            pdf.rect(x, y, PANEL_WIDTH, PANEL_HEIGHT, fill=0, stroke=1)
            pdf.setFillColor(HexColor("#24343E"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(x + 7, top - 22, f"({chr(97 + row * 2 + col)}) Period {period}")
    pdf.setFillColor(black)
    pdf.setFont("Helvetica-Bold", 22)
    for col, label in enumerate(METHODS):
        pdf.drawCentredString(14 + col * 494 + PANEL_WIDTH / 2,
                            PAGE_HEIGHT - 25, METHOD_LABELS[label])
    pdf.setFont("Helvetica", 18)
    pdf.setFillColor(HexColor("#A6AEB8"))
    pdf.setStrokeColor(HexColor("#24343E"))
    pdf.setLineWidth(0.8)
    pdf.circle(266, 37, 4.2, fill=1, stroke=1)
    pdf.rect(446, 32.8, 8.4, 8.4, fill=1, stroke=1)
    pdf.setStrokeColor(HexColor("#A6AEB8"))
    pdf.setLineWidth(2.2)
    pdf.line(633, 37, 691, 37)
    pdf.setFillColor(black)
    for x, label in ((305, "Source"), (489, "Sink"), (703, "Pipe")):
        pdf.drawString(x, 31, label)
    for label, x in (("New", 245), ("Expanded", 430), ("Existing", 640)):
        symbols(pdf, x, 12, COLORS[label])
        pdf.setFillColor(black)
        pdf.drawString(x + 67, 6, label)
    pdf.showPage()
    pdf.save()


def main(case="E", curve="Exponential"):
    prefix = f"{case.lower()}2e6"
    data_path = ROOT / "experiment_graph" / f"{prefix}_comparison_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.linewidth": 0.65, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "hatch.linewidth": 0.65})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_costs(data["methods"], prefix)
    plot_utilization(data["methods"], prefix)
    plot_network(case, curve)
    for path in OUTPUT.glob(f"{prefix}_*.pdf"):
        print(path)


if __name__ == "__main__":
    main()
