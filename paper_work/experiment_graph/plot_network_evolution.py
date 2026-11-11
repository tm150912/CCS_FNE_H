"""Assemble Section 5.3's six supplied network snapshots without changing maps."""

from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "graph" / "linear_network_evolution_periods_5_10_15.pdf"
PAGE_WIDTH, PAGE_HEIGHT = 1008.0, 1141.4671
PANEL_WIDTH = 486.0
SCALE = PANEL_WIDTH / 1892
PANEL_HEIGHT = 1354 * SCALE
COLORS = {"New": "#1A7CBB", "Expanded": "#D55E00", "Existing": "#2CA02C"}


def symbols(pdf, x, y, color):
    pdf.setFillColor(HexColor(color))
    pdf.setStrokeColor(black)
    pdf.setLineWidth(0.9)
    pdf.circle(x, y, 4.2, fill=1, stroke=1)
    pdf.rect(x + 18, y - 4.2, 8.4, 8.4, fill=1, stroke=1)
    pdf.setStrokeColor(HexColor(color))
    pdf.setLineWidth(2.2)
    pdf.line(x + 36, y, x + 57, y)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    pdf.setTitle("Network evolution: Gurobi and LRH, periods 5, 10, and 15")
    for row, period in enumerate((5, 10, 15)):
        for col, method in enumerate(("gurobi", "Heuristic")):
            source = ROOT / "5.3_graph" / f"CCS_FNE_{method}_results_periods" / f"construction_cumulative_period_{period:02d}.png"
            with Image.open(source) as image:
                assert image.size == (1892, 1354), (source, image.size)
            x = 14 + col * 494
            top = PAGE_HEIGHT - 40 - row * PANEL_HEIGHT
            y = top - PANEL_HEIGHT
            pdf.drawImage(str(source), x, y, width=PANEL_WIDTH, height=PANEL_HEIGHT)
            # Only the original title and the legends in the blank southwestern
            # margin are covered. The six map images are otherwise unchanged.
            pdf.setFillColor(white)
            pdf.rect(x + 30 * SCALE, top - 100 * SCALE, 250 * SCALE, 70 * SCALE, fill=1, stroke=0)
            pdf.rect(x + 30 * SCALE, top - 1325 * SCALE, 670 * SCALE, 215 * SCALE, fill=1, stroke=0)
            pdf.setStrokeColor(HexColor("#E1E6EA"))
            pdf.setLineWidth(0.65)
            pdf.rect(x, y, PANEL_WIDTH, PANEL_HEIGHT, fill=0, stroke=1)
            pdf.setFillColor(HexColor("#24343E"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(x + 7, top - 22, f"({chr(97 + row * 2 + col)}) Period {period}")

    pdf.setFillColor(black)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawCentredString(14 + PANEL_WIDTH / 2, PAGE_HEIGHT - 25, "Gurobi")
    pdf.drawCentredString(508 + PANEL_WIDTH / 2, PAGE_HEIGHT - 25, "LRH")

    pdf.setFont("Helvetica", 18)
    legend_y = 37
    pdf.setFillColor(HexColor("#A6AEB8"))
    pdf.setStrokeColor(HexColor("#24343E"))
    pdf.setLineWidth(0.8)
    pdf.circle(266, legend_y, 4.2, fill=1, stroke=1)
    pdf.rect(446, legend_y - 4.2, 8.4, 8.4, fill=1, stroke=1)
    pdf.setStrokeColor(HexColor("#A6AEB8"))
    pdf.setLineWidth(2.2)
    pdf.line(633, legend_y, 691, legend_y)
    pdf.setFillColor(black)
    pdf.drawString(305, legend_y - 6, "Source")
    pdf.drawString(489, legend_y - 6, "Sink")
    pdf.drawString(703, legend_y - 6, "Pipe")
    for label, x in (("New", 245), ("Expanded", 430), ("Existing", 640)):
        symbols(pdf, x, 12, COLORS[label])
        pdf.setFillColor(black)
        pdf.drawString(x + 67, 6, label)
    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
