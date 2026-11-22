"""Draw Gambar 1 (alur penelitian) and Gambar 2 (skema leave-one-source-out) for the proposal.

Usage: python paper/proposal/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = Path(__file__).parent / "figures"
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]

BLUE, GREEN, ORANGE, GREY = "#CFE3F5", "#D5ECD4", "#F9D9B5", "#EDEDED"
EDGE = "#333333"


def box(ax, x, y, w, h, title, lines=(), fill=GREY, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=size, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.52 - i * 0.28, line, ha="center", va="top", fontsize=size - 1.5)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.0, color=EDGE))


def figure1():
    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=220)
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    w, h, y = 2.4, 2.5, 4.0
    xs = [0.05 + i * 2.7 for i in range(5)]
    box(ax, xs[0], y, w, h, "Data", ["Darwin 600", "Montgomery 138", "Shenzhen 566"], BLUE, size=8.5)
    box(ax, xs[1], y, w, h, "Pra-proses", ["grayscale 256x256", "mask biner", "augmentasi geometri"], size=8.5)
    box(ax, xs[2], y, w, h, "Skenario kontras", ["S0 tanpa", "S1 HE global", "S2 CLAHE bawaan", "S3 CLAHE tuning"], ORANGE, size=8.5)
    box(ax, xs[3], y, w, h, "Pelatihan", ["U-Net, 3 seed", "tiap lipatan LOSO", "36 pelatihan"], size=8.5)
    box(ax, xs[4], y, w, h, "Pasca-proses", ["ambang 0,5", "2 komponen terbesar", "closing, isi lubang"], size=8.5)
    for i in range(4):
        arrow(ax, xs[i] + w + 0.04, y + h / 2, xs[i + 1] - 0.04, y + h / 2)
    y2, h2 = 0.2, 2.0
    bw = 4.1
    bx = [0.05, 4.65, 9.25]
    box(ax, bx[0], y2, bw, h2, "Metrik per citra", ["Dice, IoU, HD95, cakupan paru", "galat luas, tingkat gagal"], GREEN, size=8.5)
    box(ax, bx[1], y2, bw, h2, "Alarm tanpa label", ["jarak intensitas (A1), entropi (A2)", "ambang dari validasi sumber latih"], GREEN, size=8.5)
    box(ax, bx[2], y2, bw, h2, "Keputusan dan uji", ["aturan kesiapan: siap / belum siap", "Wilcoxon + Holm, bootstrap 95%"], GREEN, size=8.5)
    arrow(ax, xs[4] + w / 2, y - 0.04, bx[0] + bw / 2 + 1.0, y2 + h2 + 0.04)
    arrow(ax, bx[0] + bw + 0.04, y2 + h2 / 2, bx[1] - 0.04, y2 + h2 / 2)
    arrow(ax, bx[1] + bw + 0.04, y2 + h2 / 2, bx[2] - 0.04, y2 + h2 / 2)
    fig.savefig(OUT / "gambar1_alur.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure2():
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=220)
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    sources = ["Darwin (600)", "Montgomery (138)", "Shenzhen (566)"]
    for c, s in enumerate(sources):
        ax.text(3.8 + c * 3.1 + 1.35, 5.95, s, ha="center", fontsize=9, fontweight="bold")
    for r, fold in enumerate(sources):
        y = 3.9 - r * 1.75
        ax.text(0.1, y + 0.55, f"Lipatan {r + 1}\nuji asing:\n{fold.split(' ')[0]}", fontsize=8.5, va="center")
        for c in range(3):
            x = 3.8 + c * 3.1
            if c == r:
                ax.add_patch(FancyBboxPatch((x, y), 2.7, 1.15, boxstyle="round,pad=0.02,rounding_size=0.06", fc=ORANGE, ec=EDGE))
                ax.text(x + 1.35, y + 0.575, "uji sumber asing\n(seluruhnya)", ha="center", va="center", fontsize=8.5)
            else:
                ax.add_patch(FancyBboxPatch((x, y), 2.7, 1.15, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=EDGE))
                ax.add_patch(FancyBboxPatch((x, y), 2.7 * 0.8, 1.15, boxstyle="round,pad=0.0,rounding_size=0.0", fc=BLUE, ec=EDGE))
                ax.add_patch(FancyBboxPatch((x + 2.7 * 0.8, y), 2.7 * 0.1, 1.15, boxstyle="round,pad=0.0,rounding_size=0.0", fc=GREEN, ec=EDGE))
                ax.add_patch(FancyBboxPatch((x + 2.7 * 0.9, y), 2.7 * 0.1, 1.15, boxstyle="round,pad=0.0,rounding_size=0.0", fc="#E8D9F2", ec=EDGE))
                ax.text(x + 1.08, y + 0.575, "latih 80%", ha="center", va="center", fontsize=8)
    ax.legend(
        handles=[
            Patch(fc=BLUE, ec=EDGE, label="latih"),
            Patch(fc=GREEN, ec=EDGE, label="validasi 10%"),
            Patch(fc="#E8D9F2", ec=EDGE, label="uji dalam sumber 10%"),
            Patch(fc=ORANGE, ec=EDGE, label="uji sumber asing"),
        ],
        loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.55, -0.12),
    )
    fig.savefig(OUT / "gambar2_loso.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    print("ok")
