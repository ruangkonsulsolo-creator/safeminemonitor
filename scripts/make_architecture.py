"""Draw the system architecture and data-flow diagrams (docs/figures/*.png)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="white", lw=1.2, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=fc, ec="black", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1, linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2 + 0.005, (y1 + y2) / 2, text, fontsize=7, ha="left", va="center", style="italic")


def architecture(color):
    c = (lambda a, b: a if color else b)
    L1, L2, L3, L4 = c("#e8f1fb", "#f2f2f2"), c("#fff4e0", "#e6e6e6"), c("#eaf7ea", "#f2f2f2"), c("#f3e9fb", "#e6e6e6")
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "SafeMine Monitor - System Architecture", ha="center", fontsize=13, fontweight="bold")

    # layer frames
    for y, h, lab, fc in [(0.825, 0.135, "1. INPUT LAYER", L1), (0.57, 0.2, "2. PYTHON PROCESSING LAYER - Phase 2: computer vision", L3),
                          (0.305, 0.2, "3. PYTHON PROCESSING LAYER - Phase 1: analytics & EGT", L4),
                          (0.145, 0.14, "4. BACKEND & DATA LAYER", L2), (0.008, 0.125, "5. PRESENTATION LAYER (web)", L1)]:
        ax.add_patch(FancyBboxPatch((0.01, y), 0.98, h, boxstyle="round,pad=0.002", fc=fc, ec="black", lw=0.8,
                                    ls="--"))
        ax.text(0.02, y + h - 0.008, lab, fontsize=8, fontweight="bold", va="top")

    def poly(pts, ls="-"):
        xs, ys = zip(*pts[:-1])
        ax.plot(xs + (pts[-1][0],), ys + (pts[-1][1],), color="white", lw=0)
        for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x1, x2], [y1, y2], color="black", lw=1.1, ls=ls)
        arrow(ax, *pts[-2], *pts[-1], ls=ls)

    box(ax, 0.06, 0.84, 0.19, 0.06, "Research data (CSV)\nMSHA / site records")
    box(ax, 0.29, 0.84, 0.19, 0.06, "Image / video upload\n(mp4, avi, jpg, png)")
    box(ax, 0.52, 0.84, 0.19, 0.06, "Sample video library\n(virtual cameras CAM-01..n)")
    box(ax, 0.75, 0.84, 0.21, 0.06, "Restricted-zone polygon\n(drawn on first frame)")

    y2 = 0.585
    box(ax, 0.06, y2, 0.15, 0.13, "Frame sampling\n(OpenCV,\nstride N)")
    box(ax, 0.25, y2, 0.15, 0.13, "YOLO11 detector\n(PPE model)\n+ ByteTrack IDs")
    box(ax, 0.44, y2, 0.15, 0.13, "Rule engine\nPPE / zone /\nproximity")
    box(ax, 0.63, y2, 0.15, 0.13, "Event aggregator\n(time-stamped,\n>=0.5 s filter)")
    box(ax, 0.82, y2, 0.15, 0.13, "HIRARC risk\nL x S, level,\nrecommendation")
    for x1, x2 in [(0.214, 0.246), (0.404, 0.436), (0.594, 0.626), (0.784, 0.816)]:
        arrow(ax, x1, y2 + 0.065, x2, y2 + 0.065)

    y3 = 0.32
    box(ax, 0.06, y3, 0.17, 0.13, "Preprocessing\n(cleaning, dates,\nnarrative tagging)")
    box(ax, 0.27, y3, 0.17, 0.13, "EDA & statistics\nPareto, chi-square,\ntrend (pandas/SciPy)")
    box(ax, 0.48, y3, 0.17, 0.13, "Behavioural risk\nindicators (BRI,\nUBR, compliance)")
    box(ax, 0.69, y3, 0.28, 0.13, "Tripartite EGT model\nWorker x Supervisor x Management\nreplicator dynamics (RK4),\nJacobian stability, sensitivity")
    for x1, x2 in [(0.234, 0.266), (0.444, 0.476), (0.654, 0.686)]:
        arrow(ax, x1, y3 + 0.065, x2, y3 + 0.065)

    box(ax, 0.06, 0.16, 0.26, 0.07, "FastAPI REST API + job queue\n(background worker thread)")
    box(ax, 0.37, 0.16, 0.26, 0.07, "SQLite DB: jobs, events\n+ annotated media & snapshots")
    box(ax, 0.68, 0.16, 0.29, 0.07, "MJPEG stream: analysing / replay\n(substitute for live CCTV feed)")
    arrow(ax, 0.324, 0.195, 0.366, 0.195)
    arrow(ax, 0.634, 0.195, 0.676, 0.195)

    labels = ["Dashboard", "Upload &\nDetect", "Monitoring\n(worker panel)", "Records", "Risk\nmatrix",
              "Analytics", "EGT\nsimulation", "Validation"]
    for i, t in enumerate(labels):
        box(ax, 0.03 + i * 0.12, 0.022, 0.105, 0.065, t, fs=7.5)

    # input -> processing
    arrow(ax, 0.385, 0.836, 0.135, y2 + 0.134)
    arrow(ax, 0.615, 0.836, 0.20, y2 + 0.134)
    arrow(ax, 0.855, 0.836, 0.515, y2 + 0.134)
    poly([(0.056, 0.87), (0.025, 0.87), (0.025, y3 + 0.065), (0.056, y3 + 0.065)])
    # Phase-2 -> Phase-1 bridge (calibration)
    poly([(0.705, y2 - 0.004), (0.705, 0.505), (0.565, 0.505), (0.565, y3 + 0.134)], ls="--")
    ax.text(0.575, 0.49, "UBR, compliance rate", fontsize=7.5, style="italic", va="top")
    poly([(0.895, y2 - 0.004), (0.895, y3 + 0.134)], ls="--")
    ax.text(0.715, 0.53, "to EGT: x0 = 1 - UBR,\nrecall r, coverage phi", fontsize=7.5, style="italic")
    # results -> backend (right margin) and backend -> web
    poly([(0.974, y2 + 0.03), (0.985, y2 + 0.03), (0.985, 0.25), (0.5, 0.25), (0.5, 0.234)])
    arrow(ax, 0.5, 0.156, 0.5, 0.092)
    fig.savefig(FIG / ("architecture_color.png" if color else "architecture.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def flow():
    fig, ax = plt.subplots(figsize=(11, 2.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = ["Upload\nvideo/image", "Python\nprocessing", "CV detection\n(YOLO11+track)", "Unsafe behaviour\nidentification",
             "Risk\nclassification", "Database\nrecords", "Web\nmonitoring"]
    w = 0.12
    for i, s in enumerate(steps):
        x = 0.01 + i * 0.142
        box(ax, x, 0.2, w, 0.6, s, fs=8.5)
        if i:
            arrow(ax, x - 0.022, 0.5, x, 0.5)
    fig.savefig(FIG / "dataflow.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def framework():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.02, 0.62, 0.44, 0.32, "", fc="#f2f2f2")
    ax.text(0.24, 0.9, "PHASE 1 - Computational Analysis &\nBehavioural Modelling", ha="center", fontsize=9, fontweight="bold", va="top")
    for i, t in enumerate(["Research data", "Python analytics", "EGT simulation"]):
        box(ax, 0.04 + i * 0.145, 0.66, 0.125, 0.12, t, fs=8)
        if i:
            arrow(ax, 0.04 + i * 0.145 - 0.02, 0.72, 0.04 + i * 0.145, 0.72)
    box(ax, 0.54, 0.62, 0.44, 0.32, "", fc="#f2f2f2")
    ax.text(0.76, 0.9, "PHASE 2 - Experimental Identification &\nIntelligent Recognition", ha="center", fontsize=9, fontweight="bold", va="top")
    for i, t in enumerate(["Image / video", "CV processing", "Unsafe behaviour\nidentification"]):
        box(ax, 0.56 + i * 0.145, 0.66, 0.125, 0.12, t, fs=8)
        if i:
            arrow(ax, 0.56 + i * 0.145 - 0.02, 0.72, 0.56 + i * 0.145, 0.72)
    arrow(ax, 0.66, 0.56, 0.34, 0.56)
    ax.text(0.5, 0.545, "Phase 2 -> Phase 1: UBR & detector recall calibrate EGT (x0, phi, r)", ha="center", fontsize=8, style="italic", va="top")
    arrow(ax, 0.34, 0.47, 0.66, 0.47, None, ls=":")
    ax.text(0.5, 0.455, "Phase 1 -> Phase 2: incident indicators justify risk severity weights", ha="center", fontsize=8, style="italic", va="top")
    box(ax, 0.2, 0.03, 0.6, 0.3, "SYSTEM IMPLEMENTATION\nWeb-Based Monitoring & Risk Assessment Platform\n(SafeMine Monitor)\n"
        "Dashboard - Monitoring - Risk - Analytics - EGT - Validation", fs=9.5, fc="#e6e6e6")
    arrow(ax, 0.12, 0.615, 0.25, 0.335)
    arrow(ax, 0.88, 0.615, 0.75, 0.335)
    fig.savefig(FIG / "research_framework.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture(False)
    architecture(True)
    flow()
    framework()
    print("diagrams ->", FIG)
    sys.exit(0)
