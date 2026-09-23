#!/usr/bin/env python3
"""Generate the design diagrams for the report and the README.

These are the 'DIP design pipeline and flowcharts' deliverable: they are drawn
from the same constants the code runs on (surveillance/pipeline.py thresholds
and PROFILES), so a diagram can never silently drift from the implementation.

    python scripts/make_diagrams.py

Writes results/figures/fig_{pipeline,flowchart,interdependence}.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from surveillance.config import FIG_DIR
from surveillance.pipeline import (PROFILES, T_DARK, T_DARK_REGIONS, T_IMPULSE, T_LOST, T_NOISE)

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.bbox": "tight",
                     "savefig.dpi": 300, "figure.dpi": 150})

# Restrained academic palette, grouped by the role a stage plays.
C = {
    "diagnose": ("#E8EDF4", "#2E4A6B"),   # (fill, edge/text)
    "repair":   ("#DCE9F5", "#1F5C8B"),
    "restore":  ("#D8E8E2", "#1F6F54"),
    "analyse":  ("#E4E2F0", "#4A3F8F"),
    "protect":  ("#F7E9D8", "#9A5B1E"),
    "deliver":  ("#F2DFE0", "#8F3339"),
    "io":       ("#F0F0F0", "#333333"),
    "decision": ("#FFF6DA", "#8A6D22"),
}


def box(ax, xy, w, h, text, kind="repair", fontsize=8, bold=False, radius=0.03):
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle=f"round,pad=0.008,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize, color=edge,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.35)


def diamond(ax, xy, w, h, text, fontsize=7.5):
    fill, edge = C["decision"]
    x, y = xy
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=edge, zorder=3,
            linespacing=1.3)


def arrow(ax, p0, p1, label=None, color="#444444", style="-", lw=1.1, rad=0.0, fontsize=7,
          label_offset=(0, 0.012)):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
                                 color=color, linestyle=style, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + label_offset[0], (p0[1] + p1[1]) / 2 + label_offset[1],
                label, ha="center", va="bottom", fontsize=fontsize, color=color, zorder=3)


def elbow(ax, points, color="#444444", lw=1.1):
    """Right-angled routed connector through `points`, arrowhead on the last leg."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1], ys[:-1], color=color, linewidth=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, zorder=1, shrinkA=0, shrinkB=2))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("auto")
    return fig, ax


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> results/figures/{name}.png/.pdf")


# ---------------------------------------------------------------------------
# Figure 1: end-to-end system pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = canvas(11, 5.6)
    y1, y2 = 0.735, 0.345           # the two stage rows
    w, h = 0.172, 0.115
    xs = [0.215, 0.435, 0.655, 0.875]

    row1 = [
        ("1  DIAGNOSE  (blind)", "diagnose",
         "noise $\\sigma$ · impulse density · spectral\npeaks · mean luma · lost blocks"),
        ("2  REPAIR  @ native res.", "repair",
         "in-paint lost blocks →\n(adaptive) median for impulses"),
        ("3  RESIZE  (profile)", "repair", "0.5×  /  0.75×  /  1.0×"),
        ("4  RESTORE", "restore",
         "notch-reject → denoise →\nlow-light / homomorphic"),
    ]
    row2 = [
        ("5  RECOGNISE", "analyse", "YOLOv4-tiny (OpenCV DNN)\nor HOG + linear SVM"),
        ("6  SEGMENT", "analyse", "marker watershed in box,\nor MOG2 motion masks"),
        ("7  PROTECT", "protect", "pixelate / blur heads ·\nkeyed scramble + HMAC"),
        ("8  COMPRESS", "deliver", "ROI JPEG: q90 objects,\nq15–40 background"),
    ]
    for row, y in ((row1, y1), (row2, y2)):
        for x, (title, kind, sub) in zip(xs, row):
            box(ax, (x, y), w, h, title, kind, fontsize=8.4, bold=True)
            ax.text(x, y - h / 2 - 0.028, sub, ha="center", va="top", fontsize=6.7,
                    color="#555555", linespacing=1.45)
        for i in range(3):
            arrow(ax, (xs[i] + w / 2, y), (xs[i + 1] - w / 2, y))

    # camera input
    box(ax, (0.055, y1), 0.095, h, "camera\nframe\nCCTV / RTSP", "io", fontsize=7.2)
    arrow(ax, (0.055 + 0.0475, y1), (xs[0] - w / 2, y1))

    # wrap connector: RESTORE -> RECOGNISE, routed through a clear channel
    ych = 0.545
    elbow(ax, [(xs[3], y1 - h / 2 - 0.075), (xs[3], ych), (xs[0], ych), (xs[0], y2 + h / 2)])
    # placed under stage 3, whose caption is a single line -> no collision
    ax.text(xs[2], ych + 0.016, "enhanced frame  $\\hat{f}(x,y)$", fontsize=7.2,
            color="#444444", ha="center", va="bottom")

    # outputs: the compressed, protected stream forks to two audiences
    yo, ho = 0.082, 0.098
    box(ax, (0.620, yo), 0.30, ho,
        "PUBLIC / OPERATOR STREAM\nidentities hidden, analytics still work", "io", fontsize=7.2)
    box(ax, (0.885, yo), 0.205, ho,
        "KEY HOLDER\nexact recovery\n+ tamper check", "io", fontsize=7.2)
    ystub = 0.168
    elbow(ax, [(xs[3], y2 - h / 2 - 0.058), (xs[3], ystub), (0.620, ystub), (0.620, yo + ho / 2)])
    elbow(ax, [(xs[3], ystub), (0.885, ystub), (0.885, yo + ho / 2)])

    ax.text(0.5, 0.985,
            "Adaptive surveillance pipeline — only the stages the diagnosis calls for are executed",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#222222")
    ax.text(0.5, 0.925,
            "a profile (fast / balanced / quality) caps how expensive each selected stage may be",
            ha="center", va="top", fontsize=7.8, color="#666666", style="italic")
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: adaptive restoration flowchart (thresholds read from the code)
# ---------------------------------------------------------------------------
def fig_flowchart():
    fig, ax = canvas(9.4, 10.4)
    xc, xr = 0.375, 0.775        # main column, action column
    dw, dh = 0.295, 0.078        # decision size
    bw, bh = 0.285, 0.058        # action size

    ax.text(0.5, 0.995, "Adaptive restoration — blind diagnostics select the stages",
            ha="center", va="top", fontsize=11, fontweight="bold", color="#222222")

    box(ax, (xc, 0.930), 0.30, 0.048, "frame  $g(x,y)$  from camera", "io", fontsize=8.5)
    box(ax, (xc, 0.849), 0.355, 0.058,
        "diagnose(): $\\hat{\\sigma}$, impulse ratio, luma,\ndark regions, lost-block ratio",
        "diagnose", fontsize=7.6)
    arrow(ax, (xc, 0.906), (xc, 0.879))

    steps = [
        (f"lost blocks\n> {T_LOST}?", "Telea in-painting", "repair"),
        (f"impulse ratio\n> {T_IMPULSE}?",
         "median 3×3  |  adaptive\nmedian 7×7 (if > 0.10)", "repair"),
        ("profile.periodic\nand peaks found?", "Butterworth\nnotch-reject", "restore"),
        (f"$\\hat{{\\sigma}}\\times$gain > {T_NOISE}?",
         "Gaussian | bilateral | NLM\n(profile), + unsharp", "restore"),
        (f"mean luma\n< {T_DARK}?", "auto-gamma + CLAHE", "restore"),
        (f"dark regions\n> {T_DARK_REGIONS}?", "homomorphic | CLAHE", "restore"),
    ]
    y0, dy = 0.750, 0.124
    ys = [y0 - i * dy for i in range(len(steps))]
    prev = (xc, 0.820)
    for y, (q, action, kind) in zip(ys, steps):
        diamond(ax, (xc, y), dw, dh, q)
        arrow(ax, prev, (xc, y + dh / 2))
        box(ax, (xr, y), bw, bh, action, kind, fontsize=7.2)
        arrow(ax, (xc + dw / 2, y), (xr - bw / 2, y), label="yes", fontsize=7,
              label_offset=(0, 0.006))
        # the action rejoins the main column between this decision and the next
        yj = y - dh / 2 - 0.025
        elbow(ax, [(xr, y - bh / 2), (xr, yj), (xc + 0.004, yj)], color="#8A8A8A", lw=0.95)
        ax.text(xc - 0.020, y - dh / 2 - 0.014, "no", ha="right", va="center", fontsize=7,
                color="#444444")
        prev = (xc, y - dh / 2)

    box(ax, (xc, 0.022), 0.34, 0.050, "enhanced frame  $\\hat{f}(x,y)$   →   detector", "io",
        fontsize=8.5)
    arrow(ax, (xc, ys[-1] - dh / 2), (xc, 0.047))

    # annotations that carry the design argument
    note = dict(fontsize=7, ha="left", va="center", style="italic", linespacing=1.55)
    ax.text(0.012, (ys[0] + ys[1]) / 2, "repaired at NATIVE\nresolution, before\nany resize",
            color="#1F5C8B", **note)
    ax.text(0.012, ys[3], "gain = clip(110/luma,1,4)\njudge the noise AFTER\nthe brightening it\nwill receive",
            color="#1F6F54", **note)
    ax.text(0.012, (ys[4] + ys[5]) / 2, "luma test first:\na uniformly dark frame\nis not an unevenly\nlit one",
            color="#1F6F54", **note)
    save(fig, "fig_flowchart")


# ---------------------------------------------------------------------------
# Figure 3: module interdependence
# ---------------------------------------------------------------------------
def _clip_to_box(centre, half_w, half_h, toward):
    """Point where the centre->toward ray leaves the node's rectangle."""
    dx, dy = toward[0] - centre[0], toward[1] - centre[1]
    if dx == 0 and dy == 0:
        return centre
    t = min(half_w / abs(dx) if dx else 1e9, half_h / abs(dy) if dy else 1e9)
    return centre[0] + dx * t, centre[1] + dy * t


def fig_interdependence(couplings: str | None = None):
    fig, ax = canvas(10.2, 5.4)

    # name -> (x, y, w, h, label, kind)
    N = {
        "degradation":  (0.098, 0.555, 0.175, 0.105, "degradation.py\nforward model\n$g=H[f]+\\eta$", "io"),
        "spatial":      (0.285, 0.790, 0.150, 0.082, "spatial.py\nTask 1", "repair"),
        "frequency":    (0.443, 0.790, 0.150, 0.082, "frequency.py\nTask 2", "repair"),
        "restoration":  (0.364, 0.555, 0.190, 0.095, "restoration.py\nTask 3 + diagnostics", "restore"),
        "recognition":  (0.644, 0.555, 0.180, 0.095, "recognition.py\nTask 6", "analyse"),
        "segmentation": (0.644, 0.310, 0.180, 0.085, "segmentation.py\nTask 4", "analyse"),
        "privacy":      (0.898, 0.555, 0.168, 0.095, "privacy.py\nkey req. 5", "protect"),
        "compression":  (0.898, 0.310, 0.168, 0.085, "compression.py\nTask 5", "deliver"),
    }
    for x, y, w, h, label, kind in N.values():
        box(ax, (x, y), w, h, label, kind, fontsize=7.4)

    def edge(a, b, label=None, loff=(0, 0), fs=6.8):
        xa, ya, wa, ha, *_ = N[a]
        xb, yb, wb, hb, *_ = N[b]
        p0 = _clip_to_box((xa, ya), wa / 2, ha / 2, (xb, yb))
        p1 = _clip_to_box((xb, yb), wb / 2, hb / 2, (xa, ya))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=1.15,
                                     color="#555555", zorder=4, shrinkA=1, shrinkB=1))
        if label:
            ax.text((p0[0] + p1[0]) / 2 + loff[0], (p0[1] + p1[1]) / 2 + loff[1], label,
                    ha="center", va="center", fontsize=fs, color="#3B3B3B", zorder=5,
                    linespacing=1.35,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    # horizontal edges carry their label ABOVE the line, so a narrow gap between
    # two boxes never has to accommodate the text
    edge("degradation", "restoration", "controlled faults", loff=(0, 0.030))
    edge("spatial", "restoration", None)
    edge("frequency", "restoration", None)
    edge("restoration", "recognition", "enhanced frame", loff=(0, 0.030))
    edge("recognition", "privacy", "which pixels to hide", loff=(0, 0.030))
    edge("segmentation", "compression", "silhouettes", loff=(0, 0.028))
    edge("recognition", "segmentation", "boxes say\nWHERE", loff=(-0.068, 0))
    edge("recognition", "compression", "ROI map", loff=(0.030, 0.020))

    # cross-cutting modules
    box(ax, (0.50, 0.112), 0.80, 0.085,
        "pipeline.py  —  Profile + SurveillancePipeline\n"
        "orchestrates every module above, executing only the stages the diagnosis selects",
        "diagnose", fontsize=7.5, bold=True)
    for k in ("restoration", "recognition", "segmentation", "privacy", "compression"):
        x, y, _, h, *_ = N[k]
        ax.plot([x, x], [0.155, y - h / 2 - 0.006], color="#9AA7B4", lw=0.9, ls=(0, (2, 2)),
                zorder=0)

    ax.text(0.5, 0.985,
            "Module interdependence — every downstream task consumes an upstream product",
            ha="center", va="top", fontsize=10.2, fontweight="bold", color="#222222")
    ax.text(0.5, 0.030, couplings or "", ha="center", va="center", fontsize=7,
            color="#444444", linespacing=1.6)
    save(fig, "fig_interdependence")


def measured_couplings() -> str:
    """Build the interdependence caption from the result tables, so the numbers
    printed on the diagram are always the ones the experiments produced."""
    import csv

    from surveillance.config import TABLE_DIR

    def rows(name):
        p = TABLE_DIR / f"{name}.csv"
        return list(csv.DictReader(p.open())) if p.exists() else []

    parts = []
    r6b = {r["scenario"]: r for r in rows("exp6b_robustness")}
    bits = []
    for sc, pretty in (("saltpepper_20", "S&P 20%"), ("periodic", "periodic"),
                       ("night_compound", "night")):
        if sc in r6b:
            raw = float(r6b[sc]["raw F1"])
            best = max(float(r6b[sc][c]) for c in r6b[sc] if c.endswith(") F1"))
            bits.append(f"{pretty} F1 {raw:.3f}$\\rightarrow${best:.3f}")
    if bits:
        parts.append("restoration$\\rightarrow$recognition (exp6b):  " + "   ·   ".join(bits))

    r5d = rows("exp5d_roi_coding_cctv")
    roi = next((r for r in r5d if r["scheme"].startswith("ROI q90/q15")), None)
    uni = next((r for r in r5d if r["scheme"].startswith("uniform (<= size of q90/q15")), None)
    if roi and uni:
        gain = float(roi["PSNR_objects"]) - float(uni["PSNR_objects"])
        parts.append(f"recognition$\\rightarrow$compression (exp5d):  "
                     f"+{gain:.1f} dB on people at equal file size")
    return "measured couplings —  " + "\n".join(parts) if parts else ""


if __name__ == "__main__":
    print("Generating design diagrams:")
    fig_pipeline()
    fig_flowchart()
    fig_interdependence(measured_couplings())
    print(f"\nProfiles drawn from code: {list(PROFILES)}")
