"""Shared helpers for all experiments (paths, CLI, table/figure saving)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from surveillance.config import FIG_DIR, SEED, TABLE_DIR, ensure_dirs  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})


def parse_args(desc: str, default_n: int = 40, quick_n: int = 8, extra=None):
    """Shared CLI. `extra` is an optional callable(parser) for per-experiment flags."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--n", type=int, default=None, help="number of Penn-Fudan images")
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    if extra is not None:
        extra(ap)
    a = ap.parse_args()
    a.n = a.n or (quick_n if a.quick else default_n)
    ensure_dirs()
    return a


def pick_indices(n_total: int, n: int):
    """Evenly spaced subset (covers both the Penn and the Fudan halves)."""
    return sorted(set(np.linspace(0, n_total - 1, min(n, n_total)).astype(int).tolist()))


def rng(offset: int = 0):
    return np.random.default_rng(SEED + offset)


def save_table(df: pd.DataFrame, name: str, floatfmt: str = ".3f"):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    with open(TABLE_DIR / f"{name}.md", "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=floatfmt))
    try:  # LaTeX version for the Overleaf report
        df.to_latex(TABLE_DIR / f"{name}.tex", index=False, float_format=lambda x: f"{x:{floatfmt}}")
    except Exception:  # noqa: BLE001 (jinja2 missing -> skip LaTeX export)
        pass
    print(f"\n== {name} ==")
    print(df.to_string(index=False, float_format=lambda x: f"{x:{floatfmt}}"))


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  figure -> results/figures/{name}.png/.pdf")


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def image_grid(images, titles, ncols: int, name: str, cell: float = 2.2):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows * 0.95))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(rgb(im), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=7)
    fig.tight_layout()
    save_fig(fig, name)
