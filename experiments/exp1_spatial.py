"""Experiment 1 - Spatial filtering (Technical Task 1).

(a) PSNR / SSIM / time of mean, Gaussian, median, adaptive median, bilateral,
    NLM and Laplacian / unsharp sharpening on Gaussian and impulse noise.
    Denoiser strength uses the *blind* sigma estimate (as the live system would).
(b) Resource analysis: from-scratch NumPy vs OpenCV implementations
    (identical output, very different cost).
"""
from common import image_grid, parse_args, pick_indices, rng, save_fig, save_table, plt  # noqa: I001
import numpy as np
import pandas as pd

from surveillance import degradation as D
from surveillance import restoration as R
from surveillance import spatial as SP
from surveillance.datasets import PennFudan
from surveillance.metrics import Timer, psnr, ssim

SCEN = ["gaussian_s15", "gaussian_s30", "saltpepper_5", "saltpepper_20"]


def main():
    a = parse_args(__doc__, default_n=40, quick_n=6)
    ds = PennFudan()
    idx = pick_indices(len(ds), a.n)
    g = rng(1)
    rows = []
    examples = {}
    for i in idx:
        im = ds[i].image
        for sc in SCEN:
            noisy = D.SCENARIOS[sc](im, g)
            sigma = R.estimate_noise_sigma(noisy)
            rows.append({"scenario": sc, "filter": "(none)", "PSNR": psnr(im, noisy), "SSIM": ssim(im, noisy), "ms": 0.0})
            for name, fn in SP.SPATIAL_FILTERS.items():
                with Timer() as t:
                    out = fn(noisy, sigma)
                rows.append({"scenario": sc, "filter": name, "PSNR": psnr(im, out), "SSIM": ssim(im, out), "ms": t.ms})
                if i == idx[len(idx) // 2]:
                    examples[(sc, name)] = out
            if i == idx[len(idx) // 2]:
                examples[(sc, "(none)")] = noisy
                examples["clean"] = im
    df = pd.DataFrame(rows).groupby(["scenario", "filter"], sort=False).mean(numeric_only=True).reset_index()
    save_table(df, "exp1_spatial_filters")

    # best filter per scenario (for the report's discussion)
    best = df[df["filter"] != "(none)"].loc[lambda d: d.groupby("scenario")["PSNR"].idxmax()]
    save_table(best, "exp1_best_per_scenario")

    # PSNR bar chart
    fig, axes = plt.subplots(1, len(SCEN), figsize=(12, 2.8), sharey=False)
    for ax, sc in zip(axes, SCEN):
        d = df[df.scenario == sc]
        ax.barh(d["filter"], d["PSNR"], color=["#999"] + ["#3a7"] * (len(d) - 1))
        ax.set_title(sc, fontsize=8); ax.set_xlabel("PSNR (dB)")
        ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "exp1_spatial_psnr")

    # qualitative grid (one image, crop around the centre for visibility)
    show = ["(none)", "mean5", "median3", "adaptive_median", "bilateral", "nlm"]
    ims, titles = [], []
    for sc in ("gaussian_s30", "saltpepper_20"):
        for f in show:
            ims.append(examples[(sc, f)])
            titles.append(f"{sc}\n{f}")
    image_grid(ims, titles, len(show), "exp1_qualitative")

    # (b) from-scratch vs OpenCV
    im = ds[idx[0]].image
    gray = im.mean(axis=2).astype(np.uint8)
    sp = D.add_salt_pepper(gray, 0.05, rng=g)
    cmp = []
    for label, f_np, f_cv in [
        ("mean 5x5", lambda: SP.mean_filter(gray, 5, "numpy"), lambda: SP.mean_filter(gray, 5)),
        ("median 3x3", lambda: SP.median_filter(sp, 3, "numpy"), lambda: SP.median_filter(sp, 3)),
        ("median 5x5", lambda: SP.median_filter(sp, 5, "numpy"), lambda: SP.median_filter(sp, 5)),
        ("Laplacian", lambda: SP.laplacian(gray, True, "numpy"), lambda: SP.laplacian(gray, True)),
    ]:
        reps = 3
        with Timer() as t1:
            for _ in range(reps):
                o1 = f_np()
        with Timer() as t2:
            for _ in range(reps * 10):
                o2 = f_cv()
        cmp.append({"operation": label, "numpy_ms": t1.ms / reps, "opencv_ms": t2.ms / (reps * 10),
                    "speedup": (t1.ms / reps) / max(1e-6, t2.ms / (reps * 10)),
                    "max_abs_diff": float(np.abs(o1.astype(float) - o2.astype(float)).max())})
    save_table(pd.DataFrame(cmp), "exp1_scratch_vs_opencv", ".2f")


if __name__ == "__main__":
    main()
