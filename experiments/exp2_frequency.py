"""Experiment 2 - Frequency-domain filtering (Technical Task 2).

(a) Periodic interference: automatic notch-reject vs global low-pass filters
    (ideal / Butterworth / Gaussian) vs spatial smoothing.
(b) Ideal vs Butterworth vs Gaussian LPF on Gaussian noise (ringing study).
(c) Uneven street lighting: homomorphic filter vs HE vs CLAHE.
(d) Equivalence/cost: Gaussian LPF in frequency domain vs spatial GaussianBlur.
"""
from common import image_grid, parse_args, pick_indices, rng, save_fig, save_table, plt  # noqa: I001
import cv2
import numpy as np
import pandas as pd

from surveillance import degradation as D
from surveillance import frequency as F
from surveillance import restoration as R
from surveillance import spatial as SP
from surveillance.datasets import PennFudan
from surveillance.metrics import Timer, psnr, ssim


def evaluate(methods, pairs):
    rows = []
    for clean, deg in pairs:
        rows.append({"method": "(degraded)", "PSNR": psnr(clean, deg), "SSIM": ssim(clean, deg), "ms": 0.0})
        for name, fn in methods.items():
            with Timer() as t:
                out = fn(deg)
            rows.append({"method": name, "PSNR": psnr(clean, out), "SSIM": ssim(clean, out), "ms": t.ms})
    return pd.DataFrame(rows).groupby("method", sort=False).mean().reset_index()


def main():
    a = parse_args(__doc__, default_n=30, quick_n=5)
    ds = PennFudan()
    idx = pick_indices(len(ds), a.n)
    g = rng(2)
    clean = [ds[i].image for i in idx]

    # (a) periodic interference with random frequencies
    per = []
    for im in clean:
        fr = ((g.uniform(0.04, 0.2), g.uniform(-0.1, 0.1)), (g.uniform(0.02, 0.15), g.uniform(0.02, 0.2)))
        per.append((im, D.add_periodic_noise(im, 40, fr)))
    methods = {
        "spatial mean 5x5": lambda x: SP.mean_filter(x, 5),
        "spatial median 5x5": lambda x: SP.median_filter(x, 5),
        "ideal LPF D0=0.10": lambda x: F.frequency_lowpass(x, 0.10, "ideal"),
        "Butterworth LPF D0=0.10 n=2": lambda x: F.frequency_lowpass(x, 0.10, "butterworth"),
        "Gaussian LPF D0=0.10": lambda x: F.frequency_lowpass(x, 0.10, "gaussian"),
        "auto notch-reject": lambda x: F.remove_periodic_noise(x)[0],
    }
    save_table(evaluate(methods, per), "exp2a_periodic_noise")

    # spectrum figure for the report
    im, deg = per[len(per) // 2]
    out, peaks = F.remove_periodic_noise(deg)
    gray = cv2.cvtColor(deg, cv2.COLOR_BGR2GRAY)
    S = F.spectrum(gray)
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
    ax[0].imshow(cv2.cvtColor(deg, cv2.COLOR_BGR2RGB)); ax[0].set_title("interference")
    ax[1].imshow(S, cmap="magma"); ax[1].set_title("log |F(u,v)| (Hann)")
    P, Q = S.shape
    for du, dv in peaks:
        for s in (1, -1):
            ax[1].add_patch(plt.Circle((Q // 2 + s * dv, P // 2 + s * du), 8, fill=False, color="cyan", lw=1))
    H = F.notch_reject(S.shape, [(du / P, dv / Q) for du, dv in peaks], 4.0)
    ax[2].imshow(H, cmap="gray"); ax[2].set_title(f"notch filter H ({len(peaks)} pairs)")
    ax[3].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    ax[3].set_title(f"restored {psnr(im, deg):.1f} -> {psnr(im, out):.1f} dB")
    for x in ax:
        x.axis("off")
    fig.tight_layout()
    save_fig(fig, "exp2_notch_spectrum")

    # (b) LPF family on Gaussian noise + ringing figure
    gn = [(im, D.add_gaussian_noise(im, 25, g)) for im in clean]
    lp = {}
    for kind in ("ideal", "butterworth", "gaussian"):
        for d0 in (0.05, 0.10, 0.20):
            lp[f"{kind} D0={d0:.2f}"] = (lambda k, d: (lambda x: F.frequency_lowpass(x, d, k)))(kind, d0)
    save_table(evaluate(lp, gn), "exp2b_lowpass_family")
    edge = np.zeros((128, 128), np.uint8); edge[:, 64:] = 200
    prof = {k: F.frequency_lowpass(edge, 0.06, k)[64, 40:90] for k in ("ideal", "butterworth", "gaussian")}
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(edge[64, 40:90], "k--", lw=0.8, label="step edge")
    for k, v in prof.items():
        ax.plot(v, label=k)
    ax.set_title("Ringing of LPFs on a step edge"); ax.legend(fontsize=7)
    save_fig(fig, "exp2_ringing")

    # (c) uneven illumination
    ul = [(im, D.uneven_illumination(im, 0.7, g)) for im in clean]
    ill = {"global HE": R.equalize_color, "CLAHE": R.clahe,
           "homomorphic (ours)": F.homomorphic_filter,
           "homomorphic, % stretch": lambda x: F.homomorphic_filter(x, 0.5, 1.4, 1.0, 0.02, 0.0)}
    save_table(evaluate(ill, ul), "exp2c_uneven_illumination")
    im, deg = ul[len(ul) // 3]
    image_grid([im, deg, R.equalize_color(deg), R.clahe(deg), F.homomorphic_filter(deg)],
               ["clean", "uneven light", "global HE", "CLAHE", "homomorphic"], 5, "exp2_illumination")

    # (d) frequency vs spatial Gaussian smoothing: equivalence & cost
    rows = []
    for s in (1.0, 3.0, 8.0):
        m = min(clean[0].shape[:2])
        x = clean[0][:m, :m]  # square -> the D(u,v) grid is isotropic in cycles/pixel
        with Timer() as t1:
            # spatial sigma s <-> frequency D0 = P / (2 pi s) on a P-point DFT
            f_out = F.apply_filter(x, lambda sh: F.lowpass(sh, sh[0] / (2 * np.pi * s), "gaussian"))
        k = int(2 * np.ceil(3 * s) + 1)
        with Timer() as t2:
            s_out = cv2.GaussianBlur(x, (k, k), s, borderType=cv2.BORDER_REFLECT)
        rows.append({"sigma_px": s, "freq_ms": t1.ms, "spatial_ms": t2.ms,
                     "PSNR(freq vs spatial)": psnr(s_out, f_out)})
    save_table(pd.DataFrame(rows), "exp2d_freq_vs_spatial", ".2f")


if __name__ == "__main__":
    main()
