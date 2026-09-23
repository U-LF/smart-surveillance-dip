"""Experiment 3 - Image restoration and reconstruction (Technical Task 3).

(a) Motion deblurring: inverse vs Wiener (K sweep) vs constrained least squares,
    and robustness to a mis-estimated PSF length (a real camera never knows it exactly).
(b) Night / low-light: gamma, global HE (from scratch), CLAHE, gamma+CLAHE,
    and denoise-before-enhance (order matters: HE amplifies noise).
(c) Reconstruction of packet-loss blocks: Telea vs Navier-Stokes in-painting.
"""
from common import image_grid, parse_args, pick_indices, rng, save_fig, save_table, plt  # noqa: I001
import pandas as pd

from surveillance import degradation as D
from surveillance import restoration as R
from surveillance import spatial as SP
from surveillance.datasets import PennFudan
from surveillance.metrics import Timer, psnr, ssim


def run(methods, pairs, group=None):
    rows = []
    for clean, deg in pairs:
        rows.append({"method": "(degraded)", "PSNR": psnr(clean, deg), "SSIM": ssim(clean, deg), "ms": 0.0})
        for name, fn in methods.items():
            with Timer() as t:
                out = fn(deg)
            rows.append({"method": name, "PSNR": psnr(clean, out), "SSIM": ssim(clean, out), "ms": t.ms})
    df = pd.DataFrame(rows).groupby("method", sort=False).mean().reset_index()
    if group:
        df.insert(0, "case", group)
    return df


def main():
    a = parse_args(__doc__, default_n=30, quick_n=5)
    ds = PennFudan()
    idx = pick_indices(len(ds), a.n)
    g = rng(3)
    clean = [ds[i].image for i in idx]

    # (a) motion blur, length 15 px horizontal + sigma=2 noise
    L = 15
    psf = D.motion_psf(L, 0)
    mb = [(im, D.add_motion_blur(im, L, 0, 2.0, g)) for im in clean]
    m = {"inverse": lambda x: R.inverse_filter(x, psf)}
    for K in (0.001, 0.005, 0.01, 0.03):
        m[f"Wiener K={K}"] = (lambda k: lambda x: R.wiener_filter(x, psf, k))(K)
    for gm in (0.001, 0.005, 0.02):
        m[f"CLS gamma={gm}"] = (lambda k: lambda x: R.cls_filter(x, psf, k))(gm)
    m["Laplacian sharpen (no PSF)"] = lambda x: SP.laplacian_sharpen(x, 0.5)
    save_table(run(m, mb), "exp3a_deblurring")

    rows = []
    for L_est in (9, 12, 15, 18, 21):
        p2 = D.motion_psf(L_est, 0)
        vals = [psnr(c, R.wiener_filter(d, p2, 0.005)) for c, d in mb]
        rows.append({"assumed_length": L_est, "true_length": L, "Wiener PSNR": sum(vals) / len(vals)})
    df = pd.DataFrame(rows)
    save_table(df, "exp3a_psf_mismatch", ".2f")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(df.assumed_length, df["Wiener PSNR"], "o-")
    ax.axvline(L, color="k", ls="--", lw=0.8)
    ax.axhline(sum(psnr(c, d) for c, d in mb) / len(mb), color="r", ls=":", lw=0.8, label="blurred input")
    ax.set_xlabel("assumed PSF length (px)"); ax.set_ylabel("PSNR (dB)"); ax.legend(fontsize=7)
    save_fig(fig, "exp3_psf_mismatch")
    c, d = mb[len(mb) // 2]
    image_grid([c, d, R.inverse_filter(d, psf), R.wiener_filter(d, psf, 0.005), R.cls_filter(d, psf, 0.005)],
               ["clean", "motion blur + noise", "inverse", "Wiener", "CLS"], 5, "exp3_deblurring")

    # (b) low light
    ll = [(im, D.low_light(im, 0.3, 1.4, 6.0, g)) for im in clean]
    m = {
        "gamma (auto)": lambda x: R.enhance_low_light(x, "gamma"),
        "global HE (scratch)": lambda x: R.enhance_low_light(x, "he"),
        "CLAHE": lambda x: R.enhance_low_light(x, "clahe"),
        "gamma+CLAHE": lambda x: R.enhance_low_light(x, "gamma+clahe"),
        "gamma+CLAHE -> bilateral": lambda x: SP.bilateral_filter(R.enhance_low_light(x, "gamma+clahe"), 12),
        "bilateral -> gamma+CLAHE": lambda x: R.enhance_low_light(SP.bilateral_filter(x, 6), "gamma+clahe"),
    }
    save_table(run(m, ll), "exp3b_low_light")
    c, d = ll[len(ll) // 2]
    image_grid([c, d, R.enhance_low_light(d, "he"), R.enhance_low_light(d, "gamma+clahe"),
                R.enhance_low_light(SP.bilateral_filter(d, 6), "gamma+clahe")],
               ["clean", "night", "global HE", "gamma+CLAHE", "denoise->gamma+CLAHE"], 5, "exp3_low_light")

    # (c) packet loss
    out = []
    for ratio in (0.01, 0.03, 0.08):
        bl = [(im, D.block_loss(im, ratio, rng=g)[0]) for im in clean]
        out.append(run({"Telea in-painting": lambda x: R.reconstruct_lost_blocks(x, method="telea"),
                        "Navier-Stokes in-painting": lambda x: R.reconstruct_lost_blocks(x, method="ns"),
                        "median 5x5 (naive)": lambda x: SP.median_filter(x, 5)}, bl, f"{ratio:.0%} blocks lost"))
    save_table(pd.concat(out), "exp3c_block_loss")
    c = clean[len(clean) // 2]
    d = D.block_loss(c, 0.05, rng=g)[0]
    image_grid([c, d, R.reconstruct_lost_blocks(d)], ["clean", "5% blocks lost", "Telea in-painting"], 3,
               "exp3_block_loss")


if __name__ == "__main__":
    main()
