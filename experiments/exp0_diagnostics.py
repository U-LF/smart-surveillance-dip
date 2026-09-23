"""Experiment 0 - Problem analysis: can the system *detect* each degradation?

(a) accuracy of the edge-aware noise estimator vs. true sigma
(b) trigger matrix: how often each adaptive rule fires per scenario
    (diagonal = correct detection, off-diagonal on 'clean' = false alarms)
Justifies the thresholds in surveillance/pipeline.py.
"""
from common import parse_args, pick_indices, rng, save_fig, save_table, plt  # noqa: I001
import numpy as np
import pandas as pd

from surveillance import degradation as D
from surveillance import frequency as F
from surveillance import pipeline as PL
from surveillance import restoration as R
from surveillance.datasets import PennFudan


def main():
    a = parse_args(__doc__, default_n=60, quick_n=10)
    ds = PennFudan()
    idx = pick_indices(len(ds), a.n)
    g = rng(0)

    # (a) noise estimator
    rows = []
    for i in idx:
        im = ds[i].image
        for s in (0, 5, 10, 15, 20, 30, 40):
            x = D.add_gaussian_noise(im, s, g) if s else im
            rows.append({"true_sigma": s, "est_sigma": R.estimate_noise_sigma(x)})
    df = pd.DataFrame(rows).groupby("true_sigma")["est_sigma"].agg(["mean", "std", "min", "max"]).reset_index()
    save_table(df, "exp0_noise_estimator")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.errorbar(df.true_sigma, df["mean"], yerr=df["std"], fmt="o-", capsize=3, label="estimated")
    ax.plot([0, 40], [0, 40], "k--", lw=0.8, label="ideal")
    ax.axhline(PL.T_NOISE, color="r", lw=0.8, ls=":", label=f"threshold {PL.T_NOISE}")
    ax.set_xlabel("true noise sigma"); ax.set_ylabel("estimated sigma"); ax.legend(fontsize=7)
    save_fig(fig, "exp0_noise_estimator")

    # (b) trigger matrix
    scen = ["clean"] + list(D.SCENARIOS)
    trig = []
    for name in scen:
        c = {"scenario": name, "impulse": 0, "noise": 0, "periodic": 0, "dark": 0, "dark_regions": 0, "lost_blocks": 0}
        for i in idx:
            im = ds[i].image
            x = im if name == "clean" else D.SCENARIOS[name](im, g)
            d = R.diagnose(x)
            gray = x.mean(axis=2).astype(np.uint8)
            c["impulse"] += d.impulse_ratio > PL.T_IMPULSE
            c["noise"] += d.noise_sigma > PL.T_NOISE
            c["periodic"] += bool(F.detect_periodic_peaks(gray))
            c["dark"] += d.mean_luma < PL.T_DARK
            c["dark_regions"] += (d.mean_luma >= PL.T_DARK) and d.dark_regions > PL.T_DARK_REGIONS
            c["lost_blocks"] += d.lost_block_ratio > PL.T_LOST
        for k in list(c)[1:]:
            c[k] = 100.0 * c[k] / len(idx)
        trig.append(c)
    save_table(pd.DataFrame(trig), "exp0_trigger_matrix_percent", ".0f")

    # periodic detection rate vs amplitude (random frequencies)
    rows = []
    for amp in (0, 10, 20, 40, 60):
        hits = 0
        for i in idx:
            im = ds[i].image
            fr = ((g.uniform(0.04, 0.2), g.uniform(-0.1, 0.1)), (g.uniform(0.02, 0.15), g.uniform(0.02, 0.2)))
            x = D.add_periodic_noise(im, amp, fr) if amp else im
            hits += bool(F.detect_periodic_peaks(x.mean(axis=2).astype(np.uint8)))
        rows.append({"amplitude": amp, "detected_%": 100.0 * hits / len(idx)})
    save_table(pd.DataFrame(rows), "exp0_periodic_detection_rate", ".1f")


if __name__ == "__main__":
    main()
