#!/usr/bin/env python3
"""Print the headline numbers from results/tables, so the README and the report
quote the generated evidence instead of hand-copied values.

    python scripts/summarize_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from surveillance.config import TABLE_DIR


def load(name):
    p = TABLE_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def show(title, df, cols=None, n=None):
    print(f"\n===== {title} =====")
    if df is None:
        print("  (table missing - experiment not run)")
        return
    d = df[cols] if cols else df
    print((d.head(n) if n else d).to_string(index=False))


def main():
    for title, name, cols in [
        ("exp0 trigger matrix (%)", "exp0_trigger_matrix_percent", None),
        ("exp0 noise estimator", "exp0_noise_estimator", None),
        ("exp0 periodic detection", "exp0_periodic_detection_rate", None),
        ("exp1 best filter per scenario", "exp1_best_per_scenario", None),
        ("exp1 scratch vs OpenCV", "exp1_scratch_vs_opencv", None),
        ("exp2a periodic", "exp2a_periodic_noise", None),
        ("exp2c uneven illumination", "exp2c_uneven_illumination", None),
        ("exp3a deblurring", "exp3a_deblurring", None),
        ("exp3b low light", "exp3b_low_light", None),
        ("exp3c block loss", "exp3c_block_loss", None),
        ("exp4a segmentation", "exp4a_segmentation", None),
        ("exp4b segmentation vs noise", "exp4b_segmentation_vs_noise", None),
        ("exp4c motion segmentation", "exp4c_motion_segmentation", None),
        ("exp5a quality at fixed bpp", "exp5a_quality_at_fixed_bpp", None),
        ("exp5b detection vs JPEG", "exp5b_detection_vs_jpeg", None),
        ("exp5c ROI (Penn-Fudan)", "exp5c_roi_coding", None),
        ("exp5d ROI (wide CCTV)", "exp5d_roi_coding_cctv", None),
        ("exp6a detector benchmark", "exp6a_detector_benchmark", None),
        ("exp6b robustness", "exp6b_robustness", None),
        ("exp6c privacy vs detection", "exp6c_privacy_vs_detection", None),
        ("exp6d vehicles", "exp6d_vehicles_pseudo_gt", None),
        ("exp6e CAVIAR detection", "exp6e_caviar_detection", None),
        ("exp7a profiles latency", "exp7a_profiles_latency", None),
        ("exp7b detector input size", "exp7b_detector_input_size", None),
        ("exp7c memory", "exp7c_memory", None),
    ]:
        show(title, load(name), cols)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    main()
