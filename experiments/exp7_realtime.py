"""Experiment 7 - Real-time performance vs quality and resource constraints.

(a) Per-stage latency and FPS of the three profiles on a surveillance video,
    both on the clean stream and on a noisy 'night' stream (which triggers
    the restoration branch).
(b) Detector input size vs speed (the main real-time knob).
(c) Peak memory footprint of the whole process (edge-device budget).
"""
from common import parse_args, save_fig, save_table, plt  # noqa: I001
import time

import cv2
import numpy as np
import pandas as pd

from surveillance import degradation as D
from surveillance.config import VIDEO_DIR
from surveillance.datasets import CaviarSequence, available_caviar, iter_video
from surveillance.pipeline import PROFILES, SurveillancePipeline
from surveillance.recognition import YOLOTinyDetector
from surveillance.segmentation import MotionSegmenter


def _source_arg(ap):
    ap.add_argument("--source", default="vtest", choices=["vtest", "caviar"],
                    help="benchmark clip (default vtest.avi, 768x576)")


def peak_memory_mb() -> float:
    """Peak resident memory of this process, on Windows, Linux and macOS."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 2 ** 20  # peak_wset exists on Windows only
    except ImportError:
        pass
    try:
        import resource  # Unix only
        import sys
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / 2 ** 20 if sys.platform == "darwin" else ru / 1024.0  # bytes on macOS, KB on Linux
    except ImportError:
        return float("nan")


def frames_source(n, source: str = "vtest"):
    """Benchmark frames.

    The source is chosen explicitly, never by 'whichever dataset happens to be
    on disk': vtest.avi is 768x576 and CAVIAR is 384x288, so silently switching
    would make latency numbers incomparable between runs. vtest.avi is the
    default because every published figure uses it.
    """
    if source == "caviar":
        caviar = available_caviar()
        if not caviar:
            raise SystemExit("no CAVIAR sequences on disk - run: "
                             "python scripts/download_data.py --caviar")
        seq = CaviarSequence(caviar[0])
        return f"CAVIAR/{caviar[0]}", [fr for k, (_, fr, _) in enumerate(seq) if k < n]
    return "vtest.avi", [f for _, f in iter_video(VIDEO_DIR / "vtest.avi", max_frames=n)]


def main():
    a = parse_args(__doc__, default_n=120, quick_n=20, extra=_source_arg)
    src, frames = frames_source(a.n, a.source)
    g = np.random.default_rng(7)
    night = [D.SCENARIOS["night_compound"](f, g) for f in frames]
    h, w = frames[0].shape[:2]
    print(f"source {src}: {len(frames)} frames of {w}x{h}")

    rows, stage_rows = [], []
    for stream, fr_list in (("clean", frames), ("night+noise", night)):
        for prof in PROFILES:
            pipe = SurveillancePipeline(prof, detector="yolo", privacy="head")
            motion = MotionSegmenter("mog2") if PROFILES[prof].segmentation == "box" else None
            t0 = time.perf_counter()
            timings = []
            for f in fr_list:
                mm, tm0 = None, time.perf_counter()
                if motion is not None:  # fast profile: MOG2 silhouettes instead of per-box segmentation
                    small = cv2.resize(f, None, fx=pipe.profile.scale, fy=pipe.profile.scale,
                                       interpolation=cv2.INTER_AREA)
                    mm, _ = motion.apply(small)
                motion_ms = 1000 * (time.perf_counter() - tm0)
                r = pipe.process(f, motion_mask=mm)
                timings.append({**r.timings, "motion": motion_ms})
            wall = time.perf_counter() - t0
            td = pd.DataFrame(timings).mean()
            rows.append({"stream": stream, "profile": prof, "FPS": len(fr_list) / wall,
                         "mean_ms": 1000 * wall / len(fr_list), **{f"{k}_ms": v for k, v in td.items()}})
            stage_rows.append((stream, prof, td))
    df = pd.DataFrame(rows).fillna(0)
    save_table(df, "exp7a_profiles_latency", ".1f")

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    stages = [c for c in df.columns if c.endswith("_ms") and c != "mean_ms"]
    for ax, stream in zip(axes, ("clean", "night+noise")):
        d = df[df.stream == stream]
        bottom = np.zeros(len(d))
        for st in stages:
            ax.bar(d.profile, d[st], bottom=bottom, label=st.replace("_ms", ""))
            bottom += d[st].values
        ax.set_title(f"{stream} stream"); ax.set_ylabel("ms / frame")
        for k, (b, fps) in enumerate(zip(bottom, d.FPS)):
            ax.text(k, b + 5, f"{fps:.1f} FPS", ha="center", fontsize=7)
    axes[1].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "exp7_latency_breakdown")

    # (b) detector size vs latency
    rows = []
    for size in (224, 320, 416, 512, 608):
        det = YOLOTinyDetector(size)
        det.detect(frames[0])
        t0 = time.perf_counter()
        for f in frames[:30]:
            det.detect(f)
        ms = 1000 * (time.perf_counter() - t0) / min(30, len(frames))
        rows.append({"input_size": size, "ms/frame": ms, "max_FPS": 1000 / ms})
    save_table(pd.DataFrame(rows), "exp7b_detector_input_size", ".1f")

    # (c) memory
    peak_mb = peak_memory_mb()
    save_table(pd.DataFrame([{"peak_RSS_MB": peak_mb, "frame": f"{w}x{h}", "source": src}]), "exp7c_memory", ".1f")


if __name__ == "__main__":
    main()
