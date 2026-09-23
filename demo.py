#!/usr/bin/env python3
"""Run the complete surveillance pipeline on a video file, webcam or RTSP stream.

Examples
    python demo.py --source data/videos/vtest.avi --profile balanced
    python demo.py --source data/videos/kitti-inference-vid.mp4 --degrade night_compound
    python demo.py --source 0 --show                       # laptop webcam
    python demo.py --source rtsp://user:pw@cam/stream1 --profile fast --show
    python demo.py --source data/videos/vtest.avi --privacy scramble --key mysecret

Output: an annotated MP4 (side-by-side input | processed) and a JSON log with
per-frame actions, detections and stage timings -> ideal material for the
video presentation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from surveillance import degradation as D
from surveillance.config import RESULTS_DIR
from surveillance.datasets import iter_video
from surveillance.pipeline import PROFILES, SurveillancePipeline, overlay
from surveillance.privacy import sign_frame
from surveillance.segmentation import MotionSegmenter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/videos/vtest.avi", help="file, RTSP URL or webcam index")
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES))
    ap.add_argument("--detector", default="yolo", choices=["yolo", "hog", "ultralytics", "none"])
    ap.add_argument("--privacy", default="head", choices=["head", "body", "mask", "scramble", "none"])
    ap.add_argument("--key", default=None, help="secret key for --privacy scramble and HMAC signing")
    ap.add_argument("--degrade", default=None, choices=list(D.SCENARIOS),
                    help="simulate a camera fault on every frame (for the demo video)")
    ap.add_argument("--illumination", default="auto", choices=["auto", "always", "off"])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS_DIR / "demo"))
    ap.add_argument("--show", action="store_true", help="open a live window (press q to quit)")
    a = ap.parse_args()

    key = a.key.encode() if a.key else None
    pipe = SurveillancePipeline(a.profile, detector=a.detector, privacy=a.privacy, scramble_key=key,
                                illumination=a.illumination)
    motion = MotionSegmenter("mog2")
    rng = np.random.default_rng(0)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(str(a.source)).stem}_{a.profile}" + (f"_{a.degrade}" if a.degrade else "")
    writer, log, fps_ema = None, [], None

    for i, frame in iter_video(a.source, max_frames=a.max_frames):
        raw = D.SCENARIOS[a.degrade](frame, rng) if a.degrade else frame
        t0 = time.perf_counter()
        s = pipe.profile.scale
        small = cv2.resize(raw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else raw
        mm, _ = motion.apply(small)
        res = pipe.process(raw, motion_mask=mm if pipe.profile.segmentation == "box" else None)
        dt = time.perf_counter() - t0
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt
        vis = overlay(raw, res, fps_ema)
        left = cv2.resize(raw, (vis.shape[1], vis.shape[0]))
        cv2.putText(left, "camera input", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        canvas = np.hstack([left, vis])
        if writer is None:
            writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                     (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        entry = {"frame": i, "actions": res.actions, "ms": round(res.total_ms, 1),
                 "timings": {k: round(v, 1) for k, v in res.timings.items()},
                 "detections": [{"label": d.label, "score": round(d.score, 3),
                                 "box": [round(v, 1) for v in d.box]} for d in res.detections],
                 "bytes": res.compressed_bytes}
        if key:
            entry["hmac"] = sign_frame(res.protected, key, i)
        log.append(entry)
        if a.show:
            cv2.imshow("Smart surveillance (q to quit)", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if i % 25 == 0:
            print(f"frame {i:4d}  {res.total_ms:6.1f} ms  {fps_ema:5.1f} FPS  "
                  f"{len(res.detections)} objects  {res.actions}")

    if writer:
        writer.release()
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"source": str(a.source), "profile": a.profile, "degrade": a.degrade, "frames": log}, f, indent=1)
    if log:
        ms = np.mean([e["ms"] for e in log])
        print(f"\n{len(log)} frames, mean {ms:.1f} ms/frame ({1000 / ms:.1f} FPS pipeline-only)")
    print(f"video -> {out_dir / (stem + '.mp4')}\nlog   -> {out_dir / (stem + '.json')}")
    if a.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
