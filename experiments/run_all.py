"""Run every experiment in order and log timings.

    python experiments/run_all.py           # full run (~1-2 h on one laptop core)
    python experiments/run_all.py --quick   # smoke test (~5 min)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPS = ["exp0_diagnostics", "exp1_spatial", "exp2_frequency", "exp3_restoration",
        "exp4_segmentation", "exp5_compression", "exp5d_roi_video", "exp6_recognition", "exp7_realtime"]

if __name__ == "__main__":
    extra = sys.argv[1:]
    only = [e for e in EXPS if any(e.startswith(x) for x in extra if not x.startswith("-"))] or EXPS
    flags = [x for x in extra if x.startswith("-")]
    for e in only:
        t0 = time.time()
        print(f"\n######## {e} ########", flush=True)
        r = subprocess.run([sys.executable, str(HERE / f"{e}.py"), *flags], cwd=HERE)
        print(f"######## {e}: {'OK' if r.returncode == 0 else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
