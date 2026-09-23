#!/usr/bin/env python3
"""Download and verify every dataset / model the project needs.

    python scripts/download_data.py --all            # everything (~190 MB)
    python scripts/download_data.py --pennfudan      # 51 MB, required
    python scripts/download_data.py --models         # YOLOv4-tiny, 24 MB, required
    python scripts/download_data.py --videos         # test videos, ~20 MB
    python scripts/download_data.py --caviar         # 5 CAVIAR sequences, ~60 MB
    python scripts/download_data.py --caviar Walk1 Meet_Crowd
    python scripts/download_data.py --verify         # only check what is on disk

Reliability measures
  * every source has a fallback mirror where one exists (Penn-Fudan: official
    UPenn server -> GitHub mirror);
  * downloads go to a .part file and are renamed only when complete;
  * retries with back-off; resumes are skipped if the verified target exists;
  * integrity: MD5 for model weights, file-count checks for Penn-Fudan
    (170/170/170), XML parse + frame-count checks for CAVIAR.
Only the Python standard library is used, so this runs before `pip install`.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surveillance.config import (CAVIAR_DEFAULT, CAVIAR_DIR, CAVIAR_SEQUENCES, DATA_DIR,  # noqa: E402
                                 MODEL_DIR, MODEL_MD5, MODEL_SOURCES, PENNFUDAN_DIR,
                                 PENNFUDAN_EXPECTED, PENNFUDAN_SOURCES, VIDEO_DIR, VIDEO_SOURCES)

UA = {"User-Agent": "Mozilla/5.0 (CS406 DIP project downloader)"}


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, t0 = 0, time.time()
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(f"\r    {dest.name}: {pct:5.1f}% of {human(total)} "
                              f"({human(done / max(1e-3, time.time() - t0))}/s)", end="", flush=True)
            print()
            if total and done != total:
                raise IOError(f"incomplete download ({done}/{total} bytes)")
            part.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"\n    attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    part.unlink(missing_ok=True)
    return False


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
def verify_pennfudan() -> bool:
    ok = all((PENNFUDAN_DIR / d).is_dir() and len(list((PENNFUDAN_DIR / d).iterdir())) >= n
             for d, n in PENNFUDAN_EXPECTED.items())
    print(f"[{'OK' if ok else '--'}] Penn-Fudan  {PENNFUDAN_DIR}")
    return ok


def get_pennfudan():
    if verify_pennfudan():
        return True
    tmp = DATA_DIR / "_tmp_pennfudan"
    for url in PENNFUDAN_SOURCES:
        print(f"  -> {url}")
        z = tmp / "pf.zip"
        if not fetch(url, z):
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            print("    corrupt archive, trying next mirror")
            continue
        # locate the folder that contains PNGImages (layout differs per mirror)
        root = next((p.parent for p in tmp.rglob("PNGImages") if p.is_dir()), None)
        if root is None:
            continue
        if PENNFUDAN_DIR.exists():
            shutil.rmtree(PENNFUDAN_DIR)
        shutil.move(str(root), str(PENNFUDAN_DIR))
        shutil.rmtree(tmp, ignore_errors=True)
        if verify_pennfudan():
            return True
    print("  !! Penn-Fudan download failed. Manual fallback: download PennFudanPed.zip from\n"
          "     https://www.cis.upenn.edu/~jshi/ped_html/ (or Kaggle 'penn-fudan-database')\n"
          f"     and unzip so that {PENNFUDAN_DIR}/PNGImages exists.")
    return False


def get_models():
    ok_all = True
    for name, url in MODEL_SOURCES.items():
        dest = MODEL_DIR / name
        good = dest.exists() and (name not in MODEL_MD5 or md5(dest) == MODEL_MD5[name])
        if not good:
            print(f"  -> {url}")
            fetch(url, dest)
            good = dest.exists() and (name not in MODEL_MD5 or md5(dest) == MODEL_MD5[name])
        print(f"[{'OK' if good else '!!'}] model {name}")
        ok_all &= good
    return ok_all


def get_videos(names=None):
    for name, url in VIDEO_SOURCES.items():
        if names and name not in names:
            continue
        dest = VIDEO_DIR / name
        if not dest.exists():
            print(f"  -> {url}")
            fetch(url, dest)
        print(f"[{'OK' if dest.exists() else '!!'}] video {name}")


def verify_caviar(name: str) -> bool:
    d = CAVIAR_DIR / name
    n_jpg = len(list(d.rglob("*.jpg"))) if d.exists() else 0
    xml_ok = False
    if d.exists():
        try:
            from surveillance.datasets import parse_cvml
            xmls = list(d.glob("*.xml"))
            xml_ok = bool(xmls) and len(parse_cvml(xmls[0])) > 0
        except Exception:  # noqa: BLE001
            xml_ok = False
    ok = n_jpg > 100 and xml_ok
    print(f"[{'OK' if ok else '--'}] CAVIAR {name}: {n_jpg} frames, GT xml {'ok' if xml_ok else 'missing'}")
    return ok


def get_caviar(names):
    for name in names:
        if name not in CAVIAR_SEQUENCES:
            print(f"  unknown sequence {name}; choose from {list(CAVIAR_SEQUENCES)}")
            continue
        if verify_caviar(name):
            continue
        base, tar_name, xml_name = CAVIAR_SEQUENCES[name]
        d = CAVIAR_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        xml_url, tar_url = f"{base}/{name}/{xml_name}", f"{base}/{name}/{tar_name}"
        # the Edinburgh 'groups' host mirrors the same tree
        alt = lambda u: u.replace("homepages.inf.ed.ac.uk/rbf", "groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR")
        for u in (xml_url, alt(xml_url)):
            if fetch(u, d / xml_name):
                break
        tar_path = d / tar_name
        for u in (tar_url, alt(tar_url)):
            if fetch(u, tar_path):
                break
        if tar_path.exists():
            with tarfile.open(tar_path) as tf:
                members = [m for m in tf.getmembers() if m.name.lower().endswith(".jpg") and ".." not in m.name]
                try:
                    tf.extractall(d / "frames", members=members, filter="data")  # Python >= 3.12
                except TypeError:
                    tf.extractall(d / "frames", members=members)
            tar_path.unlink()
        verify_caviar(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pennfudan", action="store_true")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--caviar", nargs="*", default=None, help="sequence names (default set if empty)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify_pennfudan()
        for n in MODEL_SOURCES:
            p = MODEL_DIR / n
            print(f"[{'OK' if p.exists() else '--'}] model {n}")
        for n in VIDEO_SOURCES:
            print(f"[{'OK' if (VIDEO_DIR / n).exists() else '--'}] video {n}")
        for n in CAVIAR_SEQUENCES:
            if (CAVIAR_DIR / n).exists():
                verify_caviar(n)
        return
    nothing = not (a.all or a.pennfudan or a.models or a.videos or a.caviar is not None)
    if a.all or a.pennfudan or nothing:
        get_pennfudan()
    if a.all or a.models or nothing:
        get_models()
    if a.all or a.videos:
        get_videos()
    if a.all or a.caviar is not None:
        get_caviar(a.caviar or CAVIAR_DEFAULT)


if __name__ == "__main__":
    main()
