"""Dataset loaders.

PennFudan : 170 street/campus images, 345 pedestrians, pixel-accurate instance
            masks + boxes  -> IoU/Dice (segmentation), P/R/F1 (recognition),
            and clean references for PSNR/SSIM after controlled degradation.
CAVIAR    : real CCTV sequences (384x288 @ 25 fps) with hand-labelled boxes in
            CVML XML (listed in the project brief, ref. [4]) -> video,
            background subtraction, real-time evaluation.
Video     : any file / RTSP stream / webcam index via OpenCV.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CAVIAR_DIR, PENNFUDAN_DIR


@dataclass
class Sample:
    name: str
    image: np.ndarray            # BGR uint8
    boxes: list                  # [(x1,y1,x2,y2), ...]
    mask: np.ndarray | None      # bool union mask of all persons (or None)
    instance_mask: np.ndarray | None = None


class PennFudan:
    def __init__(self, root: Path = PENNFUDAN_DIR):
        self.root = Path(root)
        img_dir = self.root / "PNGImages"
        if not img_dir.exists():
            raise FileNotFoundError(f"{img_dir} not found - run: python scripts/download_data.py --pennfudan")
        self.names = sorted(p.stem for p in img_dir.glob("*.png"))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i) -> Sample:
        n = self.names[i]
        img = cv2.imread(str(self.root / "PNGImages" / f"{n}.png"), cv2.IMREAD_COLOR)
        inst = cv2.imread(str(self.root / "PedMasks" / f"{n}_mask.png"), cv2.IMREAD_UNCHANGED)
        if inst.ndim == 3:
            inst = inst[..., 0]
        boxes = []
        for k in np.unique(inst):
            if k == 0:
                continue
            ys, xs = np.where(inst == k)
            boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))
        return Sample(n, img, boxes, inst > 0, inst)

    def subset(self, limit: int | None = None, step: int = 1):
        idx = list(range(0, len(self), step))
        return idx[:limit] if limit else idx


# ----------------------------------------------------------------------------
# CAVIAR
# ----------------------------------------------------------------------------
def parse_cvml(xml_path) -> dict[int, list]:
    """Parse CAVIAR CVML ground truth -> {frame_number: [(x1,y1,x2,y2), ...]}.

    Box tags store the centre and size; the published grammar writes x/y while
    the files use xc/yc, so both spellings are accepted. Only individual
    <object>s are used (group boxes are not persons).
    """
    tree = ET.parse(str(xml_path))
    out = {}
    for fr in tree.getroot().iter("frame"):
        n = int(fr.get("number"))
        boxes = []
        ol = fr.find("objectlist")
        if ol is not None:
            for ob in ol.findall("object"):
                b = ob.find("box")
                if b is None:
                    continue
                xc = float(b.get("xc", b.get("x")))
                yc = float(b.get("yc", b.get("y")))
                w, h = float(b.get("w")), float(b.get("h"))
                boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        out[n] = boxes
    return out


def _trailing_int(p: Path) -> int:
    m = re.search(r"(\d+)$", p.stem)
    return int(m.group(1)) if m else -1


class CaviarSequence:
    """Frames of one CAVIAR sequence + per-frame GT boxes."""

    def __init__(self, name: str, root: Path = CAVIAR_DIR):
        self.dir = Path(root) / name
        frames = sorted(self.dir.rglob("*.jpg"), key=lambda p: (_trailing_int(p), p.name))
        if not frames:
            raise FileNotFoundError(f"No frames in {self.dir} - run: python scripts/download_data.py --caviar")
        self.frames = frames
        xmls = list(self.dir.glob("*.xml"))
        self.gt = parse_cvml(xmls[0]) if xmls else {}
        # JPEG numbering may start at 0 or 1; GT frame numbers start at 0
        self.offset = _trailing_int(frames[0]) if _trailing_int(frames[0]) >= 0 else 0
        self.name = name

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        for i, p in enumerate(self.frames):
            n = _trailing_int(p) - self.offset if _trailing_int(p) >= 0 else i
            yield n, cv2.imread(str(p)), self.gt.get(n, [])


def available_caviar(root: Path = CAVIAR_DIR):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.rglob("*.jpg")))


def iter_video(source, max_frames: int | None = None, scale: float = 1.0):
    """Yield (index, frame) from a file path, RTSP URL or webcam index."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise IOError(f"Cannot open video source {source}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        yield i, frame
        i += 1
    cap.release()
