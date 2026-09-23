"""Central configuration: paths, dataset sources and global constants.

Every experiment and script imports its paths from here so the whole project
can be relocated by changing only ROOT (or the SURV_DATA / SURV_RESULTS env vars).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SURV_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SURV_RESULTS", ROOT / "results"))
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

PENNFUDAN_DIR = DATA_DIR / "PennFudanPed"
CAVIAR_DIR = DATA_DIR / "caviar"
VIDEO_DIR = DATA_DIR / "videos"
MODEL_DIR = DATA_DIR / "models"

SEED = 42  # every random degradation is seeded -> results are reproducible

# --------------------------------------------------------------------------
# Dataset / model sources (primary first, then fallbacks)
# --------------------------------------------------------------------------
PENNFUDAN_SOURCES = [
    # Official host (University of Pennsylvania) - used by the PyTorch tutorial
    "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
    # Full mirror of the same 170 images / masks / annotations on GitHub
    "https://codeload.github.com/swallan/PennFudanPed/zip/refs/heads/master",
]
PENNFUDAN_EXPECTED = {"PNGImages": 170, "PedMasks": 170, "Annotation": 170}

CAVIAR_BASE_INRIA = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1"
CAVIAR_BASE_LISBON = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2"
# name -> (base url, jpeg archive, ground-truth xml)
CAVIAR_SEQUENCES = {
    # INRIA lobby (wide-angle, 384x288, 25 fps)
    "Walk1": (CAVIAR_BASE_INRIA, "Walk1_jpg.tar.gz", "wk1gt.xml"),
    "Meet_Crowd": (CAVIAR_BASE_INRIA, "Meet_Crowd_jpg.tar.gz", "mc1gt.xml"),
    "Meet_WalkSplit": (CAVIAR_BASE_INRIA, "Meet_WalkSplit_jpg.tar.gz", "mws1gt.xml"),
    "Fight_Chase": (CAVIAR_BASE_INRIA, "Fight_Chase_jpg.tar.gz", "fcgt.xml"),
    "Browse1": (CAVIAR_BASE_INRIA, "Browse1_jpg.tar.gz", "br1gt.xml"),
    # Lisbon shopping-centre corridor view
    "OneLeaveShop1cor": (CAVIAR_BASE_LISBON, "OneLeaveShop1cor.tar.gz", "cols1gt.xml"),
    "EnterExitCrossingPaths1cor": (CAVIAR_BASE_LISBON, "EnterExitCrossingPaths1cor.tar.gz", "ceecp1gt.xml"),
    "TwoLeaveShop2cor": (CAVIAR_BASE_LISBON, "TwoLeaveShop2cor.tar.gz", "c2ls2gt.xml"),
}
CAVIAR_DEFAULT = ["Walk1", "Meet_Crowd", "Fight_Chase", "OneLeaveShop1cor", "EnterExitCrossingPaths1cor"]

VIDEO_SOURCES = {
    # OpenCV's own pedestrian surveillance clip (768x576, 795 frames, static camera)
    "vtest.avi": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
    # 16.6 s street clip from the KITTI benchmark (brief ref. [3]), 1226x370 @ 30 fps,
    # hosted on the Ultralytics assets release -> cars / trucks / cyclists / pedestrians
    "kitti-inference-vid.mp4":
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti-inference-vid.mp4",
}

MODEL_SOURCES = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights":
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}
MODEL_MD5 = {"yolov4-tiny.weights": "8911bf808aed305d6854c4ea48fcc731"}

# COCO class ids relevant to street surveillance
SURVEILLANCE_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}


def ensure_dirs() -> None:
    for d in (FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
