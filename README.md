# Smart City Surveillance System — CS406 Digital Image Processing (Option 1)

An adaptive image-processing pipeline for CCTV in Islamabad that copes with **noise, poor
lighting, interference, blur and transmission loss**, then **detects, segments, protects and
compresses** pedestrians and vehicles — with every design trade-off measured.

```
camera frame
  │
  ├─[1] DIAGNOSE (blind): noise σ · impulse density · spectral spikes · brightness · lost blocks
  ├─[2] REPAIR @ native res: in-paint lost blocks → (adaptive) median for impulses
  ├─[3] RESIZE (profile)
  ├─[4] RESTORE: notch-reject (periodic) → denoise (Gaussian/bilateral/NLM, gain-aware)
  │               → low-light (gamma+CLAHE) / illumination (homomorphic, per camera)
  ├─[5] RECOGNISE: YOLOv4-tiny (OpenCV DNN) or HOG+SVM  → persons, cars, buses, trucks, bikes
  ├─[6] SEGMENT: marker watershed inside boxes (GrabCut/Otsu/edges selectable), or MOG2 motion masks
  ├─[7] PROTECT: pixelate heads | blur silhouettes | keyed reversible scrambling + HMAC
  └─[8] COMPRESS: ROI JPEG (people/vehicles q90, background q15–40)
```

Only the steps the diagnostics call for are executed, and a **profile** decides how expensive
each one may be:

| profile  | resolution | impulse | denoise | periodic | illumination | detector | segmentation |
|----------|-----------:|---------|---------|:--------:|--------------|----------|--------------|
| fast     | 0.5×  | median 3×3 | Gaussian | – | gamma | YOLO@320, every 3rd frame | MOG2 masks |
| balanced | 0.75× | median 3×3 | bilateral | ✓ | gamma+CLAHE | YOLO@416 | watershed in box |
| quality  | 1.0×  | adaptive median | NLM + unsharp | ✓ | + homomorphic | YOLO@608 | watershed in box |

---

## 1. Setup (5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py --all                  # ~150 MB, verified, with fallbacks
python -m pytest -q tests                              # 17 correctness tests
```

No GPU and no PyTorch are needed. Tested with Python 3.12 / OpenCV 4.13 on Linux and Windows.

> **OpenCV must be 4.x.** OpenCV 5.0 (2026) removed `HOGDescriptor` and the Darknet loader used by
> YOLOv4-tiny. If you see `module 'cv2.dnn' has no attribute 'readNetFromDarknet'`, run
> `python -m pip uninstall -y opencv-python` then `python -m pip install "opencv-python>=4.8,<5"`.

## 2. Datasets — how they are obtained reliably

| Data | Used for | Source (primary → fallback) | Size | Verification |
|------|----------|-----------------------------|-----:|--------------|
| **Penn-Fudan Pedestrian** (170 images, 345 people, pixel masks) | PSNR/SSIM references, IoU/Dice, P/R/F1 | UPenn server → GitHub mirror (`swallan/PennFudanPed`) → manual Kaggle | 51 MB | 170/170/170 files |
| **CAVIAR** (brief ref. [4]) real CCTV, hand-labelled boxes | video detection & motion segmentation | Edinburgh `homepages` host → `groups` mirror | 6–30 MB / sequence | XML parses, >100 frames |
| **KITTI street clip** (brief ref. [3]) | vehicle robustness, demo | Ultralytics assets release (GitHub) | 6 MB | opens in OpenCV |
| **vtest.avi** (OpenCV sample) | real-time benchmark, demo | OpenCV GitHub | 8 MB | opens in OpenCV |
| **YOLOv4-tiny** COCO weights | recognition | AlexeyAB/darknet GitHub release | 24 MB | MD5 `8911bf80…` |

Everything downloads without an account. Cityscapes / KITTI-object (12 GB) need registration and
are therefore not required; the Lahore Traffic dataset in the brief is tabular sensor data, not images.

**Why synthetic degradations?** PSNR/SSIM need a clean reference that real night footage cannot
provide. Clean frames are degraded with the standard model *g = H[f] + η* using seeded,
documented parameters (`surveillance/degradation.py`), so every number is reproducible.

## 3. Mapping to the project brief

| Brief item | Where | Evidence |
|------------|-------|----------|
| Spatial filtering (mean, median, Laplacian) | `spatial.py` (from scratch + OpenCV) | exp1 |
| Frequency-domain filtering (Fourier) | `frequency.py`: ideal/Butterworth/Gaussian LPF/HPF, auto notch, homomorphic | exp2 |
| Restoration & reconstruction | `restoration.py`: inverse, Wiener, CLS, HE/CLAHE/gamma, in-painting | exp3 |
| Segmentation (edge & region) | `segmentation.py`: Canny/Sobel+morphology, Otsu, region growing, watershed, GrabCut, MOG2 | exp4 |
| Compression (JPEG, wavelets) | `compression.py`: libjpeg, from-scratch DCT codec, DWT codec, ROI coding | exp5 |
| Object recognition (YOLO optional) | `recognition.py`: HOG+SVM, YOLOv4-tiny, optional Ultralytics | exp6 |
| PSNR, SSIM, IoU, Dice, P/R/F1, speed | `metrics.py` (checked against scikit-image) | all |
| Real-time vs quality · resources | `pipeline.py` profiles | exp7 |
| Noise removal vs detail | adaptive median, bilateral/NLM, gain-aware trigger | exp1, exp3b |
| Compression vs fidelity | RD curves + detection-vs-quality + ROI coding | exp5 |
| Security & privacy vs accessibility | `privacy.py`: anonymisation, keyed scrambling, HMAC | exp6c, demo |

## 4. Running

```bash
python experiments/run_all.py --quick     # smoke test, ~5 min
python experiments/run_all.py             # full results, ~40 min on one core
python experiments/run_all.py exp6        # a single experiment
python demo.py --source data/videos/vtest.avi --degrade night_compound --profile balanced
python demo.py --source 0 --show          # webcam, live window
```

Outputs: `results/tables/*.csv|.md|.tex` (LaTeX tables drop straight into Overleaf) and
`results/figures/*.png|.pdf`. The demo writes a side-by-side MP4 plus a JSON log with per-frame
actions, detections, timings and (with `--key`) an HMAC-SHA256 tag per frame.

## 5. Repository layout

```
surveillance/        the library (one module per technical task)
  config.py          paths, dataset URLs, checksums
  degradation.py     noise / blur / low-light / interference / packet-loss models
  spatial.py         Task 1      frequency.py   Task 2      restoration.py  Task 3 + diagnostics
  segmentation.py    Task 4      compression.py Task 5      recognition.py  Task 6
  privacy.py         security & privacy      metrics.py     all evaluation metrics
  pipeline.py        the integrated adaptive system    datasets.py   loaders
experiments/         exp0 … exp7 + run_all.py (every table/figure in the report)
scripts/download_data.py
tests/test_core.py   from-scratch code vs OpenCV / scikit-image
demo.py              live / video demo for the presentation
```

## 6. Results (full run, single CPU core; all numbers reproducible with `run_all.py`)

**Problem analysis — can each fault be detected blindly?** (exp0, 60 images) Gaussian noise,
impulse noise, periodic interference, night exposure and packet loss are flagged in **100 %** of
degraded frames, with ≤ 5 % false alarms on clean frames. Periodic interference of amplitude ≥ 20
is found 98–100 % of the time. Uneven illumination is *not* separable (see limitations).

**Restoration quality** (PSNR dB, mean over 30–40 images)

| fault | degraded | best method | restored |
|-------|---------:|-------------|---------:|
| salt & pepper 20 % | 11.9 | adaptive median (median 3×3: 25.0) | **29.4** |
| periodic interference | 22.3 | auto notch-reject (best global LPF: 23.4) | **34.8** |
| motion blur 15 px + noise | 22.5 | Wiener K=0.01 (inverse filter: 6.7) | **25.5** |
| night / low light | 8.6 | denoise → gamma+CLAHE (reverse order: 16.1) | **18.4** |
| uneven illumination | 16.8 / SSIM 0.902 | homomorphic (CLAHE SSIM 0.859) | 18.2 / **0.922** |
| 3 % blocks lost | 21.6 | Telea in-painting | **35.0** |

Trade-off highlights: at σ = 30 a 1 ms Gaussian blur equals NLM (≈330 ms) in PSNR; our
from-scratch filters are bit-identical to OpenCV but 10–400× slower (exp1b).

**Segmentation** (exp4, 100 images, IoU with ground-truth / YOLO boxes)

| baseline box | Canny+morph (edge) | Otsu (region) | region growing | **watershed** | GrabCut |
|---:|---:|---:|---:|---:|---:|
| 0.50 / 0.47 | 0.29 / 0.28 | 0.43 / 0.41 | 0.54 / 0.52 | **0.69 / 0.63** (13 ms) | 0.64 / 0.59 (1.4 s) |

Region-based methods beat edge-based ones on cluttered streets; watershed is the default.
Denoising before segmentation did **not** help consistently (NLM erases the weak edges Canny needs).

**Compression** (exp5): at 1 bpp libjpeg 30.7 dB, our DCT codec 29.1 dB (same quantiser; the gap
is the entropy coder), bior4.4 wavelet 29.0 dB, Haar 28.0 dB. Pedestrian F1 is unchanged down to
JPEG q = 40 (0.95 bpp, 25× smaller than raw) and collapses below q = 10. ROI coding gives only
+0.5 dB on people in Penn-Fudan close-ups but **+7.2 dB on people at equal file size in the wide
CCTV view** where people cover 4 % of the frame (exp5d).

**Recognition** (exp6a, all 170 images, 423 mask instances)

| detector | Precision | Recall | F1 | AP50 | ms/img |
|----------|---------:|------:|---:|-----:|------:|
| HOG + linear SVM | 0.666 | 0.508 | 0.576 | 0.427 | 59 |
| YOLOv4-tiny @320 | 0.838 | 0.939 | 0.885 | 0.917 | 123 |
| **YOLOv4-tiny @416** | 0.855 | 0.960 | **0.904** | **0.926** | 120 |
| YOLOv4-tiny @608 | 0.724 | 0.917 | 0.809 | 0.807 | 228 |

**Module interdependence — restoration → recognition** (exp6b, pedestrian F1, 30 images)

| scenario | raw | + balanced | + quality |
|----------|----:|-----------:|----------:|
| clean | 0.900 | 0.917 | 0.906 |
| Gaussian σ=30 | 0.803 | 0.828 | **0.867** |
| salt & pepper 20 % | 0.624 | **0.917** | 0.910 |
| periodic | 0.634 | 0.885 | **0.890** |
| motion blur | 0.706 | **0.750** | 0.706 |
| night compound | 0.509 | 0.722 | **0.733** |

Vehicles on the KITTI clip (pseudo-GT): Gaussian 0.65 → 0.79, S&P 0.71 → 0.84, periodic 0.57 → 0.81.

**Privacy vs accessibility** (exp6c): pixelating heads keeps pedestrian F1 at 0.889 (0.900
unprotected), so counting/tracking still work on the public stream; pixelating whole bodies drops
it to 0.026. Keyed scrambling restores the original bit-exactly for key holders.

**Real-time vs quality** (exp7, vtest.avi 768×576, one CPU core, full pipeline incl. privacy + ROI coding)

| profile | FPS clean | FPS night+noise | restore ms | detect ms |
|---------|---------:|----------------:|-----------:|----------:|
| fast     | **16.9** | **15.3** | 5 | 25 |
| balanced | 4.1 | 3.9 | 97–104 | 115 |
| quality  | 2.3 | 0.7 | 176–932 | 232 |

The detector input size is the main speed knob (224 px: 22 FPS → 608 px: 4.2 FPS). Peak memory is
≈ 1 GB including the network, within a Jetson-/Raspberry-Pi-5-class budget. In the balanced
profile most of the clean-stream restore time is the per-frame FFT used for interference
detection; running that check every N frames is an easy further speed-up.



## 7. Known limitations (worth stating in the report)

* **Uneven illumination is not reliably auto-detectable** from frame statistics (exp0): shadows in
  clean scenes look the same. It is therefore a per-camera setting (`illumination="always"`).
* **Deblurring needs the PSF.** Wiener is excellent with the right blur length but loses 2–4 dB when
  it is ±20 % wrong (exp3a); blind PSF estimation is future work.
* The noise estimator under-reads heavy noise (≈19.5 for true σ=30) because of clipping; the
  thresholds were calibrated on the estimator's actual output, not on the true σ.
* `KeyedScrambler` is a transparent teaching implementation; production should encrypt ROI tiles
  with AES-GCM and keep keys in a vault.
* Penn-Fudan does not label small/occluded people, so unmatched detections under 90 px tall are
  ignored (documented in `experiments/exp6_recognition.py`). Vehicle numbers use pseudo ground
  truth (YOLO@608 on the clean clip) because vehicle labels need registered datasets.

## 8. Data credits

Penn-Fudan: Wang, Shi, Song, Shen, "Object detection combining recognition and segmentation", ACCV 2007.
CAVIAR: EC Funded CAVIAR project/IST 2001 37540, http://homepages.inf.ed.ac.uk/rbf/CAVIAR/ (CC BY-SA).
KITTI: Geiger, Lenz, Urtasun, CVPR 2012. YOLOv4-tiny: Bochkovskiy et al., 2020 (AlexeyAB/darknet).
