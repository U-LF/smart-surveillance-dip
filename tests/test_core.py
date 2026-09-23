"""Correctness tests: every from-scratch implementation is checked against a
reference library (OpenCV / scikit-image) or an exact mathematical property.
Run:  python -m pytest -q   (or: python tests/test_core.py)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import compression as C  # noqa: E402
from surveillance import degradation as D  # noqa: E402
from surveillance import frequency as F  # noqa: E402
from surveillance import metrics as M  # noqa: E402
from surveillance import privacy as PV  # noqa: E402
from surveillance import restoration as R  # noqa: E402
from surveillance import segmentation as SG  # noqa: E402
from surveillance import spatial as SP  # noqa: E402
from surveillance.datasets import parse_cvml  # noqa: E402

RNG = np.random.default_rng(0)


def _test_image(h=96, w=128):
    """Deterministic synthetic scene with edges, gradients and texture."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.dstack([(x * 2) % 256, (y * 2) % 256, ((x + y) * 1.5) % 256]).astype(np.float64)
    cv2.rectangle(img, (20, 20), (60, 70), (200, 40, 40), -1)
    cv2.circle(img, (95, 50), 18, (30, 220, 90), -1)
    return img.astype(np.uint8)


IMG = _test_image()
GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)


def test_psnr_ssim_match_skimage():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    noisy = D.add_gaussian_noise(GRAY, 15, RNG)
    assert abs(M.psnr(GRAY, noisy) - peak_signal_noise_ratio(GRAY, noisy, data_range=255)) < 1e-6
    ref = structural_similarity(GRAY, noisy, data_range=255, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
    assert abs(M.ssim(GRAY, noisy) - ref) < 0.01
    assert M.psnr(GRAY, GRAY) == float("inf")


def test_mask_and_box_metrics():
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[:5, :5] = True
    assert M.iou_mask(a, b) == 0.5 and abs(M.dice_mask(a, b) - 2 / 3) < 1e-9
    assert M.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150
    m = M.match_detections([((0, 0, 10, 10), 0.9), ((50, 50, 60, 60), 0.8)], [(1, 1, 10, 10)])
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)


def test_convolution_matches_filter2d():
    k = RNG.normal(size=(5, 5))
    ours = SP.convolve2d(GRAY, k)
    ref = cv2.filter2D(GRAY.astype(np.float64), -1, k, borderType=cv2.BORDER_REFLECT_101)
    assert np.allclose(ours, ref, atol=1e-8)


def test_mean_median_numpy_equal_opencv():
    assert np.abs(SP.mean_filter(IMG, 5, "numpy").astype(int) - SP.mean_filter(IMG, 5).astype(int)).max() <= 1
    sp = D.add_salt_pepper(GRAY, 0.1, rng=RNG)
    assert np.array_equal(SP.median_filter(sp, 3, "numpy"), SP.median_filter(sp, 3))


def test_adaptive_median_beats_median_on_dense_impulses():
    sp = D.add_salt_pepper(IMG, 0.3, rng=RNG)
    assert M.psnr(IMG, SP.adaptive_median_filter(sp, 7)) > M.psnr(IMG, SP.median_filter(sp, 3))


def test_otsu_matches_opencv():
    t_cv, _ = cv2.threshold(GRAY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert abs(SG.otsu_threshold(GRAY) - t_cv) <= 1


def test_histogram_equalization_matches_opencv():
    dark = (GRAY * 0.3).astype(np.uint8)
    assert np.abs(R.histogram_equalization(dark).astype(int) - cv2.equalizeHist(dark).astype(int)).max() <= 1


def test_frequency_identity_and_lowpass():
    assert np.abs(F.apply_filter(GRAY, lambda s: np.ones(s)).astype(int) - GRAY.astype(int)).max() <= 1
    noisy = D.add_gaussian_noise(GRAY, 25, RNG)
    assert M.psnr(GRAY, F.frequency_lowpass(noisy, 0.15)) > M.psnr(GRAY, noisy)


def test_periodic_noise_detected_and_removed():
    big = cv2.resize(IMG, (256, 192))
    noisy = D.add_periodic_noise(big, 50, freqs=((0.15, 0.0),))
    out, peaks = F.remove_periodic_noise(noisy)
    assert peaks and M.psnr(big, out) > M.psnr(big, noisy) + 3


def test_wiener_beats_inverse_under_noise():
    psf = D.motion_psf(9, 0)
    g = D.add_motion_blur(GRAY, 9, 0, 2.0, RNG)
    assert M.psnr(GRAY, R.wiener_filter(g, psf, 0.01)) > M.psnr(GRAY, R.inverse_filter(g, psf))


def test_noise_estimator_tracks_sigma():
    flat = np.full((200, 200), 128, np.uint8)
    for s in (5, 15, 30):
        est = R.estimate_noise_sigma(D.add_gaussian_noise(flat, s, RNG))
        assert abs(est - s) / s < 0.35


def test_dct_codec_roundtrip_and_monotonic_quality():
    lo, n_lo = C.DCTCodec(20).roundtrip(IMG)
    hi, n_hi = C.DCTCodec(90).roundtrip(IMG)
    assert M.psnr(IMG, hi) > M.psnr(IMG, lo) and n_hi > n_lo
    # quantisation stage should be close to libjpeg at the same quality
    ref, _ = C.jpeg_roundtrip(IMG, 90)
    assert abs(M.psnr(IMG, hi) - M.psnr(IMG, ref)) < 3.0


def test_wavelet_codec_rate_distortion():
    fine, n1 = C.WaveletCodec(delta=4).roundtrip(IMG)
    coarse, n2 = C.WaveletCodec(delta=40).roundtrip(IMG)
    assert n2 < n1 and M.psnr(IMG, fine) > M.psnr(IMG, coarse) and M.psnr(IMG, fine) > 35


def test_scrambler_is_exactly_reversible_and_keyed():
    s = PV.KeyedScrambler(b"k1")
    boxes = [(10, 10, 70, 80)]
    enc = s.scramble(IMG, boxes, 7)
    assert not np.array_equal(enc, IMG)
    assert np.array_equal(s.unscramble(enc, boxes, 7), IMG)
    assert not np.array_equal(PV.KeyedScrambler(b"wrong").unscramble(enc, boxes, 7), IMG)
    tag = PV.sign_frame(IMG, b"k1")
    tampered = IMG.copy(); tampered[0, 0, 0] ^= 1
    assert PV.verify_frame(IMG, b"k1", tag) and not PV.verify_frame(tampered, b"k1", tag)


def test_block_loss_reconstruction():
    big = cv2.resize(IMG, (256, 192))
    lost, mask = D.block_loss(big, 0.05, rng=RNG)
    assert (R.lost_block_mask(lost) == mask).all()
    assert M.psnr(big, R.reconstruct_lost_blocks(lost)) > M.psnr(big, lost) + 5


def test_cvml_parser(tmp_path=None):
    import tempfile
    xml = """<?xml version="1.0"?><dataset name="t">
      <frame number="0"><objectlist>
        <object id="0"><orientation>90</orientation><box h="40" w="20" xc="100" yc="50"/></object>
      </objectlist><grouplist/></frame>
      <frame number="1"><objectlist>
        <object id="0"><box h="40" w="20" x="102" y="50"/></object>
        <object id="1"><box h="10" w="10" xc="10" yc="10"/></object>
      </objectlist></frame></dataset>"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
    gt = parse_cvml(f.name)
    assert gt[0] == [(90.0, 30.0, 110.0, 70.0)] and len(gt[1]) == 2 and gt[1][0][0] == 92.0


def test_segmentation_methods_return_masks():
    box = [(20, 20, 60, 70)]
    for m in SG.ROI_METHODS:
        mask = SG.segment_people(IMG, box, m)
        assert mask.shape == IMG.shape[:2] and mask.dtype == bool


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
