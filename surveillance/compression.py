"""Technical Task 5 - Image compression (JPEG and wavelets).

  jpeg_*      : baseline JPEG through libjpeg (OpenCV) - the production codec.
  DCTCodec    : a JPEG-like codec written from scratch (YCbCr, 4:2:0 chroma
                subsampling, 8x8 block DCT, IJG-scaled quantisation tables,
                zig-zag scan, zlib as the entropy coder) - shows every stage.
  WaveletCodec: multi-level 2-D DWT (PyWavelets) + dead-zone uniform
                quantisation + zlib - the JPEG-2000 idea.
  roi_*       : region-of-interest coding - people/vehicles at high quality,
                background at low quality (compression vs fidelity trade-off
                resolved *per region* instead of per frame).
Every codec returns real compressed byte counts so bits-per-pixel (bpp) and
compression ratios are measured, not estimated.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

try:
    import pywt
except ImportError:  # wavelet codec becomes unavailable, everything else works
    pywt = None


def bpp(n_bytes: int, shape) -> float:
    return 8.0 * n_bytes / (shape[0] * shape[1])


# ----------------------------------------------------------------------------
# Production JPEG
# ----------------------------------------------------------------------------
def jpeg_encode(img, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return buf.tobytes()


def jpeg_decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def jpeg_roundtrip(img, quality: int = 75):
    data = jpeg_encode(img, quality)
    return jpeg_decode(data), len(data)


# ----------------------------------------------------------------------------
# From-scratch DCT codec
# ----------------------------------------------------------------------------
Q_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float64)
Q_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], np.float64)


def _dct_matrix(n: int = 8) -> np.ndarray:
    """Orthonormal DCT-II basis C, so that DCT2(B) = C B C^T."""
    C = np.zeros((n, n))
    for k in range(n):
        a = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        C[k] = a * np.cos(np.pi * (2 * np.arange(n) + 1) * k / (2 * n))
    return C


_C = _dct_matrix()


def _zigzag_order(n: int = 8) -> np.ndarray:
    idx = sorted(((i, j) for i in range(n) for j in range(n)),
                 key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else p[0]))
    return np.array([i * n + j for i, j in idx])


_ZZ = _zigzag_order()


def _scaled_table(base: np.ndarray, quality: int) -> np.ndarray:
    """IJG quality scaling (the one libjpeg uses)."""
    q = int(np.clip(quality, 1, 100))
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((base * s + 50) / 100), 1, 255)


def _blockify(ch):
    h, w = ch.shape
    return ch.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


def _unblockify(blocks):
    hb, wb = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(hb * 8, wb * 8)


class DCTCodec:
    def __init__(self, quality: int = 75, subsample: bool = True):
        self.quality, self.subsample = quality, subsample
        self.tq_l = _scaled_table(Q_LUMA, quality)
        self.tq_c = _scaled_table(Q_CHROMA, quality)

    @staticmethod
    def _pad8(ch):
        h, w = ch.shape
        return np.pad(ch, ((0, (-h) % 8), (0, (-w) % 8)), mode="edge")

    def _encode_channel(self, ch, table):
        ch = self._pad8(ch.astype(np.float64) - 128.0)
        B = _blockify(ch)
        D = _C @ B @ _C.T                                  # forward 2-D DCT per block
        Q = np.round(D / table).astype(np.int16)           # quantisation (the lossy step)
        zz = Q.reshape(Q.shape[0], Q.shape[1], 64)[..., _ZZ]  # zig-zag scan
        return zz, ch.shape

    def _decode_channel(self, zz, padded_shape, table, out_shape):
        flat = np.zeros_like(zz)
        flat[..., _ZZ] = zz
        Q = flat.reshape(zz.shape[0], zz.shape[1], 8, 8).astype(np.float64)
        B = _C.T @ (Q * table) @ _C                        # dequantise + inverse DCT
        ch = _unblockify(B)[:out_shape[0], :out_shape[1]] + 128.0
        return ch

    def encode(self, img):
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        h, w = ycc.shape[:2]
        chans = [(ycc[..., 0], self.tq_l, (h, w))]
        for c in (1, 2):
            ch = ycc[..., c]
            if self.subsample:
                ch = cv2.resize(ch, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
            chans.append((ch, self.tq_c, ch.shape))
        streams, meta = [], []
        for ch, table, shp in chans:
            zz, pshape = self._encode_channel(ch, table)
            streams.append(zz)
            meta.append((zz.shape, pshape, shp))
        payload = zlib.compress(b"".join(self._serialize(s) for s in streams), 9)
        return payload, (meta, (h, w))

    @staticmethod
    def _serialize(zz):
        """Entropy-coding front end (stands in for JPEG's Huffman stage):
        DC coefficients are DPCM-coded, and coefficients are written
        *plane by plane* (all blocks' k-th zig-zag coefficient together) so the
        high-frequency zeros form very long runs that zlib's LZ77+Huffman
        compresses almost for free."""
        planes = zz.reshape(-1, 64).T.copy()           # (64, n_blocks)
        planes[0] = np.diff(planes[0], prepend=0)      # DPCM on DC
        return planes.astype(np.int16).tobytes()

    @staticmethod
    def _deserialize(raw, zshape):
        planes = raw.reshape(64, -1).copy()
        planes[0] = np.cumsum(planes[0])
        return planes.T.reshape(zshape)

    def decode(self, payload, header):
        meta, (h, w) = header
        raw = np.frombuffer(zlib.decompress(payload), np.int16)
        chans, off = [], 0
        for i, (zshape, pshape, shp) in enumerate(meta):
            n = int(np.prod(zshape))
            zz = self._deserialize(raw[off:off + n], zshape)
            off += n
            table = self.tq_l if i == 0 else self.tq_c
            ch = self._decode_channel(zz, pshape, table, shp)
            if ch.shape != (h, w):
                ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR)
            chans.append(ch)
        ycc = np.clip(np.rint(np.dstack(chans)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    def roundtrip(self, img):
        payload, header = self.encode(img)
        return self.decode(payload, header), len(payload)


# ----------------------------------------------------------------------------
# Wavelet codec
# ----------------------------------------------------------------------------
class WaveletCodec:
    """DWT -> dead-zone quantiser (step `delta`, finer for the approximation
    band) -> int16 -> zlib. Larger delta = smaller file, lower quality."""

    def __init__(self, wavelet: str = "bior4.4", level: int = 3, delta: float = 20.0,
                 chroma_factor: float = 2.0):
        if pywt is None:
            raise ImportError("PyWavelets is required: pip install PyWavelets")
        self.wavelet, self.level, self.delta, self.cf = wavelet, level, delta, chroma_factor

    def _q(self, c, d):
        return (np.sign(c) * np.floor(np.abs(c) / d)).astype(np.int32)

    def _dq(self, q, d):
        return np.sign(q) * (np.abs(q) + 0.5) * d * (q != 0)

    def encode(self, img):
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float64)
        parts, header = [], []
        for c in range(3):
            d = self.delta * (1.0 if c == 0 else self.cf)
            coeffs = pywt.wavedec2(ycc[..., c], self.wavelet, level=self.level, mode="periodization")
            arr, slices = pywt.coeffs_to_array(coeffs)
            q = self._q(arr, d)
            a_sl = slices[0]  # approximation band -> 4x finer step, it carries most energy
            q[a_sl] = self._q(arr[a_sl], d / 4)
            parts.append(np.clip(q, -32768, 32767).astype(np.int16))
            header.append((arr.shape, slices, d))
        payload = zlib.compress(b"".join(p.tobytes() for p in parts), 9)
        return payload, (header, img.shape[:2])

    def decode(self, payload, hdr):
        header, (h, w) = hdr
        raw = np.frombuffer(zlib.decompress(payload), np.int16).astype(np.int32)
        chans, off = [], 0
        for shape, slices, d in header:
            n = int(np.prod(shape))
            q = raw[off:off + n].reshape(shape)
            off += n
            arr = self._dq(q, d).astype(np.float64)
            a_sl = slices[0]
            arr[a_sl] = self._dq(q[a_sl], d / 4)
            coeffs = pywt.array_to_coeffs(arr, slices, output_format="wavedec2")
            chans.append(pywt.waverec2(coeffs, self.wavelet, mode="periodization")[:h, :w])
        ycc = np.clip(np.rint(np.dstack(chans)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    def roundtrip(self, img):
        payload, header = self.encode(img)
        return self.decode(payload, header), len(payload)


# ----------------------------------------------------------------------------
# ROI-aware coding
# ----------------------------------------------------------------------------
def roi_mask_from_boxes(shape, boxes, pad: int = 4):
    m = np.zeros(shape[:2], bool)
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(y1) - pad):int(y2) + pad, max(0, int(x1) - pad):int(x2) + pad] = True
    return m


def roi_jpeg(img, boxes, q_roi: int = 90, q_bg: int = 25):
    """Background frame at q_bg + each ROI crop at q_roi (sent as separate
    JPEG tiles, like a surveillance 'smart codec'). Returns (recon, total_bytes)."""
    recon, total = jpeg_roundtrip(img, q_bg)
    recon = recon.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        tile, n = jpeg_roundtrip(img[y1:y2, x1:x2], q_roi)
        recon[y1:y2, x1:x2] = tile
        total += n + 8  # + 8 bytes for tile coordinates
    return recon, total
