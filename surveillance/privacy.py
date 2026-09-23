"""Key requirement 5 - Security & privacy vs system accessibility.

Tiered access model implemented here:
  * Public / operator view : identities hidden (pixelated or blurred heads or
                             whole bodies) - analytics (counting, tracking,
                             vehicle flow) still work on the protected frame.
  * Authorised investigator: holds a secret key and can *exactly* recover the
                             original pixels from the scrambled region.
  * Evidence integrity     : every stored frame carries an HMAC-SHA256 tag so
                             tampering is detectable.

NOTE: KeyedScrambler (block permutation + keyed XOR) is a transparent teaching
implementation of 'reversible privacy masking'. A production system should
encrypt the ROI tiles with an authenticated cipher (e.g. AES-GCM from the
`cryptography` package) and keep keys in an HSM / key vault.
"""
from __future__ import annotations

import hashlib
import hmac

import cv2
import numpy as np


def _clip_box(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))


def head_region(box, frac: float = 0.22):
    """Approximate head/face area of an upright pedestrian box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 + 0.15 * w, y1, x2 - 0.15 * w, y1 + frac * h)


def pixelate(img, box, blocks: int = 8):
    out = img.copy()
    x1, y1, x2, y2 = _clip_box(box, img.shape)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return out
    roi = out[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, blocks), max(1, int(blocks * (y2 - y1) / (x2 - x1)))),
                       interpolation=cv2.INTER_AREA)
    out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def blur_region(img, box, k: int = 31):
    out = img.copy()
    x1, y1, x2, y2 = _clip_box(box, img.shape)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return out
    out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k | 1, k | 1), 0)
    return out


def anonymize(img, dets, region: str = "head", method: str = "pixelate", mask=None):
    """Hide identities of detected persons (vehicles: optional plate area).

    region: 'head' | 'body' | 'mask' (use a segmentation mask -> silhouette only)
    """
    out = img.copy()
    if region == "mask" and mask is not None:
        blurred = cv2.GaussianBlur(img, (41, 41), 0)
        out[mask] = blurred[mask]
        return out
    for d in dets:
        if d.label != "person":
            continue
        box = head_region(d.box) if region == "head" else d.box
        out = pixelate(out, box) if method == "pixelate" else blur_region(out, box)
    return out


class KeyedScrambler:
    """Reversible privacy masking: permute 8x8 blocks and XOR with a keystream
    derived from (key, frame_id, box). Without the key the region is noise;
    with the key the original is recovered bit-exactly (PSNR = inf)."""

    def __init__(self, key: bytes, block: int = 8):
        self.key, self.block = key, block

    def _rng(self, frame_id: int, box):
        seed = hashlib.sha256(self.key + f"{frame_id}:{box}".encode()).digest()
        return np.random.default_rng(int.from_bytes(seed[:8], "little"))

    def _region(self, img, box):
        x1, y1, x2, y2 = _clip_box(box, img.shape)
        b = self.block
        x2 = x1 + ((x2 - x1) // b) * b
        y2 = y1 + ((y2 - y1) // b) * b
        return x1, y1, x2, y2

    def scramble(self, img, boxes, frame_id: int = 0):
        out = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = self._region(img, box)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]
            b = self.block
            hb, wb = roi.shape[0] // b, roi.shape[1] // b
            blocks = roi.reshape(hb, b, wb, b, -1).transpose(0, 2, 1, 3, 4).reshape(hb * wb, b, b, -1)
            rng = self._rng(frame_id, (x1, y1, x2, y2))
            perm = rng.permutation(hb * wb)
            ks = rng.integers(0, 256, blocks.shape, dtype=np.uint8)
            blocks = blocks[perm] ^ ks
            out[y1:y2, x1:x2] = blocks.reshape(hb, wb, b, b, -1).transpose(0, 2, 1, 3, 4).reshape(roi.shape)
        return out

    def unscramble(self, img, boxes, frame_id: int = 0):
        out = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = self._region(img, box)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]
            b = self.block
            hb, wb = roi.shape[0] // b, roi.shape[1] // b
            blocks = roi.reshape(hb, b, wb, b, -1).transpose(0, 2, 1, 3, 4).reshape(hb * wb, b, b, -1)
            rng = self._rng(frame_id, (x1, y1, x2, y2))
            perm = rng.permutation(hb * wb)
            ks = rng.integers(0, 256, blocks.shape, dtype=np.uint8)
            restored = np.empty_like(blocks)
            restored[perm] = blocks ^ ks
            out[y1:y2, x1:x2] = restored.reshape(hb, wb, b, b, -1).transpose(0, 2, 1, 3, 4).reshape(roi.shape)
        return out


def sign_frame(img, key: bytes, frame_id: int = 0) -> str:
    """HMAC-SHA256 tag over the raw pixels + frame id (tamper evidence)."""
    return hmac.new(key, frame_id.to_bytes(8, "little") + img.tobytes(), hashlib.sha256).hexdigest()


def verify_frame(img, key: bytes, tag: str, frame_id: int = 0) -> bool:
    return hmac.compare_digest(sign_frame(img, key, frame_id), tag)
