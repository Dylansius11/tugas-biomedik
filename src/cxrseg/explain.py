"""Explainable alarm (Percobaan 2): says *why* a new source looks unready, not just ready / not ready.

Two kinds of evidence, both compared with the fold's own validation images (never with the held-out source):
1. factors from the image and from the predicted mask, as robust z-scores against validation;
2. a model uncertainty map: per-pixel spread of the probability over small shifts and zooms (test-time augmentation).

A3 score = largest |z| over all factors. Threshold tau = 95th percentile of A3 on validation, the same rule as A1/A2.
A factor is named as a reason when its own |z| is above tau.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from scipy import ndimage as ndi

from .postprocess import postprocess_mask

# id, Indonesian label, text when higher than usual, text when lower than usual
FACTORS = [
    ("brightness", "kecerahan foto", "lebih terang", "lebih gelap"),
    ("contrast", "kontras foto", "lebih tajam kontrasnya", "lebih datar kontrasnya"),
    ("hist_dist", "jarak histogram ke data latih (A1)", "gaya foto jauh berbeda", "-"),
    ("area", "luas paru prediksi", "lebih besar", "lebih kecil"),
    ("bottom", "posisi batas bawah paru", "lebih rendah di foto", "lebih tinggi di foto"),
    ("lr_balance", "keseimbangan paru kiri-kanan", "-", "tidak seimbang"),
    ("fragments", "jumlah potongan mask mentah", "terpecah-pecah", "-"),
    ("entropy", "keraguan rata-rata model (A2)", "model lebih ragu", "-"),
    ("tta_unstable", "bagian paru yang goyah saat foto digeser", "prediksi tidak stabil", "-"),
]
ONE_SIDED = {"hist_dist": 1, "lr_balance": -1, "fragments": 1, "entropy": 1, "tta_unstable": 1}
MIN_SCALE = {"fragments": 0.5}  # a count: on validation it is almost always 2, so MAD is 0; one extra piece = z 2


def hist(x_u8: np.ndarray) -> np.ndarray:
    return np.bincount(x_u8.reshape(-1), minlength=256).astype(float) / x_u8.size


def tta_probs(predict: Callable[[np.ndarray], np.ndarray], x: np.ndarray, shift: int, scales: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of probabilities over identity, 4 shifts and the given zooms, each mapped back to the original frame."""
    h, w = x.shape[1:]
    mats = [np.float32([[1, 0, 0], [0, 1, 0]])]
    mats += [np.float32([[1, 0, dx], [0, 1, dy]]) for dx, dy in ((shift, 0), (-shift, 0), (0, shift), (0, -shift))]
    mats += [cv2.getRotationMatrix2D((w / 2, h / 2), 0, s).astype(np.float32) for s in scales]
    outs = []
    for m in mats:
        xi = np.stack([cv2.warpAffine(i, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0) for i in x])
        p = predict(xi)
        inv = cv2.invertAffineTransform(m)
        outs.append(np.stack([cv2.warpAffine(pi, inv, (w, h), flags=cv2.INTER_LINEAR, borderValue=0) for pi in p]))
    s = np.stack(outs)
    return s.mean(0), s.std(0)


def features(x_u8: np.ndarray, prob: np.ndarray, std: np.ndarray, ref_hist: np.ndarray, unstable_std: float) -> dict:
    """All factors for one image. prob/std are float maps in [0,1] (prob from the plain, un-augmented pass)."""
    m = postprocess_mask(prob)
    h, w = m.shape
    rows = np.where(m.any(1))[0]
    left, right = m[:, : w // 2].sum(), m[:, w // 2 :].sum()
    lab, n = ndi.label(prob >= 0.5)
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) if n else np.array([])
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "brightness": float(x_u8.mean() / 255),
        "contrast": float(x_u8.std() / 255),
        "hist_dist": float(np.abs(np.cumsum(hist(x_u8)) - np.cumsum(ref_hist)).sum() / 255),
        "area": float(m.mean()),
        "bottom": float(rows.max() / h) if len(rows) else 0.0,
        "lr_balance": float(min(left, right) / max(left, right, 1)),
        "fragments": float((sizes >= 50).sum()),
        "entropy": float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean()),
        "tta_unstable": float((std > unstable_std).sum() / max(m.sum(), 1)),
    }


class Reference:
    """Robust centre and spread of every factor on validation images, and the A3 threshold."""

    def __init__(self, val_feats: list[dict], percentile: float = 95):
        self.keys = [f[0] for f in FACTORS]
        v = np.array([[f[k] for k in self.keys] for f in val_feats])
        self.med = np.median(v, 0)
        lo, hi = np.percentile(v, [2.5, 97.5], 0)
        mad = 1.4826 * np.median(np.abs(v - self.med), 0)
        floor = np.array([MIN_SCALE.get(k, 1e-4) for k in self.keys])
        self.scale = np.maximum.reduce([mad, (hi - lo) / 4, floor])
        self.lo, self.hi = lo, hi
        self.tau = float(np.percentile(self.a3(val_feats), percentile))

    def z(self, feats: list[dict]) -> np.ndarray:
        """Signed z (N, factors); one-sided factors are zero on their harmless side."""
        v = np.array([[f[k] for k in self.keys] for f in feats])
        z = (v - self.med) / self.scale
        for j, k in enumerate(self.keys):
            if k in ONE_SIDED:
                z[:, j] = np.where(z[:, j] * ONE_SIDED[k] > 0, z[:, j], 0)
        return z

    def a3(self, feats: list[dict]) -> np.ndarray:
        return np.abs(self.z(feats)).max(1)

    def reasons(self, f: dict) -> list[str]:
        """Plain-language reasons for one image, strongest first."""
        z = self.z([f])[0]
        out = []
        for j in np.argsort(-np.abs(z)):
            if abs(z[j]) <= self.tau:
                break
            k, label, up, down = FACTORS[j]
            out.append(f"{label} {up if z[j] > 0 else down}: {f[k]:.3f} (normal {self.lo[j]:.3f}-{self.hi[j]:.3f}, z={z[j]:+.1f})")
        return out


def source_verdict(ref: Reference, feats: list[dict], max_alarm_rate: float) -> dict:
    """Readiness of a whole new source with the reasons that appear most often."""
    z = np.abs(ref.z(feats))
    alarm = z.max(1) > ref.tau
    share = (z > ref.tau).mean(0)
    order = [j for j in np.argsort(-share) if share[j] > 0.05]
    zs = ref.z(feats)
    reasons = []
    for j in order[:3]:
        k, label, up, down = FACTORS[j]
        flagged = z[:, j] > ref.tau
        med = float(np.median([f[k] for f, fl in zip(feats, flagged) if fl]))
        reasons.append(f"{share[j] * 100:.0f}% foto: {label} {up if np.median(zs[flagged, j]) > 0 else down} (median foto beralarm {med:.3f}, normal {ref.lo[j]:.3f}-{ref.hi[j]:.3f})")
    return {"alarm_rate": float(alarm.mean()), "ready": bool(alarm.mean() <= max_alarm_rate), "reasons": reasons}
