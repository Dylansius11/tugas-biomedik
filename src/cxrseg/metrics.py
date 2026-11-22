"""Per-image segmentation metrics (docs/PROTOCOL.md section 7). All inputs are boolean HxW masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _b(x: np.ndarray) -> np.ndarray:
    return x.astype(bool)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _b(pred), _b(gt)
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2.0 * (p & g).sum() / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _b(pred), _b(gt)
    union = (p | g).sum()
    return 1.0 if union == 0 else float((p & g).sum() / union)


def coverage(pred: np.ndarray, gt: np.ndarray) -> float:
    """Share of the true lung that the prediction captures (recall)."""
    p, g = _b(pred), _b(gt)
    return float((p & g).sum() / g.sum())


def area_error_pct(pred: np.ndarray, gt: np.ndarray) -> float:
    """Signed lung area error in percent of the true area. Negative means under-segmentation."""
    p, g = _b(pred), _b(gt)
    return float(100.0 * (int(p.sum()) - int(g.sum())) / g.sum())


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask)


def hd95(pred: np.ndarray, gt: np.ndarray, empty_value: float = 362.0) -> float:
    """95th percentile of symmetric boundary distances, in pixels.

    Returns empty_value when the prediction is empty (maximum possible distance on 256x256).
    """
    p, g = _b(pred), _b(gt)
    if not p.any():
        return float(empty_value)
    sp, sg = _surface(p), _surface(g)
    d_to_g = ndi.distance_transform_edt(~sg)
    d_to_p = ndi.distance_transform_edt(~sp)
    dists = np.concatenate([d_to_g[sp], d_to_p[sg]])
    return float(np.percentile(dists, 95))


def evaluate_pair(pred: np.ndarray, gt: np.ndarray, failure_dice: float = 0.90, hd95_empty: float = 362.0) -> dict:
    d = dice(pred, gt)
    return {
        "dice": d,
        "iou": iou(pred, gt),
        "hd95": hd95(pred, gt, hd95_empty),
        "coverage": coverage(pred, gt),
        "area_error_pct": area_error_pct(pred, gt),
        "failed": bool(d < failure_dice),
    }
