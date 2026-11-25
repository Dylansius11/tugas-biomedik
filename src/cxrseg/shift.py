"""Label-free domain shift alarms (A1, A2) and deployment readiness rules (docs/PROTOCOL.md section 8)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _image_hist_256(img_u8: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_u8)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {arr.dtype}")
    if arr.size == 0:
        raise ValueError("Cannot compute histogram of an empty array")
    return np.bincount(arr.reshape(-1), minlength=256).astype(np.float64) / float(arr.size)


def pooled_hist(images_u8: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Compute a combined 256-bin normalized histogram across training images after a scenario."""
    arr = np.asarray(images_u8)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 images, got {arr.dtype}")
    if arr.size == 0:
        raise ValueError("Cannot compute pooled histogram from empty images")
    counts = np.bincount(arr.reshape(-1), minlength=256).astype(np.float64)
    return counts / float(counts.sum())


def intensity_distance(img_u8: np.ndarray, reference_hist: np.ndarray) -> float:
    """A1: Wasserstein-1 distance between a 256-bin image histogram and reference_hist.

    Computed as sum(|CDF(img) - CDF(ref)|) / 255.0.
    """
    img_hist = _image_hist_256(img_u8)
    ref = np.asarray(reference_hist, dtype=np.float64)
    if ref.shape != (256,):
        raise ValueError(f"Expected reference_hist of shape (256,), got {ref.shape}")
    ref_sum = ref.sum()
    if ref_sum <= 0:
        raise ValueError("reference_hist must have positive sum")
    ref = ref / ref_sum
    cdf_img = np.cumsum(img_hist)
    cdf_ref = np.cumsum(ref)
    return float(np.abs(cdf_img - cdf_ref).sum() / 255.0)


def mean_entropy(prob_u8: np.ndarray) -> float:
    """A2: Mean pixel-wise binary entropy -(p ln p + (1-p) ln(1-p)).

    Accepts uint8 probabilities in [0, 255] or float probabilities in [0, 1].
    Returns 0.0 for p in {0, 1} and ln(2) (~0.6931) for p = 0.5.
    """
    arr = np.asarray(prob_u8)
    if arr.size == 0:
        raise ValueError("Cannot compute entropy of an empty probability map")
    if arr.dtype == np.uint8:
        p = arr.astype(np.float64) / 255.0
    else:
        p = np.clip(arr.astype(np.float64), 0.0, 1.0)

    ent = np.zeros_like(p, dtype=np.float64)
    mask = (p > 0.0) & (p < 1.0)
    pm = p[mask]
    ent[mask] = -(pm * np.log(pm) + (1.0 - pm) * np.log(1.0 - pm))
    return float(ent.mean())


def alarm_threshold(val_scores: Sequence[float] | np.ndarray, percentile: float = 95) -> float:
    """Threshold tau from validation scores of the two training sources (never the held-out source)."""
    scores = np.asarray(val_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("val_scores must not be empty")
    return float(np.percentile(scores, percentile))


def alarm_rate(scores: Sequence[float] | np.ndarray, tau: float) -> float:
    """Fraction in [0, 1] of images with score > tau."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("scores must not be empty")
    return float((arr > tau).mean())


def readiness_label(
    failure_rate: float,
    median_dice: float,
    in_source_median_dice: float,
    cfg: dict,
) -> str:
    """Label-based ground-truth readiness ("siap" / "belum siap").

    "siap" iff failure_rate <= max_failure_rate AND (in_source_median_dice - median_dice) <= max_median_dice_drop.
    """
    rcfg = cfg["readiness"]
    drop = float(in_source_median_dice) - float(median_dice)
    is_ready = (float(failure_rate) <= float(rcfg["max_failure_rate"])) and (
        drop <= float(rcfg["max_median_dice_drop"])
    )
    return "siap" if is_ready else "belum siap"


def decision_from_alarm(rate: float, cfg: dict) -> str:
    """Label-free deployment decision ("siap" / "belum siap") from alarm fraction rate."""
    max_rate = float(cfg["readiness"]["max_alarm_rate"])
    return "siap" if float(rate) <= max_rate else "belum siap"


def compute_alarm_evaluation(
    scores: Sequence[float] | np.ndarray,
    dice: Sequence[float] | np.ndarray,
    failed: Sequence[bool] | np.ndarray,
    min_failed_for_auroc: int = 10,
) -> dict:
    """Compute Spearman correlation (score vs Dice) and AUROC (score vs failed).

    Returns auroc="tidak dapat dihitung" when fewer than `min_failed_for_auroc`
    failed or non-failed images are present (PROTOCOL section 8).
    """
    s = np.asarray(scores, dtype=np.float64)
    d = np.asarray(dice, dtype=np.float64)
    f = np.asarray(failed, dtype=bool)

    if len(s) < 2 or np.allclose(s, s[0]) or np.allclose(d, d[0]):
        rho, sp_p = float("nan"), float("nan")
    else:
        res = spearmanr(s, d)
        rho, sp_p = float(res.statistic), float(res.pvalue)

    n_fail = int(f.sum())
    n_pass = int((~f).sum())
    if n_fail < min_failed_for_auroc or n_pass < min_failed_for_auroc:
        auroc: float | str = "tidak dapat dihitung"
    else:
        auroc = float(roc_auc_score(f.astype(int), s))

    return {
        "n_images": int(len(s)),
        "n_failed": n_fail,
        "spearman_rho": rho,
        "spearman_p": sp_p,
        "auroc": auroc,
    }
