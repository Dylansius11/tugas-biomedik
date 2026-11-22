"""Post-processing of model probabilities (docs/PROTOCOL.md section 6)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

_EIGHT = np.ones((3, 3), dtype=int)


def postprocess_mask(
    prob: np.ndarray,
    threshold: float = 0.5,
    keep_components: int = 2,
    closing_kernel: int = 5,
) -> np.ndarray:
    """Probability map (float in [0,1] or uint8 in [0,255]) -> boolean mask.

    threshold -> keep the largest components -> closing -> fill holes.
    """
    if prob.dtype == np.uint8:
        prob = prob.astype(np.float32) / 255.0
    binary = prob >= threshold
    labels, n = ndi.label(binary, structure=_EIGHT)
    if n > keep_components:
        sizes = ndi.sum(binary, labels, index=np.arange(1, n + 1))
        keep = np.argsort(sizes)[::-1][:keep_components] + 1
        binary = np.isin(labels, keep)
    kernel = np.ones((closing_kernel, closing_kernel), dtype=np.uint8)
    closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return ndi.binary_fill_holes(closed.astype(bool))
