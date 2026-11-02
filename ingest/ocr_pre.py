"""Image preprocessing for hard OCR pages: tables with grid lines, faint or skewed scans.

Pipeline (grayscale in, binary out): deskew -> adaptive threshold -> remove long
horizontal/vertical rules. Pure numpy + PIL so it runs anywhere Tesseract does.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

THRESH_BLOCK = 31  # local-mean window (px at 400 dpi)
THRESH_C = 12.0  # how much darker than the local mean a pixel must be to count as ink
LINE_FRACTION = 0.5  # a row/column with > 50% ink is a table rule, not text
MAX_SKEW_DEG = 3.0
SKEW_STEP_DEG = 0.25
SKEW_DOWNSCALE = 4


def adaptive_threshold(
    gray: np.ndarray, block: int = THRESH_BLOCK, c: float = THRESH_C
) -> np.ndarray:
    """Binarise against the local mean (robust to uneven scan lighting). 0 = ink."""
    mean = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.BoxBlur(block // 2)), dtype=np.float32
    )
    return np.where(gray.astype(np.float32) < mean - c, 0, 255).astype(np.uint8)


def remove_rules(binary: np.ndarray, fraction: float = LINE_FRACTION) -> np.ndarray:
    """Whiten table rules: rows/columns that are mostly ink (plus 1px neighbours)."""
    ink = binary == 0
    out = binary.copy()
    for axis in (1, 0):  # 1: rows (horizontal rules), 0: columns (vertical rules)
        lines = np.flatnonzero(ink.mean(axis=axis) > fraction)
        idx = np.unique(np.clip(np.concatenate([lines - 1, lines, lines + 1]), 0, None))
        idx = idx[idx < ink.shape[1 - axis]]
        if axis == 1:
            out[idx, :] = 255
        else:
            out[:, idx] = 255
    return out


def _row_profile_score(binary_small: Image.Image, angle: float) -> float:
    rotated = np.asarray(binary_small.rotate(angle, expand=False, fillcolor=255)) == 0
    return float(np.var(rotated.sum(axis=1)))


def estimate_skew(gray: np.ndarray) -> float:
    """Angle (degrees) that makes text lines horizontal: max row-profile variance."""
    small = Image.fromarray(adaptive_threshold(gray)).reduce(SKEW_DOWNSCALE)
    angles = np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-9, SKEW_STEP_DEG)
    return float(max(angles, key=lambda a: _row_profile_score(small, float(a))))


def preprocess(gray: np.ndarray) -> Image.Image:
    """Grayscale page -> deskewed, binarised, rule-free image for Tesseract."""
    angle = estimate_skew(gray)
    if angle:
        gray = np.asarray(Image.fromarray(gray).rotate(angle, expand=False, fillcolor=255))
    return Image.fromarray(remove_rules(adaptive_threshold(gray)))


def ink_fraction(gray: np.ndarray, level: int = 128) -> float:
    """Share of dark pixels; ~0 for blank pages."""
    return float((gray < level).mean())
