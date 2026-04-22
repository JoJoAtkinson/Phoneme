"""256-entry colormap LUT for the spectrogram widget.

A perceptually-uniform-ish dark→purple→pink→yellow ramp (viridis-adjacent).
Pre-computed as a numpy uint8 array of shape [256, 4] (RGBA) so we can
index with normalized float intensity and produce a QImage in one step.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# 9 control points from perceptually-uniform magma. We linearly interpolate
# in RGB space between them. Good enough visually, zero dependencies.
_STOPS = [
    (0.00, (0, 0, 4)),
    (0.12, (28, 16, 68)),
    (0.25, (79, 18, 123)),
    (0.38, (129, 37, 129)),
    (0.50, (181, 54, 122)),
    (0.62, (229, 80, 100)),
    (0.75, (251, 135, 97)),
    (0.88, (254, 194, 135)),
    (1.00, (252, 253, 191)),
]


@lru_cache(maxsize=1)
def magma_lut() -> np.ndarray:
    """Return a (256, 4) uint8 RGBA array."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Find bracketing stops
        lo = _STOPS[0]
        hi = _STOPS[-1]
        for j in range(len(_STOPS) - 1):
            if _STOPS[j][0] <= t <= _STOPS[j + 1][0]:
                lo = _STOPS[j]
                hi = _STOPS[j + 1]
                break
        span = hi[0] - lo[0]
        u = 0.0 if span == 0 else (t - lo[0]) / span
        r = lo[1][0] + u * (hi[1][0] - lo[1][0])
        g = lo[1][1] + u * (hi[1][1] - lo[1][1])
        b = lo[1][2] + u * (hi[1][2] - lo[1][2])
        lut[i] = (int(r), int(g), int(b), 255)
    return lut


def apply_colormap(norm: np.ndarray) -> np.ndarray:
    """Map a float array in [0, 1] to an RGBA uint8 array of shape (..., 4)."""
    idx = np.clip((norm * 255.0), 0, 255).astype(np.uint8)
    return magma_lut()[idx]
