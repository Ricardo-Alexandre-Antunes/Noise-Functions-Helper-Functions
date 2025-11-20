from pathlib import Path
import numpy as np
from typing import Tuple

#!/usr/bin/env python3
"""
examples/value_noise.py

Create a simple plot showing a constant value series and the same series
with 1D value noise applied. Save and show the resulting figure.

This version ensures the same raw (sparse) noise points are used for both
the non-interpolated series and an interpolated version of those same points.
"""

import matplotlib.pyplot as plt


def point_noise_1d(x: np.ndarray, seed: int = 0, prob: float = 0.05, amplitude: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse point noise: only change specific samples in x (no interpolation).
    - prob: probability each sample is changed
    - amplitude: maximum absolute change
    Returns a tuple (noise, mask) where noise is an array of same shape as x
    with zeros except random offsets at selected indices, and mask is a boolean
    array marking which indices were changed.
    """
    rng = np.random.RandomState(seed)
    mask = rng.rand(x.size) < prob
    noise = np.zeros_like(x)
    if mask.any():
        noise[mask] = rng.uniform(-amplitude, amplitude, size=mask.sum())
    return noise, mask


def interpolate_from_sparse_points(x: np.ndarray, knot_x: np.ndarray, knot_vals: np.ndarray, smooth: bool = False) -> np.ndarray:
    """
    Interpolate a noise signal from a set of sparse (x, value) points using
    a smoothstep interpolation (6t^5 - 15t^4 + 10t^3) between knots.
    If there are no knots, returns zeros. If one knot, returns a constant array.
    Optionally apply a small moving-average smoothing to the interpolated result.
    """
    if knot_x.size == 0:
        return np.zeros_like(x)
    if knot_x.size == 1:
        return np.full_like(x, knot_vals[0])

    # ensure knots are sorted
    order = np.argsort(knot_x)
    kx = knot_x[order]
    kv = knot_vals[order]

    noise = np.empty_like(x)

    # left of first knot -> constant
    left_mask = x <= kx[0]
    noise[left_mask] = kv[0]

    # right of last knot -> constant
    right_mask = x >= kx[-1]
    noise[right_mask] = kv[-1]

    # smoothstep function
    def smoothstep(t: np.ndarray) -> np.ndarray:
        # clamp t to [0,1] to be safe
        t = np.clip(t, 0.0, 1.0)
        return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3

    # interpolate between each pair of knots using smoothstep
    for i in range(len(kx) - 1):
        x0 = kx[i]
        x1 = kx[i + 1]
        v0 = kv[i]
        v1 = kv[i + 1]
        # include left edge, exclude right to avoid double-assign; last interval handled by right_mask
        if i < len(kx) - 2:
            mask = (x >= x0) & (x < x1)
        else:
            # for the last interval include the right endpoint (but those >= kx[-1] were already set by right_mask)
            mask = (x >= x0) & (x <= x1)
        if not np.any(mask):
            continue
        t = (x[mask] - x0) / (x1 - x0)
        s = smoothstep(t)
        noise[mask] = (1.0 - s) * v0 + s * v1

    if smooth:
        kernel_size = max(3, int(len(x) * 0.002))
        kernel = np.ones(kernel_size) / kernel_size
        noise = np.convolve(noise, kernel, mode="same")
    return noise


if __name__ == "__main__":
    # domain and base (constant) signal
    x = np.linspace(0, 10, 2000)
    base_value = 0.5
    base = np.full_like(x, base_value)

    # Create sparse point noise (shared raw noise for both displays)
    noise_points, mask = point_noise_1d(x, seed=42, prob=0.02, amplitude=0.25)
    noisy_points = base + noise_points

    # Interpolate the exact same sparse noise points across the domain using smoothstep
    knot_x = x[mask]
    knot_vals = noise_points[mask]
    noise_interp = interpolate_from_sparse_points(x, knot_x, knot_vals, smooth=True)
    noisy_interp = base + noise_interp

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, base, color="#333333", lw=1.0, label="base values")

    # show sparse point changes as scatter to make them visible, and also show a connecting line
    idx_changed = mask
    plt.plot(x, noisy_points, color="#1f77b4", lw=0.8, alpha=0.7, label="value noise")
    plt.scatter(x[idx_changed], noisy_points[idx_changed], color="#1f77b4", s=12)

    plt.plot(x, noisy_interp, color="#ff7f0e", lw=1.0, alpha=0.9, label="interpolation of noise (smoothstep)")
    plt.title("Value Noise applied to 1D (constant set of values) with smoothstep")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.ylim(-0.1, 1.1)
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path = Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to: {out_path}")
    plt.show()
