import math
import numpy as np
from random import Random
import matplotlib.pyplot as plt

class Simplex1D:
    def __init__(self, seed=None):
        self.rng = Random(seed)
        self.grad = {}

    def _ensure_grad(self, i):
        if i not in self.grad:
            # gradient in [-1, 1]
            self.grad[i] = self.rng.uniform(-1.0, 1.0)

    def noise(self, x):
        # find cell corners
        i0 = math.floor(x)
        i1 = i0 + 1
        x0 = x - i0
        x1 = x0 - 1.0

        # ensure gradients exist
        self._ensure_grad(i0)
        self._ensure_grad(i1)
        g0 = self.grad[i0]
        g1 = self.grad[i1]

        # compute falloff function
        t0 = 0.5 - x0 * x0
        t1 = 0.5 - x1 * x1

        n0 = n1 = 0.0
        if t0 > 0:
            t0 *= t0
            n0 = t0 * t0 * g0 * x0
        if t1 > 0:
            t1 *= t1
            n1 = t1 * t1 * g1 * x1

        return 70.0 * (n0 + n1)  # scale factor to normalize roughly to [-1, 1]


if __name__ == "__main__":
    # Parameters
    base_value = 0.5
    amplitude = 0.25
    x_start, x_end = 0.0, 10.0
    n_samples = 400

    simplex = Simplex1D(seed=0)

    # Precompute gradients for all integer lattice points in the domain
    i_start = math.floor(x_start)
    i_end = math.ceil(x_end)
    for ii in range(i_start, i_end + 2):
        simplex._ensure_grad(ii)

    x = np.linspace(x_start, x_end, n_samples)
    raw_noise = np.array([simplex.noise(xx) for xx in x])
    y_noisy = base_value + amplitude * raw_noise
    y_base = np.full_like(x, base_value)

    # integer lattice points (for visualization)
    ints = np.arange(i_start, i_end + 1)
    ints = ints[(ints >= x_start) & (ints <= x_end)]
    xi = ints.astype(float)
    yi = np.full_like(xi, base_value)

    # Plot (use same colors/visuals as the provided Perlin snippet)
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y_base, color="gray", linewidth=2, label="base values")
    plt.plot(x, y_noisy, color="orange", linewidth=2, label="simplex noise")

    # blue points: integer lattice points (where noise is zero)
    plt.scatter(xi, yi, color="blue", s=30, zorder=5, label="lattice points")

    # green lines: show gradient applied from each lattice point across a fixed symmetric segment
    segment_half_width = 0.5  # half-length of each green segment; all segments will be same length

    for idx, i in enumerate(ints):
        x0 = i - segment_half_width
        x1 = i + segment_half_width
        # skip segments that are completely outside the plotting domain
        if x1 <= x_start or x0 >= x_end:
            continue
        xs_line = np.linspace(max(x0, x_start), min(x1, x_end), 10)
        # gradient at lattice i applied as linear contribution g_i * (x - i)
        g_i = simplex.grad[int(i)]
        ys_line = base_value + amplitude * (g_i * (xs_line - i))
        # label only on first plotted gradient to avoid duplicates
        label = "gradient" if idx == 0 else ""
        plt.plot(xs_line, ys_line, color="green", linewidth=1.0, alpha=0.8, label=label)

    plt.title("Simplex Noise applied to 1D (constant set of values)")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.ylim(base_value - amplitude * 1.2, base_value + amplitude * 1.2)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")

    plt.tight_layout()
    out_filename = "examples/simplex_noise.png"
    plt.savefig(out_filename, dpi=150)
    print(f"Saved image to {out_filename}")
    plt.show()
