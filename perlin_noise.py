import math
import numpy as np
from random import Random
import matplotlib.pyplot as plt

def fade(t):
    # 6t^5 - 15t^4 + 10t^3 smoothstep
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b - a)

class Perlin1D:
    def __init__(self):
        self.rng = Random()
        self.grad = {}

    def _ensure_grad(self, i):
        if i not in self.grad:
            # gradient in [-1, 1]
            self.grad[i] = self.rng.uniform(-1.0, 1.0)

    def noise(self, x):
        # 1D Perlin noise value at x
        xi = math.floor(x)
        xf = x - xi
        self._ensure_grad(xi)
        self._ensure_grad(xi + 1)
        g0 = self.grad[xi]
        g1 = self.grad[xi + 1]
        # dot product: in 1D it's gradient * distance
        d0 = g0 * xf
        d1 = g1 * (xf - 1.0)
        u = fade(xf)
        return lerp(d0, d1, u)

if __name__ == "__main__":
    # Parameters
    base_value = 0.5         # original constant
    amplitude = 0.25         # how strong the noise is
    x_start, x_end = 0.0, 10.0
    n_samples = 400          # number of plotted sample points

    perlin = Perlin1D()

    # ensure gradients for all integer lattice points that cover the domain
    i_start = math.floor(x_start)
    i_end = math.ceil(x_end)  # we'll ensure grads up to i_end (so i_end+1 can exist when needed)
    for ii in range(i_start, i_end + 2):
        perlin._ensure_grad(ii)

    x = np.linspace(x_start, x_end, n_samples)
    raw_noise = np.array([perlin.noise(xx) for xx in x])  # in roughly [-1,1]
    y_noisy = base_value + amplitude * raw_noise
    y_base = np.full_like(x, base_value)

    # integer lattice points where noise is zero (xf == 0)
    ints = np.arange(i_start, i_end + 1)
    ints = ints[(ints >= x_start) & (ints <= x_end)]
    xi = ints.astype(float)
    yi = np.full_like(xi, base_value)  # noise at integer x is zero, so y == base_value

    # Plot
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y_base, color="gray", linewidth=2, label="base values")
    plt.plot(x, y_noisy, color="orange", linewidth=2, label="perlin noise")

    # blue points: integer lattice points (where noise is zero)
    plt.scatter(xi, yi, color="blue", s=30, zorder=5, label="lattice points")

    # green lines: show gradient applied from each lattice point across a fixed symmetric segment
    segment_half_width = 0.5  # half-length of each green segment; all segments will be same length

    for i in ints:
        x0 = i - segment_half_width
        x1 = i + segment_half_width
        # skip segments that are completely outside the plotting domain
        if x1 <= x_start or x0 >= x_end:
            continue
        xs_line = np.linspace(x0, x1, 10)
        # gradient at lattice i applied as linear contribution g_i * (x - i)
        g_i = perlin.grad[i]
        ys_line = base_value + amplitude * (g_i * (xs_line - i))
        plt.plot(xs_line, ys_line, color="green", linewidth=1.0, alpha=0.8, label="gradient" if i == ints[0] else "")

    plt.title("Perlin Noise applied to 1D (constant set of values)")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.ylim(base_value - amplitude * 1.2, base_value + amplitude * 1.2)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")

    plt.tight_layout()
    out_filename = "examples/perlin_noise.png"
    plt.savefig(out_filename, dpi=150)
    print(f"Saved image to {out_filename}")
    plt.show()