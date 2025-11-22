"""Generate a 3D terrain: Perlin noise base + value-noise mountains.

Run with the project's venv (PowerShell):

    & ".\.venv\Scripts\python.exe" "examples\terrain_perlin_value_mountains.py"

This script writes `examples/terrain_perlin_value_mountains.png`.
"""
import math
import numpy as np
from random import Random
import matplotlib.pyplot as plt


class Perlin2D:
    def __init__(self, seed=None):
        self.rng = Random(seed)
        p = list(range(256))
        self.rng.shuffle(p)
        self.perm = p + p
        # 8 gradient directions (unit-ish)
        self.grad = [
            (1,1), (-1,1), (1,-1), (-1,-1),
            (1,0), (-1,0), (0,1), (0,-1)
        ]

    @staticmethod
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    def grad_dot(self, gi, x, y):
        g = self.grad[gi % len(self.grad)]
        return g[0] * x + g[1] * y

    def noise(self, x, y):
        # classic Perlin on integer lattice
        xi = math.floor(x)
        yi = math.floor(y)
        xf = x - xi
        yf = y - yi

        xi &= 255
        yi &= 255

        aa = self.perm[xi + self.perm[yi]]
        ab = self.perm[xi + self.perm[yi + 1]]
        ba = self.perm[xi + 1 + self.perm[yi]]
        bb = self.perm[xi + 1 + self.perm[yi + 1]]

        u = Perlin2D.fade(xf)
        v = Perlin2D.fade(yf)

        x1 = self.lerp(self.grad_dot(aa, xf, yf), self.grad_dot(ba, xf - 1, yf), u)
        x2 = self.lerp(self.grad_dot(ab, xf, yf - 1), self.grad_dot(bb, xf - 1, yf - 1), u)
        return self.lerp(x1, x2, v)

    def fbm(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        value = 0.0
        amp = 1.0
        freq = 1.0
        max_amp = 0.0
        for _ in range(octaves):
            value += amp * self.noise(x * freq, y * freq)
            max_amp += amp
            amp *= gain
            freq *= lacunarity
        return value / max_amp


class ValueNoise2D:
    def __init__(self, seed=None):
        self.rng = Random(seed)

    @staticmethod
    def smoothstep(t):
        return t * t * (3 - 2 * t)

    def value_noise(self, x, y, seed):
        xi = math.floor(x)
        yi = math.floor(y)
        xf = x - xi
        yf = y - yi

        # get four corner random values deterministically using seed
        def corner(i, j):
            r = Random((i * 1836311903) ^ (j * 2971215073) ^ seed)
            return r.random()

        v00 = corner(xi, yi)
        v10 = corner(xi + 1, yi)
        v01 = corner(xi, yi + 1)
        v11 = corner(xi + 1, yi + 1)

        u = ValueNoise2D.smoothstep(xf)
        v = ValueNoise2D.smoothstep(yf)

        ix0 = v00 * (1 - u) + v10 * u
        ix1 = v01 * (1 - u) + v11 * u
        return ix0 * (1 - v) + ix1 * v

    def fbm(self, x, y, seed=0, octaves=4, lacunarity=2.0, gain=0.5):
        value = 0.0
        amp = 1.0
        freq = 1.0
        max_amp = 0.0
        for _ in range(octaves):
            value += amp * self.value_noise(x * freq, y * freq, int(seed + freq * 1000))
            max_amp += amp
            amp *= gain
            freq *= lacunarity
        return value / max_amp


def generate_terrain(nx=200, ny=200, seed=0):
    # domain
    x_start, x_end = 0.0, 6.0
    y_start, y_end = 0.0, 6.0

    xs = np.linspace(x_start, x_end, nx)
    ys = np.linspace(y_start, y_end, ny)
    X, Y = np.meshgrid(xs, ys)

    perlin = Perlin2D(seed=seed)
    value = ValueNoise2D(seed=seed + 1)

    # base terrain from Perlin FBM (smooth hills)
    base_freq = 0.6
    base_octaves = 5
    base_amp = 1.2

    Z_base = np.zeros_like(X)
    for iy in range(ny):
        for ix in range(nx):
            x = X[iy, ix] * base_freq
            y = Y[iy, ix] * base_freq
            Z_base[iy, ix] = perlin.fbm(x, y, octaves=base_octaves)
    Z_base = (Z_base - Z_base.min()) / (Z_base.max() - Z_base.min())  # normalize 0..1
    Z_base = (Z_base - 0.5) * base_amp  # center and scale

    # --- subtle value-noise bumps/dips (low-intensity) using smoothstep ---
    # Use Perlin FBM result as main terrain (already computed in Z_base).
    # Use value-noise FBM to create small, smooth increases/dips in altitude.
    bump_freq = 8.0
    bump_octaves = 4
    bump_amp = 0.12  # small amplitude so bumps are subtle

    Z_val = np.zeros_like(X)
    for iy in range(ny):
        for ix in range(nx):
            x = X[iy, ix] * bump_freq
            y = Y[iy, ix] * bump_freq
            Z_val[iy, ix] = value.fbm(x, y, seed=seed + 200, octaves=bump_octaves)

    # normalize to 0..1
    Z_val = (Z_val - Z_val.min()) / (Z_val.max() - Z_val.min() + 1e-12)

    # Center around 0 so we have positive (bumps) and negative (dips)
    v = Z_val - 0.5

    # Smoothstep on absolute magnitude to get soft transition near zero
    s = ValueNoise2D.smoothstep(np.clip(np.abs(v) * 2.0, 0.0, 1.0))

    # restore sign and scale by small amplitude
    bump = np.sign(v) * s
    Z_perturb = bump * bump_amp

    # Optionally modulate perturbations by base elevation so very low-lying areas
    # are less affected (keeps lakes smooth). Use a gentle base mask.
    base_norm = (Z_base - Z_base.min()) / (Z_base.max() - Z_base.min() + 1e-12)
    base_mask = np.clip((base_norm - 0.15) / 0.7, 0.0, 1.0)
    Z_perturb *= (0.4 + 0.6 * base_mask)  # slightly prefer higher ground

    Z_mountains = Z_perturb

    # Final terrain: Perlin base + small value-noise bumps/dips
    Z = Z_base + Z_perturb

    return X, Y, Z, Z_base, Z_mountains


def plot_terrain(X, Y, Z, Z_base, Z_mountains, out_filename="examples/terrain_perlin_value_mountains.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        rcount=200,
        ccount=200,
        zorder=1,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, label="height")

    # Optionally overlay mountain mask contour lines for emphasis
    ax.contour(X, Y, Z_mountains, levels=6, zdir="z", offset=Z.min() - 1.0, cmap="binary", linewidths=1.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title("Perlin base terrain + Value-noise mountains")

    # adjust view
    ax.view_init(elev=45, azim=-120)

    plt.tight_layout()
    plt.savefig(out_filename, dpi=200)
    print(f"Saved terrain image to: {out_filename}")
    plt.show()


if __name__ == "__main__":
    X, Y, Z, Z_base, Z_mountains = generate_terrain(nx=220, ny=220, seed=42)
    plot_terrain(X, Y, Z, Z_base, Z_mountains)
