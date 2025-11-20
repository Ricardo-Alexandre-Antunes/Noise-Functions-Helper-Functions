import numpy as np
from PIL import Image

"""
simplex_noise_normal_map.py

Generate 2D Simplex noise on a triangular (skewed) lattice and save:
 - bump map (grayscale height)
 - bump map with lattice control point markers (colored by gradient direction)
 - directional RGB normal map where color encodes the surface normal.
This is a vectorized 2D Simplex implementation (based on Stefan Gustavson),
using a controllable spacing (frequency) in pixels. Markers are placed at
the integer simplex lattice points mapped back to image space (triangle/hex lattice).
"""


# Skew/unskew factors for 2D simplex
F2 = 0.5 * (np.sqrt(3.0) - 1.0)
G2 = (3.0 - np.sqrt(3.0)) / 6.0


def fade(t):
    # kept for compatibility (not used by simplex)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a + t * (b - a)


def _make_grad_grid(gy, gx, seed):
    rng = np.random.default_rng(seed)
    angles = rng.random(size=(gy, gx)).astype(np.float32) * (2.0 * np.pi)
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=2).astype(np.float32)
    return grads


def simplex_noise(width, height, spacing=32, seed=None, return_grads=False):
    """
    Generate 2D Simplex noise (approx range [-1,1]) for an image of size (width, height).
    spacing: distance between integer simplex lattice points in pixels (controls frequency).
    If return_grads is True, returns (values, grads, lattice_origin_transform) where
    grads is shape (gy, gx, 2) giving unit gradient vectors at lattice integer points,
    and lattice_origin_transform is a callable f(i,j)->(x_px,y_px) mapping lattice indices to image pixels
    (useful to draw triangular-lattice control markers).
    """
    if seed is None:
        seed = None  # rng will pick nondet seed
    # scale image coordinates to lattice coordinates (units = lattice spacing)
    xv, yv = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    xs = xv / float(spacing)
    ys = yv / float(spacing)

    # skew to simplex grid
    s = (xs + ys) * F2
    i = np.floor(xs + s).astype(np.int32)
    j = np.floor(ys + s).astype(np.int32)

    t = (i + j) * G2
    x0 = xs - (i - t)
    y0 = ys - (j - t)

    # determine simplex corner offsets
    i1 = (x0 > y0).astype(np.int32)  # if true, (1,0) else (0,1)
    j1 = 1 - i1

    # coordinates of other two simplex corners relative to (x0,y0)
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    # prepare gradient grid large enough to index i,i+1 and j,j+1 and also i+i1, j+j1
    # i and j are >= floor(min(xs+s)) which is >= 0 for xs>=0
    i_min = int(np.min(i))
    j_min = int(np.min(j))
    i_max = int(np.max(i)) + 2
    j_max = int(np.max(j)) + 2
    gx = i_max - i_min + 1
    gy = j_max - j_min + 1

    # build gradient vectors for lattice integer points (indexing offset by i_min,j_min)
    grads = _make_grad_grid(gy, gx, seed)

    # helper to fetch gradients for arrays of indices
    def take_grad(i_idx, j_idx):
        # convert to local indices
        ii = (i_idx - i_min).astype(np.int32)
        jj = (j_idx - j_min).astype(np.int32)
        return grads[jj, ii]  # shape (...,2)

    # gradient at corner 0 (i,j)
    g0 = take_grad(i, j)
    g1 = take_grad(i + i1, j + j1)
    g2 = take_grad(i + 1, j + 1)

    # dot products
    dot0 = g0[..., 0] * x0 + g0[..., 1] * y0
    dot1 = g1[..., 0] * x1 + g1[..., 1] * y1
    dot2 = g2[..., 0] * x2 + g2[..., 1] * y2

    # contributions
    t0 = 0.5 - x0 * x0 - y0 * y0
    t1 = 0.5 - x1 * x1 - y1 * y1
    t2 = 0.5 - x2 * x2 - y2 * y2

    n0 = np.where(t0 > 0, (t0 ** 4) * dot0, 0.0)
    n1 = np.where(t1 > 0, (t1 ** 4) * dot1, 0.0)
    n2 = np.where(t2 > 0, (t2 ** 4) * dot2, 0.0)

    # scale factor chosen to keep values roughly in [-1,1] for visual use
    values = 70.0 * (n0 + n1 + n2)  # Gustavson uses ~70 for 2D

    # normalize to [-1,1] using max absolute
    max_abs = np.max(np.abs(values))
    if max_abs > 0:
        values = (values / max_abs).astype(np.float32)
    else:
        values = values.astype(np.float32)

    if return_grads:
        # lattice-to-pixel transform: unskew lattice point (I,J) to image pixel coords
        def lattice_to_pixel(I, J):
            # unskew
            tIJ = (I + J) * G2
            ux = (I - tIJ) * spacing
            uy = (J - tIJ) * spacing
            return ux, uy

        return values, grads, lattice_to_pixel

    return values


def simplex_direction_rgb(width, height, spacing=32, seed=None):
    """
    Build a directional RGB image by computing the numeric gradient of the simplex noise field
    and mapping direction to RGB (R = nx mapped, G = ny, B = angle).
    This uses np.gradient on the scalar field for simplicity.
    """
    noise = simplex_noise(width, height, spacing=spacing, seed=seed, return_grads=False)
    # numeric derivatives (gy, gx) = d/dy, d/dx
    gy, gx = np.gradient(noise.astype(np.float32))
    mag = np.sqrt(gx * gx + gy * gy)
    inv = 1.0 / (mag + 1e-12)
    nx = gx * inv
    ny = gy * inv

    r = ((nx * 0.5) + 0.5) * 255.0
    g = ((ny * 0.5) + 0.5) * 255.0
    angle = np.arctan2(ny, nx)
    b = ((angle + np.pi) / (2.0 * np.pi)) * 255.0

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    rgb[rgb == 0] = 1
    return rgb


# keep bump_to_normal_map, save functions from the original file (no changes needed)

def bump_to_normal_map(values, strength=1.0):
    gy, gx = np.gradient(values.astype(np.float32))
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx, dtype=np.float32)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm

    r = ((nx * 0.5) + 0.5) * 255.0
    g = ((ny * 0.5) + 0.5) * 255.0
    b = ((nz * 0.5) + 0.5) * 255.0

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    rgb[rgb == 0] = 1
    return rgb


def save_bump_map(values, path):
    clamped = np.clip(values, -1.0, 1.0)
    img = ((clamped + 1.0) * 0.5 * 255.0).astype(np.uint8)
    img[img == 0] = 1
    Image.fromarray(img, mode="L").save(path)


def save_bump_map_with_grid_markers(values, path, spacing, marker_color=(255, 0, 0), marker_radius=1, grads=None, lattice_to_pixel=None):
    """
    Save bump with markers at simplex lattice integer points.
    If grads is provided, color markers by gradient direction at that lattice point.
    lattice_to_pixel is a function (I,J)->(x_px,y_px) to place markers on image.
    """
    height, width = values.shape
    clamped = np.clip(values, -1.0, 1.0)
    gray = ((clamped + 1.0) * 0.5 * 255.0).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)

    if grads is None or lattice_to_pixel is None:
        # fall back to square grid markers as before
        gx = int(np.ceil(width / spacing)) + 1
        gy = int(np.ceil(height / spacing)) + 1
        for j in range(gy):
            for i in range(gx):
                x = i * spacing
                y = j * spacing
                if x < width and y < height:
                    x0 = max(0, x - marker_radius)
                    x1 = min(width, x + marker_radius + 1)
                    y0 = max(0, y - marker_radius)
                    y1 = min(height, y + marker_radius + 1)
                    rgb[y0:y1, x0:x1, :] = marker_color
    else:
        gy_count, gx_count = grads.shape[0], grads.shape[1]
        # lattice indices likely range from i_min..i_max used when creating grads;
        # But grads here is the raw grads array returned from simplex with local indexing.
        # The lattice_to_pixel function should be the one returned from simplex_noise
        for J in range(gy_count):
            for I in range(gx_count):
                # map lattice integer (I,J) to pixel coordinates
                x_f, y_f = lattice_to_pixel(I, J)
                x = int(round(x_f))
                y = int(round(y_f))
                if 0 <= x < width and 0 <= y < height:
                    x0 = max(0, x - marker_radius)
                    x1 = min(width, x + marker_radius + 1)
                    y0 = max(0, y - marker_radius)
                    y1 = min(height, y + marker_radius + 1)
                    nx = float(grads[J, I, 0])
                    ny = float(grads[J, I, 1])
                    r = int(np.clip(((nx * 0.5) + 0.5) * 255.0, 1, 255))
                    g = int(np.clip(((ny * 0.5) + 0.5) * 255.0, 1, 255))
                    angle = np.arctan2(ny, nx)
                    b = int(np.clip(((angle + np.pi) / (2.0 * np.pi)) * 255.0, 1, 255))
                    color = (r, g, b)
                    rgb[y0:y1, x0:x1, 0] = color[0]
                    rgb[y0:y1, x0:x1, 1] = color[1]
                    rgb[y0:y1, x0:x1, 2] = color[2]

    rgb[rgb == 0] = 1
    Image.fromarray(rgb, mode="RGB").save(path)


def save_direction_rgb(rgb_array, path):
    rgb = rgb_array.copy()
    rgb[rgb == 0] = 1
    Image.fromarray(rgb, mode="RGB").save(path)


def save_normal_map_from_bump(values, path, strength=1.0):
    normal_rgb = bump_to_normal_map(values, strength=strength)
    save_direction_rgb(normal_rgb, path)


if __name__ == "__main__":
    # Example usage (reduce resolution for quick testing)
    W, H = 4096, 4096
    SPACING = 32
    SEED = 42
    out_gray = "examples/simplex_noise_bump.png"
    out_marked = "examples/simplex_noise_bump_with_lattice.png"
    out_dir = "examples/simplex_direction_rgb.png"
    out_normal = "examples/simplex_normal_map.png"

    noise, grads, lattice_to_pixel = simplex_noise(W, H, spacing=SPACING, seed=SEED, return_grads=True)
    save_bump_map(noise, out_gray)
    save_bump_map_with_grid_markers(noise, out_marked, spacing=SPACING, marker_radius=1, grads=grads, lattice_to_pixel=lattice_to_pixel)

    direction_rgb = simplex_direction_rgb(W, H, spacing=SPACING, seed=SEED)
    save_direction_rgb(direction_rgb, out_dir)

    save_normal_map_from_bump(noise, out_normal, strength=8.0)

    print(f"Saved simplex bump map to {out_gray}")
    print(f"Saved simplex bump map with lattice markers to {out_marked}")
    print(f"Saved simplex directional RGB map to {out_dir}")
    print(f"Saved simplex normal map (RGB) to {out_normal}")
