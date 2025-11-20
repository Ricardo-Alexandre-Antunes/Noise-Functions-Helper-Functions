import numpy as np
from PIL import Image

"""
value_noise_texture.py

Generate a simple -1..1 value noise on a grid and save:
 - an 8-bit grayscale bump map (height),
 - an RGB "directional" map (normal-like) where R/G encode X/Y directions and B encodes Z,
 - and an RGB image where grid control points are marked with a color that encodes their (x, y, z) components.

Interpolation is bilinear between grid points. The directional map is produced by computing
image gradients and encoding a normalized normal-like vector into RGB.
"""


def value_noise(width, height, spacing=32, seed=None):
    """
    Generate value noise in range [-1, 1] for an image of size (width, height).
    spacing: distance between random grid control points in pixels.
    Returns a numpy float32 array of shape (height, width).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    gx = int(np.ceil(width / spacing)) + 1  # grid columns
    gy = int(np.ceil(height / spacing)) + 1  # grid rows

    grid = rng.uniform(-1.0, 1.0, size=(gy, gx)).astype(np.float32)

    # integer cell coords for each pixel
    xv, yv = np.meshgrid(np.arange(width, dtype=np.int32),
                         np.arange(height, dtype=np.int32))
    cell_x = (xv // spacing).astype(np.int32)
    cell_y = (yv // spacing).astype(np.int32)

    # local coordinates in [0,1] (use linear interpolation weights)
    fx = (xv - cell_x * spacing) / float(spacing)
    fy = (yv - cell_y * spacing) / float(spacing)
    sx = fx.astype(np.float32)
    sy = fy.astype(np.float32)

    # fetch grid corner values
    v00 = grid[cell_y, cell_x]
    v10 = grid[cell_y, cell_x + 1]
    v01 = grid[cell_y + 1, cell_x]
    v11 = grid[cell_y + 1, cell_x + 1]

    # bilinear interpolation (linear in each axis)
    a = v00 * (1.0 - sx) + v10 * sx
    b = v01 * (1.0 - sx) + v11 * sx
    values = a * (1.0 - sy) + b * sy

    return values


def save_bump_map(values, path):
    """
    Save a float array in range [-1,1] as an 8-bit grayscale bump map.
    """
    clamped = np.clip(values, -1.0, 1.0)
    img = ((clamped + 1.0) * 0.5 * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def save_normal_map(values, path, strength=1.0):
    """
    Save an RGB directional map derived from the height map 'values'.
    - Computes gradients dx, dy (per-pixel) and constructs a normal-like vector:
      n = normalize([-dx*strength, -dy*strength, 1.0])
    - Encodes n into RGB with components mapped from [-1,1] -> [0,255].
    strength controls how strongly slopes affect the RGB direction.
    """
    # values assumed shape (H, W)
    height, width = values.shape

    # compute gradients: np.gradient returns (dy, dx)
    dy, dx = np.gradient(values.astype(np.float32))
    # Optionally scale gradients by pixel size (we assume 1.0 per pixel) and strength
    nx = -dx * float(strength)
    ny = -dy * float(strength)
    nz = np.ones_like(nx, dtype=np.float32)

    # normalize
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm

    # map [-1,1] -> [0,255]
    r = ((nx * 0.5 + 0.5) * 255.0).astype(np.uint8)
    g = ((ny * 0.5 + 0.5) * 255.0).astype(np.uint8)
    b = ((nz * 0.5 + 0.5) * 255.0).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=2)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_bump_map_with_grid_markers(values, path, spacing, marker_radius=1):
    """
    Save the bump map as an RGB image and mark grid control points (multiples of spacing)
    with a color that encodes the control point's (x, y, z) as RGB:
      R = normalized x position (0..255)
      G = normalized y position (0..255)
      B = height at that control point mapped from [-1,1] -> [0..255]
    marker_radius sets the half-size of the square marker.
    """
    height, width = values.shape
    clamped = np.clip(values, -1.0, 1.0)
    gray = ((clamped + 1.0) * 0.5 * 255.0).astype(np.uint8)

    # create RGB image from grayscale
    rgb = np.stack([gray, gray, gray], axis=2)

    # compute grid extents (same logic as value_noise)
    gx = int(np.ceil(width / spacing)) + 1
    gy = int(np.ceil(height / spacing)) + 1

    for gy_i in range(gy):
        for gx_i in range(gx):
            x = gx_i * spacing
            y = gy_i * spacing
            if x < width and y < height:
                # compute marker color from x,y position and height (z)
                x_norm = float(x) / max(1, width - 1)
                y_norm = float(y) / max(1, height - 1)
                z_val = float(values[y, x])  # in [-1,1]
                z_norm = (np.clip(z_val, -1.0, 1.0) + 1.0) * 0.5

                marker_color = (
                    int(np.round(x_norm * 255.0)),
                    int(np.round(y_norm * 255.0)),
                    int(np.round(z_norm * 255.0)),
                )

                x0 = max(0, x - marker_radius)
                x1 = min(width, x + marker_radius + 1)
                y0 = max(0, y - marker_radius)
                y1 = min(height, y + marker_radius + 1)
                rgb[y0:y1, x0:x1, 0] = marker_color[0]
                rgb[y0:y1, x0:x1, 1] = marker_color[1]
                rgb[y0:y1, x0:x1, 2] = marker_color[2]

    Image.fromarray(rgb, mode="RGB").save(path)


if __name__ == "__main__":
    # example usage
    W, H = 4096, 4096  # reduced example size for quicker run; set to 4096 if you need full res
    SPACING = 32
    SEED = 42
    out_gray = "examples/value_noise_bump.png"
    out_directional = "examples/value_noise_directional_rgb.png"
    out_marked = "examples/value_noise_bump_with_grid_colored.png"

    noise = value_noise(W, H, spacing=SPACING, seed=SEED)
    save_bump_map(noise, out_gray)
    save_normal_map(noise, out_directional, strength=4.0)
    save_bump_map_with_grid_markers(noise, out_marked, spacing=SPACING, marker_radius=1)
    print(f"Saved bump map to {out_gray}")
    print(f"Saved directional RGB map to {out_directional}")
    print(f"Saved bump map with colored grid markers to {out_marked}")
