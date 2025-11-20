import numpy as np
from PIL import Image

"""
perlin_noise_texture.py

Generate 2D Perlin noise on a grid and save:
 - bump map (grayscale height)
 - bump map with grid control point markers (colored by gradient direction)
 - directional RGB normal map where color encodes the surface normal:
    R = Nx mapped to [0,255]
    G = Ny mapped to [0,255]
    B = Nz mapped to [0,255]
Interpolation uses a quintic fade function (classic Perlin).
"""


def fade(t):
    # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a + t * (b - a)


def perlin_noise(width, height, spacing=32, seed=None, return_grads=False):
    """
    Generate Perlin noise in range approximately [-1, 1] for an image of size
    (width, height). spacing: distance between gradient grid control points in pixels.
    If return_grads is True, returns (values, grads) where grads is shape (gy, gx, 2).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    gx = int(np.ceil(width / spacing)) + 1  # grid columns
    gy = int(np.ceil(height / spacing)) + 1  # grid rows

    # random gradient unit vectors at grid points (angles)
    angles = rng.random(size=(gy, gx)).astype(np.float32) * (2.0 * np.pi)
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=2).astype(np.float32)  # shape (gy, gx, 2)

    # integer cell coords for each pixel
    xv, yv = np.meshgrid(np.arange(width, dtype=np.int32),
                         np.arange(height, dtype=np.int32))
    cell_x = (xv // spacing).astype(np.int32)
    cell_y = (yv // spacing).astype(np.int32)

    # local coordinates in [0,1]
    fx = (xv - cell_x * spacing) / float(spacing)
    fy = (yv - cell_y * spacing) / float(spacing)
    fx = fx.astype(np.float32)
    fy = fy.astype(np.float32)

    # fade curves
    sx = fade(fx)
    sy = fade(fy)

    # fetch gradient vectors at corners
    g00 = grads[cell_y, cell_x]         # top-left
    g10 = grads[cell_y, cell_x + 1]     # top-right
    g01 = grads[cell_y + 1, cell_x]     # bottom-left
    g11 = grads[cell_y + 1, cell_x + 1] # bottom-right

    # offset vectors from grid corner to pixel (in cell-local coords)
    dx00 = fx
    dy00 = fy
    dx10 = fx - 1.0
    dy10 = fy
    dx01 = fx
    dy01 = fy - 1.0
    dx11 = fx - 1.0
    dy11 = fy - 1.0

    # dot products: influence of each corner
    v00 = g00[..., 0] * dx00 + g00[..., 1] * dy00
    v10 = g10[..., 0] * dx10 + g10[..., 1] * dy10
    v01 = g01[..., 0] * dx01 + g01[..., 1] * dy01
    v11 = g11[..., 0] * dx11 + g11[..., 1] * dy11

    # bilinear interpolation using fade weights
    a = lerp(v00, v10, sx)
    b = lerp(v01, v11, sx)
    values = lerp(a, b, sy)

    # normalize to [-1, 1] using the max absolute value to fill full range
    max_abs = np.max(np.abs(values))
    if max_abs > 0:
        values = (values / max_abs).astype(np.float32)
    else:
        values = values.astype(np.float32)

    if return_grads:
        return values, grads
    return values


def perlin_direction_rgb(width, height, spacing=32, seed=None):
    """
    Generate an RGB image encoding the per-pixel interpolated gradient direction.
    R,G encode the x,y components of the interpolated gradient (mapped from [-1,1] to [0,255]).
    B encodes the angle atan2(y,x) mapped to [0,255].
    Returns a uint8 array shape (height, width, 3). Ensures there are no pure black pixels.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    gx = int(np.ceil(width / spacing)) + 1  # grid columns
    gy = int(np.ceil(height / spacing)) + 1  # grid rows

    # random gradient unit vectors at grid points (angles)
    angles = rng.random(size=(gy, gx)).astype(np.float32) * (2.0 * np.pi)
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=2).astype(np.float32)  # shape (gy, gx, 2)

    # integer cell coords for each pixel
    xv, yv = np.meshgrid(np.arange(width, dtype=np.int32),
                         np.arange(height, dtype=np.int32))
    cell_x = (xv // spacing).astype(np.int32)
    cell_y = (yv // spacing).astype(np.int32)

    # local coordinates in [0,1]
    fx = (xv - cell_x * spacing) / float(spacing)
    fy = (yv - cell_y * spacing) / float(spacing)
    fx = fx.astype(np.float32)
    fy = fy.astype(np.float32)

    # fade curves
    sx = fade(fx)
    sy = fade(fy)

    # fetch gradient vectors at corners
    g00 = grads[cell_y, cell_x]         # top-left
    g10 = grads[cell_y, cell_x + 1]     # top-right
    g01 = grads[cell_y + 1, cell_x]     # bottom-left
    g11 = grads[cell_y + 1, cell_x + 1] # bottom-right

    # interpolate gradient field components using same fade (this gives per-pixel direction)
    ax = lerp(g00[..., 0], g10[..., 0], sx)
    bx = lerp(g01[..., 0], g11[..., 0], sx)
    gx_field = lerp(ax, bx, sy)

    ay = lerp(g00[..., 1], g10[..., 1], sx)
    by = lerp(g01[..., 1], g11[..., 1], sx)
    gy_field = lerp(ay, by, sy)

    # normalize direction to unit vectors (avoid div by zero)
    mag = np.sqrt(gx_field * gx_field + gy_field * gy_field)
    inv_mag = 1.0 / (mag + 1e-12)
    nx = gx_field * inv_mag
    ny = gy_field * inv_mag

    # map to 0..255
    r = ((nx * 0.5) + 0.5) * 255.0
    g = ((ny * 0.5) + 0.5) * 255.0
    angle = np.arctan2(ny, nx)  # -pi..pi
    b = ((angle + np.pi) / (2.0 * np.pi)) * 255.0

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    # avoid pure black pixels: replace any 0 with 1
    rgb[rgb == 0] = 1
    return rgb


def bump_to_normal_map(values, strength=1.0):
    """
    Convert a bump (height) map values (float in [-1,1]) to a normal map (uint8 RGB).
    strength controls how strong the surface slopes are (higher = more pronounced normals).
    Uses central differences (np.gradient) to compute per-pixel derivatives.
    """
    # values: shape (H, W), dtype float32 expected
    # compute derivatives: gy = d/dy, gx = d/dx (rows, cols)
    gy, gx = np.gradient(values.astype(np.float32))
    # build normal: (-dx * strength, -dy * strength, 1)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx, dtype=np.float32)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm

    # map from [-1,1] to [0,255]
    r = ((nx * 0.5) + 0.5) * 255.0
    g = ((ny * 0.5) + 0.5) * 255.0
    b = ((nz * 0.5) + 0.5) * 255.0

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    rgb[rgb == 0] = 1
    return rgb


def save_bump_map(values, path):
    """
    Save a float array in range [-1,1] as an 8-bit grayscale bump map.
    Avoid pure black by mapping to [1,255].
    """
    clamped = np.clip(values, -1.0, 1.0)
    img = ((clamped + 1.0) * 0.5 * 255.0).astype(np.uint8)
    img[img == 0] = 1
    Image.fromarray(img, mode="L").save(path)


def save_bump_map_with_grid_markers(values, path, spacing, marker_color=(255, 0, 0), marker_radius=1, grads=None):
    """
    Save the bump map as an RGB image and mark grid control points.
    If grads is provided (shape gy x gx x 2), color each marker by the gradient direction at that control point.
    marker_radius sets the half-size of the square marker.
    Avoid pure black in final image.
    """
    height, width = values.shape
    clamped = np.clip(values, -1.0, 1.0)
    gray = ((clamped + 1.0) * 0.5 * 255.0).astype(np.uint8)

    # create RGB image from grayscale
    rgb = np.stack([gray, gray, gray], axis=2)

    if grads is not None:
        gy, gx = grads.shape[0], grads.shape[1]
    else:
        gx = int(np.ceil(width / spacing)) + 1
        gy = int(np.ceil(height / spacing)) + 1

    for gy_i in range(gy):
        for gx_i in range(gx):
            x = gx_i * spacing
            y = gy_i * spacing
            if x < width and y < height:
                x0 = max(0, x - marker_radius)
                x1 = min(width, x + marker_radius + 1)
                y0 = max(0, y - marker_radius)
                y1 = min(height, y + marker_radius + 1)

                if grads is not None:
                    # extract gradient unit vector at this control point
                    nx = float(grads[gy_i, gx_i, 0])
                    ny = float(grads[gy_i, gx_i, 1])
                    # map to RGB (same scheme as perlin_direction_rgb)
                    r = int(np.clip(((nx * 0.5) + 0.5) * 255.0, 1, 255))
                    g = int(np.clip(((ny * 0.5) + 0.5) * 255.0, 1, 255))
                    angle = np.arctan2(ny, nx)
                    b = int(np.clip(((angle + np.pi) / (2.0 * np.pi)) * 255.0, 1, 255))
                    color = (r, g, b)
                else:
                    color = marker_color

                rgb[y0:y1, x0:x1, 0] = color[0]
                rgb[y0:y1, x0:x1, 1] = color[1]
                rgb[y0:y1, x0:x1, 2] = color[2]

    # avoid pure black pixels in whole image
    rgb[rgb == 0] = 1
    Image.fromarray(rgb, mode="RGB").save(path)


def save_direction_rgb(rgb_array, path):
    """
    Save a uint8 RGB array (height, width, 3) to path. Ensure no pure black pixels.
    """
    rgb = rgb_array.copy()
    rgb[rgb == 0] = 1
    Image.fromarray(rgb, mode="RGB").save(path)


def save_normal_map_from_bump(values, path, strength=1.0):
    """
    Compute normal map from bump (height) values and save as RGB.
    strength controls the bump-to-normal scale.
    """
    normal_rgb = bump_to_normal_map(values, strength=strength)
    save_direction_rgb(normal_rgb, path)


if __name__ == "__main__":
    # example usage
    W, H = 4096, 4096  # reduce size for faster testing; set to 4096 for high-res
    SPACING = 32
    SEED = 42
    out_gray = "examples/perlin_noise_bump.png"
    out_marked = "examples/perlin_noise_bump_with_grid.png"
    out_dir = "examples/perlin_direction_rgb.png"
    out_normal = "examples/perlin_normal_map.png"

    noise, grads = perlin_noise(W, H, spacing=SPACING, seed=SEED, return_grads=True)
    save_bump_map(noise, out_gray)
    save_bump_map_with_grid_markers(noise, out_marked, spacing=SPACING, marker_radius=1, grads=grads)

    # direction image from interpolated gradients (visualizes direction / angle)
    direction_rgb = perlin_direction_rgb(W, H, spacing=SPACING, seed=SEED)
    save_direction_rgb(direction_rgb, out_dir)

    # create a proper normal map (RGB) from the bump map (you can adjust strength)
    save_normal_map_from_bump(noise, out_normal, strength=8.0)

    print(f"Saved bump map to {out_gray}")
    print(f"Saved bump map with grid markers to {out_marked}")
    print(f"Saved directional RGB map to {out_dir}")
    print(f"Saved normal map (RGB) to {out_normal}")
