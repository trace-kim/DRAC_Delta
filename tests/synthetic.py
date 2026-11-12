"""Synthetic SEM-like images with known contact positions, for the tests (from AffineRANSAC's tests); and how far
found centres are from the known ones (centre_errors)."""

import cv2
import numpy as np


def write_sem_like(path, rows=768, cols=1024, pitch_px=64, radius_px=14, databar_px=64, seed=0):
    """Write a grayscale JPEG of a contact-hole array and return the true centres, (N, 2) (x, y) in pixels.

    Contacts are dark discs with a bright rim, on a grey background, with blur and noise. A black data bar with
    text is drawn along the bottom edge.
    """
    image = np.full((rows, cols), 110, dtype=np.uint8)
    usable_rows = rows - databar_px
    xs = np.arange(pitch_px // 2, cols - radius_px, pitch_px)
    ys = np.arange(pitch_px // 2, usable_rows - radius_px, pitch_px)
    centers = np.array([(x, y) for y in ys for x in xs])
    for x, y in centers:
        cv2.circle(image, (int(x), int(y)), radius_px + 2, 200, thickness=3)  # bright rim
        cv2.circle(image, (int(x), int(y)), radius_px, 40, thickness=-1)  # dark hole

    image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    noise = np.random.default_rng(seed).normal(0, 8, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    if databar_px > 0:
        image[usable_rows:] = 0
        cv2.putText(image, "SEM  HV 1.0 kV  x100k  FOV 3.0 um", (10, rows - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    encoded.tofile(str(path))  # works for non-ASCII paths, unlike cv2.imwrite
    return centers.astype(np.float64)


def wide_band_image(band_px, pitch_px=64, radius_px=14, hole_grey=40, seed=0):
    """Holes (grey level hole_grey) inside bright bands band_px wide on a grey (110) background; (image, centres)."""
    image = np.full((512, 512), 110, dtype=np.uint8)
    centers = np.array([(x, y) for y in range(40, 480, pitch_px) for x in range(40, 480, pitch_px)])
    for x, y in centers:
        cv2.circle(image, (int(x), int(y)), radius_px + band_px // 2 + 1, 200, thickness=band_px)
        cv2.circle(image, (int(x), int(y)), radius_px, hole_grey, thickness=-1)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    noise = np.random.default_rng(seed).normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8), centers.astype(float)


def render_tile(lattice_nm, true_center_nm, fov_nm=720.0, size_px=512, radius_nm=20.0, seed=0):
    """SEM-like image of the contacts in lattice_nm ((N, 2), nm, y up) seen by a tile whose true centre is
    true_center_nm; contacts drawn at sub-pixel positions."""
    pixel_nm = fov_nm / size_px
    image = np.full((size_px, size_px), 110, dtype=np.uint8)
    local = np.asarray(lattice_nm, dtype=float) - true_center_nm
    radius_px = radius_nm / pixel_nm
    for x_nm, y_nm in local:
        x = x_nm / pixel_nm + (size_px - 1) / 2
        y = -y_nm / pixel_nm + (size_px - 1) / 2
        if -2 * radius_px < x < size_px + 2 * radius_px and -2 * radius_px < y < size_px + 2 * radius_px:
            center = (int(round(x * 16)), int(round(y * 16)))  # 4 fractional bits: sub-pixel drawing
            cv2.circle(image, center, int(round((radius_px + 2) * 16)), 200, thickness=3, shift=4)
            cv2.circle(image, center, int(round(radius_px * 16)), 40, thickness=-1, shift=4)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    noise = np.random.default_rng(seed).normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def exact_ellipse_image(center, semi_axes, size=80, dark=40, bright=180, blur=2.0, oversample=16):
    """Blurred dark ellipse with exact sub-pixel area coverage (cv2's anti-aliased drawing is slightly asymmetric,
    which would bias a test by ~0.1 px)."""
    fine_y, fine_x = (np.indices((size * oversample, size * oversample)) + 0.5) / oversample - 0.5
    inside = ((fine_x - center[0]) / semi_axes[0]) ** 2 + ((fine_y - center[1]) / semi_axes[1]) ** 2 < 1
    coverage = inside.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    image = (bright - (bright - dark) * coverage).astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), blur).round().astype(np.uint8)


def banded_image(centres, radii, size, shading=0.0, noise=0.0, seed=0, oversample=4):
    """Contacts as an SEM shows them: a dark interior (radius px, sub-pixel coverage) and a bright band just outside
    its edge, the band's brightness x (1 + shading cos(angle)) (detector shading), on a grey background; blurred,
    Gaussian noise added."""
    fine_y, fine_x = (np.indices((size * oversample, size * oversample)) + 0.5) / oversample - 0.5
    signal = np.full(fine_x.shape, 110.0)
    for (cx, cy), radius in zip(centres, radii):
        dx, dy = fine_x - cx, fine_y - cy
        edge = np.hypot(dx, dy) - radius  # signed distance from the edge, px
        band = np.exp(-0.5 * ((edge - 2.0) / 1.5) ** 2) * (1 + shading * np.cos(np.arctan2(dy, dx)))
        signal += -50.0 * (edge < 0) + 80.0 * band
    image = cv2.GaussianBlur(signal.reshape(size, oversample, size, oversample).mean(axis=(1, 3)), (0, 0), 1.2)
    image += np.random.default_rng(seed).normal(0, noise, image.shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def no_band_image(centres, radius, shape=(80, 140), noise=0.0, seed=0, oversample=8):
    """Dark disks (60, exact sub-pixel coverage) on a grey background (130), no bright band (wafer ADI images);
    beam blur 1 px, Gaussian noise."""
    fine_y, fine_x = (np.indices((shape[0] * oversample, shape[1] * oversample)) + 0.5) / oversample - 0.5
    inside = np.zeros(fine_x.shape, bool)
    for x, y in centres:
        inside |= np.hypot(fine_x - x, fine_y - y) < radius
    coverage = inside.reshape(shape[0], oversample, shape[1], oversample).mean(axis=(1, 3))
    image = cv2.GaussianBlur((130 - 70 * coverage).astype(np.float32), (0, 0), 1.0)
    image += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def ramp_image():
    """Dark disks (radius 7 px) 50 grey levels below their surroundings, on a background that brightens from 60 to
    200 across the image (charging, shading); the disks' centres; each pixel's distance to the nearest centre."""
    yy, xx = np.mgrid[:200, :300]
    centres = np.array([(x, y) for x in range(30, 300, 40) for y in range(30, 200, 40)], float)
    distance = np.min([np.hypot(xx - x, yy - y) for x, y in centres], axis=0)
    return np.clip(60 + 140 * xx / 299 - 50 * (distance <= 7), 0, 255).astype(np.uint8), centres, distance


# --- Patterns of any shape (extract_any_contours), from AffineRANSAC's tests/test_contours.py ------------------------

def shape_image(inside, shape=(200, 300), dark=60.0, bright=180.0, blur=1.0, noise=0.0, seed=0) -> np.ndarray:
    """A bright pattern on a dark background (uint8): inside(x, y) says where the pattern is (px, y down); each pixel's
    brightness from the share of it the pattern covers (8 x 8 samples), blurred (Gaussian, sigma blur px), with
    Gaussian noise."""
    rows, cols = shape
    fine = (np.arange(8) + 0.5) / 8 - 0.5
    y = (np.arange(rows)[:, None] + fine).reshape(-1, 1)
    x = (np.arange(cols)[:, None] + fine).reshape(1, -1)
    cover = np.broadcast_to(inside(x, y), (rows * 8, cols * 8)).reshape(rows, 8, cols, 8).mean(axis=(1, 3))
    image = cv2.GaussianBlur(dark + (bright - dark) * cover, (0, 0), blur)
    image += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


VERTICAL_EDGES_X = (60.3, 85.7, 180.0, 201.4)  # vertical_lines: the x of each edge, from the left


def vertical_lines(x, y):
    """Two lines from the top to the bottom of the image, between the edges VERTICAL_EDGES_X."""
    return ((x > 60.3) & (x < 85.7)) | ((x > 180.0) & (x < 201.4))


def diagonal_line(x, y):
    """A line 24 px wide, 30 degrees off the x axis, across the whole image: its middle through (150, 100)."""
    return np.abs(across_diagonal(x, y)) < 12


def across_diagonal(x, y):
    """Distance (px) of (x, y) from the diagonal line's middle, across the line."""
    angle = np.radians(30)
    return -(x - 150) * np.sin(angle) + (y - 100) * np.cos(angle)


def ring(x, y):
    """A ring around (150, 100): from radius 30.4 to 55.2."""
    r = np.hypot(x - 150, y - 100)
    return (r > 30.4) & (r < 55.2)


def centre_errors(found: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(N, 2) found - true centre of every true contact (its nearest found centre)."""
    nearest = np.linalg.norm(truth[:, None, :] - found[None, :, :], axis=2).argmin(axis=1)
    return found[nearest] - truth
