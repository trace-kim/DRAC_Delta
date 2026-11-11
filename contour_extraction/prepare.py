"""Step 2, prepare: the image the contacts are found in (prepared_image).

Each part is optional: flatten_background() removes brightness that varies across the image (charging, shading), so
that one threshold fits the whole image; denoised_image() removes pixel noise; upscaled_image() interpolates the image
to more pixels, and in_image_pixels() puts the contacts found in it back in the image's own pixels.
"""

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle

from .contacts import DetectedContacts

DENOISERS = ("none", "median", "non-local means", "total variation")  # denoised_image methods
NLM_PATCH_PX, NLM_SEARCH_PX = 7, 21  # non-local means: patch and search window, OpenCV's recommended sizes


def flatten_background(image: np.ndarray, size_px: float, dark_contacts: bool = True) -> np.ndarray:
    """The image (uint8) with its slowly varying background removed, so that one threshold fits the whole image.

    The background: a grey-level closing with a disk size_px across (an opening for bright contacts), which fills
    every contact narrower than the disk with the brightness around it, then smoothed (Gaussian, sigma 2 x size_px:
    the closing of a noisy image is noisy itself, and that noise at the scale of a contact would move the centres).
    For the smoothing the background is continued past the image's edges with its own slope (odd reflection: a
    mirror would bend a ramp there and move the contacts near the edges). Returned: the image minus the background
    plus the background's mean, so that grey levels stay in the same range.
    """
    side = max(3, int(round(size_px)) | 1)  # odd
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE if dark_contacts else cv2.MORPH_OPEN, disk)
    sigma = 2 * size_px
    pad = int(np.ceil(3 * sigma))
    continued = np.pad(background.astype(np.float32), pad, mode="reflect", reflect_type="odd")
    background = cv2.GaussianBlur(continued, (0, 0), sigma)[pad:-pad, pad:-pad]
    flat = image.astype(np.float32) - background + background.mean()
    return np.clip(np.round(flat), 0, 255).astype(np.uint8)


def denoised_image(image: np.ndarray, method: str = "none", median_px: int = 3, nlm_h: float = 10.0,
                   tv_weight: float = 0.1) -> np.ndarray:
    """The image (uint8) with its pixel noise removed by `method` (DENOISERS); "none": the image itself.

    "median": each pixel the median of a median_px x median_px window (odd): removes isolated bright or dark pixels
    and spots narrower than half the window, and keeps straight edges where they are.
    "non-local means": each pixel a weighted mean of the pixels whose NLM_PATCH_PX patch looks like its own, within
    NLM_SEARCH_PX (cv2.fastNlMeansDenoising); nlm_h (grey levels) sets how alike: larger removes more noise, and detail.
    "total variation": the image closest to it with the least total variation (Chambolle's algorithm on the grey
    levels / 255): flat regions with sharp edges between them; tv_weight: larger is flatter.
    """
    if method == "none":
        return image
    if method == "median":
        return cv2.medianBlur(image, median_px)
    if method == "non-local means":
        return cv2.fastNlMeansDenoising(image, None, h=nlm_h, templateWindowSize=NLM_PATCH_PX,
                                        searchWindowSize=NLM_SEARCH_PX)
    if method == "total variation":
        flat = denoise_tv_chambolle(image / 255.0, weight=tv_weight)
        return np.clip(np.round(flat * 255), 0, 255).astype(np.uint8)
    raise ValueError(f"denoise must be one of {DENOISERS}, not {method!r}")


def upscaled_image(image: np.ndarray, factor: int) -> np.ndarray:
    """The image interpolated to `factor` times as many pixels per side (bicubic); factor 1: the image itself.

    Both cover the same area: pixel (x, y) of the result lies at ((x + 0.5) / factor - 0.5, (y + 0.5) / factor - 0.5)
    of the image (in_image_pixels).
    """
    if factor == 1:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def in_image_pixels(contacts: DetectedContacts, factor: int) -> DetectedContacts:
    """Contacts found in an image upscaled by `factor` (upscaled_image), in the pixel frame of the image itself."""
    if factor == 1:
        return contacts

    def back(xy):
        return (xy + 0.5) / factor - 0.5
    return DetectedContacts(back(contacts.centers), contacts.areas / factor ** 2, [back(c) for c in contacts.contours],
                            contacts.threshold, contacts.left_out)


def prepared_image(image: np.ndarray, background_px: float = 0.0, dark_contacts: bool = True, denoise: str = "none",
                   median_px: int = 3, nlm_h: float = 10.0, tv_weight: float = 0.1, upscale: int = 1) -> np.ndarray:
    """The image the contacts are found in: its background flattened over background_px (flatten_background) if that
    is > 0, then denoised (denoised_image: method `denoise` with its option), then upscaled `upscale` times
    (upscaled_image); with none of them, the image itself.

    Denoised before it is upscaled: interpolation spreads each noisy pixel over several, which no longer look like
    noise.
    """
    flat = flatten_background(image, background_px, dark_contacts) if background_px > 0 else image
    return upscaled_image(denoised_image(flat, denoise, median_px, nlm_h, tv_weight), upscale)
