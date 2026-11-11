"""Step 4, check: tell real contacts from regions that thresholding cut out of plain noise.

Thresholding always finds two classes, even in an image without pattern, so detection cuts regions out of noise too.
contact_contrast() gives the contrast of each contour against a ring around it, in noise sigma: about one for those
regions, many for real contacts; keep_contacts() keeps the contacts chosen.
"""

import cv2
import numpy as np

from .contacts import DetectedContacts


def noise_sigma(image: np.ndarray, blur_sigma: float = 1.5) -> float:
    """Robust pixel noise of the image (grey levels): 1.4826 x the median absolute deviation of the image minus its
    blurred copy. Spatially correlated noise makes it come out lower than the real noise."""
    detail = image.astype(np.float64) - cv2.GaussianBlur(image.astype(np.float64), (0, 0), sigmaX=blur_sigma)
    return 1.4826 * float(np.median(np.abs(detail - np.median(detail))))


def contact_contrast(image: np.ndarray, contacts: DetectedContacts, ring_px: int = 3) -> np.ndarray:
    """(N,) |mean grey level inside each contour - mean in a ring ring_px wide around it| / noise_sigma.

    Real contacts stand out by many noise sigma; regions that thresholding cut out of plain noise (an image without
    pattern) by about one.
    """
    sigma = max(noise_sigma(image), 1e-6)
    kernel = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    rows, cols = image.shape
    contrast = np.zeros(len(contacts.contours))
    for k, contour in enumerate(contacts.contours):
        low = np.maximum(contour.min(axis=0) - ring_px - 1, 0).astype(int)
        high = np.minimum(contour.max(axis=0) + ring_px + 2, (cols, rows)).astype(int)
        crop = image[low[1]:high[1], low[0]:high[0]].astype(np.float64)
        inside = np.zeros(crop.shape, np.uint8)
        cv2.fillPoly(inside, [np.round(contour - low).astype(np.int32)], 1)
        ring = cv2.dilate(inside, kernel) - inside
        if inside.any() and ring.any():
            contrast[k] = abs(crop[inside > 0].mean() - crop[ring > 0].mean()) / sigma
    return contrast


def keep_contacts(contacts: DetectedContacts, keep: np.ndarray) -> DetectedContacts:
    """The contacts where keep (N,) bool is True, in the same order."""
    return DetectedContacts(contacts.centers[keep], contacts.areas[keep],
                            [c for c, k in zip(contacts.contours, keep) if k], contacts.threshold, contacts.left_out)
