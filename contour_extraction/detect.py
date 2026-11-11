"""Step 3, detect: the contacts of an image by thresholding (detect_contacts).

Blur slightly, threshold to a mask of candidate contact pixels (contact_mask: "otsu", "otsu3" or "band"), split the
mask into connected regions, drop the regions that are not contacts (touching the border, too small or too large, not
solid enough); each region left is a contact: centre = centroid of its pixels, contour = its outer boundary, a pixel
staircase that step 5 (refine) moves to the sub-pixel edge.
"""

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

from .contacts import DetectedContacts

METHODS = ("otsu", "otsu3", "band")  # detect_contacts threshold methods
BAND_MIN_SOLIDITY = 0.9  # "band": regions less solid than this are background pinched off between bands
LEFT_OUT = ("border", "small", "large", "not solid")  # why a region is not a contact


def contact_mask(image: np.ndarray, dark_contacts: bool = True, blur_sigma: float = 1.5, method: str = "otsu"):
    """Blur, then threshold by method. Returns (mask, threshold); mask is 1 on candidate contact pixels.

    "otsu":  plain Otsu (two grey-level classes).
    "otsu3": three classes (hole interior | background | bright band at the hole edge, multi-Otsu); the threshold
             between the hole and the background. Use it when wide bands make plain Otsu split background | band.
    "band":  the bright band: the mask is every pixel NOT above the threshold between background and band. The
             background is the large region touching the border (dropped by detect_contacts), so the regions left
             are those enclosed by a band. It does not need the hole interior to be darker than the background.
    Bright contacts (dark_contacts False): the same with the grey levels the other way round.
    Comparisons as in cv2: <= threshold is the dark side.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma) if blur_sigma > 0 else image
    if method == "otsu":
        mode = cv2.THRESH_BINARY_INV if dark_contacts else cv2.THRESH_BINARY
        threshold, mask = cv2.threshold(blurred, 0, 1, mode + cv2.THRESH_OTSU)
        return mask, threshold
    low, high = threshold_multiotsu(blurred, classes=3)
    if method == "otsu3":
        threshold = low if dark_contacts else high
    else:  # band: bright band around dark contacts, dark band around bright ones
        threshold = high if dark_contacts else low
    mask = blurred <= threshold if dark_contacts else blurred > threshold
    return mask.astype(np.uint8), float(threshold)


def solidity(contour: np.ndarray, area: float) -> float:
    """Pixel area / area of the convex hull of the contour (slightly above 1 for convex pixel regions)."""
    hull_area = cv2.contourArea(cv2.convexHull(contour.astype(np.int32)))
    return area / hull_area if hull_area > 0 else 0.0


def outer_contour(labels: np.ndarray, label: int, left: int, top: int, width: int, height: int) -> np.ndarray:
    """Outer boundary of one labelled region (bounding box left, top, width, height) as an (M, 2) array of (x, y)
    pixel coordinates."""
    region = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
    found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = max(found, key=len).reshape(-1, 2)  # one region -> one outer contour
    return boundary + [left, top]  # crop coordinates -> image coordinates


def detect_contacts(
    image: np.ndarray,
    dark_contacts: bool = True,
    blur_sigma: float = 1.5,
    min_area_px: int = 20,
    max_area_px: int | None = None,
    method: str = "otsu",
    min_solidity: float | None = None,
) -> DetectedContacts:
    """Find contacts in a grayscale uint8 image.

    The mask of contact_mask is split into connected regions ("band": 4-connected, so a region cannot leak
    diagonally through a thin band), each one candidate contact. A region is left out (counted in left_out, under the
    first reason that applies) if it touches the image border (a cut-off contact, the data bar, the background for
    "band"), its area is outside [min_area_px, max_area_px] (max None: no limit), or its solidity is below
    min_solidity (contacts are convex, ~1; background pinched off between nearly touching bands is concave, ~0.7-0.88;
    "band" uses BAND_MIN_SOLIDITY unless given). Centre: the centroid of the region's pixels; contour: its outer
    boundary.
    """
    mask, threshold = contact_mask(image, dark_contacts, blur_sigma, method)
    if min_solidity is None and method == "band":
        min_solidity = BAND_MIN_SOLIDITY
    connectivity = 4 if method == "band" else 8
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    rows, cols = image.shape

    centers, areas, contours = [], [], []
    left_out = dict.fromkeys(LEFT_OUT, 0)
    for label in range(1, count):  # label 0 is the background
        left, top, width, height, area = stats[label]
        if left == 0 or top == 0 or left + width == cols or top + height == rows:
            left_out["border"] += 1
        elif area < min_area_px:
            left_out["small"] += 1
        elif max_area_px is not None and area > max_area_px:
            left_out["large"] += 1
        else:
            contour = outer_contour(labels, label, left, top, width, height)
            if min_solidity is not None and solidity(contour, area) < min_solidity:
                left_out["not solid"] += 1
            else:
                centers.append(centroids[label])
                areas.append(area)
                contours.append(contour)

    return DetectedContacts(
        centers=np.array(centers, dtype=np.float64).reshape(-1, 2),
        areas=np.array(areas, dtype=np.float64),
        contours=contours,
        threshold=float(threshold),
        left_out=left_out,
    )
