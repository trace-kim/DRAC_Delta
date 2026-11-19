"""Step 3, detect: the contacts of an image by thresholding (detect_contacts).

Blur slightly, threshold to a mask of candidate contact pixels (contact_mask: "otsu", "otsu3" or "band"), split the
mask into connected regions, drop the regions that are not contacts (touching the border, too small or too large, not
solid enough); each region left is a contact: centre = centroid of its pixels, contour = its outer boundary, a pixel
staircase that step 5 (refine) moves to the sub-pixel edge.

Or the contours of any pattern (detect_contours, AffineRANSAC D155), nothing assumed about it: no shape, size or
direction, and a region need not lie inside the image. Every region of the same mask (within the area limits) gives its
whole boundary: its outer one and that of each hole in it. The points on the image's outermost rows and columns are not
edges (the region runs out of the image there) and are taken out (inside_image): a boundary that never gets there is
one closed contour, the others fall into open contours, each ending at the image border. Each point has the direction
its edge faces, out of its region, which step 5 moves it along (refine_contours). mask_contours() is its second half,
for a mask made any other way (a workflow: modules.py).

Or without a threshold: canny_edges() marks the pixels where the brightness changes fastest, as lines one pixel wide,
and edge_contours() follows those lines (edge_chains) into the same kind of contours, each point put where the
gradient peaks between the pixels (gradient_peaks) and facing the brighter side. This is the Canny/Devernay edge
detector (Grompone von Gioi and Randall, IPOL 2017, doi 10.5201/ipol.2017.216) with its own way of chaining.
"""

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.morphology import thin

from .contacts import Contours, DetectedContacts

METHODS = ("otsu", "otsu3", "band")  # detect_contacts threshold methods
BAND_MIN_SOLIDITY = 0.9  # "band": regions less solid than this are background pinched off between bands
LEFT_OUT = ("border", "small", "large", "not solid")  # why a region is not a contact
CONTOUR_LEFT_OUT = ("small", "large")  # why a region gives no contour (detect_contours)
CANNY_TOP_SHARE = 0.3  # canny_edges, high 0: this share of the steepest gradient in the image


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


def detect_contours(image: np.ndarray, dark: bool = True, blur_sigma: float = 1.5, min_area_px: int = 20,
                    max_area_px: int | None = None, method: str = "otsu",
                    normal_span: int = 3) -> tuple[Contours, np.ndarray]:
    """The contours of any pattern in a grayscale uint8 image (module docstring), and the region number of every pixel
    (0: none), which refine_contours needs. dark: the regions are the darker side.

    The mask of contact_mask is split into connected regions ("band": 4-connected) as for contacts; a region of an area
    outside [min_area_px, max_area_px] (max None: no limit) is left out (counted in left_out). The normals: a tangent
    across +-normal_span points (right_of_tangent), out of the region (a hole's: into the hole).
    """
    mask, threshold = contact_mask(image, dark, blur_sigma, method)
    return mask_contours(mask, min_area_px, max_area_px, 4 if method == "band" else 8, normal_span,
                         threshold=float(threshold))


def mask_contours(mask: np.ndarray, min_area_px: int = 20, max_area_px: int | None = None, connectivity: int = 8,
                  normal_span: int = 3, holes: bool = True, open_at_border: bool = True,
                  threshold: float = np.nan) -> tuple[Contours, np.ndarray]:
    """The contours of the regions of a mask (uint8, 1 = region), and the region number of every pixel (0: none):
    what detect_contours does with its threshold mask. threshold: the grey level the mask was cut at, kept in the
    Contours (NaN: none).

    connectivity 8: pixels that touch by a corner belong to one region; 4: they do not. holes False: only the outer
    boundary of each region, not those of the holes in it. open_at_border False: a region that touches the image
    border is left out (counted under "border", before its area is looked at, as detect_contacts does) instead of
    giving open contours.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    rows, cols = mask.shape
    points, normals, closed, region = [], [], [], []
    left_out = dict.fromkeys(CONTOUR_LEFT_OUT if open_at_border else ("border",) + CONTOUR_LEFT_OUT, 0)
    for label in range(1, count):  # label 0 is not mask
        left, top, width, height, area = stats[label]
        if not open_at_border and (left == 0 or top == 0 or left + width == cols or top + height == rows):
            left_out["border"] += 1
            continue
        if area < min_area_px:
            left_out["small"] += 1
            continue
        if max_area_px is not None and area > max_area_px:
            left_out["large"] += 1
            continue
        pixels = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
        boundaries, hierarchy = cv2.findContours(pixels, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for boundary, (_, _, _, parent) in zip(boundaries, hierarchy[0]):
            if parent >= 0 and not holes:
                continue
            boundary = boundary.reshape(-1, 2).astype(float) + [left, top]  # crop -> image coordinates
            side = out_of_polygon(boundary) * (-1.0 if parent >= 0 else 1.0)  # a hole: out of the region is into it
            for piece, whole in inside_image(boundary, mask.shape):
                points.append(piece)
                normals.append(side * right_of_tangent(piece, normal_span, whole))
                closed.append(whole)
                region.append(label)
    return Contours(points, normals, np.array(closed, bool), np.array(region, int), float(threshold), left_out), labels


def canny_edges(image: np.ndarray, blur_sigma: float = 2.0, high: float = 0.0,
                low_ratio: float = 0.5) -> tuple[np.ndarray, float]:
    """Canny's edges of a grayscale uint8 image: (mask, high). The mask (uint8) is 1 on the edge pixels, which form
    lines one pixel wide along the steepest brightness changes; high is the threshold used.

    The image is blurred (Gaussian, blur_sigma) and its gradient taken (Sobel), in grey levels per pixel. An edge
    pixel is a local maximum of the gradient's size across the edge that is above `high`, or above low_ratio x high
    and joined to one above `high` (hysteresis; cv2.Canny, with the gradient's Euclidean size).
    high 0: CANNY_TOP_SHARE of the steepest gradient in the image, so that it follows the image's contrast: fit for a
    pattern whose edges are about equally strong; an image without pattern gives edges of noise.
    """
    smooth = image.astype(np.float32)
    if blur_sigma > 0:
        smooth = cv2.GaussianBlur(smooth, (0, 0), blur_sigma)
    dx, dy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(smooth, cv2.CV_32F, 0, 1) / 8  # grey levels per pixel
    if high <= 0:
        high = CANNY_TOP_SHARE * float(np.hypot(dx, dy).max())
    # cv2.Canny takes the gradient as 16-bit integers: 128 steps per grey level per pixel (at most 127.5 of them)
    edges = cv2.Canny(np.round(dx * 128).astype(np.int16), np.round(dy * 128).astype(np.int16),
                      low_ratio * high * 128, high * 128, L2gradient=True)
    return (edges > 0).astype(np.uint8), float(high)


def edge_chains(edges: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """The pixels of thin lines (a mask, 1 = line; canny_edges) in order along each line: [(pixels (M, 2) as (x, y),
    closed)]. A line that closes on itself is closed; the others are open, each from one end to the other, a line
    that branches as several (walk_chains).

    The mask is thinned first, to lines no pixel of which can go without breaking them (skimage's thin): Canny's
    slanted edges are staircases whose steps overlap by a pixel, and round such a line the walk comes to other pixels
    on its way back than on its way out. (A mask of wider regions is thinned to their middle lines.)
    """
    walks, hierarchy = cv2.findContours(thin(edges).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:  # no line
        return []
    # the walk round the outside of every group of joined pixels (those round the holes in it walk the same pixels)
    return [chain for walk, (_, _, _, parent) in zip(walks, hierarchy[0]) if parent < 0
            for chain in walk_chains(walk.reshape(-1, 2))]


def walk_chains(walk: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """The lines in one walk (N, 2) of cv2.findContours round a group of joined line pixels: [(pixels, closed)].

    The walk goes once round a closed line, and comes to each of its pixels once. Along an open line it goes out and
    back, and comes to each pixel twice. So the walk is followed, keeping the path of the pixels it comes to for the
    first time; where it is on pixels it has been on before, either it has closed a loop (it comes back to the path
    far behind: the path from there is a closed line), or it has turned round and walks back (what it walks back over
    is taken off the path, as an open line: a spur of a closed line, a branch, or the line itself). The path left at
    the end is a closed line, or the pixel the walk began at, part-way along an open line: the two parts of that line
    are joined there.
    """
    n = len(walk)
    key = walk[:, 0].astype(np.int64) + (walk[:, 1].astype(np.int64) << 32)
    _, index, inverse = np.unique(key, return_index=True, return_inverse=True)
    first = index[inverse]  # for each step of the walk, the step that first came to its pixel
    new = first == np.arange(n)
    if new.all():
        return [(walk, n >= 3)]
    new &= np.roll(new, 1) | np.roll(new, -1)  # one new pixel between old ones: a corner cut short on the way out
    pieces = np.split(np.arange(n), np.flatnonzero(np.diff(new)) + 1)  # in turn new, old, new, ... steps
    path = np.empty(0, int)  # steps of the walk, rising
    closed, hanging = [], []  # hanging: (steps walked back over, the step of the path they hang from)
    for number, piece in enumerate(pieces):
        if new[piece[0]]:
            path = np.concatenate([path, piece])
            continue
        at = np.searchsorted(path, first[piece[0]])
        if at < len(path) - 3 and path[at] == first[piece[0]]:  # back on the path, and not by turning round: a loop
            closed.append(path[at:])
            path = path[:at + 1]
        back_to = 0 if number == len(pieces) - 1 else first[piece[-1]]  # the walk ends where it began
        at = np.searchsorted(path, back_to)
        if at < len(path) - 1 and path[at] == back_to:
            hanging.append((path[at + 1:], back_to))
            path = path[:at + 1]
    if len(path) >= 3:  # never walked back over: a closed line
        closed.append(path)
        opened = [steps for steps, _ in hanging]
    else:  # one open line through what is left: a line that hangs from its start, backwards, and one from its end
        at_start = [k for k, (_, back_to) in enumerate(hanging) if back_to == path[0]][:1]
        at_end = [k for k, (_, back_to) in enumerate(hanging) if back_to == path[-1] and k not in at_start][-1:]
        before = hanging[at_start[0]][0][::-1] if at_start else path[:0]
        after = hanging[at_end[0]][0] if at_end else path[:0]
        opened = [np.concatenate([before, path, after])]
        opened += [steps for k, (steps, _) in enumerate(hanging) if k not in at_start + at_end]  # the branches
    return [(walk[steps], True) for steps in closed] + [(walk[steps], False) for steps in opened]


def edge_contours(edges: np.ndarray, image: np.ndarray, min_points: int = 20, normal_span: int = 3,
                  blur_sigma: float = 2.0) -> Contours:
    """The lines of an edge mask (canny_edges) as contours: each line of min_points pixels or more (edge_chains; the
    shorter ones are counted in left_out), closed or open.

    Each pixel of a line is moved across the line to where the gradient of the image (blurred with blur_sigma, as
    canny_edges blurs it) is steepest between the pixels (gradient_peaks): Canny's edges with Devernay's correction.
    At each point the direction its edge faces: across the line (right_of_tangent, +-normal_span points), towards the
    brighter side (one side for the whole line: the one most of its points have). The contours bound no region
    (region 0) and have no threshold (NaN).
    """
    smooth = image.astype(np.float32)
    if blur_sigma > 0:
        smooth = cv2.GaussianBlur(smooth, (0, 0), blur_sigma)
    gradient = np.dstack([cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)])
    size = np.hypot(gradient[..., 0], gradient[..., 1])
    points, normals, closed = [], [], []
    left_out = {"short": 0}
    for chain, whole in edge_chains(edges):
        if len(chain) < min_points:
            left_out["short"] += 1
            continue
        uphill = gradient[chain[:, 1], chain[:, 0]].astype(float)  # at each pixel: towards the brighter side
        placed = gradient_peaks(size, chain.astype(float), uphill)
        normal = right_of_tangent(placed, normal_span, whole)
        points.append(placed)
        normals.append(normal if np.sign((normal * uphill).sum(axis=1)).sum() >= 0 else -normal)
        closed.append(whole)
    return Contours(points, normals, np.array(closed, bool), np.zeros(len(points), int), np.nan, left_out)


def gradient_peaks(size: np.ndarray, pixels: np.ndarray, uphill: np.ndarray) -> np.ndarray:
    """Edge pixels (M, 2) moved to the peak of the gradient's size across their edge, between the pixels (Devernay's
    correction of Canny's edges): (M, 2).

    size: the size of the image's gradient at every pixel; uphill (M, 2): the gradient at each edge pixel, which
    points across the edge. A parabola through the size at the pixel and one pixel before and after it across the
    edge (sampled between the pixels) has its top where the edge is: at most one pixel away; a pixel that is no peak
    (no top, no gradient) stays. Taking the gradient at the pixels first, and its peak after, keeps what sampling the
    image along a line and differentiating that loses: there the gradient is flat within a pixel.
    """
    length = np.linalg.norm(uphill, axis=1, keepdims=True)
    across = np.divide(uphill, length, out=np.zeros_like(uphill), where=length > 0)
    behind, at, ahead = (sampled(size, pixels + k * across) for k in (-1, 0, 1))
    bend = behind - 2 * at + ahead
    shift = np.where(bend < 0, 0.5 * (behind - ahead) / np.where(bend < 0, bend, 1.0), 0.0)
    return pixels + np.clip(shift, -1, 1)[:, None] * across


def sampled(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(M,) the image between its pixels at points (M, 2) (x, y), bilinear; beyond the image: at its border."""
    rows, cols = image.shape
    x, y = np.clip(points[:, 0], 0, cols - 1), np.clip(points[:, 1], 0, rows - 1)
    left, top = np.minimum(x.astype(int), cols - 2), np.minimum(y.astype(int), rows - 2)
    fx, fy = x - left, y - top
    return (image[top, left] * (1 - fx) * (1 - fy) + image[top, left + 1] * fx * (1 - fy)
            + image[top + 1, left] * (1 - fx) * fy + image[top + 1, left + 1] * fx * fy)


def out_of_polygon(boundary: np.ndarray) -> float:
    """+1 if right_of_tangent's vectors point out of a closed boundary (M, 2), -1 if into it: by the sign of its area,
    as refine.outward_normals."""
    x, y = boundary[:, 0], boundary[:, 1]
    signed_area = (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2
    return 1.0 if signed_area > 0 else -1.0


def inside_image(boundary: np.ndarray, shape: tuple[int, int]) -> list[tuple[np.ndarray, bool]]:
    """A closed boundary (M, 2) without its points on the outermost rows and columns of an image of this shape:
    [(points, closed)]. None there: the boundary itself, closed. Otherwise the runs of points between them, each open
    (runs of fewer than 2 points are dropped: no direction)."""
    rows, cols = shape
    x, y = boundary[:, 0], boundary[:, 1]
    on_border = (x == 0) | (y == 0) | (x == cols - 1) | (y == rows - 1)
    if not on_border.any():
        return [(boundary, True)]
    start = np.flatnonzero(on_border)[0]  # begin on the border, so that no run goes on from the last point to the first
    boundary, on_border = np.roll(boundary, -start, axis=0), np.roll(on_border, -start)
    runs = np.split(boundary, np.flatnonzero(np.diff(on_border)) + 1)  # in turn on the border, off it, on it, ...
    return [(run, False) for run in runs[1::2] if len(run) >= 2]


def right_of_tangent(points: np.ndarray, span: int, closed: bool) -> np.ndarray:
    """(M, 2) unit vectors at right angles to a contour (M, 2): (ty, -tx) of its unit tangent (tx, ty), all on the same
    side of it. The tangent at point k runs from point k - span to point k + span, which smooths the pixel staircase;
    an open contour's stops at its ends. Where the two coincide (the tip of a 1 px wide spur) there is no tangent:
    (0, 0), so the point is not moved."""
    k = np.arange(len(points))
    if closed:
        ahead, behind = points[(k + span) % len(points)], points[(k - span) % len(points)]
    else:
        ahead, behind = points[np.minimum(k + span, len(points) - 1)], points[np.maximum(k - span, 0)]
    tangent = ahead - behind
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 0)
    return np.column_stack([tangent[:, 1], -tangent[:, 0]])
