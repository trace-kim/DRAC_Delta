"""The result of every step: contacts with their centres, areas and contours (DetectedContacts), or the contours of any
pattern (Contours); and the polygon measures of a contour.

Coordinates are in the pixel frame: (x, y) = (column, row) in pixels, y pointing DOWN, pixel centres at integer
values, (0, 0) the centre of the top-left pixel.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectedContacts:
    centers: np.ndarray  # (N, 2) centroid (x, y) in pixels
    areas: np.ndarray  # (N,) area in pixels
    contours: list[np.ndarray]  # N arrays of shape (M, 2), boundary points (x, y) in pixels
    threshold: float  # threshold that was used (grey level; for "band" the band threshold)
    left_out: dict[str, int] = field(default_factory=dict)  # regions not taken as contacts, per reason (detect.LEFT_OUT)


@dataclass
class Contours:
    """The contours of any pattern in one image (detect.detect_contours): each a run of edge points in order, each
    point with the direction its edge faces, out of its region."""
    points: list[np.ndarray]   # N arrays (M, 2): the points of each contour, in order, (x, y) px
    normals: list[np.ndarray]  # N arrays (M, 2): at each point the unit direction its edge faces, out of its region
    closed: np.ndarray         # (N,) bool: the last point joins the first; False: both ends at the image border
    region: np.ndarray         # (N,) number of the region each contour bounds (with a hole, or cut: several of one)
    threshold: float           # the detection's threshold (grey level)
    left_out: dict[str, int] = field(default_factory=dict)  # regions without a contour, per reason (CONTOUR_LEFT_OUT)


# --- Polygon measures (shoelace formula); vertices (N, 2), not repeated at the end -----------------------------------

def polygon_area(vertices: np.ndarray) -> float:
    """Area of a simple polygon (positive for either vertex order)."""
    x, y = vertices[:, 0], vertices[:, 1]
    return abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Area centroid (x, y) of a simple polygon."""
    x, y = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])
