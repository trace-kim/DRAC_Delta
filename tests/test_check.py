import cv2
import numpy as np

from contour_extraction.check import contact_contrast, contour_solidity, keep_contacts, keep_contours
from contour_extraction.contacts import Contours
from contour_extraction.detect import detect_contacts
from synthetic import render_tile


def test_contact_contrast_tells_contacts_from_regions_cut_out_of_noise():
    lattice = np.array([(x, y) for x in np.arange(-1530, 1530, 90.0) for y in np.arange(-1530, 1530, 90.0)])
    pattern = render_tile(lattice, (0.0, 0.0), seed=1)
    noise = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 12, pattern.shape), (0, 0), 1.0)  # correlated
    blank = np.clip(110 + noise, 0, 255).astype(np.uint8)

    real, false = (detect_contacts(image, method="otsu") for image in (pattern, blank))

    assert len(real.centers) > 20 and len(false.centers) > 20  # Otsu always cuts something out of noise
    assert contact_contrast(pattern, real).min() > 10
    assert contact_contrast(blank, false).max() < 5
    kept = keep_contacts(real, np.arange(len(real.centers)) % 2 == 0)
    assert len(kept.contours) == len(kept.centers) == len(kept.areas) == (len(real.centers) + 1) // 2


def test_contour_solidity_is_one_for_a_convex_contour_and_less_for_a_dented_one():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    dented = np.array([[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]], float)  # a quarter of it cut out
    assert contour_solidity(square) == 1.0 and contour_solidity(square[::-1]) == 1.0
    assert contour_solidity(dented) == 0.75
    assert contour_solidity(np.array([[0, 0], [5, 0], [10, 0]], float)) == 0.0  # no area


def test_keep_contours_keeps_the_chosen_ones_and_counts_the_others():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    contours = Contours([square, square + 20, square + 40], [square * 0] * 3, np.array([True, False, True]),
                        np.array([1, 2, 3]), 97.0, {"small": 2})

    kept = keep_contours(contours, np.array([True, False, True]), {"open": 1})

    assert len(kept.points) == len(kept.normals) == 2 and list(kept.closed) == [True, True]
    np.testing.assert_array_equal(kept.points[1], square + 40)
    assert list(kept.region) == [1, 3] and kept.threshold == 97.0 and kept.left_out == {"small": 2, "open": 1}
