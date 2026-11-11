import cv2
import numpy as np

from contour_extraction.check import contact_contrast, keep_contacts
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
