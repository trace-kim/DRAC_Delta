import cv2
import numpy as np
import pytest
from skimage.restoration import denoise_tv_chambolle

from contour_extraction.contacts import DetectedContacts
from contour_extraction.prepare import (denoised_image, flatten_background, in_image_pixels, prepared_image,
                                        upscaled_image)
from synthetic import ramp_image


def test_flatten_background_removes_a_ramp_and_keeps_the_contacts_contrast():
    image, _, distance = ramp_image()
    flat = flatten_background(image, 20)
    background, holes = distance > 10, distance <= 4
    assert np.ptp(image[background]) == 140
    assert np.ptp(flat[background]) < 10  # measured: 6
    assert flat[background].mean() - flat[holes].mean() > 45  # measured: 49.8 of 50
    assert abs(float(flat.mean()) - float(image.mean())) < 5  # grey levels stay in the image's range


def test_each_denoising_method_is_the_librarys_own():
    image = np.random.default_rng(0).integers(0, 256, (40, 50)).astype(np.uint8)
    assert denoised_image(image) is image
    np.testing.assert_array_equal(denoised_image(image, "median", median_px=5), cv2.medianBlur(image, 5))
    np.testing.assert_array_equal(denoised_image(image, "non-local means", nlm_h=12.0),
                                  cv2.fastNlMeansDenoising(image, None, 12.0, 7, 21))
    flat = denoise_tv_chambolle(image / 255.0, weight=0.2)
    np.testing.assert_array_equal(denoised_image(image, "total variation", tv_weight=0.2),
                                  np.clip(np.round(flat * 255), 0, 255).astype(np.uint8))
    with pytest.raises(ValueError, match="denoise must be one of"):
        denoised_image(image, "wavelet")


def test_the_median_removes_isolated_pixels_and_keeps_an_edge_where_it_is():
    edge = np.full((20, 20), 60, np.uint8)
    edge[:, 10:] = 130
    spotted = edge.copy()
    spotted[5, 4], spotted[12, 15] = 200, 0  # a bright pixel in the dark side, a dark one in the bright side
    np.testing.assert_array_equal(denoised_image(spotted, "median"), edge)


def test_an_upscaled_image_covers_the_same_area_and_its_contacts_map_back():
    """Pixel (x, y) of the image upscaled f times lies at ((x + 0.5) / f - 0.5, (y + 0.5) / f - 0.5) of the image; a
    spot's centroid in the upscaled image maps back onto its pixel, and areas shrink by f squared."""
    image = np.zeros((9, 11), np.uint8)
    image[4, 6] = 200
    assert upscaled_image(image, 1) is image
    for factor in (2, 3, 4):
        big = upscaled_image(image, factor)
        assert big.shape == (9 * factor, 11 * factor)
        rows, cols = np.indices(big.shape)
        weight = big.astype(float)
        centre = np.array([[(weight * cols).sum(), (weight * rows).sum()]]) / weight.sum()
        found = in_image_pixels(DetectedContacts(centre, np.array([16.0 * factor ** 2]), [centre], 100.0), factor)
        np.testing.assert_allclose(found.centers, [[6, 4]], atol=1e-6)
        np.testing.assert_allclose(found.contours[0], [[6, 4]], atol=1e-6)
        assert found.areas[0] == 16 and found.threshold == 100.0


def test_prepared_image_with_no_option_is_the_image_itself():
    image = np.random.default_rng(0).integers(0, 256, (30, 40)).astype(np.uint8)
    assert prepared_image(image) is image
    assert prepared_image(image, upscale=2).shape == (60, 80)
