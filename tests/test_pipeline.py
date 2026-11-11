import cv2
import numpy as np
import pytest

from contour_extraction.detect import detect_contacts
from contour_extraction.load import crop_databar, load_sem_image
from contour_extraction.pipeline import REFINEMENTS, extract_contours
from contour_extraction.refine import refine_edges
from synthetic import centre_errors, write_sem_like


def sem_like_tile(tmp_path):
    """A synthetic SEM tile with its data bar cropped, and its true contact centres."""
    path = tmp_path / "tile.jpg"
    truth = write_sem_like(path, databar_px=64)
    return crop_databar(load_sem_image(path), 64), truth


def test_extract_contours_is_detection_then_refinement(tmp_path):
    image, truth = sem_like_tile(tmp_path)

    detected, refined, dropped = extract_contours(image)

    plain = detect_contacts(image)
    np.testing.assert_array_equal(detected.centers, plain.centers)
    np.testing.assert_array_equal(refined.centers, refine_edges(image, plain).centers)
    assert dropped == 0
    assert len(refined.centers) == len(truth)
    assert centre_errors(refined.centers, truth).max() < 0.2


def test_extract_contours_without_refinement_returns_the_detected_contacts(tmp_path):
    image, _ = sem_like_tile(tmp_path)
    detected, refined, _ = extract_contours(image, refinement="none")
    assert refined is detected


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_extract_contours_upscaled_finds_the_same_contacts_in_the_image_pixels(tmp_path, refinement):
    """The options in pixels are those of the image as loaded: upscaling needs no other option changed, and the
    contacts come back in the image's own pixel frame."""
    image, truth = sem_like_tile(tmp_path)
    _, refined, _ = extract_contours(image, upscale=2, refinement=refinement)
    assert len(refined.centers) == len(truth)
    assert centre_errors(refined.centers, truth).max() < 0.2
    assert refined.areas.min() > 300  # in the image's pixels, not the upscaled ones (x4)


def test_extract_contours_contrast_check_drops_regions_cut_out_of_noise():
    noise = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 12, (512, 512)), (0, 0), 1.0)
    blank = np.clip(110 + noise, 0, 255).astype(np.uint8)

    found_without, _, _ = extract_contours(blank, refinement="none")
    detected, refined, dropped = extract_contours(blank, min_contrast=5.0)

    assert len(found_without.centers) > 20  # Otsu always cuts something out of noise
    assert len(detected.centers) == len(refined.centers) == 0
    assert dropped == len(found_without.centers)
