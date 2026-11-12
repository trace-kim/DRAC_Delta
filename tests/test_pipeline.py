import cv2
import numpy as np
import pytest

from contour_extraction.detect import detect_contacts
from contour_extraction.load import crop_databar, load_sem_image
from contour_extraction.pipeline import REFINEMENTS, extract_any_contours, extract_contours
from contour_extraction.refine import RULES, refine_edges
from synthetic import (VERTICAL_EDGES_X, across_diagonal, centre_errors, diagonal_line, render_tile, ring,
                       shape_image, vertical_lines, write_sem_like)

BRIGHT = dict(dark_contacts=False)  # extract_any_contours: the bright regions


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


# --- extract_any_contours: the contours of any pattern (AffineRANSAC's tests/test_contours.py) ----------------------

def test_any_contours_of_lines_running_out_of_the_image_are_open_one_per_edge():
    image = shape_image(vertical_lines)
    detected, refined = extract_any_contours(image, **BRIGHT)
    assert len(refined.points) == 4 and not refined.closed.any()
    assert len(np.unique(refined.region)) == 2 and detected.left_out == {"small": 0, "large": 0}
    order = np.argsort([points[:, 0].mean() for points in refined.points])
    for k, x_edge, faces in zip(order, VERTICAL_EDGES_X, (-1, 1, -1, 1)):
        points, normals = refined.points[k], refined.normals[k]
        np.testing.assert_allclose(points[:, 0], x_edge, atol=0.05)
        np.testing.assert_allclose(normals, np.tile([faces, 0.0], (len(points), 1)), atol=1e-9)  # out of its line
        # from one image border to the other, without the outermost rows
        assert points[:, 1].min() == 1 and points[:, 1].max() == image.shape[0] - 2
        assert np.all(np.abs(detected.points[k][:, 0] - x_edge) <= 1)


def test_any_contours_of_a_diagonal_line_are_its_two_edges():
    _, refined = extract_any_contours(shape_image(diagonal_line), **BRIGHT)
    assert len(refined.points) == 2 and not refined.closed.any()
    for points, normals in zip(refined.points, refined.normals):
        away = points[5:-5]  # at the image border the blurred line is cut, and its edge bends
        distance = across_diagonal(away[:, 0], away[:, 1])
        np.testing.assert_allclose(np.abs(distance), 12, atol=0.1)
        facing = across_diagonal(150 + normals[5:-5, 0], 100 + normals[5:-5, 1])  # +1 or -1: straight out of the line
        np.testing.assert_allclose(facing, np.sign(distance), atol=0.02)


def test_any_contours_of_a_region_with_a_hole_are_closed_one_per_boundary():
    _, refined = extract_any_contours(shape_image(ring), **BRIGHT)
    assert len(refined.points) == 2 and refined.closed.all() and refined.region[0] == refined.region[1]
    radii = sorted(np.hypot(*(points - [150, 100]).T).mean() for points in refined.points)
    np.testing.assert_allclose(radii, (30.4, 55.2), atol=0.1)
    for points, normals in zip(refined.points, refined.normals):
        radial = points - [150, 100]
        radius = np.hypot(*radial.T)
        np.testing.assert_allclose(radius, radius.mean(), atol=0.15)
        outward = (normals * radial).sum(axis=1) / radius  # +1: away from the centre
        np.testing.assert_allclose(outward, -1 if radius.mean() < 40 else 1, atol=0.05)  # the hole's: into the hole


@pytest.mark.parametrize("rule", RULES)
def test_any_contours_every_rule_finds_the_edges_within_a_pixel(rule):
    _, refined = extract_any_contours(shape_image(vertical_lines), refinement=rule, **BRIGHT)
    x = np.sort([points[:, 0].mean() for points in refined.points])
    np.testing.assert_allclose(x, VERTICAL_EDGES_X, atol=1)


def test_any_contours_of_dark_regions_are_those_of_bright_ones_in_the_negative_image():
    _, bright = extract_any_contours(shape_image(ring), **BRIGHT)
    _, dark = extract_any_contours(255 - shape_image(ring), dark_contacts=True)
    for a, b in zip(dark.points, bright.points):
        np.testing.assert_allclose(a, b, atol=1e-3)


def test_any_contours_without_refinement_are_the_detected_ones_and_small_regions_are_left_out():
    def with_speck(x, y):
        return vertical_lines(x, y) | (np.hypot(x - 130, y - 100) < 4)  # about 50 px
    image = shape_image(with_speck)
    options = dict(min_area_px=100, **BRIGHT)
    detected, refined = extract_any_contours(image, refinement="none", **options)
    assert refined is detected and len(detected.points) == 4
    assert detected.left_out == {"small": 1, "large": 0}
    assert len(extract_any_contours(image, **BRIGHT)[0].points) == 5  # default: 20 px, the speck is kept
    assert extract_any_contours(image, max_area_px=1000, **options)[0].left_out == {"small": 1, "large": 2}


def test_any_contours_upscaled_are_in_the_pixels_of_the_image():
    image = shape_image(vertical_lines)
    _, plain = extract_any_contours(image, **BRIGHT)
    _, upscaled = extract_any_contours(image, upscale=2, **BRIGHT)
    assert len(upscaled.points) == 4
    x = [np.sort([points[:, 0].mean() for points in found.points]) for found in (plain, upscaled)]
    np.testing.assert_allclose(x[1], x[0], atol=0.05)
    assert max(points[:, 1].max() for points in upscaled.points) < image.shape[0] - 1  # inside the image


def test_any_contours_of_contacts_inside_the_image_are_closed_and_of_cut_ones_open():
    """extract_contours drops a contact cut by the image border; extract_any_contours gives its edge inside the
    image."""
    lattice = np.array([(x, y) for x in np.arange(-1530, 1530, 90.0) for y in np.arange(-1530, 1530, 90.0)])
    image = render_tile(lattice, (0.0, 0.0), seed=1)
    _, contacts, _ = extract_contours(image)
    _, refined = extract_any_contours(image)
    assert int(refined.closed.sum()) == len(contacts.centers)
    assert int((~refined.closed).sum()) >= contacts.left_out["border"] > 0
    centres = np.array([points.mean(axis=0) for points, closed in zip(refined.points, refined.closed) if closed])
    nearest = np.linalg.norm(centres[:, None] - contacts.centers[None], axis=2).min(axis=1)
    assert nearest.max() < 0.2  # the same contacts, the same edges
