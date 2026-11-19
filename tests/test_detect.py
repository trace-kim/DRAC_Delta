import cv2
import numpy as np
import pytest

from contour_extraction.detect import (canny_edges, contact_mask, detect_contacts, detect_contours, edge_chains,
                                       edge_contours, inside_image, mask_contours)
from contour_extraction.load import crop_databar, load_sem_image
from contour_extraction.prepare import flatten_background
from synthetic import (VERTICAL_EDGES_X, across_diagonal, banded_image, diagonal_line, ramp_image, ring, shape_image,
                       vertical_lines, wide_band_image, write_sem_like)


def nearest_distances(found, truth):
    """Distance from each found point to its nearest true point."""
    return np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2).min(axis=1)


def test_finds_all_contacts_accurately(tmp_path):
    path = tmp_path / "tile.jpg"
    truth = write_sem_like(path, databar_px=64)
    image = crop_databar(load_sem_image(path), 64)

    found = detect_contacts(image)

    assert len(found.centers) == len(found.contours) == len(truth)
    assert nearest_distances(found.centers, truth).max() < 0.2  # pixels
    assert found.areas.min() > 300  # radius-14 holes, ~460 px after blur + threshold


def test_contact_cut_by_image_border_is_dropped():
    image = np.full((100, 100), 120, dtype=np.uint8)
    cv2.circle(image, (50, 50), 10, 30, thickness=-1)  # fully inside
    cv2.circle(image, (0, 50), 10, 30, thickness=-1)  # cut by the left border
    found = detect_contacts(image, blur_sigma=0)
    np.testing.assert_allclose(found.centers, [[50, 50]], atol=1e-9)


def test_a_boundary_loses_its_points_on_the_outermost_rows_and_columns():
    # A region cut by the left border of a 10 x 10 image: its boundary runs along column 0 from y = 4 back to y = 2.
    boundary = np.array([[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [3, 4], [2, 4], [1, 4], [0, 4], [0, 3]], float)
    (piece, closed), = inside_image(np.roll(boundary, 3, axis=0), (10, 10))  # wherever the boundary starts
    assert not closed
    np.testing.assert_array_equal(piece, boundary[1:8])  # the edge inside the image, from one end to the other
    (whole, closed), = inside_image(boundary + 2, (10, 10))  # off the border: the boundary itself, closed
    assert closed
    np.testing.assert_array_equal(whole, boundary + 2)


def test_contours_are_every_region_cut_by_the_border_included():
    image = np.full((100, 100), 120, dtype=np.uint8)
    cv2.circle(image, (50, 50), 10, 30, thickness=-1)  # fully inside
    cv2.circle(image, (0, 50), 10, 30, thickness=-1)  # cut by the left border
    found, labels = detect_contours(image, blur_sigma=0)
    assert list(found.closed) == [True, False] or list(found.closed) == [False, True]
    assert labels.max() == 2 and sorted(found.region) == [1, 2]
    cut = found.points[int(np.flatnonzero(~found.closed)[0])]
    assert cut[:, 0].min() == 1  # column 0 is the image border, not an edge


def test_regions_left_out_are_counted_per_reason():
    image = np.full((100, 100), 120, dtype=np.uint8)
    cv2.circle(image, (30, 30), 6, 30, thickness=-1)  # a contact, ~113 px
    cv2.circle(image, (70, 70), 12, 30, thickness=-1)  # larger, ~450 px
    cv2.circle(image, (0, 50), 6, 30, thickness=-1)  # cut by the border
    image[80:82, 20:22] = 30  # 4-pixel speck
    image[10:12, 60:62] = 30  # another

    found = detect_contacts(image, blur_sigma=0, min_area_px=20, max_area_px=300)

    np.testing.assert_allclose(found.centers, [[30, 30]], atol=1e-9)
    assert found.left_out == {"border": 1, "small": 2, "large": 1, "not solid": 0}


def test_flattening_the_background_lets_one_threshold_find_every_contact():
    image, centres, _ = ramp_image()
    assert len(detect_contacts(image, blur_sigma=0).centers) < len(centres) / 2

    found = detect_contacts(flatten_background(image, 20), blur_sigma=0)

    assert len(found.centers) == len(centres)
    np.testing.assert_allclose(found.centers[np.lexsort(found.centers.T[::-1])],
                               centres[np.lexsort(centres.T[::-1])], atol=0.05)


def test_bright_contacts():
    image = np.full((100, 100), 60, dtype=np.uint8)
    cv2.circle(image, (40, 60), 10, 220, thickness=-1)
    assert len(detect_contacts(image, blur_sigma=0).centers) == 0  # wrong polarity
    found = detect_contacts(image, dark_contacts=False, blur_sigma=0)
    np.testing.assert_allclose(found.centers, [[40, 60]], atol=1e-9)


def test_contour_surrounds_centre():
    image = np.full((60, 60), 120, dtype=np.uint8)
    cv2.circle(image, (25, 35), 8, 30, thickness=-1)
    contour = detect_contacts(image, blur_sigma=0).contours[0]
    radii = np.linalg.norm(contour - [25, 35], axis=1)
    assert 7 <= radii.min() and radii.max() <= 9


def test_three_classes_find_the_holes_when_bands_are_wide():
    # Bands 16 px wide nearly touch: plain Otsu splits background | band and picks up the background pinched off
    # between the bands; the three-class threshold (hole | background) does not.
    image, truth = wide_band_image(band_px=16)
    assert len(detect_contacts(image).centers) > len(truth)
    found = detect_contacts(image, method="otsu3")
    assert len(found.centers) == len(truth)
    assert nearest_distances(found.centers, truth).max() < 0.2


def test_three_classes_agree_with_plain_otsu_on_narrow_bands():
    image, truth = wide_band_image(band_px=3)
    plain, three = detect_contacts(image), detect_contacts(image, method="otsu3")
    assert len(plain.centers) == len(three.centers) == len(truth)
    assert nearest_distances(three.centers, truth).max() < 0.2


def test_three_classes_bright_contacts_and_bad_method():
    image, truth = wide_band_image(band_px=16)
    found = detect_contacts(255 - image, dark_contacts=False, method="otsu3")  # bright holes, dark bands
    assert len(found.centers) == len(truth)
    with pytest.raises(ValueError, match="method must be one of"):
        detect_contacts(image, method="nope")


def test_band_method_finds_the_regions_enclosed_by_bands():
    for band_px in (3, 16):  # narrow, and wide bands that nearly touch (background pinched off)
        image, truth = wide_band_image(band_px)
        found = detect_contacts(image, method="band")
        assert len(found.centers) == len(truth), band_px
        assert nearest_distances(found.centers, truth).max() < 0.2


def test_band_method_does_not_need_holes_darker_than_the_background():
    image, truth = wide_band_image(band_px=16, hole_grey=110)  # interior = background grey
    found = detect_contacts(image, method="band")
    assert len(found.centers) == len(truth)
    assert nearest_distances(found.centers, truth).max() < 0.2


def test_solidity_filter_drops_the_pinched_off_background():
    image, truth = wide_band_image(band_px=16)
    assert len(detect_contacts(image, method="band", min_solidity=0.0).centers) > len(truth)


# --- mask_contours: the contours of a mask made any way ---------------------------------------------------------------

def test_mask_contours_of_the_threshold_mask_are_those_of_detect_contours():
    image = shape_image(ring, noise=4)
    mask, threshold = contact_mask(image, dark_contacts=False)

    contours, labels = mask_contours(mask, threshold=threshold)

    detected, their_labels = detect_contours(image, dark=False)
    assert len(contours.points) == len(detected.points) == 2
    for ours, theirs in zip(contours.points + contours.normals, detected.points + detected.normals):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(labels, their_labels)
    assert contours.threshold == detected.threshold and contours.left_out == detected.left_out


def test_mask_contours_can_leave_out_the_holes_and_the_regions_at_the_image_border():
    mask = np.zeros((60, 80), np.uint8)
    cv2.circle(mask, (25, 30), 15, 1, 6)  # a ring: an outer boundary and a hole's
    mask[20:40, 60:] = 1  # a region that runs out of the image

    everything, _ = mask_contours(mask)
    outer, _ = mask_contours(mask, holes=False)
    inside, _ = mask_contours(mask, open_at_border=False)

    assert sorted(everything.closed) == [False, True, True]
    assert sorted(outer.closed) == [False, True]
    assert max(map(len, outer.points)) == max(map(len, everything.points))  # the ring's outer boundary
    assert list(inside.closed) == [True, True] and inside.left_out == {"border": 1, "small": 0, "large": 0}


# --- Canny's edges, followed into contours ---------------------------------------------------------------------------

def line_mask(pixels, shape=(30, 30)):
    mask = np.zeros(shape, np.uint8)
    for x, y in pixels:
        mask[y, x] = 1
    return mask


def joined(pixels: np.ndarray, closed: bool) -> bool:
    """Whether each pixel of a chain touches the next (and the last the first, if closed)."""
    steps = np.abs(np.diff(np.vstack([pixels, pixels[:1]]) if closed else pixels, axis=0)).max(axis=1)
    return bool((steps == 1).all())


def test_edge_chains_follow_an_open_line_from_one_end_to_the_other_wherever_the_walk_begins():
    """cv2.findContours begins at the topmost pixel: an end of the vertical line, the middle of the ^."""
    vertical = [(10, y) for y in range(5, 20)]
    hat = [(15 - k, 10 + k) for k in range(8)] + [(15 + k, 10 + k) for k in range(1, 8)]
    for pixels, ends in ((vertical, {(10, 5), (10, 19)}), (hat, {(8, 17), (22, 17)})):
        (chain, closed), = edge_chains(line_mask(pixels))
        assert not closed and len(chain) == 15 and joined(chain, closed)
        assert {tuple(map(int, chain[0])), tuple(map(int, chain[-1]))} == ends


def test_edge_chains_keep_a_closed_line_closed_and_take_its_spur_off():
    loop = np.zeros((30, 30), np.uint8)
    cv2.rectangle(loop, (5, 5), (20, 18), 1, 1)
    (chain, closed), = edge_chains(loop)
    assert closed and len(chain) == 52 and joined(chain, closed)  # 56 pixels, without its four corners (thinned)

    for spur in ([(x, 5) for x in range(21, 27)], [(12, y) for y in range(1, 5)]):  # the walk begins on the second
        with_spur = loop.copy()
        for x, y in spur:
            with_spur[y, x] = 1
        chains = edge_chains(with_spur)
        assert [closed for _, closed in chains] == [True, False]
        assert len(chains[0][0]) >= 52 and joined(*chains[0]) and len(chains[1][0]) <= len(spur)


def test_edge_chains_follow_a_staircase_whose_steps_overlap_as_one_line():
    """Canny's slanted edges: rows of pixels, each beginning under the end of the one above. Round such a line the
    walk comes back over other pixels than it went out on, unless the line is thinned first."""
    stairs = line_mask([(3 * k + dx, 5 + k) for k in range(7) for dx in range(4)])
    (chain, closed), = edge_chains(stairs)
    assert not closed and joined(chain, closed) and 15 <= len(chain) < stairs.sum()
    assert {int(chain[0][0]), int(chain[-1][0])} == {0, 21}  # from its one end to its other

    image = shape_image(diagonal_line, noise=8)
    contours = edge_contours(canny_edges(image)[0], image)
    assert len(contours.points) == 2 and not contours.closed.any() and min(map(len, contours.points)) > 280
    for points in contours.points:  # each pixel put on the edge, between the pixels: 12 px from the line's middle
        np.testing.assert_allclose(np.abs(across_diagonal(points[5:-5, 0], points[5:-5, 1])), 12, atol=0.25)


def test_edge_chains_find_nested_closed_lines_and_split_a_branching_one():
    nested = np.zeros((30, 30), np.uint8)
    cv2.circle(nested, (15, 15), 10, 1, 1)
    cv2.circle(nested, (15, 15), 5, 1, 1)
    assert sorted((len(chain), closed) for chain, closed in edge_chains(nested)) == [(28, True), (56, True)]

    tee = line_mask([(x, 15) for x in range(3, 27)] + [(15, y) for y in range(5, 15)])
    chains = edge_chains(tee)
    assert len(chains) == 2 and not any(closed for _, closed in chains)
    assert sum(len(chain) for chain, _ in chains) == 34 and all(joined(*chain) for chain in chains)
    assert edge_chains(np.zeros((30, 30), np.uint8)) == []


@pytest.mark.parametrize("noise", [0, 8, 20])
def test_canny_edges_of_a_ring_are_two_closed_lines_on_its_edges(noise):
    image = shape_image(ring, noise=noise)

    edges, high = canny_edges(image)
    contours = edge_contours(edges, image)

    assert high > 0 and set(np.unique(edges)) == {0, 1}
    assert len(contours.points) == 2 and contours.closed.all()
    radii = sorted(np.hypot(*(points - [150, 100]).T).mean() for points in contours.points)
    np.testing.assert_allclose(radii, (30.4, 55.2), atol=0.12)  # between the pixels (gradient_peaks)
    for points, normals in zip(contours.points, contours.normals):  # facing the brighter side: into the ring
        radius = np.hypot(*(points - [150, 100]).T)
        outward = (normals * (points - [150, 100])).sum(axis=1) / radius
        np.testing.assert_allclose(outward, 1 if radius.mean() < 40 else -1, atol=0.1)
    assert np.isnan(contours.threshold) and not contours.region.any()


def test_canny_edges_of_lines_running_out_of_the_image_are_open_one_per_edge():
    image = shape_image(vertical_lines, noise=8)
    contours = edge_contours(canny_edges(image)[0], image)
    assert len(contours.points) == 4 and not contours.closed.any()
    order = np.argsort([points[:, 0].mean() for points in contours.points])
    for k, x_edge, faces in zip(order, VERTICAL_EDGES_X, (1, -1, 1, -1)):  # each faces into its (bright) line
        np.testing.assert_allclose(contours.points[k][:, 0].mean(), x_edge, atol=0.08)
        np.testing.assert_allclose(contours.points[k][:, 0], x_edge, atol=0.4)
        np.testing.assert_allclose(contours.normals[k][10:-10], np.tile([faces, 0.0], (len(contours.points[k]) - 20, 1)),
                                   atol=0.35)


def test_canny_edges_give_each_bright_band_two_contours_and_short_lines_are_left_out():
    centres = [(30, 30), (70, 30), (30, 70), (70, 70)]
    image = banded_image(centres, [10] * 4, 100, noise=6)
    edges, high = canny_edges(image)

    contours = edge_contours(edges, image)
    assert len(contours.points) == 8 and contours.closed.all()  # the inner and the outer slope of each band
    longer = edge_contours(edges, image, min_points=70)
    assert len(longer.points) == 4 and longer.left_out == {"short": 4}  # the outer ones only
    assert canny_edges(image, high=2 * high)[0].sum() < edges.sum()  # a higher threshold: fewer edge pixels
