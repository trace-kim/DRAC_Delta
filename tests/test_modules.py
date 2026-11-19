"""The modules of a workflow (modules.py): what a window is told about each, and what each makes of a State."""

import cv2
import numpy as np
import pytest

from contour_extraction.contacts import polygon_area, polygon_centroid
from contour_extraction.detect import contact_mask
from contour_extraction.modules import MODULES, STAGES, Option, State, WorkflowError, image_contours
from synthetic import banded_image, no_band_image, ring, shape_image, wide_band_image


def run(state, *steps) -> State:
    """The state after these modules in turn: each (name, options) or just its name, the other options at their
    defaults."""
    if isinstance(state, np.ndarray):
        state = State(state)
    for entry in steps:
        name, given = (entry, {}) if isinstance(entry, str) else entry
        module = MODULES[name]
        state = module.function(state, **{key: option.default for key, option in module.options.items()} | given)
    return state


def radii(points, centre):
    return np.hypot(*(points - np.asarray(centre)).T)


CONTACTS = np.array([(30.3, 30.6), (70.8, 30.2), (30.1, 70.4), (70.5, 70.9)])


def contacts_image(noise=6):
    """Four contacts of radius 10 with a bright band, at CONTACTS."""
    return banded_image(CONTACTS, [10] * 4, 100, noise=noise)


# --- What a window is told about the modules --------------------------------------------------------------------------

def test_every_module_says_what_it_is_and_what_its_options_are():
    assert len(MODULES) >= 20 and {module.stage for module in MODULES.values()} == set(STAGES)
    assert len({module.title for module in MODULES.values()}) == len(MODULES)
    for name, module in MODULES.items():
        assert module.name == name == module.function.__name__
        assert module.title and module.stage in STAGES and len(module.help) > 40 and "\n" not in module.help
        for key, option in module.options.items():
            assert option.help, (name, key)
            assert option.checked(option.default, key) == option.default
            if isinstance(option.default, str):
                assert option.default in option.choices
            elif not isinstance(option.default, bool):
                assert option.low <= option.default <= option.high and option.step > 0, (name, key)


def test_an_option_takes_only_values_of_its_kind():
    whole, real = Option(3, "", 1, 9), Option(1.5, "", 0, 9)
    assert whole.checked(4, "n") == 4 and isinstance(whole.checked(4.0, "n"), int)
    assert real.checked(2, "x") == 2.0 and isinstance(real.checked(2, "x"), float)
    for option, value, why in ((whole, 4.5, "whole number"), (whole, 10, "between 1 and 9"), (real, "2", "number"),
                               (real, True, "number"), (Option(True, ""), 1, "true or false"),
                               (Option("a", "", choices=("a", "b")), "c", "one of a, b")):
        with pytest.raises(ValueError, match=why):
            option.checked(value, "x")


@pytest.mark.parametrize("name", list(MODULES))
def test_every_module_runs_with_its_defaults_and_leaves_what_it_was_given_as_it_was(name):
    before = {"Image": [], "Mask": ["otsu_threshold"], "Contours": ["otsu_threshold", "trace_regions"]}
    given = run(contacts_image(), *before[MODULES[name].stage])
    arrays = [a.copy() for a in (given.image, given.mask) if a is not None]
    points = [p.copy() for p in given.contours.points] if given.contours else []

    after = run(given, name)

    assert isinstance(after, State) and after.image.dtype == np.uint8 and after.image.ndim == 2
    for array, copy in zip((given.image, given.mask), arrays):
        np.testing.assert_array_equal(array, copy)
    for array, copy in zip(given.contours.points if given.contours else [], points):
        np.testing.assert_array_equal(array, copy)
    if after.mask is not None:
        assert after.mask.shape == after.image.shape and after.mask.dtype == np.uint8 and after.mask.max() <= 1


# --- Image modules ----------------------------------------------------------------------------------------------------

def test_the_image_modules_change_the_image_as_they_say():
    image = contacts_image(noise=12)
    noise = lambda picture: float(picture[:8, :8].std())  # a corner without pattern

    for name in ("gaussian_blur", "median_filter", "bilateral_filter", "non_local_means", "total_variation"):
        assert noise(run(image, name).image) < 0.75 * noise(image), name
    assert run(image, ("gaussian_blur", dict(sigma_px=0.0))).image is image
    np.testing.assert_array_equal(run(image, ("median_filter", dict(size_px=4))).image, cv2.medianBlur(image, 5))
    np.testing.assert_array_equal(run(image, "invert").image, 255 - image)
    np.testing.assert_array_equal(run(image, "equalize_histogram").image, cv2.equalizeHist(image))
    assert run(image, "clahe").image.std() > image.std()
    stretched = run(image // 2 + 60, "stretch_contrast")
    assert stretched.image.min() == 0 and stretched.image.max() == 255 and "grey levels" in stretched.note
    assert "one grey level" in run(np.full((20, 20), 7, np.uint8), "stretch_contrast").note


def test_flatten_background_lets_one_threshold_find_the_pattern_on_a_ramp():
    from synthetic import ramp_image
    image, centres, _ = ramp_image()
    whole = dict(holes=False, open_at_border=False)
    plain = run(image, "otsu_threshold", ("trace_regions", whole))
    flat = run(image, "flatten_background", "otsu_threshold", ("trace_regions", whole))
    assert len(plain.contours.points) < len(centres) == len(flat.contours.points)


def test_crop_and_upscale_change_the_size_of_every_image_and_come_before_the_mask():
    image = contacts_image()

    cropped = run(image, "edge_image", ("gaussian_blur", dict(sigma_px=2.0)), ("crop_databar", dict(rows=20)))
    assert cropped.image.shape == cropped.edge_image.shape == (80, 100) and "80 rows" in cropped.note
    np.testing.assert_array_equal(cropped.edge_image, image[:80])

    big = run(image, ("upscale", dict(factor=2)), ("crop_databar", dict(rows=20)))
    assert big.scale == 2 and big.image.shape == (160, 200) and big.unscaled.shape == (80, 100)
    np.testing.assert_array_equal(big.unscaled, image[:80])  # the image as it was before Upscale
    assert run(image, "edge_image", ("upscale", dict(factor=2))).edge_image.shape == (200, 200)

    for name in ("crop_databar", "upscale"):
        with pytest.raises(WorkflowError, match="before the module that makes the mask"):
            run(image, "otsu_threshold", name)
    with pytest.raises(WorkflowError, match="only 100 rows"):
        run(image, ("crop_databar", dict(rows=100)))


def test_edge_image_keeps_the_image_the_edges_are_measured_on():
    """The pattern is found in the image as the later modules made it, its edges in the image kept."""
    image = contacts_image(noise=12)
    blur = ("gaussian_blur", dict(sigma_px=3.0))
    plain = run(image, "otsu_threshold", "trace_regions", "refine_edges")
    blurred = run(image, blur, "otsu_threshold", "trace_regions", "refine_edges")
    kept = run(image, "edge_image", blur, "otsu_threshold", "trace_regions", "refine_edges")

    assert kept.edge_image is image and kept.measured is image and blurred.measured is blurred.image
    area = lambda state: np.mean([polygon_area(p) for p in state.contours.points])
    assert len(plain.contours.points) == len(blurred.contours.points) == len(kept.contours.points) == 4
    assert abs(area(kept) / area(plain) - 1) < 0.01 < abs(area(blurred) / area(plain) - 1)


def test_invert_turns_dark_into_bright_and_is_refused_after_edge_image():
    image = contacts_image()
    state = run(image, "otsu_threshold", "invert")
    assert state.dark is False and state.threshold == 255 - run(image, "otsu_threshold").threshold
    with pytest.raises(WorkflowError, match="before Edge image"):
        run(image, "edge_image", "invert")


# --- Mask modules -----------------------------------------------------------------------------------------------------

def test_otsu_threshold_is_contact_mask_and_says_its_grey_level():
    image, _ = wide_band_image(16)
    for method in ("otsu", "otsu3", "band"):
        state = run(image, ("otsu_threshold", dict(method=method, blur_px=2.0)))
        mask, level = contact_mask(image, True, 2.0, method)
        np.testing.assert_array_equal(state.mask, mask)
        assert state.threshold == level and state.dark is True and f"grey level {level:.0f}" in state.note
        assert state.connectivity == (4 if method == "band" else 8)
    bright = run(image, ("otsu_threshold", dict(dark=False)))
    assert bright.dark is False and bright.mask.sum() == image.size - run(image, "otsu_threshold").mask.sum()


def test_a_threshold_found_in_an_image_changed_after_edge_image_is_no_level_of_the_edge_image():
    image = contacts_image()
    assert np.isfinite(run(image, "edge_image", "otsu_threshold").threshold)
    changed = run(image, "edge_image", "gaussian_blur", "otsu_threshold", "trace_regions")
    assert np.isnan(changed.threshold)
    with pytest.raises(WorkflowError, match="threshold needs the grey level"):
        run(changed, ("refine_edges", dict(rule="threshold")))
    assert len(run(changed, "refine_edges").contours.points) == 4  # mid-level: each region's own levels


def test_fixed_and_adaptive_thresholds_take_the_darker_or_the_brighter_side():
    image = np.tile(np.linspace(40, 200, 120).astype(np.uint8), (60, 1))  # brightens to the right
    cv2.circle(image, (30, 30), 8, 10, -1)   # a dark disk on the dark side
    cv2.circle(image, (90, 30), 8, 110, -1)  # one on the bright side, brighter than the dark side's background

    fixed = run(image, ("fixed_threshold", dict(level=100)))
    np.testing.assert_array_equal(fixed.mask, (image <= 100).astype(np.uint8))
    assert fixed.threshold == 100 and fixed.mask[30, 30] == 1 and fixed.mask[30, 90] == 0
    np.testing.assert_array_equal(run(image, ("fixed_threshold", dict(level=100, dark=False))).mask, 1 - fixed.mask)

    local = run(image, ("adaptive_threshold", dict(block_px=41, offset=20.0)))
    assert local.mask[30, 30] == local.mask[30, 90] == 1 and local.mask[5, 60] == 0  # both disks, not the ramp
    assert np.isnan(local.threshold)
    bright = run(255 - image, ("adaptive_threshold", dict(block_px=41, offset=20.0, dark=False)))
    np.testing.assert_array_equal(bright.mask, local.mask)


def test_canny_edges_are_thin_lines_and_their_threshold_is_per_pixel_of_the_image_as_loaded():
    image = shape_image(ring, noise=8)
    state = run(image, "canny_edges")
    assert state.dark is None and np.isnan(state.threshold) and "grey levels / px" in state.note
    assert 0.005 < state.mask.mean() < 0.02  # two circles, a pixel wide
    fewer = run(image, ("canny_edges", dict(high=40.0)))
    assert fewer.mask.sum() < state.mask.sum()
    big = run(image, ("upscale", dict(factor=2)), ("canny_edges", dict(high=10.0)))
    assert "above 10.0 grey levels" in big.note and 0.5 < big.mask.sum() / (2 * state.mask.sum()) < 1.5


def test_morphology_and_fill_holes_change_the_mask():
    mask = np.zeros((60, 60), np.uint8)
    cv2.circle(mask, (30, 30), 15, 1, 5)  # a ring
    mask[5, 5] = 1                        # a speck
    state = run(State(mask * 200, mask=mask, dark=False), ("morphology", dict(operation="open", size_px=3)))
    assert state.mask[5, 5] == 0 and state.mask[30, 45] == 1 and state.dark is False
    assert run(state, ("morphology", dict(operation="dilate", size_px=5))).mask.sum() > state.mask.sum()
    assert run(state, ("morphology", dict(operation="erode", size_px=3))).mask.sum() < state.mask.sum()
    assert run(state, ("morphology", dict(operation="close", size_px=35))).mask[30, 30] == 1
    assert run(state, "fill_holes").mask[30, 30] == 1 and state.mask[30, 30] == 0
    assert run(state, "trace_regions", "fill_holes").contours is None  # a new mask: its contours are to be traced
    for name in ("morphology", "fill_holes", "trace_regions", "trace_edges"):
        with pytest.raises(WorkflowError, match="needs a mask"):
            run(mask, name)


# --- Contours modules -------------------------------------------------------------------------------------------------

def test_trace_regions_gives_whole_boundaries_or_only_the_whole_features():
    image = shape_image(ring, shape=(200, 300), noise=4)
    image[80:120, 270:] = 180  # a bright bar that runs out of the image
    bright = ("otsu_threshold", dict(dark=False))

    everything = run(image, bright, "trace_regions")
    assert sorted(everything.contours.closed) == [False, True, True] and everything.labels is not None
    assert "3 contours (2 closed, 1 open)" in everything.note
    whole = run(image, bright, ("trace_regions", dict(holes=False, open_at_border=False)))
    assert list(whole.contours.closed) == [True] and "left out: 1 at the image border" in whole.note
    assert len(run(image, bright, ("trace_regions", dict(min_area_px=3000))).contours.points) == 2
    assert len(run(image, bright, ("trace_regions", dict(max_area_px=3000))).contours.points) == 1


def test_canny_edges_filled_are_regions_whose_side_is_taken_from_the_image():
    """A mask of edges does not say which side is the darker: Trace regions looks."""
    image = no_band_image([(40.3, 39.6), (100.8, 40.2)], 12.0, noise=4)
    for picture, dark in ((image, True), (255 - image, False)):
        state = run(picture, "canny_edges", "fill_holes", "trace_regions", "refine_edges")
        assert state.dark is dark and len(state.contours.points) == 2 and state.contours.closed.all()
        for points, centre in zip(state.contours.points, [(40.3, 39.6), (100.8, 40.2)]):
            assert abs(radii(points, centre).mean() - 12) < 0.3


def test_trace_edges_puts_the_contours_on_the_edges_between_the_pixels():
    image = shape_image(ring, noise=8)
    traced = run(image, "canny_edges", "trace_edges")
    assert traced.dark is True and traced.labels is None and "2 contours (2 closed, 0 open)" in traced.note
    found = sorted(radii(points, (150, 100)).mean() for points in traced.contours.points)
    np.testing.assert_allclose(found, (30.4, 55.2), atol=0.12)
    assert max(radii(points, (150, 100)).std() for points in traced.contours.points) < 0.15  # off the pixel grid

    with pytest.raises(WorkflowError, match="mid-level needs the regions"):
        run(traced, "refine_edges")
    moved = run(traced, ("refine_edges", dict(rule="steepest gradient", search_px=2.0, blur_px=2.0)))
    assert np.abs(moved.contours.points[0] - traced.contours.points[0]).max() < 1  # allowed, and hardly moves them
    assert run(image, "canny_edges", ("trace_edges", dict(min_length_px=250))).contours.left_out == {"short": 1}

    blurred = run(image, ("upscale", dict(factor=2)), ("canny_edges", dict(blur_px=3.0)))
    assert blurred.edge_blur == 6.0  # Trace edges takes the gradient of the image blurred as Canny edges blurred it


def test_filter_contours_leaves_out_the_open_the_dented_and_the_faint():
    image, truth = wide_band_image(16)
    band = ("otsu_threshold", dict(method="band"))
    traced = run(image, band, ("trace_regions", dict(holes=False, open_at_border=False)))
    solid = run(traced, ("filter_contours", dict(min_solidity=0.9)))
    assert len(traced.contours.points) > len(truth) == len(solid.contours.points)
    assert solid.contours.left_out["not solid"] == len(traced.contours.points) - len(truth)
    assert "not solid" in solid.note and "of low contrast" not in solid.note

    cut = run(no_band_image([(4.0, 40.0), (70.0, 40.0)], 12.0), "otsu_threshold", "trace_regions")  # one runs out
    closed = run(cut, ("filter_contours", dict(closed_only=True)))
    assert list(cut.contours.closed) == [False, True] and list(closed.contours.closed) == [True]
    assert closed.contours.left_out["open"] == 1 and "left out: 1 open" in closed.note

    noise = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 12, (256, 256)), (0, 0), 1.0)
    blank = np.clip(110 + noise, 0, 255).astype(np.uint8)
    for before in ([], [("upscale", dict(factor=2))]):  # upscaled: measured on the image as it was before
        false = run(blank, *before, "otsu_threshold", "trace_regions")
        kept = run(false, ("filter_contours", dict(min_contrast=5.0)))
        assert len(false.contours.points) > 5 and len(kept.contours.points) == int((~false.contours.closed).sum())
    real = run(contacts_image(), "otsu_threshold", "trace_regions", ("filter_contours", dict(min_contrast=5.0)))
    assert len(real.contours.points) == 4
    for name in ("filter_contours", "refine_edges", "smooth_contours"):
        with pytest.raises(WorkflowError, match="needs contours"):
            run(blank, "otsu_threshold", name)


def test_refine_edges_moves_the_contours_to_the_edges_and_smooth_contours_takes_out_their_scatter():
    image = contacts_image(noise=20)
    traced = run(image, "otsu_threshold", "trace_regions")
    refined = run(traced, "refine_edges")
    smooth = run(refined, ("smooth_contours", dict(window=9)))

    rough = lambda state: np.mean([radii(p, c).std() for p, c in zip(state.contours.points, centres(state))])
    centres = lambda state: [CONTACTS[np.argmin(np.hypot(*(CONTACTS - p.mean(axis=0)).T))] for p in state.contours.points]
    assert rough(smooth) < rough(refined) < rough(traced)
    mean = lambda state: np.mean([radii(p, c).mean() for p, c in zip(state.contours.points, centres(state))])
    assert abs(mean(smooth) - mean(refined)) < 0.02  # smoothing does not shrink them
    assert refined.note == "mid-level" and refined.contours is not traced.contours


def test_the_options_in_pixels_are_those_of_the_image_as_loaded_whatever_the_scale():
    """Upscale put into a workflow needs no other option changed: the modules after it scale their own."""
    image = contacts_image()
    steps = (("gaussian_blur", dict(sigma_px=1.0)), ("otsu_threshold", dict(blur_px=1.0)),
             ("morphology", dict(operation="open", size_px=3)), ("trace_regions", dict(min_area_px=200)),
             "refine_edges", "smooth_contours")
    plain, big = run(image, *steps), run(image, ("upscale", dict(factor=3)), *steps)

    assert big.scale == 3 and big.image.shape == (300, 300)
    assert len(image_contours(plain).points) == len(image_contours(big).points) == 4
    for ours, theirs in zip(image_contours(big).points, image_contours(plain).points):
        assert abs(polygon_area(ours) / polygon_area(theirs) - 1) < 0.02
        np.testing.assert_allclose(polygon_centroid(ours), polygon_centroid(theirs), atol=0.05)
    assert image_contours(State(image)) is None and image_contours(plain).points[0] is plain.contours.points[0]
