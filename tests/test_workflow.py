"""Workflows (workflow.py): steps run in turn, the workflows to start from, and workflow files."""

import json

import numpy as np
import pytest

from contour_extraction.contacts import polygon_area, polygon_centroid
from contour_extraction.load import crop_databar, load_sem_image
from contour_extraction.modules import MODULES, WorkflowError, image_contours
from contour_extraction.pipeline import extract_any_contours, extract_contours
from contour_extraction.workflow import (FORMAT, PRESETS, Step, load_workflow, preset, run_workflow, save_workflow,
                                         step, workflow_contours, workflow_data, workflow_steps)
from synthetic import (VERTICAL_EDGES_X, across_diagonal, banded_image, centre_errors, diagonal_line, no_band_image,
                       ramp_image, ring, shape_image, vertical_lines, wide_band_image, write_sem_like)


@pytest.fixture(scope="module")
def tile(tmp_path_factory):
    """A synthetic SEM tile of 176 contact holes with its data bar cropped, and their true centres."""
    path = tmp_path_factory.mktemp("tile") / "tile.jpg"
    truth = write_sem_like(path, databar_px=64)
    return crop_databar(load_sem_image(path), 64), truth


def centres(contours) -> np.ndarray:
    """The area centroids of the closed contours."""
    return np.array([polygon_centroid(p) for p, closed in zip(contours.points, contours.closed) if closed])


# --- Steps ------------------------------------------------------------------------------------------------------------

def test_a_step_has_every_option_of_its_module_and_refuses_what_the_module_has_not():
    blur = step("gaussian_blur")
    assert blur == Step("gaussian_blur", {"sigma_px": 1.0}, True)
    assert step("otsu_threshold", blur_px=2).options == {"method": "otsu", "dark": True, "blur_px": 2.0}
    assert step("upscale", enabled=False, factor=3) == Step("upscale", {"factor": 3}, False)
    for module, options, why in (("sharpen", {}, "no module 'sharpen'"), ("upscale", {"times": 2}, "no option 'times'"),
                                 ("upscale", {"factor": 2.5}, "upscale.factor must be a whole number"),
                                 ("otsu_threshold", {"method": "li"}, "must be one of otsu, otsu3, band")):
        with pytest.raises(ValueError, match=why):
            step(module, **options)


# --- Running ----------------------------------------------------------------------------------------------------------

def test_run_workflow_gives_the_state_after_every_step(tile):
    image, truth = tile
    steps = [step("gaussian_blur"), step("otsu_threshold"), step("trace_regions"), step("refine_edges")]

    run = run_workflow(image, steps)

    assert run.error is None and run.failure == "" and len(run.states) == 5 and len(run.seconds) == 4
    assert run.states[0].image is image and run.states[1].mask is None and run.states[2].contours is None
    assert run.states[3].contours is not run.final.contours and len(run.final.contours.points) == len(truth)
    assert run.steps == steps and run.steps[0] is not steps[0]  # as they were: changing a step later changes no run


def test_a_step_that_is_not_enabled_passes_on_what_it_was_given(tile):
    image, _ = tile
    steps = [step("gaussian_blur", enabled=False, sigma_px=5.0), step("otsu_threshold")]
    run = run_workflow(image, steps)
    assert run.states[1].image is image and run.states[1].note == "off"
    np.testing.assert_array_equal(run.final.mask, run_workflow(image, steps[1:]).final.mask)


def test_a_step_that_fails_ends_the_run_and_says_why(tile):
    image, _ = tile
    run = run_workflow(image, [step("gaussian_blur"), step("refine_edges"), step("otsu_threshold")])
    assert isinstance(run.error, WorkflowError) and len(run.states) == 2 and len(run.seconds) == 1
    assert run.failure == "Step 2 (Refine edges) failed: Refine edges needs contours: put Trace regions or Trace " \
                          "edges before it"
    with pytest.raises(WorkflowError, match="Step 2 .Refine edges. failed"):
        workflow_contours(image, [step("gaussian_blur"), step("refine_edges")])
    with pytest.raises(WorkflowError, match="makes no contours"):
        workflow_contours(image, [step("otsu_threshold")])
    broken = run_workflow(image, [Step("median_filter", {"size_px": "wide"})])  # not a WorkflowError: named
    assert broken.failure.startswith("Step 1 (Median filter) failed: TypeError")


def test_a_run_takes_up_an_earlier_one_where_their_steps_begin_alike(tile):
    image, _ = tile
    steps = [step("median_filter"), step("otsu_threshold"), step("trace_regions"), step("refine_edges")]
    first = run_workflow(image, steps)

    steps[3].options["rule"] = "threshold"  # the last step changed: the three before it are not run again
    second = run_workflow(image, steps, first)
    assert all(a is b for a, b in zip(second.states[:4], first.states[:4])) and second.seconds[:3] == first.seconds[:3]
    assert second.final is not first.final and second.final.note == "threshold"

    steps[1].options["blur_px"] = 3.0  # an early one: everything from it on is
    third = run_workflow(image, steps, second)
    assert third.states[1] is second.states[1] and all(a is not b for a, b in zip(third.states[2:], second.states[2:]))
    np.testing.assert_array_equal(third.final.contours.points[0], run_workflow(image, steps).final.contours.points[0])

    other = run_workflow(image.copy(), steps, third)  # another image: nothing is taken
    assert other.states[1] is not third.states[1]
    failed = run_workflow(image, steps[:1] + [step("refine_edges")], third)  # and from a run that failed
    assert failed.error is not None and failed.states[1] is third.states[1]
    again = run_workflow(image, steps, failed)
    assert again.error is None and again.states[1] is third.states[1] and len(again.states) == 5


# --- The workflows to start from --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_finds_the_contact_holes_of_an_sem_image(tile, name):
    image, truth = tile
    description, steps = PRESETS[name]
    assert len(description) > 60 and all(s.module in MODULES for s in steps)
    assert preset(name) == steps and preset(name)[0] is not steps[0]  # a copy, to change

    contours = workflow_contours(image, preset(name))

    per_hole = 2 if "Canny" in name else 1  # Canny: the inner and the outer slope of each bright band
    assert len(contours.points) == per_hole * len(truth) and contours.closed.all()
    assert np.abs(centre_errors(centres(contours), truth)).max() < 0.25


def test_the_contact_holes_preset_finds_what_extract_contours_finds(tile):
    image, _ = tile
    contours = workflow_contours(image, preset("Contact holes"))
    _, refined, _ = extract_contours(image)
    np.testing.assert_allclose(centres(contours), refined.centers, atol=1e-3)
    np.testing.assert_allclose([polygon_area(p) for p in contours.points], refined.areas, rtol=1e-3)


@pytest.mark.parametrize("prepare, options", [
    ([], {}),
    ([step("flatten_background", size_px=60.0), step("median_filter"), step("upscale", factor=2)],
     dict(background_px=60.0, denoise="median", upscale=2)),
])
def test_the_any_pattern_preset_is_extract_any_contours_step_by_step(tile, prepare, options):
    """The modules are the pipeline's own steps: the same contours to the last bit, prepared and upscaled too."""
    image, _ = tile
    contours = workflow_contours(image, prepare + preset("Any pattern"))
    _, refined = extract_any_contours(image, **options)
    assert len(contours.points) == len(refined.points) > 100
    for ours, theirs in zip(contours.points + contours.normals, refined.points + refined.normals):
        np.testing.assert_array_equal(ours, theirs)
    assert contours.threshold == refined.threshold and contours.left_out == refined.left_out


def test_the_band_preset_finds_holes_as_grey_as_the_background():
    image, truth = wide_band_image(band_px=16, hole_grey=110)
    found = centres(workflow_contours(image, preset("Contact holes by their bright band")))
    assert len(found) == len(truth) and np.abs(centre_errors(found, truth)).max() < 0.2


def test_the_no_band_preset_puts_the_edges_on_the_threshold():
    truth = np.array([[40.3, 39.6], [100.8, 40.2]])
    contours = workflow_contours(no_band_image(truth, 12.0), preset("Holes without a bright band"))
    np.testing.assert_allclose(centres(contours), truth, atol=0.01)
    # Otsu's level is not quite half-way between the hole and the background: the radius comes out 0.1 px wide
    np.testing.assert_allclose([np.sqrt(polygon_area(p) / np.pi) for p in contours.points], 12, atol=0.15)


def test_the_uneven_background_preset_finds_the_pattern_on_a_brightness_ramp():
    image, truth, _ = ramp_image()
    assert len(workflow_contours(image, preset("Contact holes")).points) < len(truth)  # one threshold does not fit
    found = centres(workflow_contours(image, preset("Uneven background")))
    assert len(found) == len(truth) and np.abs(centre_errors(found, truth)).max() < 0.1


def test_the_noisy_image_preset_keeps_noise_out_of_the_mask_and_measures_the_edges_as_the_default_does():
    truth = np.array([(x + 0.3, y - 0.2) for x in range(20, 190, 34) for y in range(20, 190, 34)], float)
    image = banded_image(truth, [10] * len(truth), 200, noise=35, seed=1)
    radius = lambda contours: np.mean([np.sqrt(polygon_area(p) / np.pi) for p in contours.points])

    plain = workflow_contours(image, preset("Contact holes"))
    denoised = workflow_contours(image, preset("Noisy image"))
    without_edge_image = workflow_contours(image, preset("Noisy image")[1:])

    assert len(plain.points) > len(truth) == len(denoised.points)  # regions cut out of noise | none
    quiet = banded_image(truth, [10] * len(truth), 200, noise=6, seed=1)
    default, kept, moved = (radius(workflow_contours(quiet, steps)) for steps in (
        preset("Contact holes"), preset("Noisy image"), preset("Noisy image")[1:]))
    assert abs(kept - default) < 0.03 < abs(moved - default)  # measured on the denoised image, the edges move
    assert len(without_edge_image.points) == len(truth)


def test_the_canny_preset_follows_edges_of_any_shape_open_or_closed():
    canny = preset("Edges without a threshold (Canny)")
    rings = workflow_contours(shape_image(ring, noise=8), canny)
    assert len(rings.points) == 2 and rings.closed.all()
    found = sorted(np.hypot(*(points - [150, 100]).T).mean() for points in rings.points)
    np.testing.assert_allclose(found, (30.4, 55.2), atol=0.12)  # the blur draws a curved edge in by sigma² / 2R

    lines = workflow_contours(shape_image(vertical_lines, noise=8), canny)
    assert len(lines.points) == 4 and not lines.closed.any()
    np.testing.assert_allclose(np.sort([points[:, 0].mean() for points in lines.points]), VERTICAL_EDGES_X, atol=0.08)

    slanted = workflow_contours(shape_image(diagonal_line, noise=8), canny)  # a staircase of pixels
    assert len(slanted.points) == 2 and not slanted.closed.any() and min(map(len, slanted.points)) > 280
    for points in slanted.points:
        np.testing.assert_allclose(np.abs(across_diagonal(points[5:-5, 0], points[5:-5, 1])), 12, atol=0.25)


def test_the_small_features_preset_gives_its_contours_in_the_pixels_of_the_image():
    truth = np.array([(x + 0.3, y - 0.2) for x in range(12, 170, 12) for y in range(12, 110, 12)], float)
    image = no_band_image(truth, 3.0, shape=(120, 180), noise=5, seed=2)
    run = run_workflow(image, preset("Small features (upscaled)"))
    found = centres(image_contours(run.final))
    assert run.final.scale == 4 and run.final.image.shape == (480, 720)
    assert len(found) == len(truth) and np.abs(centre_errors(found, truth)).max() < 0.3


# --- Workflow files ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(PRESETS))
def test_a_workflow_saved_is_the_workflow_loaded(tmp_path, name):
    steps = preset(name)
    steps[0].enabled = False
    save_workflow(tmp_path / "workflow.json", steps)
    assert load_workflow(tmp_path / "workflow.json") == steps
    data = json.loads((tmp_path / "workflow.json").read_text(encoding="utf-8"))
    assert data["format"] == FORMAT and data["version"] == 1 and len(data["steps"]) == len(steps)
    assert data == workflow_data(steps) and workflow_steps(data) == steps


def test_a_workflow_file_may_leave_out_options_and_must_name_what_exists(tmp_path):
    path = tmp_path / "workflow.json"
    data = {"format": FORMAT, "version": 1, "steps": [{"module": "otsu_threshold", "options": {"blur_px": 3}},
                                                       {"module": "trace_regions", "enabled": False}]}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_workflow(path) == [step("otsu_threshold", blur_px=3.0), step("trace_regions", enabled=False)]

    for change, why in (({"format": "something else"}, "not a workflow file"),
                        ({"version": 2}, "version 2"),
                        ({"steps": [{"module": "otsu_threshold"}, {"module": "sharpen"}]}, "step 2: there is no module"),
                        ({"steps": [{"module": "upscale", "options": {"factor": 99}}]}, "step 1: upscale.factor must be"),
                        ({"steps": [{"options": {}}]}, "step 1: 'module'")):
        path.write_text(json.dumps(data | change), encoding="utf-8")
        with pytest.raises(ValueError, match=why):
            load_workflow(path)
    path.write_text("[1, 2", encoding="utf-8")
    with pytest.raises(ValueError):
        load_workflow(path)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a workflow file"):
        load_workflow(path)
