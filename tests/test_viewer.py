"""The viewer (viewer.py, outside the package): a window to design a workflow in. It offers every module and every
workflow to start from, runs the workflow on the image when it changes (or when asked), and shows what each step
made."""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window on the screen; read when Qt starts

import cv2
import numpy as np
import pytest

for package in ("PySide6", "pyqtgraph"):  # the viewer's packages are optional: pip install -e ".[viewer]"
    pytest.importorskip(package, reason="the viewer's packages are not installed (pip install -e .[viewer])")

from pyqtgraph.Qt import QtCore, QtWidgets

import viewer
from contour_extraction.load import crop_databar, load_sem_image
from contour_extraction.modules import MODULES, STAGES, image_contours
from contour_extraction.workflow import PRESETS, load_workflow, preset, run_workflow, step, workflow_contours
from synthetic import shape_image, vertical_lines, write_sem_like

CHECKED, UNCHECKED = QtCore.Qt.CheckState.Checked, QtCore.Qt.CheckState.Unchecked


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def tile(tmp_path_factory):
    """The file of a synthetic SEM tile of 176 contact holes with a data bar of 64 rows, and their true centres."""
    path = tmp_path_factory.mktemp("viewer") / "tile.jpg"
    return path, write_sem_like(path, databar_px=64)


@pytest.fixture
def window(app):
    window = viewer.Viewer()
    yield window
    window.close()


def wait_for(condition, app, seconds=60.0):
    """Run the Qt events until condition() holds (the background run sends its result as one)."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        app.processEvents()
        time.sleep(0.01)


def settle(window, app):
    """Wait until the window shows what its workflow makes of its image."""
    wait_for(lambda: not (window.running or window.timer.isActive() or window.waiting()), app)


def contact_holes() -> list:
    """The workflow for the tile: its data bar cropped, then the first example."""
    return [step("crop_databar", rows=64)] + preset("Contact holes")


def rows(window, column=0) -> list[str]:
    return [window.list.topLevelItem(row).text(column) for row in range(window.list.topLevelItemCount())]


def select(window, row):
    window.list.setCurrentItem(window.list.topLevelItem(row))


def write_lines(path):
    """Write a PNG of two bright lines from the top to the bottom of the image (synthetic.vertical_lines), whose four
    edges are open contours."""
    ok, encoded = cv2.imencode(".png", shape_image(vertical_lines, noise=4))
    assert ok
    encoded.tofile(str(path))


# --- What the window offers -------------------------------------------------------------------------------------------

def test_the_window_offers_every_module_by_stage_and_every_workflow_to_start_from(window):
    stages = window.add_menu.actions()
    assert [action.text() for action in stages] == list(STAGES)
    for action in stages:
        offered = [(entry.text(), entry.toolTip()) for entry in action.menu().actions()]
        modules = [m for m in MODULES.values() if m.stage == action.text()]
        assert [title for title, _ in offered] == [m.title for m in modules]
        assert all(tooltip for _, tooltip in offered)
    examples = window.examples_menu.actions()
    assert [action.text() for action in examples] == list(PRESETS) and all(action.toolTip() for action in examples)
    assert window.steps == preset(viewer.START) and not window.modified  # it opens with the first example
    assert viewer.START in window.workflow_label.text() and window.auto.isChecked()


def test_a_new_step_goes_after_the_one_selected_or_where_a_module_of_its_stage_belongs():
    steps = [step("crop_databar"), step("otsu_threshold"), step("trace_regions"), step("refine_edges")]
    after_the_last = len(steps)
    assert viewer.place_for(steps, "gaussian_blur", after_the_last) == 1  # an image module: before the mask is made
    assert viewer.place_for(steps, "gaussian_blur", 0) == 0 and viewer.place_for(steps, "upscale", 1) == 1
    assert viewer.place_for(steps, "fill_holes", after_the_last) == 2  # a mask module: before the contours are traced
    assert viewer.place_for(steps, "fill_holes", 0) == 1
    assert viewer.place_for(steps, "smooth_contours", after_the_last) == 4 and viewer.place_for(steps, "filter_contours", 3) == 3
    assert viewer.place_for(steps, "smooth_contours", 0) == 2
    assert viewer.place_for([], "refine_edges", 0) == 0
    mixed = [step("otsu_threshold"), step("gaussian_blur"), step("trace_regions")]  # not in that order: as wanted
    assert viewer.place_for(mixed, "median_filter", 2) == 2 and viewer.place_for(mixed, "median_filter", 3) == 3


def test_every_option_of_every_module_has_an_editor_that_shows_its_default(app):
    for module in MODULES.values():
        for name, option in module.options.items():
            widget = viewer.editor(option)
            assert viewer.value(widget) == option.default, (module.name, name)
            assert type(viewer.value(widget)) is type(option.default), (module.name, name)


def test_outlines_draws_every_contour_closed_with_a_break_after_it():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)

    line = viewer.outlines([square, square + 5])

    assert line.shape == (12, 2)
    np.testing.assert_array_equal(line[4], square[0])
    np.testing.assert_array_equal(line[10], square[0] + 5)
    assert np.isnan(line[5]).all() and np.isnan(line[11]).all()
    assert viewer.outlines([]).shape == (0, 2)


def test_outlines_leaves_an_open_contour_open():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)

    line = viewer.outlines([square, square + 5], closed=np.array([False, True]))

    assert line.shape == (11, 2)
    assert np.isnan(line[4]).all()  # the open one ends at its last point
    np.testing.assert_array_equal(line[9], square[0] + 5)


def test_a_dropped_file_is_the_one_opened(tmp_path):
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path / "tile.jpg"))])

    assert Path(viewer.dropped_file(mime)) == tmp_path / "tile.jpg"
    assert viewer.dropped_file(QtCore.QMimeData()) is None


# --- Running the workflow, and what is shown ----------------------------------------------------------------------------

def test_the_window_shows_the_contours_the_workflow_finds_in_an_image(tile, window, app):
    path, truth = tile
    window.set_workflow(contact_holes(), "Contact holes")

    window.open_image(str(path))
    settle(window, app)

    found = workflow_contours(load_sem_image(path), contact_holes())  # what the package finds without the window
    assert f"<b>{len(truth)} contours</b> after step 4 (Refine edges): {len(truth)} closed" in window.summary.text()
    x, y = window.items["Centres"].getData()
    assert len(x) == len(truth) == len(found.points)
    x, y = window.items["Contours"].getData()
    np.testing.assert_array_equal(np.column_stack([x, y]), viewer.outlines(found.points, found.closed))
    assert window.displayed.shape == (704, 1024)  # the data bar cropped
    assert window.legend_names == {"Contours": "4. Refine edges", "Contours before": "3. Trace regions (before)"}
    assert all(window.items[name].isVisible() for name in ("Contours", "Contours before", "Centres"))
    assert not window.items["Mask"].isVisible() and window.boxes["Mask"].isEnabled()  # there, and not ticked

    assert rows(window) == ["1. Crop data bar", "2. Otsu threshold", "3. Trace regions", "4. Refine edges"]
    notes = rows(window, 1)
    assert notes[0] == "704 rows left" and notes[1].startswith("grey level ") and notes[3] == "mid-level"
    assert notes[2] == f"{len(truth)} contours ({len(truth)} closed, 0 open)"
    assert all(text.endswith(" s") for text in rows(window, 2))
    assert "tile.jpg" in window.windowTitle() and "Contact holes" in window.windowTitle()


def test_the_steps_are_added_after_the_one_selected_removed_moved_and_turned_off(tile, window, app):
    path, truth = tile
    window.set_workflow(contact_holes(), "Contact holes")
    window.open_image(str(path))
    settle(window, app)

    select(window, 0)
    window.add_step("gaussian_blur")
    assert rows(window)[:3] == ["1. Crop data bar", "2. Gaussian blur", "3. Otsu threshold"]
    assert window.selected() == 1 and list(window.editors) == ["sigma_px"] and window.modified
    assert "(changed)" in window.workflow_label.text()
    settle(window, app)
    assert len(window.result.states) == 6 and f"<b>{len(truth)} contours</b>" in window.summary.text()

    window.move_step(1)  # the blur after the threshold: Down
    assert rows(window)[1:3] == ["2. Otsu threshold", "3. Gaussian blur"] and window.selected() == 2
    assert window.step_buttons["Up"].isEnabled() and window.step_buttons["Down"].isEnabled()
    window.move_step(-1)
    window.move_step(-1)
    assert rows(window)[0] == "1. Gaussian blur" and not window.step_buttons["Up"].isEnabled()
    window.move_step(-1)  # at the top: stays
    assert window.selected() == 0

    window.list.topLevelItem(0).setCheckState(0, UNCHECKED)  # the tick: off
    assert window.steps[0].enabled is False
    settle(window, app)
    assert rows(window, 1)[0] == "off" and window.result.states[1].image is window.loaded

    window.remove_step()
    assert rows(window) == ["1. Crop data bar", "2. Otsu threshold", "3. Trace regions", "4. Refine edges"]
    assert window.steps == contact_holes()
    for _ in range(4):
        window.remove_step()
    assert window.steps == [] and window.selected() is None and not window.step_buttons["Remove"].isEnabled()
    settle(window, app)
    assert "<b>No contours</b> in the image as loaded" in window.summary.text()
    window.add_step("otsu_threshold")  # into an empty list
    assert rows(window) == ["1. Otsu threshold"] and window.selected() == 0


def test_changing_an_option_runs_the_workflow_again_from_that_step_on(tile, window, app):
    path, _ = tile
    window.set_workflow(contact_holes(), "Contact holes")
    window.open_image(str(path))
    settle(window, app)
    before = window.result

    select(window, 3)
    assert list(window.editors) == ["rule", "search_px", "step_px", "blur_px"]
    assert all(window.editors[name].toolTip() for name in window.editors)
    viewer.set_value(window.editors["rule"], "threshold")
    assert window.steps[3].options["rule"] == "threshold" and window.waiting()
    settle(window, app)

    assert rows(window, 1)[3] == "threshold" and window.result is not before
    assert all(a is b for a, b in zip(window.result.states[:4], before.states[:4]))  # the steps before it: not again
    assert window.result.final.contours is not before.final.contours

    select(window, 1)  # another step: its own editors, showing its options
    viewer.set_value(window.editors["blur_px"], 3.0)
    select(window, 3)
    select(window, 1)
    assert viewer.value(window.editors["blur_px"]) == 3.0 and window.steps[1].options["blur_px"] == 3.0
    settle(window, app)
    assert window.result.states[1] is before.states[1] and window.result.states[2] is not before.states[2]

    viewer.set_value(window.editors["method"], "otsu3")
    window.defaults_button.click()  # the step's options back to its module's
    assert window.steps[1] == step("otsu_threshold") and viewer.value(window.editors["blur_px"]) == 1.5
    assert viewer.value(window.editors["method"]) == "otsu"
    settle(window, app)
    np.testing.assert_array_equal(window.result.states[2].mask, before.states[2].mask)

    select(window, 3)
    window.add_step("median_filter")  # an image module, the last step selected: it goes before the threshold
    assert rows(window)[:3] == ["1. Crop data bar", "2. Median filter", "3. Otsu threshold"] and window.selected() == 1


def test_without_auto_process_the_changes_wait_for_the_process_button(tile, window, app):
    path, truth = tile
    window.set_workflow(contact_holes(), "Contact holes")
    window.auto.setChecked(False)

    window.open_image(str(path))  # an image, too
    app.processEvents()
    assert window.result is None and not window.running and "press Process" in window.busy.text()
    assert window.displayed is window.loaded
    window.process_button.click()
    settle(window, app)
    assert f"<b>{len(truth)} contours</b>" in window.summary.text() and window.busy.text() == ""

    shown = window.result
    select(window, 1)
    viewer.set_value(window.editors["dark"], False)
    window.add_step("fill_holes")
    time.sleep(0.3)  # longer than the pause Auto process waits for
    app.processEvents()
    assert window.result is shown and not window.running and "press Process" in window.busy.text()
    assert rows(window, 1)[0] == "704 rows left" and rows(window, 1)[1:] == ["", "", "", ""]  # not run as they are now

    window.process_button.click()
    settle(window, app)
    assert window.result is not shown and window.busy.text() == "" and rows(window, 1)[1] != ""

    window.remove_step()
    assert window.waiting() and not window.running
    window.auto.setChecked(True)  # ticked again: what waits is run
    settle(window, app)
    assert len(window.result.steps) == 4


def test_the_selected_step_can_be_shown_instead_of_the_result(tile, window, app):
    path, truth = tile
    window.set_workflow([step("crop_databar", rows=64), step("gaussian_blur", sigma_px=3.0)] + preset("Contact holes"))
    window.open_image(str(path))
    settle(window, app)
    blurred = window.result.states[2].image
    assert window.displayed is blurred  # the result: after the last step, whose image the blur made

    window.view_buttons["selected step"].setChecked(True)
    select(window, 0)
    assert window.displayed is window.result.states[1].image and "No contours</b> after step 1" in window.summary.text()
    assert not window.boxes["Mask"].isEnabled() and not window.boxes["Contours"].isEnabled()
    assert not window.legend.isVisible()

    window.boxes["Mask"].setChecked(True)
    select(window, 2)  # the threshold: its mask, no contours yet
    assert window.displayed is blurred and window.items["Mask"].isVisible()
    np.testing.assert_array_equal(window.items["Mask"].image, window.result.states[3].mask)
    assert not window.items["Contours"].isVisible() and "of it mask" in window.summary.text()

    select(window, 3)  # the tracing: the contours it made, none before
    assert window.legend_names == {"Contours": "4. Trace regions"} and not window.boxes["Contours before"].isEnabled()
    x, _ = window.items["Centres"].getData()
    assert len(x) == len(truth)

    window.view_buttons["result"].setChecked(True)
    assert window.legend_names == {"Contours": "5. Refine edges", "Contours before": "4. Trace regions (before)"}


def test_the_contours_can_be_drawn_over_the_original_image_instead_of_the_processed_one(tile, window, app):
    path, truth = tile
    window.set_workflow([step("crop_databar", rows=64), step("upscale", factor=2)] + preset("Contact holes"))
    window.open_image(str(path))
    settle(window, app)
    covered = lambda: window.image_item.mapRectToParent(window.image_item.boundingRect())
    assert window.displayed is window.result.final.image and window.displayed.shape == (1408, 2048)  # more pixels,
    assert covered() == QtCore.QRectF(-0.5, -0.5, 1024, 704)  # over the same area: the image's own pixel frame
    x, y = window.items["Centres"].getData()
    assert len(x) == len(truth) and np.abs(np.column_stack([x, y])[:, None] - truth[None]).max(axis=2).min(axis=1).max() < 0.3

    window.image_buttons["original"].setChecked(True)
    assert window.displayed is window.loaded and covered() == QtCore.QRectF(-0.5, -0.5, 1024, 768)  # with its data bar
    x, _ = window.items["Centres"].getData()  # the layers stay over it
    assert len(x) == len(truth) and window.items["Centres"].isVisible()

    window.image_buttons["processed"].setChecked(True)
    assert window.displayed is window.result.final.image


def test_the_window_shows_the_open_contours_of_lines_running_out_of_the_image(tmp_path, window, app):
    path = tmp_path / "lines.png"
    write_lines(path)
    steps = preset("Any pattern")
    steps[0].options["dark"] = False
    window.set_workflow(steps)

    window.open_image(str(path))
    settle(window, app)

    assert "<b>4 contours</b> after step 3 (Refine edges): 0 closed, 4 open" in window.summary.text()
    x, y = window.items["Contours"].getData()
    assert np.isnan(x).sum() == 4  # one break after each of the four edges, none closed back to its start
    assert not window.items["Centres"].isVisible() and not window.boxes["Centres"].isEnabled()  # no closed contour


def test_a_step_that_fails_is_named_and_the_steps_before_it_are_shown(tile, window, app):
    path, _ = tile
    window.set_workflow([step("crop_databar", rows=64), step("otsu_threshold"), step("refine_edges"),
                         step("smooth_contours")])

    window.open_image(str(path))
    settle(window, app)

    assert "Step 3 (Refine edges) failed: Refine edges needs contours" in window.summary.text()
    assert "<b>No contours</b> after step 2 (Otsu threshold)" in window.summary.text()
    notes = rows(window, 1)
    assert notes[2].startswith("Refine edges needs contours") and notes[3] == "" and notes[1].startswith("grey level")
    assert window.list.topLevelItem(2).foreground(0).color().name() == viewer.FAILURE
    assert window.displayed.shape == (704, 1024) and window.boxes["Mask"].isEnabled()
    x, _ = window.items["Contours"].getData()
    assert x is None or len(x) == 0

    select(window, 1)
    window.add_step("trace_regions")  # what was missing
    settle(window, app)
    assert "failed" not in window.summary.text() and "<b>176 contours</b>" in window.summary.text()
    assert window.list.topLevelItem(3).foreground(0).color().name() != viewer.FAILURE


# --- Workflow files -----------------------------------------------------------------------------------------------------

def test_a_workflow_is_saved_in_a_file_and_opened_again(tmp_path, window, monkeypatch):
    window.set_workflow(preset("Noisy image"), "Noisy image")
    select(window, 2)
    viewer.set_value(window.editors["h"], 22.0)
    window.list.topLevelItem(1).setCheckState(0, UNCHECKED)
    assert window.modified and "Noisy image (changed)" in window.workflow_label.text()

    assert window.save_workflow(str(tmp_path / "my noisy.json"))
    assert not window.modified and window.workflow_name == "my noisy" and "(changed)" not in window.workflow_label.text()
    saved = load_workflow(tmp_path / "my noisy.json")  # the package reads it without the window
    assert saved == window.steps and saved[2].options["h"] == 22.0 and saved[1].enabled is False

    window.set_workflow(preset("Contact holes"), "Contact holes")
    assert window.open_workflow(str(tmp_path / "my noisy.json"))
    assert window.steps == saved and window.workflow_name == "my noisy" and not window.modified
    assert window.list.topLevelItem(1).checkState(0) == UNCHECKED and window.list.topLevelItem(0).checkState(0) == CHECKED
    select(window, 2)
    assert viewer.value(window.editors["h"]) == 22.0

    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    (tmp_path / "other.json").write_text('{"format": "something else"}', encoding="utf-8")
    assert not window.open_workflow(str(tmp_path / "other.json")) and "not a workflow file" in warnings[0]
    assert not window.open_workflow(str(tmp_path / "missing.json")) and len(warnings) == 2
    assert window.steps == saved  # as it was


def test_an_example_replaces_a_changed_workflow_only_if_the_user_says_so(window, monkeypatch):
    answers = []
    reply = QtWidgets.QMessageBox.StandardButton.No
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: answers.append(args[2]) or reply)

    window.choose_preset("Any pattern")  # nothing changed: not asked
    assert window.steps == preset("Any pattern") and answers == [] and "Any pattern" in window.workflow_label.text()

    window.add_step("smooth_contours")
    window.choose_preset("Contact holes")  # asked, and refused
    assert len(answers) == 1 and "not saved" in answers[0] and window.steps[-1].module == "smooth_contours"

    reply = QtWidgets.QMessageBox.StandardButton.Yes
    window.choose_preset("Contact holes")
    assert len(answers) == 2 and window.steps == preset("Contact holes") and not window.modified


def test_a_json_file_named_or_dropped_is_a_workflow_and_any_other_file_an_image(tile, tmp_path, window, app):
    path, truth = tile
    window.set_workflow(contact_holes())
    assert window.save_workflow(str(tmp_path / "holes.json"))
    window.set_workflow(preset("Any pattern"))

    window.open_file(str(path))
    window.open_file(str(tmp_path / "holes.json"))
    settle(window, app)

    assert window.workflow_name == "holes" and window.steps == contact_holes()
    assert f"<b>{len(truth)} contours</b>" in window.summary.text()
    run = run_workflow(crop_databar(load_sem_image(path), 0), contact_holes())
    np.testing.assert_array_equal(image_contours(window.result.final).points[0], image_contours(run.final).points[0])
