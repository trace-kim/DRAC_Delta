"""The viewer (viewer.py, outside the package): its options are those of extract_contours, it finds what
extract_contours finds, and its window shows the contacts of an image."""

import inspect
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
from contour_extraction.pipeline import extract_any_contours, extract_contours
from synthetic import shape_image, vertical_lines, write_sem_like


def write_lines(path):
    """Write a PNG of two bright lines from the top to the bottom of the image (synthetic.vertical_lines), whose four
    edges are open contours."""
    ok, encoded = cv2.imencode(".png", shape_image(vertical_lines, noise=4))
    assert ok
    encoded.tofile(str(path))


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(condition, app, seconds=60.0):
    """Run the Qt events until condition() holds (the background search sends its result as one)."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        app.processEvents()
        time.sleep(0.01)


def test_the_options_are_the_pattern_the_data_bar_and_those_of_extract_contours():
    """An option added to extract_contours must be added to the viewer too, with its help and limits;
    extract_any_contours takes some of them, with the same defaults."""
    names = [name for _, group in viewer.GROUPS for name in group]
    keywords = [name for name, p in inspect.signature(extract_contours).parameters.items()
                if p.kind is p.KEYWORD_ONLY]
    assert sorted(names) == sorted(["pattern", "databar_rows", *keywords])  # each once
    assert set(viewer.HELP) == set(names)
    for name, p in inspect.signature(extract_any_contours).parameters.items():
        if p.kind is p.KEYWORD_ONLY:
            assert viewer.DEFAULTS[name] == p.default, name
    for name in names:
        default = viewer.DEFAULTS[name]
        if name in viewer.CHOICES:
            assert default in viewer.CHOICES[name]
        elif not isinstance(default, bool):
            low, high, _ = viewer.LIMITS[name]
            assert low <= default <= high, name


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


def test_find_with_pattern_contours_finds_what_extract_any_contours_finds(tmp_path):
    path = tmp_path / "lines.png"
    write_lines(path)
    options = viewer.DEFAULTS | {"pattern": "contours", "dark_contacts": False, "upscale": 2}

    found = viewer.find(load_sem_image(path), options)

    keywords = {name: options[name] for name in viewer.CONTOUR_OPTIONS}
    detected, refined = extract_any_contours(load_sem_image(path), **keywords)
    for ours, theirs in ((found.detected, detected), (found.refined, refined)):
        assert len(ours.points) == len(theirs.points) == 4
        for a, b in zip(ours.points, theirs.points):
            np.testing.assert_array_equal(a, b)
    assert found.dropped == 0


@pytest.mark.parametrize("prepare", [{}, {"background_px": 60.0, "denoise": "median", "upscale": 2}])
def test_find_finds_what_extract_contours_finds(tmp_path, prepare):
    """find prepares the image itself, to show it, and gives extract_contours the prepared one: the same contacts."""
    path = tmp_path / "tile.jpg"
    write_sem_like(path, databar_px=64)
    loaded = load_sem_image(path)
    options = viewer.DEFAULTS | {"databar_rows": 64} | prepare

    found = viewer.find(loaded, options)

    keywords = {name: options[name] for name in viewer.CONTACT_OPTIONS}
    detected, refined, dropped = extract_contours(crop_databar(loaded, 64), **keywords)
    np.testing.assert_array_equal(found.detected.centers, detected.centers)
    np.testing.assert_array_equal(found.refined.centers, refined.centers)
    assert found.dropped == dropped
    f = options["upscale"]
    assert found.image.shape == (704, 1024)
    assert found.prepared.shape == found.mask.shape == (704 * f, 1024 * f)


def test_the_window_shows_the_contacts_of_an_image(tmp_path, app):
    path = tmp_path / "tile.jpg"
    truth = write_sem_like(path, databar_px=64)
    window = viewer.Viewer()
    viewer.set_value(window.editors["databar_rows"], 64)

    window.open_image(str(path))
    wait_for(lambda: window.result is not None and not window.running, app)

    assert f"<b>{len(truth)} contacts</b>" in window.summary.text()
    x, y = window.items["Refined centres"].getData()
    assert len(x) == len(truth)
    assert window.displayed.shape == (704, 1024)  # the data bar cropped
    window.close()


def test_the_window_shows_the_open_contours_of_lines_running_out_of_the_image(tmp_path, app):
    path = tmp_path / "lines.png"
    write_lines(path)
    window = viewer.Viewer()
    viewer.set_value(window.editors["pattern"], "contours")
    viewer.set_value(window.editors["dark_contacts"], False)

    window.open_image(str(path))
    wait_for(lambda: window.result is not None and not window.running, app)

    assert "<b>4 contours</b>: 0 closed, 4 open" in window.summary.text()
    x, y = window.items["Refined contours"].getData()
    assert np.isnan(x).sum() == 4  # one break after each of the four edges, none closed back to its start
    assert not window.items["Refined centres"].isVisible()  # contours have no centre
    window.close()


def test_a_failed_search_says_why_and_shows_no_contacts(tmp_path, app):
    path = tmp_path / "tile.jpg"
    write_sem_like(path, databar_px=64)
    window = viewer.Viewer()
    viewer.set_value(window.editors["median_px"], 4)  # median windows must be odd
    viewer.set_value(window.editors["denoise"], "median")

    window.open_image(str(path))
    wait_for(lambda: not window.running and window.summary.text(), app)

    assert "No contacts found" in window.summary.text()
    assert window.result is None
    x, _ = window.items["Refined centres"].getData()
    assert x is None or len(x) == 0
    window.close()


def test_options_that_change_nothing_are_greyed_out(app):
    window = viewer.Viewer()

    viewer.set_value(window.editors["denoise"], "median")
    viewer.set_value(window.editors["refinement"], "none")

    assert window.editors["median_px"].isEnabled() and not window.editors["nlm_h"].isEnabled()
    assert not window.editors["min_solidity"].isEnabled()  # band only
    assert not window.editors["search_px"].isEnabled()
    assert not window.boxes["Refined contours"].isEnabled()

    viewer.set_value(window.editors["pattern"], "contours")
    viewer.set_value(window.editors["detection"], "band")
    assert not window.editors["min_solidity"].isEnabled() and not window.editors["min_contrast"].isEnabled()
    assert window.editors["min_area_px"].isEnabled()
    assert not window.boxes["Detected centres"].isEnabled()
    window.close()


def test_a_dropped_file_is_the_one_opened(tmp_path):
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path / "tile.jpg"))])

    assert Path(viewer.dropped_file(mime)) == tmp_path / "tile.jpg"
    assert viewer.dropped_file(QtCore.QMimeData()) is None
