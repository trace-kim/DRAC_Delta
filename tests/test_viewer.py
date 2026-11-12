"""The viewer (viewer.py, outside the package): its options are those of extract_contours, it finds what
extract_contours finds, and its window shows the contacts of an image."""

import inspect
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window on the screen; read when Qt starts

import numpy as np
import pytest

for package in ("PySide6", "pyqtgraph"):  # the viewer's packages are optional: pip install -e ".[viewer]"
    pytest.importorskip(package, reason="the viewer's packages are not installed (pip install -e .[viewer])")

from pyqtgraph.Qt import QtCore, QtWidgets

import viewer
from contour_extraction.load import crop_databar, load_sem_image
from contour_extraction.pipeline import extract_contours
from synthetic import write_sem_like


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


def test_the_options_are_the_data_bar_and_those_of_extract_contours():
    """An option added to extract_contours must be added to the viewer too, with its help and limits."""
    names = [name for _, group in viewer.GROUPS for name in group]
    keywords = [name for name, p in inspect.signature(extract_contours).parameters.items()
                if p.kind is p.KEYWORD_ONLY]
    assert sorted(names) == sorted(["databar_rows", *keywords])  # each once
    assert set(viewer.HELP) == set(names)
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


@pytest.mark.parametrize("prepare", [{}, {"background_px": 60.0, "denoise": "median", "upscale": 2}])
def test_find_finds_what_extract_contours_finds(tmp_path, prepare):
    """find prepares the image itself, to show it, and gives extract_contours the prepared one: the same contacts."""
    path = tmp_path / "tile.jpg"
    write_sem_like(path, databar_px=64)
    loaded = load_sem_image(path)
    options = viewer.DEFAULTS | {"databar_rows": 64} | prepare

    found = viewer.find(loaded, options)

    keywords = {name: value for name, value in options.items() if name != "databar_rows"}
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
    window.close()


def test_a_dropped_file_is_the_one_opened(tmp_path):
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path / "tile.jpg"))])

    assert Path(viewer.dropped_file(mime)) == tmp_path / "tile.jpg"
    assert viewer.dropped_file(QtCore.QMimeData()) is None
