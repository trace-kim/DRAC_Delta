"""A window to design a workflow that finds the contours in SEM images: open an image, put modules in a list, change
their options, and see what each makes of the image.

    .venv\\Scripts\\python viewer.py [image] [workflow.json]

A workflow is a list of steps, each a module of contour_extraction (modules.py) with its options: image modules
(crop, flatten, denoise, enhance, upscale), mask modules (thresholds, Canny edges, morphology) and contours modules
(trace, filter, refine, smooth). Add step puts a module into the list, after the step selected; the ticks turn steps
off and on; Up and Down move them; the options of the step selected are below the list, each explained in its
tooltip. Examples holds workflows to start from (workflow.PRESETS); Open and Save keep a workflow in a JSON file,
which contour_extraction.load_workflow reads without this window:

    contours = workflow_contours(load_sem_image(path), load_workflow("my workflow.json"))

With Auto process ticked, every change runs the workflow again, in a background thread, for the newest steps only,
from the first step that changed; unticked, Process (F5) does. The list says what every step found and how long it
took. The plot shows what the last step made (View: result), or the step selected: its image, or the image as loaded,
with its mask, its contours, the contours as they were before the step that last changed them, and the centres of
the closed ones.
Open an image with Open image... (Ctrl+O), by dropping its file on the window, or by naming it on the command line;
a .json file dropped or named is a workflow.

Not part of the contour_extraction package, which needs no Qt: a caller of it like any other.
Coordinates are in the pixel frame: (x, y) = (column, row), y pointing down, pixel centres at integers.
"""

import copy
import html
import sys
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from contour_extraction import (MODULES, PRESETS, STAGES, Option, Run, State, Step, image_contours, load_sem_image,
                                load_workflow, polygon_area, polygon_centroid, preset, run_workflow, save_workflow,
                                step)
from contour_extraction.modules import LEFT_OUT

START = "Contact holes"  # the workflow (of PRESETS) the window opens with
VIEWS = {  # what the plot shows, and what that is (its tooltip)
    "result": "What the whole workflow made: the state after its last step (that ran)",
    "selected step": "What the workflow had made after the step selected in the list: to see what each step does",
}
IMAGES = {  # the image under the layers, and what it is (its tooltip)
    "processed": "The image as the image modules made it, up to the step shown: what the pattern is found in",
    "original": "The image as loaded: the contours over the image itself",
}
LAYERS = {  # the layers over the image, and what each is (its tooltip)
    "Mask": "The mask as it is at the step shown: the regions a threshold took, or Canny's edge pixels",
    "Contours": "The contours as they are at the step shown",
    "Contours before": "The contours as they were before the step that last changed them (the pixel outline before "
                       "Refine edges, all of them before Filter contours)",
    "Centres": "The area centroid of every closed contour",
}
BEFORE, AFTER, MASK = "#c2189b", "#00a8c8", "#e8b500"  # the colours of AffineRANSAC's contour diagnosis
FAILURE, MUTED = "#e34948", "#77766f"
IMAGE_FILES = "SEM images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)"
WORKFLOW_FILES = "Workflows (*.json);;All files (*)"
NO_IMAGE = ("Open an SEM image (<i>Open image...</i>, Ctrl+O, or drop its file on the window) to see the contours "
            "the workflow on the right finds in it. Hover a step, an option or a menu entry to see what it does.")


def step_title(steps: list[Step], row: int) -> str:
    """A step as the list names it: its number (from 1) and its module."""
    name = steps[row].module
    return f"{row + 1}. {MODULES[name].title if name in MODULES else name}"


def place_for(steps: list[Step], module: str, wanted: int) -> int:
    """The row to put a new step of this module at: `wanted`, or the nearest row that keeps the image modules before
    the mask modules before the contours modules (a blur added while the last step is selected goes before the
    threshold, not after the contours). A list that is not in that order already: `wanted`, as it is."""
    stages = [STAGES.index(MODULES[s.module].stage) for s in steps]
    if stages != sorted(stages):
        return wanted
    stage = STAGES.index(MODULES[module].stage)
    first = sum(1 for other in stages if other < stage)  # after every step of an earlier stage
    last = sum(1 for other in stages if other <= stage)  # before every step of a later one
    return min(max(wanted, first), last)


def alike(steps: list[Step], run: Run | None) -> int:
    """How many steps, from the first, are those the run ran: its states after them are still theirs."""
    count = 0
    if run is not None:
        while count < min(len(steps), len(run.states) - 1) and steps[count] == run.steps[count]:
            count += 1
    return count


def made_by(run: Run, viewed: int) -> int | None:
    """The step (from 1) that made the contours of run.states[viewed] what they are; None: there are none."""
    contours = run.states[viewed].contours
    if contours is None:
        return None
    return next(k for k in range(viewed + 1) if run.states[k].contours is contours)


def summary(run: Run, viewed: int) -> str:
    """What the workflow found, as far as run.states[viewed], in words (HTML): how many contours, how large the
    closed ones are, which regions were left out and why, and how long it all took; a step that failed first."""
    state = run.states[viewed]
    where = "in the image as loaded" if viewed == 0 else \
        f"after step {viewed} ({html.escape(MODULES[run.steps[viewed - 1].module].title)})"
    contours = image_contours(state)
    if contours is None:
        rows, cols = state.image.shape
        text = f"<b>No contours</b> {where}: an image of {cols} × {rows} px"
        if state.mask is not None:
            text += f", {100 * float(state.mask.mean()):.0f} % of it mask"
    else:
        areas = [polygon_area(points) for points, closed in zip(contours.points, contours.closed) if closed]
        text = f"<b>{len(contours.points)} contours</b> {where}: {len(areas)} closed"
        if areas:
            median = float(np.median(areas))
            text += f" (median area {median:.1f} px², diameter {2 * np.sqrt(median / np.pi):.2f} px)"
        text += f", {len(contours.points) - len(areas)} open"
        left_out = [f"{count} {LEFT_OUT.get(why, why)}" for why, count in contours.left_out.items() if count]
        if left_out:
            text += f"; left out: {', '.join(left_out)}"
    text += f"; {sum(run.seconds):.2f} s."
    if run.error is not None:
        text = f"<span style='color:{FAILURE}'>{html.escape(run.failure)}.</span> Shown: {text}"
    return text


def outlines(contours: list[np.ndarray], closed: np.ndarray | None = None) -> np.ndarray:
    """The contours (each (M, 2)) as one (rows, 2) line to draw with connect="finite": each closed (its first point
    again; closed (N,) bool given: only those True, the others open, as they are) and followed by a NaN row, so that
    one plot item draws them all."""
    nan = np.full((1, 2), np.nan)
    closed = np.ones(len(contours), bool) if closed is None else closed
    parts = [part for contour, whole in zip(contours, closed)
             for part in ((contour, contour[:1], nan) if whole else (contour, nan))]
    return np.concatenate(parts) if parts else np.empty((0, 2))


def editor(option: Option) -> QtWidgets.QWidget:
    """A widget to edit an option, showing its default: a list of its choices, a check box (bool) or a spin box (int
    or float) that tells its value when the typing is done."""
    if option.choices:
        widget = QtWidgets.QComboBox()
        widget.addItems(option.choices)
    elif isinstance(option.default, bool):  # before int: a bool is an int
        widget = QtWidgets.QCheckBox()
    else:
        widget = QtWidgets.QSpinBox() if isinstance(option.default, int) else QtWidgets.QDoubleSpinBox(decimals=3)
        widget.setRange(option.low, option.high)
        widget.setSingleStep(option.step)
        widget.setKeyboardTracking(False)  # 12 typed is 12, not 1 and then 12
    set_value(widget, option.default)
    return widget


def value(widget: QtWidgets.QWidget):
    """The value an editor shows."""
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentText()
    if isinstance(widget, QtWidgets.QCheckBox):
        return widget.isChecked()
    return widget.value()


def set_value(widget: QtWidgets.QWidget, new):
    if isinstance(widget, QtWidgets.QComboBox):
        widget.setCurrentText(new)
    elif isinstance(widget, QtWidgets.QCheckBox):
        widget.setChecked(new)
    else:
        widget.setValue(new)


def changed_signal(widget: QtWidgets.QWidget):
    """The signal of an editor that says its value changed."""
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentTextChanged
    if isinstance(widget, QtWidgets.QCheckBox):
        return widget.toggled
    return widget.valueChanged


def place(item: pg.ImageItem, shape):
    """Put an image item over an image of this shape in the pixel frame: pixel (row, column) centred at (x, y) =
    (column, row). An item with more pixels (the image upscaled) covers the same area."""
    item.setRect(QtCore.QRectF(-0.5, -0.5, shape[1], shape[0]))


def dropped_file(mime: QtCore.QMimeData) -> str | None:
    """The first local file dragged, if any."""
    files = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()] if mime.hasUrls() else []
    return files[0] if files else None


def tip(text: str) -> str:
    """A tooltip that wraps its lines (Qt wraps rich text only)."""
    return f"<p>{html.escape(text)}</p>"


class Viewer(QtWidgets.QMainWindow):
    processed = QtCore.Signal(int, int, object)  # (request, version of the steps, the Run), from the background thread

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.loaded = None        # the image as loaded (uint8), data bar included
        self.image_name = ""
        self.steps = []           # the workflow: list[Step], changed in place by the list and the options
        self.workflow_name = ""   # of the example or the file it came from
        self.modified = False     # changed since
        self.version = 0          # counts the changes of the steps
        self.result = None        # Run: the newest finished run of the image loaded (None: none yet)
        self.result_version = -1  # the version of the steps it ran
        self.displayed = None     # the image shown, and the shape of the image whose pixel frame it covers
        self.frame = None
        self.image_folder = ""    # where Open image starts
        self.workflow_folder = ""  # where Open and Save (workflow) start
        self.request = 0          # number of the newest run asked for
        self.running = False      # a run is in the background thread
        self.filling = False      # the list is being filled: its ticks are not the user's
        self.editors = {}         # the editors of the step selected, by option
        self.processed.connect(self._processed)
        self.timer = QtCore.QTimer(self, singleShot=True, interval=150)  # a run once the changes pause
        self.timer.timeout.connect(self.process)

        # The plot: the image, the mask over it, the contours and centres on top.
        self.plot = pg.PlotWidget()
        self.plot.invertY(True)  # image rows run down
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "x (px)")
        self.plot.setLabel("left", "y (px)")
        self.legend = self.plot.addLegend(offset=(10, 10), brush=pg.mkBrush(0, 0, 0, 190), labelTextColor="w")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setZValue(-10)
        self.plot.addItem(self.image_item)
        mask = pg.ImageItem(axisOrder="row-major")
        red, green, blue, _ = pg.mkColor(MASK).getRgb()
        mask.setLookupTable(np.array([[0, 0, 0, 0], [red, green, blue, 110]], np.uint8))
        mask.setZValue(-5)
        self.items = {"Mask": mask,
                      "Contours before": pg.PlotDataItem(pen=pg.mkPen(BEFORE, width=1.5), connect="finite"),
                      "Contours": pg.PlotDataItem(pen=pg.mkPen(AFTER, width=1.5), connect="finite"),
                      "Centres": pg.PlotDataItem(pen=None, symbol="+", symbolSize=9, symbolPen=pg.mkPen(AFTER, width=1.5))}
        for item in self.items.values():
            self.plot.addItem(item)
        self.legend_names = {}  # of the contour layers, by layer
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)

        # What the plot shows: of which step, over which image, which layers.
        show = QtWidgets.QHBoxLayout()
        self.view_buttons = self._radio_buttons(show, "View:", VIEWS)
        show.addSpacing(20)
        self.image_buttons = self._radio_buttons(show, "Image:", IMAGES)
        show.addSpacing(20)
        show.addWidget(QtWidgets.QLabel("Show:"))
        self.boxes = {}
        for name, help_text in LAYERS.items():
            box = QtWidgets.QCheckBox(name)
            box.setChecked(name != "Mask")  # the mask covers the image: off at start
            box.setToolTip(tip(help_text))
            box.toggled.connect(self.update_layers)
            show.addWidget(box)
            self.boxes[name] = box
        show.addStretch(1)

        open_button = QtWidgets.QPushButton("Open image...")
        open_button.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open))
        open_button.setToolTip("Open an SEM image (Ctrl+O); or drop its file on the window")
        open_button.clicked.connect(self.choose_image)
        self.name_label = QtWidgets.QLabel("")
        self.summary = QtWidgets.QLabel(NO_IMAGE, wordWrap=True)
        self.busy = QtWidgets.QLabel("")
        self.busy.setStyleSheet(f"color: {MUTED}")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(open_button)
        top.addWidget(self.name_label, stretch=1)
        status = QtWidgets.QHBoxLayout()
        status.addWidget(self.summary, stretch=1)
        status.addWidget(self.busy)
        left = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        for row in (top, status, show):
            column.addLayout(row)
        column.addWidget(self.plot, stretch=1)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._workflow_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1040, 440])
        self.setCentralWidget(splitter)
        self.statusBar()  # the cursor's position and grey level
        self.set_workflow(preset(START), START)
        self.resize(1500, 900)

    def _radio_buttons(self, layout, label: str, choices: dict) -> dict:
        """Radio buttons for one of the choices (the first at start), each with its tooltip; any change shows the
        result again."""
        layout.addWidget(QtWidgets.QLabel(label))
        group = QtWidgets.QButtonGroup(self)  # one of them at a time
        buttons = {}
        for name, help_text in choices.items():
            button = QtWidgets.QRadioButton(name)
            button.setChecked(not buttons)
            button.setToolTip(tip(help_text))
            group.addButton(button)
            layout.addWidget(button)
            buttons[name] = button
        # Called for the button unticked and for the one ticked: show once, for the one ticked.
        group.buttonToggled.connect(lambda _, checked: checked and self.show_result())
        return buttons

    def _workflow_panel(self) -> QtWidgets.QWidget:
        """The right side: the workflow's name and files, its steps, the options of the step selected, and when it
        runs."""
        self.workflow_label = QtWidgets.QLabel("")
        examples = QtWidgets.QPushButton("Examples")
        examples.setToolTip("Workflows to start from, for different kinds of SEM image")
        menu = QtWidgets.QMenu(examples)
        menu.setToolTipsVisible(True)
        for name, (description, _) in PRESETS.items():
            action = menu.addAction(name)
            action.setToolTip(tip(description))
            action.triggered.connect(lambda _=False, name=name: self.choose_preset(name))
        examples.setMenu(menu)
        self.examples_menu = menu
        open_button = QtWidgets.QPushButton("Open...")
        open_button.setToolTip("Open a workflow saved in a file; or drop the file on the window")
        open_button.clicked.connect(self.choose_workflow)
        save_button = QtWidgets.QPushButton("Save...")
        save_button.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save))
        save_button.setToolTip("Save the workflow in a file (Ctrl+S), to open it here another day, or to run it "
                               "without this window (contour_extraction.load_workflow)")
        save_button.clicked.connect(self.choose_save)
        files = QtWidgets.QHBoxLayout()
        files.addWidget(self.workflow_label, stretch=1)
        for button in (examples, open_button, save_button):
            files.addWidget(button)

        self.list = QtWidgets.QTreeWidget()
        self.list.setHeaderLabels(["Step", "Result", "Time"])
        self.list.setRootIsDecorated(False)
        self.list.setUniformRowHeights(True)
        self.list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        header = self.list.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.list.itemChanged.connect(self._ticked)
        self.list.currentItemChanged.connect(self._selected)

        add = QtWidgets.QPushButton("Add step")
        add.setToolTip(tip("Put a module into the workflow: after the step selected, or at the nearest place where a "
                           "module of its kind belongs (image modules before the mask is made, mask modules before the "
                           "contours are traced). Up and Down move it anywhere"))
        menu = QtWidgets.QMenu(add)
        for stage in STAGES:
            stage_menu = menu.addMenu(stage)
            stage_menu.setToolTipsVisible(True)
            for module in MODULES.values():
                if module.stage == stage:
                    action = stage_menu.addAction(module.title)
                    action.setToolTip(tip(module.help))
                    action.triggered.connect(lambda _=False, name=module.name: self.add_step(name))
        add.setMenu(menu)
        self.add_menu = menu  # a menu per stage, in it an entry per module
        self.step_buttons = {}
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(add)
        for name, shortcut, slot, help_text in (
                ("Remove", "Ctrl+Del", self.remove_step, "Take the step selected out of the workflow"),
                ("Up", "Alt+Up", lambda: self.move_step(-1), "Run the step selected one step earlier"),
                ("Down", "Alt+Down", lambda: self.move_step(1), "Run the step selected one step later")):
            button = QtWidgets.QPushButton(name)
            button.setShortcut(QtGui.QKeySequence(shortcut))
            button.setToolTip(f"{help_text} ({shortcut})")
            button.clicked.connect(slot)
            buttons.addWidget(button)
            self.step_buttons[name] = button

        self.options_area = QtWidgets.QScrollArea(widgetResizable=True)
        self.options_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.auto = QtWidgets.QCheckBox("Auto process")
        self.auto.setChecked(True)
        self.auto.setToolTip(tip("Run the workflow again after every change: of an option, of the steps, of the image. "
                                 "Unticked (a slow workflow, several changes at once): Process runs it"))
        self.auto.toggled.connect(self._auto_toggled)
        self.process_button = QtWidgets.QPushButton("Process")
        self.process_button.setShortcut(QtGui.QKeySequence("F5"))
        self.process_button.setToolTip("Run the workflow on the image now (F5)")
        self.process_button.clicked.connect(self.process)
        when = QtWidgets.QHBoxLayout()
        when.addWidget(self.auto)
        when.addStretch(1)
        when.addWidget(self.process_button)

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.addLayout(files)
        layout.addWidget(self.list, stretch=2)
        layout.addLayout(buttons)
        layout.addWidget(self.options_area, stretch=3)
        layout.addLayout(when)
        panel.setMinimumWidth(380)
        return panel

    # --- The workflow: its steps and their options --------------------------------------------------------------------

    def selected(self) -> int | None:
        """The row of the step selected (None: none)."""
        row = self.list.indexOfTopLevelItem(self.list.currentItem())
        return row if row >= 0 else None

    def set_workflow(self, steps: list[Step], name: str = ""):
        """Replace the workflow by these steps (its name: of the example or file they are from), and run it."""
        self.steps, self.workflow_name = copy.deepcopy(list(steps)), name
        self.fill_list(len(self.steps) - 1)
        self.changed()
        self.modified = False
        self.update_names()

    def fill_list(self, select: int | None):
        """Show the steps in the list, and select one of them."""
        self.filling = True
        self.list.clear()
        for row, this in enumerate(self.steps):
            item = QtWidgets.QTreeWidgetItem([step_title(self.steps, row), "", ""])
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
                          | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if this.enabled else QtCore.Qt.CheckState.Unchecked)
            item.setTextAlignment(2, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.list.addTopLevelItem(item)
        self.filling = False
        if select is not None and self.steps:
            self.list.setCurrentItem(self.list.topLevelItem(max(0, min(select, len(self.steps) - 1))))
        else:
            self._selected()

    def add_step(self, module: str):
        """Put a step of this module, with its defaults, after the step selected (none: at the end), or at the
        nearest row where a module of its stage belongs (place_for)."""
        row = place_for(self.steps, module, len(self.steps) if self.selected() is None else self.selected() + 1)
        self.steps.insert(row, step(module))
        self.fill_list(row)
        self.changed()

    def remove_step(self):
        row = self.selected()
        if row is not None:
            del self.steps[row]
            self.fill_list(row)
            self.changed()

    def move_step(self, by: int):
        """Move the step selected up (-1) or down (1) the list."""
        row = self.selected()
        if row is not None and 0 <= row + by < len(self.steps):
            self.steps.insert(row + by, self.steps.pop(row))
            self.fill_list(row + by)
            self.changed()

    def _ticked(self, item, column):
        """A step ticked or unticked: on or off."""
        row = self.list.indexOfTopLevelItem(item)
        enabled = item.checkState(0) == QtCore.Qt.CheckState.Checked
        if not self.filling and column == 0 and self.steps[row].enabled != enabled:
            self.steps[row].enabled = enabled
            self.changed()

    def _selected(self, *_):
        """Another step is selected: show its options, and what it made if that is what the plot shows."""
        if self.filling:
            return
        row = self.selected()
        for name, button in self.step_buttons.items():
            button.setEnabled(row is not None and (name == "Remove" or 0 <= row + (-1 if name == "Up" else 1) < len(self.steps)))
        self.show_options()
        if self.view_buttons["selected step"].isChecked():
            self.show_result()

    def show_options(self):
        """The options of the step selected, each with its tooltip, under what its module does."""
        row = self.selected()
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        self.editors = {}
        if row is None:
            text = QtWidgets.QLabel("Add a step, or choose a workflow to start from under Examples.", wordWrap=True)
            text.setStyleSheet(f"color: {MUTED}")
            layout.addWidget(text)
        else:
            this, module = self.steps[row], MODULES[self.steps[row].module]
            layout.addWidget(QtWidgets.QLabel(f"<b>{html.escape(step_title(self.steps, row))}</b> "
                                              f"<span style='color:{MUTED}'>({module.stage.lower()} module)</span>"))
            layout.addWidget(QtWidgets.QLabel(module.help, wordWrap=True, textFormat=QtCore.Qt.TextFormat.PlainText))
            form = QtWidgets.QFormLayout()
            for name, option in module.options.items():
                widget = editor(option)
                set_value(widget, this.options[name])
                label = QtWidgets.QLabel(name)
                for item in (label, widget):
                    item.setToolTip(tip(option.help))
                form.addRow(label, widget)
                changed_signal(widget).connect(lambda new, name=name, this=this: self._edited(this, name, new))
                self.editors[name] = widget
            layout.addLayout(form)
            if module.options:
                self.defaults_button = QtWidgets.QPushButton("Defaults")
                self.defaults_button.setToolTip("Set the options of this step back to the module's defaults")
                self.defaults_button.clicked.connect(self.reset_step)
                layout.addWidget(self.defaults_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addStretch(1)
        self.options_area.setWidget(box)  # deletes the one before

    def _edited(self, this: Step, name: str, new):
        """An option of a step was changed in its editor."""
        if this.options[name] != new:
            this.options[name] = new
            self.changed()

    def reset_step(self, *_):
        """The options of the step selected back to its module's defaults; one run."""
        row = self.selected()
        if row is not None and self.steps[row].options != step(self.steps[row].module).options:
            self.steps[row].options = step(self.steps[row].module).options
            for name, widget in self.editors.items():  # shown without each editor reporting a change of its own
                widget.blockSignals(True)
                set_value(widget, self.steps[row].options[name])
                widget.blockSignals(False)
            self.changed()

    def changed(self):
        """The workflow changed: run it again once the changes pause (Auto process), or say that it waits for
        Process."""
        self.version += 1
        self.modified = True
        self.update_names()
        self.update_list()
        if self.auto.isChecked():
            self.timer.start()
        self.update_waiting()

    def waiting(self) -> bool:
        """Whether what is shown is not what the workflow, as it is now, makes of the image loaded."""
        return self.loaded is not None and (self.result is None or self.result_version != self.version)

    def update_waiting(self):
        """Say whether a run is on its way, or waits for Process."""
        waits = self.waiting() and not self.running and not self.timer.isActive()
        self.busy.setText("Processing..." if self.running or self.timer.isActive() and self.loaded is not None
                          else "Changed: press Process (F5)" if waits else "")
        self.process_button.setStyleSheet("font-weight: bold" if waits else "")

    def _auto_toggled(self, on: bool):
        """Ticked: run what waits. Unticked: a run that waited for the changes to pause waits for Process instead."""
        if on and self.waiting():
            self.process()
        elif not on:
            self.timer.stop()
        self.update_waiting()

    def update_names(self):
        """The names of the image and of the workflow, in the window's title and over the list."""
        name = (self.workflow_name or "Workflow") + (" (changed)" if self.modified else "")
        self.workflow_label.setText(f"<b>{html.escape(name)}</b>")
        self.setWindowTitle(" - ".join(part for part in (self.image_name, name, "Contour extraction") if part))

    def update_list(self):
        """Beside every step, what it found and how long it took, as far as the newest run ran these steps; the step
        that failed in red, with the reason."""
        run = self.result
        same = alike(self.steps, run)
        self.filling = True
        for row, this in enumerate(self.steps):
            item = self.list.topLevelItem(row)
            failed = run is not None and run.error is not None and row == same == len(run.states) - 1 \
                and row < len(run.steps) and run.steps[row] == this
            if failed:
                note, seconds = run.failure.split(": ", 1)[1], ""
            elif row < same:
                note, seconds = run.states[row + 1].note, f"{run.seconds[row]:.2f} s"
            else:
                note, seconds = "", ""
            item.setText(1, note)
            item.setText(2, seconds)
            module = MODULES.get(this.module)
            options = ", ".join(f"{name} = {value}" for name, value in this.options.items())
            item.setToolTip(0, tip(f"{module.help if module else ''} {'Options: ' + options if options else ''}"))
            item.setToolTip(1, tip(note) if note else "")
            colour = QtGui.QBrush(QtGui.QColor(FAILURE if failed else MUTED)) if failed or not this.enabled \
                else QtGui.QBrush()
            for column in range(3):
                item.setForeground(column, colour)
        self.filling = False

    # --- Workflow files -----------------------------------------------------------------------------------------------

    def may_replace(self) -> bool:
        """Whether the workflow may be replaced: it has no changes, or the user says so."""
        if not self.modified or not self.steps:
            return True
        answer = QtWidgets.QMessageBox.question(self, "Contour extraction",
                                                "Replace the workflow? Its changes are not saved.")
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def choose_preset(self, name: str):
        if self.may_replace():
            self.set_workflow(preset(name), name)

    def choose_workflow(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a workflow", self.workflow_folder, WORKFLOW_FILES)
        if path and self.may_replace():
            self.open_workflow(path)

    def open_workflow(self, path: str) -> bool:
        """Replace the workflow by the one of a file; False (and a message) if the file is none."""
        try:
            steps = load_workflow(path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Contour extraction", f"The workflow could not be read:\n{error}")
            return False
        self.workflow_folder = str(Path(path).parent)
        self.set_workflow(steps, Path(path).stem)
        return True

    def choose_save(self):
        start = str(Path(self.workflow_folder) / f"{self.workflow_name or 'workflow'}.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the workflow", start, WORKFLOW_FILES)
        if path:
            self.save_workflow(path)

    def save_workflow(self, path: str) -> bool:
        try:
            save_workflow(path, self.steps)
        except OSError as error:
            QtWidgets.QMessageBox.warning(self, "Contour extraction", f"The workflow could not be saved:\n{error}")
            return False
        self.workflow_folder, self.workflow_name, self.modified = str(Path(path).parent), Path(path).stem, False
        self.update_names()
        return True

    # --- The image, and running the workflow on it --------------------------------------------------------------------

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open an SEM image", self.image_folder, IMAGE_FILES)
        if path:
            self.open_image(path)

    def open_image(self, path: str):
        """Show the image, whole, and run the workflow on it (Auto process; else it waits for Process)."""
        try:
            loaded = load_sem_image(path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Contour extraction", f"The image could not be read:\n{error}")
            return
        self.loaded, self.result, self.image_folder = loaded, None, str(Path(path).parent)
        self.image_name = Path(path).name
        self.name_label.setText(f"{self.image_name}   ({loaded.shape[1]} × {loaded.shape[0]} px)")
        self.update_names()
        self.summary.setText("")
        self.update_list()
        self.show_result()
        self.plot.setRange(xRange=(-0.5, loaded.shape[1] - 0.5), yRange=(-0.5, loaded.shape[0] - 0.5), padding=0.02)
        if self.auto.isChecked():
            self.process()
        self.update_waiting()

    def process(self):
        """Run the workflow as it is now on the image, in the background. While a run is on its way, the newest
        steps wait for it (_processed starts them)."""
        self.timer.stop()
        if self.loaded is None:
            return
        self.request += 1  # the result of any earlier run is out of date now
        if not self.running:
            self._start()
        self.update_waiting()

    def _start(self):
        self.running = True
        number, version, loaded, steps, earlier = self.request, self.version, self.loaded, copy.deepcopy(self.steps), \
            self.result

        def work():
            try:
                run = run_workflow(loaded, steps, earlier)  # a step that fails is in the run
            except Exception as error:  # anything else: shown as a run that failed at once, not a window that waits
                run = Run(loaded, steps, [State(loaded)], [], error)
            try:
                self.processed.emit(number, version, run)
            except RuntimeError:  # the window was closed meanwhile
                pass
        threading.Thread(target=work, daemon=True).start()

    def _processed(self, number: int, version: int, run: Run):
        self.running = False
        if number != self.request:  # the image or the steps changed meanwhile: run the newest
            if run.image is self.loaded:
                self.result, self.result_version = run, version  # its first steps may serve the next run
            self._start()
            return
        self.result, self.result_version = run, version
        self.update_list()
        self.show_result()
        self.update_waiting()

    # --- The plot -----------------------------------------------------------------------------------------------------

    def viewed(self) -> int:
        """Which state of the run the plot shows: after the last step that ran, or after the step selected (as far
        as the run got)."""
        last = len(self.result.states) - 1
        if self.view_buttons["selected step"].isChecked() and self.selected() is not None:
            return min(self.selected() + 1, last)
        return last

    def show_result(self):
        """Show the image and what the workflow made of it, at the step shown (no run yet: the image as loaded)."""
        self.show_image()
        run = self.result
        self.legend_names = {}
        if run is None:
            self.items["Mask"].clear()
            for name in ("Contours", "Contours before", "Centres"):
                self.items[name].setData([], [])
        else:
            viewed = self.viewed()
            state = run.states[viewed]
            if state.mask is None:
                self.items["Mask"].clear()
            else:
                self.items["Mask"].setImage(state.mask, levels=(0, 1))
                place(self.items["Mask"], np.array(state.mask.shape) / state.scale)
            after = made_by(run, viewed)
            before = None if after is None else made_by(run, after - 1)
            for name, number in (("Contours", after), ("Contours before", before)):
                contours = None if number is None else image_contours(run.states[number])
                xy = np.empty((0, 2)) if contours is None else outlines(contours.points, contours.closed)
                self.items[name].setData(xy[:, 0], xy[:, 1], connect="finite")
                if contours is not None:  # named by the step that made them
                    self.legend_names[name] = f"{step_title(run.steps, number - 1)}" + (" (before)" if "before" in name else "")
            contours = image_contours(state)
            centres = [polygon_centroid(points) for points, closed in zip(contours.points, contours.closed) if closed] \
                if contours is not None else []
            centres = np.array(centres, float).reshape(-1, 2)
            self.items["Centres"].setData(centres[:, 0], centres[:, 1])
            self.summary.setText(summary(run, viewed))
        self.update_layers()

    def show_image(self):
        """Show the image as the workflow made it at the step shown, or the original, as loaded (Image), under the
        layers."""
        if self.loaded is None:
            return
        if self.result is None or self.image_buttons["original"].isChecked():
            self.frame, self.displayed = self.loaded.shape, self.loaded
        else:
            state = self.result.states[self.viewed()]
            self.frame, self.displayed = tuple(np.array(state.image.shape) / state.scale), state.image
        self.image_item.setImage(self.displayed, levels=(0, 255))
        place(self.image_item, self.frame)

    def update_layers(self, *_):
        """Show the layers ticked that there are; the legend names the contours shown by the step that made them."""
        state = None if self.result is None else self.result.states[self.viewed()]
        there = {"Mask": state is not None and state.mask is not None,
                 "Contours": "Contours" in self.legend_names,
                 "Contours before": "Contours before" in self.legend_names,
                 "Centres": state is not None and state.contours is not None and bool(state.contours.closed.any())}
        for name, item in self.items.items():
            self.boxes[name].setEnabled(there[name])
            item.setVisible(self.boxes[name].isChecked() and there[name])
        self.legend.clear()
        for name, text in self.legend_names.items():
            if self.items[name].isVisible():
                self.legend.addItem(self.items[name], text)
        self.legend.setVisible(bool(self.legend.items))  # no empty box

    def _show_cursor(self, scene_pos):
        """The cursor's position and the grey level of the pixel under it in the image shown (upscaled: of its own,
        smaller pixel)."""
        if self.displayed is None:
            return
        pos = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        f = self.displayed.shape[1] / self.frame[1]
        column, row = int(np.floor((pos.x() + 0.5) * f)), int(np.floor((pos.y() + 0.5) * f))
        rows, cols = self.displayed.shape
        grey = f", grey level {self.displayed[row, column]}" if 0 <= row < rows and 0 <= column < cols else ""
        self.statusBar().showMessage(f"x = {pos.x():.1f} px, y = {pos.y():.1f} px{grey}")

    def open_file(self, path: str):
        """Open a file dropped or named: a workflow if it is a .json file, an image otherwise."""
        if Path(path).suffix.lower() == ".json":
            if self.may_replace():
                self.open_workflow(path)
        else:
            self.open_image(path)

    def dragEnterEvent(self, event):
        if dropped_file(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = dropped_file(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.open_file(path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    window = Viewer()
    window.show()
    for path in argv[1:]:
        window.open_file(path)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
