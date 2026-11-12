"""A window to try extract_contours on real SEM images: open an image, change the options, see the contacts.

    .venv\\Scripts\\python viewer.py [image]

It shows the image (as loaded, or as prepared: the image the contacts are found in), the threshold mask, and the
detected and refined contours and centres over it. Every change of an option finds the contacts again, in a background
thread, for the newest options only; one line says how many were found and why the other regions were left out.
Open an image with Open image... (Ctrl+O), by dropping its file on the window, or by naming it on the command line.

Not part of the contour_extraction package, which needs no Qt: a caller of it like any other. The window follows
AffineRANSAC's contour diagnosis (release/contour_dialog.py).
Coordinates are in the pixel frame: (x, y) = (column, row), y pointing down, pixel centres at integers.
"""

import html
import inspect
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from contour_extraction import (DENOISERS, METHODS, REFINEMENTS, DetectedContacts, contact_mask, crop_databar,
                                extract_contours, load_sem_image, prepared_image, upscaled_image)

# The options: the data bar to crop (load step), then every option of extract_contours with its default, by step.
DEFAULTS = {"databar_rows": 0} | {name: p.default for name, p in inspect.signature(extract_contours).parameters.items()
                                  if p.kind is p.KEYWORD_ONLY}
GROUPS = (("Load", ("databar_rows",)),
          ("Prepare", ("background_px", "denoise", "median_px", "nlm_h", "tv_weight", "upscale")),
          ("Detect", ("detection", "dark_contacts", "detection_blur_px", "min_area_px", "max_area_px", "min_solidity")),
          ("Check", ("min_contrast",)),
          ("Refine", ("refinement", "search_px", "edge_step_px", "edge_blur_px", "normal_span")))
CHOICES = {"denoise": DENOISERS, "detection": METHODS, "refinement": REFINEMENTS}
LIMITS = {"databar_rows": (0, 10000, 1), "background_px": (0, 2000, 5), "median_px": (3, 99, 2), "nlm_h": (0, 100, 1),
          "tv_weight": (0, 10, 0.01), "upscale": (1, 8, 1), "detection_blur_px": (0, 50, 0.1),
          "min_area_px": (0, 10 ** 8, 5), "max_area_px": (0, 10 ** 8, 100), "min_solidity": (0, 1, 0.01),
          "min_contrast": (0, 100, 0.5), "search_px": (0.5, 100, 0.5), "edge_step_px": (0.01, 10, 0.05),
          "edge_blur_px": (0, 50, 0.1), "normal_span": (1, 50, 1)}  # (minimum, maximum, step) of each number
DENOISER_OPTIONS = {"median_px": "median", "nlm_h": "non-local means", "tv_weight": "total variation"}  # each's own
REFINE_OPTIONS = ("search_px", "edge_step_px", "edge_blur_px", "normal_span")  # unused with refinement none
HELP = {  # what each option does (its tooltip); as AffineRANSAC's analysis.Settings says it
    "databar_rows": "Rows of the SEM data bar along the bottom of the image, cropped off before anything else; 0 = "
                    "none. Until the contacts are found the whole image shows: the cursor's y says where the bar starts",
    "background_px": "Flatten the background first, so that one threshold fits the whole image when its brightness "
                     "varies (charging, shading): the brightness around the contacts (filled in over a disk this many "
                     "pixels across, then smoothed) is subtracted. Wider than a contact, narrower than the brightness "
                     "changes; 0 = off",
    "denoise": "Remove pixel noise before the contacts are found (after the flattening, before the upscaling). median: "
               "each pixel the median of a window median_px wide: removes isolated bright or dark pixels and small "
               "spots, keeps edges. non-local means: each pixel a mean of the pixels whose surroundings look alike "
               "(nlm_h); can wipe out contacts only a few pixels across. total variation: flat regions with sharp "
               "edges between them (tv_weight). none: off",
    "median_px": "Denoise median: window width in pixels (odd); removes spots narrower than half of it",
    "nlm_h": "Denoise non-local means: strength in grey levels (OpenCV h); larger removes more noise, and more detail",
    "tv_weight": "Denoise total variation: weight (on grey levels / 255); larger gives flatter regions and removes "
                 "more detail",
    "upscale": "Interpolate the image (bicubic) to this many times as many pixels per side, then find the contacts in "
               "it: finer threshold outlines and centres for contacts only a few pixels across. The other options in "
               "pixels stay pixels of the image as loaded. Time and memory grow with its square; 1 = off",
    "detection": "How contacts are found (each a region with a first, pixel outline; refinement then places its "
                 "edge). otsu: the dark regions, for images without a bright band. otsu3: the regions darker than the "
                 "background; finds contacts whose band noise breaks up, but more false regions. band: the regions a "
                 "bright band encloses, however grey the hole",
    "dark_contacts": "Contacts darker than the background (holes, with a bright band); off: brighter. Detection and "
                     "edge refinement",
    "detection_blur_px": "Detection: Gaussian blur (sigma, px) of the image before thresholding; 0 = none",
    "min_area_px": "Detection: regions of fewer pixels (px²) are left out (noise)",
    "max_area_px": "Detection: regions of more pixels (px²) are left out; 0 = no limit",
    "min_solidity": "Detection, band only: regions less solid than this (area / convex hull area) are left out. "
                    "Contacts are ~1, background pinched off between nearly touching bands 0.7-0.88; 0 = off",
    "min_contrast": "Drop detected contacts whose mean grey level differs from a 3 px ring around them by less than "
                    "this many noise sigma (false contacts in images without pattern); 0 = off. Noisy images give "
                    "real contacts only a few sigma",
    "refinement": "Edge of each contact. mid-level: where the smoothed image is half-way between the contact's "
                  "interior and its bright band. steepest gradient: where it rises fastest (noisier). threshold: where "
                  "it crosses the detection's threshold, one level for the whole image: for images without a bright "
                  "band (flatten the background if the brightness varies). none: no refinement, the detected contours",
    "search_px": "Edge refinement searches this many pixels in/out along the contour normal",
    "edge_step_px": "Edge refinement: the image is sampled along each normal every this many pixels (<= search_px)",
    "edge_blur_px": "Edge refinement: Gaussian blur (sigma, px) of the image the edges are found on; 0 = none",
    "normal_span": "Edge refinement: each contour point's normal runs across the contour points this many before and "
                   "after it (smooths the pixel staircase)",
}

LAYERS = ("Prepared image", "Threshold mask", "Detected contours", "Detected centres", "Refined contours",
          "Refined centres")
DETECTED, REFINED, THRESHOLD = "#c2189b", "#00a8c8", "#e8b500"  # the colours of AffineRANSAC's contour diagnosis
FAILURE, MUTED = "#e34948", "#77766f"
IMAGE_FILES = "SEM images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)"
NO_IMAGE = ("Open an SEM image (<i>Open image...</i>, Ctrl+O, or drop its file on the window) to see the contacts "
            "extract_contours finds in it. Hover an option to see what it does.")


@dataclass
class Found:
    """The contacts of one image with one set of options, and the images to show them on."""
    options: dict               # the options they were found with
    image: np.ndarray           # the image as loaded, its data bar cropped: the pixel frame of everything below
    prepared: np.ndarray        # the image the contacts are found in (prepared_image), upscaled: more pixels, same area
    mask: np.ndarray            # uint8, the pixels of `prepared`: 1 = taken as contact by the threshold
    detected: DetectedContacts  # after the contrast check
    refined: DetectedContacts   # refinement none: the detected ones
    dropped: int                # detected contacts the contrast check dropped
    seconds: float              # time extract_contours took (with the preparation)


def find(loaded: np.ndarray, options: dict) -> Found:
    """The contacts extract_contours finds in the loaded image with these options (databar_rows: cropped off first),
    and the prepared image and threshold mask to show them on."""
    options = dict(options)
    image = crop_databar(loaded, options.pop("databar_rows"))
    start = time.perf_counter()
    # Prepared here, to show it; extract_contours, given it with background_px 0 and denoise none, uses it as it is.
    flat = prepared_image(image, options["background_px"], options["dark_contacts"], options["denoise"],
                          options["median_px"], options["nlm_h"], options["tv_weight"])
    detected, refined, dropped = extract_contours(flat, **(options | {"background_px": 0.0, "denoise": "none"}))
    seconds = time.perf_counter() - start
    f = options["upscale"]
    prepared = upscaled_image(flat, f)  # as extract_contours upscales it, its blur in pixels of the image as given
    mask, _ = contact_mask(prepared, options["dark_contacts"], options["detection_blur_px"] * f, options["detection"])
    return Found(options, image, prepared, mask, detected, refined, dropped, seconds)


def summary(found: Found) -> str:
    """How many contacts were found, how large, and why the other regions were left out, in words (HTML)."""
    o = found.options
    reasons = {"border": "touching the image border", "small": f"smaller than {o['min_area_px']} px²",
               "large": f"larger than {o['max_area_px']} px²", "not solid": f"less solid than {o['min_solidity']:g}"}
    left_out = [f"{count} {reasons.get(why, why)}" for why, count in found.detected.left_out.items() if count]
    if found.dropped:
        left_out.append(f"{found.dropped} below the contrast of {o['min_contrast']:g} noise σ")
    areas = found.refined.areas
    text = f"<b>{len(areas)} contacts</b>"
    if len(areas):
        median = float(np.median(areas))
        text += f", median area {median:.1f} px² (diameter {2 * np.sqrt(median / np.pi):.2f} px)"
    text += (f"; {o['detection']} threshold at grey level {found.detected.threshold:.0f}{of_the_image(o)}; "
             f"{found.seconds:.2f} s. ")
    return text + (f"Left out: {', '.join(left_out)}." if left_out else "")


def of_the_image(options: dict) -> str:
    """What was done to the image before the threshold, in words ("" if nothing)."""
    steps = [step for step, done in (("flattened", options["background_px"] > 0),
                                     (f"denoised ({options['denoise']})", options["denoise"] != "none"),
                                     (f"upscaled {options['upscale']}×", options["upscale"] > 1)) if done]
    if len(steps) > 1:
        steps = [", ".join(steps[:-1]) + " and " + steps[-1]]
    return f" of the image {steps[0]}" if steps else ""


def outlines(contours: list[np.ndarray]) -> np.ndarray:
    """The contours (each (M, 2)) as one (rows, 2) line to draw with connect="finite": each closed (its first point
    again) and followed by a NaN row, so that one plot item draws them all."""
    nan = np.full((1, 2), np.nan)
    parts = [part for contour in contours for part in (contour, contour[:1], nan)]
    return np.concatenate(parts) if parts else np.empty((0, 2))


def applies(name: str, options: dict) -> bool:
    """Whether an option changes anything, given the others: min_solidity for band only, each denoiser's option for
    that denoiser only, the refinement's options unless refinement is none."""
    if name == "min_solidity":
        return options["detection"] == "band"
    if name in DENOISER_OPTIONS:
        return options["denoise"] == DENOISER_OPTIONS[name]
    return name not in REFINE_OPTIONS or options["refinement"] != "none"


def editor(name: str, value) -> QtWidgets.QWidget:
    """A widget to edit an option: a list of its choices, a check box (bool) or a spin box (int or float)."""
    if name in CHOICES:
        widget = QtWidgets.QComboBox()
        widget.addItems(CHOICES[name])
    elif isinstance(value, bool):  # before int: a bool is an int
        widget = QtWidgets.QCheckBox()
    else:
        widget = QtWidgets.QSpinBox() if isinstance(value, int) else QtWidgets.QDoubleSpinBox(decimals=3)
        low, high, step = LIMITS[name]
        widget.setRange(low, high)
        widget.setSingleStep(step)
    set_value(widget, value)
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


class Viewer(QtWidgets.QMainWindow):
    searched = QtCore.Signal(int, object)  # (search number, Found or the exception), from the background thread

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contour extraction")
        self.setAcceptDrops(True)
        self.loaded = None     # the image as loaded (uint8), data bar included
        self.result = None     # Found: what is shown (None: the loaded image alone)
        self.displayed = None  # the image shown, and the shape of the image whose pixel frame it covers
        self.frame = None
        self.folder = ""       # where Open image starts
        self.request = 0       # number of the newest search asked for
        self.running = False   # a search runs in the background thread
        self.searched.connect(self._searched)
        self.timer = QtCore.QTimer(self, singleShot=True, interval=150)  # a search once the typing pauses
        self.timer.timeout.connect(self.search)

        # The plot: the image, the threshold mask over it, the contours and centres on top.
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
        red, green, blue, _ = pg.mkColor(THRESHOLD).getRgb()
        mask.setLookupTable(np.array([[0, 0, 0, 0], [red, green, blue, 110]], np.uint8))
        mask.setZValue(-5)
        self.items = {"Threshold mask": mask}
        for name in LAYERS[2:]:
            color = DETECTED if name.startswith("Detected") else REFINED
            if name.endswith("contours"):
                self.items[name] = pg.PlotDataItem(pen=pg.mkPen(color, width=1.5), connect="finite")
            else:
                self.items[name] = pg.PlotDataItem(pen=None, symbol="+", symbolSize=9,
                                                   symbolPen=pg.mkPen(color, width=1.5))
        for item in self.items.values():
            self.plot.addItem(item)
        self.plot.scene().sigMouseMoved.connect(self._show_cursor)

        self.boxes = {}
        show = QtWidgets.QHBoxLayout()
        show.addWidget(QtWidgets.QLabel("Show:"))
        for name in LAYERS:
            box = QtWidgets.QCheckBox(name)
            box.setChecked(name != "Threshold mask")  # the mask covers the image: off at start
            box.toggled.connect(self.show_image if name == "Prepared image" else self.update_layers)
            show.addWidget(box)
            self.boxes[name] = box
        self.boxes["Prepared image"].setToolTip(tip("The image the contacts are found in: flattened, denoised and "
                                                    "upscaled as the options say. Off: the image as loaded"))
        self.boxes["Threshold mask"].setToolTip(tip("The pixels the threshold takes as contact (for band: everything "
                                                    "but the bright band), before the regions are checked"))
        show.addStretch(1)

        # The options, by step; what each does in its tooltip.
        self.editors = {}
        panel = QtWidgets.QVBoxLayout()
        for title, names in GROUPS:
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            for name in names:
                widget = editor(name, DEFAULTS[name])
                label = QtWidgets.QLabel(name)
                for item in (label, widget):
                    item.setToolTip(tip(HELP[name]))
                form.addRow(label, widget)
                changed_signal(widget).connect(self.changed)
                self.editors[name] = widget
            panel.addWidget(group)
        defaults = QtWidgets.QPushButton("Defaults")
        defaults.setToolTip("Set every option back to extract_contours' default (and databar_rows to 0)")
        defaults.clicked.connect(self.reset)
        panel.addWidget(defaults)
        panel.addStretch(1)
        side = QtWidgets.QWidget()
        side.setLayout(panel)
        scroll = QtWidgets.QScrollArea(widgetResizable=True)
        scroll.setWidget(side)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # as wide as the options
        scroll.setFixedWidth(side.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)

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
        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(self.plot, stretch=1)
        middle.addWidget(scroll)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        for row in (top, status, show):
            layout.addLayout(row)
        layout.addLayout(middle, stretch=1)
        self.setCentralWidget(central)
        self.statusBar()  # the cursor's position and grey level
        self.changed()
        self.resize(1400, 900)

    def options(self) -> dict:
        """The options shown."""
        return {name: value(widget) for name, widget in self.editors.items()}

    def changed(self, *_):
        """An option changed: grey out the options that change nothing now, then search again once the typing
        pauses."""
        o = self.options()
        for name, widget in self.editors.items():
            widget.setEnabled(applies(name, o))
        self.update_layers()
        self.timer.start()

    def reset(self):
        """Every option back to its default; one search."""
        for name, widget in self.editors.items():
            widget.blockSignals(True)
            set_value(widget, DEFAULTS[name])
            widget.blockSignals(False)
        self.changed()

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open an SEM image", self.folder, IMAGE_FILES)
        if path:
            self.open_image(path)

    def open_image(self, path: str):
        """Show the image, whole, and find its contacts."""
        try:
            loaded = load_sem_image(path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Contour extraction", f"The image could not be read:\n{error}")
            return
        self.loaded, self.result, self.folder = loaded, None, str(Path(path).parent)
        self.name_label.setText(f"{Path(path).name}   ({loaded.shape[1]} × {loaded.shape[0]} px)")
        self.setWindowTitle(f"{Path(path).name} - Contour extraction")
        self.summary.setText("")
        self.show_result()
        self.plot.setRange(xRange=(-0.5, loaded.shape[1] - 0.5), yRange=(-0.5, loaded.shape[0] - 0.5), padding=0.02)
        self.search()

    def search(self):
        """Find the contacts with the options shown, in the background. While a search runs, the newest options wait
        for it (_searched starts them)."""
        self.timer.stop()
        if self.loaded is None:
            return
        self.request += 1  # the result of any earlier search is out of date now
        if not self.running:
            self._start()

    def _start(self):
        self.running = True
        self.busy.setText("Finding the contacts...")
        number, loaded, options = self.request, self.loaded, self.options()

        def work():
            try:
                found = find(loaded, options)
            except Exception as error:  # shown in the window
                found = error
            try:
                self.searched.emit(number, found)
            except RuntimeError:  # the window was closed meanwhile
                pass
        threading.Thread(target=work, daemon=True).start()

    def _searched(self, number: int, found):
        self.running = False
        if number != self.request:  # the image or the options changed meanwhile: search with the newest
            self._start()
            return
        self.busy.setText("")
        self.result = None if isinstance(found, Exception) else found
        self.show_result()
        if isinstance(found, Exception):
            self.summary.setText(f"<span style='color:{FAILURE}'>No contacts found: {type(found).__name__}: "
                                 f"{html.escape(str(found))}</span>")

    def show_result(self):
        """Show the image and the contacts found in it (none yet: the image as loaded alone)."""
        self.show_image()
        found = self.result
        if found is None:
            self.items["Threshold mask"].clear()
            for name in LAYERS[2:]:
                self.items[name].setData([], [])
        else:
            self.items["Threshold mask"].setImage(found.mask, levels=(0, 1))
            place(self.items["Threshold mask"], found.image.shape)
            for name, contacts in (("Detected", found.detected), ("Refined", found.refined)):
                xy = outlines(contacts.contours)
                centres = np.asarray(contacts.centers, float).reshape(-1, 2)
                self.items[f"{name} contours"].setData(xy[:, 0], xy[:, 1], connect="finite")
                self.items[f"{name} centres"].setData(centres[:, 0], centres[:, 1])
            self.summary.setText(summary(found))
        self.update_layers()

    def show_image(self, *_):
        """Show the image as prepared (Prepared image ticked) or as loaded; data bar cropped once the contacts are
        found."""
        if self.loaded is None:
            return
        found = self.result
        if found is None:
            self.frame, self.displayed = self.loaded.shape, self.loaded
        else:
            prepared = self.boxes["Prepared image"].isChecked()
            self.frame, self.displayed = found.image.shape, found.prepared if prepared else found.image
        self.image_item.setImage(self.displayed, levels=(0, 255))
        place(self.image_item, self.frame)

    def update_layers(self, *_):
        """Show the layers ticked (and offered); the legend lists the contours and centres shown, each with the
        detection method or edge rule it comes from."""
        o = self.result.options if self.result else self.options()
        for name in ("Refined contours", "Refined centres"):  # refinement none: the refined ones are the detected
            self.boxes[name].setEnabled(self.options()["refinement"] != "none")
        for name, item in self.items.items():
            item.setVisible(self.boxes[name].isChecked() and self.boxes[name].isEnabled())
        self.legend.clear()
        for name in LAYERS[2:]:
            if self.items[name].isVisible():
                self.legend.addItem(self.items[name], f"{name} ({o['detection' if 'Detected' in name else 'refinement']})")

    def _show_cursor(self, scene_pos):
        """The cursor's position and the grey level of the pixel under it in the image shown (upscaled: of its own,
        smaller pixel)."""
        if self.displayed is None:
            return
        pos = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        f = self.displayed.shape[1] // self.frame[1]
        column, row = int(np.floor((pos.x() + 0.5) * f)), int(np.floor((pos.y() + 0.5) * f))
        rows, cols = self.displayed.shape
        grey = f", grey level {self.displayed[row, column]}" if 0 <= row < rows and 0 <= column < cols else ""
        self.statusBar().showMessage(f"x = {pos.x():.1f} px, y = {pos.y():.1f} px{grey}")

    def dragEnterEvent(self, event):
        if dropped_file(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = dropped_file(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.open_image(path)


def tip(text: str) -> str:
    """A tooltip that wraps its lines (Qt wraps rich text only)."""
    return f"<p>{html.escape(text)}</p>"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    window = Viewer()
    window.show()
    if len(argv) > 1:
        window.open_image(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
