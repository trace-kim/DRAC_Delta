"""The modules of a workflow: single operations on an SEM image, to be put in a list and run in turn (workflow.py).

Each module is a function `name(state, **options) -> State`: it takes what the modules before it made of the image (a
State) and returns what it makes of that. There are three kinds (STAGES), usually run in this order:
  Image     changes the grey-level image: crop, flatten, denoise, enhance, upscale.
  Mask      makes a mask of the image (a threshold: the pattern's regions; Canny: its edge pixels), or changes the mask.
  Contours  makes contours of the mask, and chooses, moves or smooths them.
@module registers a function in MODULES under its own name (the name workflow files use), with what a window needs to
offer it: a title, its stage, and for each option its default, its limits and what it does. The function's docstring
says what the module does and when to use it: the window shows it. The step files do the work; a module only fits one
of their functions to the State.

Options in pixels (`_px`) are pixels of the image as loaded: after Upscale a module multiplies them by State.scale
(areas by its square), so that putting Upscale into a workflow needs no other option changed, as extract_contours'.
A module never changes the arrays it is given: States share them.
"""

import inspect
from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage

from . import check, detect, load, prepare, refine
from .contacts import Contours, DetectedContacts

STAGES = ("Image", "Mask", "Contours")


class WorkflowError(ValueError):
    """A module cannot work on what the modules before it made; the message says what is missing."""


@dataclass(frozen=True)
class State:
    """What a workflow has made of an image so far."""
    image: np.ndarray                     # grey levels (uint8): what the pattern is found in
    scale: int = 1                        # pixels of `image` per pixel of the image as loaded, per side (Upscale)
    edge_image: np.ndarray | None = None  # the image the edges are measured on, if not `image` (Edge image)
    unscaled: np.ndarray | None = None    # the image the edges are measured on, as it was before Upscale
    mask: np.ndarray | None = None        # uint8, the pixels of `image`: 1 = region (a threshold) or edge (Canny)
    dark: bool | None = None              # the regions are the darker side; None: not known (a mask of edges)
    edge_blur: float = 2.0                # the blur (sigma, pixels of `image`) Canny edges found the mask's edges with
    connectivity: int = 8                 # pixels touching by a corner are one region (8) or not (4: Otsu band)
    threshold: float = np.nan             # the grey level of the measured image the mask was cut at; NaN: none
    contours: Contours | None = None      # in the pixels of `image`
    labels: np.ndarray | None = None      # the region number of every pixel (Trace regions); None: no regions
    note: str = ""                        # what the last module says of what it did

    @property
    def measured(self) -> np.ndarray:
        """The image the edges are measured on."""
        return self.image if self.edge_image is None else self.edge_image


def image_contours(state: State) -> Contours | None:
    """The state's contours in the pixels of the image as loaded (None: it has none yet)."""
    return None if state.contours is None else prepare.in_image_contours(state.contours, state.scale)


@dataclass(frozen=True)
class Option:
    """One option of a module, as a window offers it: a bool, a number (int or float, by its default) or a choice."""
    default: bool | int | float | str
    help: str                      # what it does
    low: float = 0.0               # a number: the least and the most it can be, and the step of its spin box
    high: float = 0.0
    step: float = 1.0
    choices: tuple[str, ...] = ()  # a choice: what it can be

    def checked(self, value, name: str):
        """The value as the option's type (a whole number for a float option becomes a float); a ValueError says
        why it cannot be this option's."""
        if isinstance(self.default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, not {value!r}")
            return value
        if isinstance(self.default, str):
            if value not in self.choices:
                raise ValueError(f"{name} must be one of {', '.join(self.choices)}, not {value!r}")
            return value
        whole = isinstance(self.default, int)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (whole and value != int(value)):
            raise ValueError(f"{name} must be a {'whole ' if whole else ''}number, not {value!r}")
        if not self.low <= value <= self.high:
            raise ValueError(f"{name} must be between {self.low:g} and {self.high:g}, not {value!r}")
        return type(self.default)(value)


def number(default: int | float, low: float, high: float, step: float, help: str) -> Option:
    return Option(default, help, low, high, step)


def choice(default: str, choices: tuple[str, ...], help: str) -> Option:
    return Option(default, help, choices=tuple(choices))


def flag(default: bool, help: str) -> Option:
    return Option(default, help)


@dataclass(frozen=True)
class Module:
    name: str                       # its name in MODULES and in workflow files: the function's
    title: str                      # its name in a window
    stage: str                      # one of STAGES
    help: str                       # what it does and when to use it: the function's docstring
    options: dict[str, Option]      # by name, in the function's order
    function: Callable[..., State]  # function(state, **options) -> State


MODULES: dict[str, Module] = {}  # by name, in the order a window lists them


def module(title: str, stage: str, **options: Option):
    """Register the function it decorates as a module (module docstring): its options are the function's arguments
    after the state."""
    def register(function):
        assert stage in STAGES and list(inspect.signature(function).parameters)[1:] == list(options), function
        MODULES[function.__name__] = Module(function.__name__, title, stage, " ".join(function.__doc__.split()),
                                            options, function)
        return function
    return register


def need_mask(state: State, title: str) -> np.ndarray:
    if state.mask is None:
        raise WorkflowError(f"{title} needs a mask: put a threshold or Canny edges before it")
    return state.mask


def need_contours(state: State, title: str) -> Contours:
    if state.contours is None:
        raise WorkflowError(f"{title} needs contours: put Trace regions or Trace edges before it")
    return state.contours


def before_the_mask(state: State, title: str):
    """Refuse a module that changes the size of the image once a mask of the image exists."""
    if state.mask is not None:
        raise WorkflowError(f"{title} changes the size of the image: put it before the module that makes the mask")


def resized(state: State, change: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    """A change of size (a crop, an upscaling) done to the image and to the image the edges are measured on: (image,
    edge_image), one array for both where they were one."""
    image = change(state.image)
    kept = state.edge_image
    return image, None if kept is None else image if kept is state.image else change(kept)


# --- Image modules ----------------------------------------------------------------------------------------------------

@module("Crop data bar", "Image",
        rows=number(0, 0, 10000, 1, "Rows of the data bar along the bottom of the image as loaded; the cursor's y in "
                                    "the window says where the bar starts. 0: nothing is cut off"))
def crop_databar(state: State, rows: int) -> State:
    """Cut the SEM data bar (magnification, scale bar, text) off the bottom of the image, before anything else: its
    black and white would be taken for pattern, and would decide the threshold. The pixels left keep their
    coordinates."""
    before_the_mask(state, "Crop data bar")
    f = state.scale
    if rows * f >= len(state.image):
        raise WorkflowError(f"Crop data bar: the image has only {len(state.image) // f} rows")
    image, kept = resized(state, lambda image: load.crop_databar(image, rows * f))
    unscaled = None if state.unscaled is None else load.crop_databar(state.unscaled, rows)
    return replace(state, image=image, edge_image=kept, unscaled=unscaled, note=f"{len(image) // f} rows left")


@module("Flatten background", "Image",
        size_px=number(60.0, 3, 2000, 5, "Width of the disk the brightness around the pattern is filled in over, in "
                                         "pixels: wider than a feature of the pattern, narrower than the brightness "
                                         "changes"),
        dark=flag(True, "The pattern is darker than what is around it (holes); off: brighter"))
def flatten_background(state: State, size_px: float, dark: bool) -> State:
    """Remove brightness that varies slowly across the image (charging, shading), so that one threshold fits the
    whole image: the brightness around the pattern, filled in over the pattern and smoothed, is subtracted."""
    return replace(state, image=prepare.flatten_background(state.image, size_px * state.scale, dark), note="")


@module("Gaussian blur", "Image",
        sigma_px=number(1.0, 0, 50, 0.1, "Sigma of the Gaussian, in pixels; 0: none"))
def gaussian_blur(state: State, sigma_px: float) -> State:
    """Average each pixel with its neighbours, weighted by a Gaussian: the plain way to remove noise. It widens
    every edge as it removes noise, on both sides alike, so an edge taken half-way up its slope stays where it is;
    features narrower than about twice sigma fade."""
    sigma = sigma_px * state.scale
    return replace(state, image=cv2.GaussianBlur(state.image, (0, 0), sigma) if sigma > 0 else state.image, note="")


@module("Median filter", "Image",
        size_px=number(3, 3, 99, 2, "Width of the window, in pixels (odd; an even width is taken one wider)"))
def median_filter(state: State, size_px: int) -> State:
    """Replace each pixel by the median of the window around it: removes isolated bright or dark pixels and spots
    narrower than half the window, and keeps straight edges sharp and where they are (it rounds corners). Use it
    before Upscale: interpolation spreads a noisy pixel over several."""
    return replace(state, image=cv2.medianBlur(state.image, int(round(size_px * state.scale)) | 1), note="")


@module("Bilateral filter", "Image",
        sigma_px=number(2.0, 0.5, 20, 0.5, "How far the neighbours averaged reach: sigma in pixels"),
        sigma_grey=number(25.0, 1, 255, 1, "How alike in grey level they must be: sigma in grey levels. Less than the "
                                           "step of an edge keeps the edge; a few times the noise removes the noise"))
def bilateral_filter(state: State, sigma_px: float, sigma_grey: float) -> State:
    """Average each pixel with the neighbours of about its own grey level only: smooths the flat areas and leaves
    the edges sharp. It also leaves alone an isolated very bright or dark pixel (no neighbour is like it): put a
    Median filter before it for those."""
    return replace(state, image=cv2.bilateralFilter(state.image, -1, sigma_grey, sigma_px * state.scale), note="")


@module("Non-local means", "Image",
        h=number(10.0, 0, 100, 1, "Strength, in grey levels (OpenCV's h): about the sigma of the noise. Larger removes "
                                  "more noise, and more detail"))
def non_local_means(state: State, h: float) -> State:
    """Replace each pixel by a mean of the pixels whose surroundings (a 7 px patch) look like its own, within 21 px:
    removes much noise from a repeating pattern and keeps its edges. It can wipe out features only a few pixels
    across, and it is made for noise that differs from pixel to pixel: use it before Upscale. Slow (about a second
    for 2048 x 2048 px)."""
    return replace(state, image=prepare.denoised_image(state.image, "non-local means", nlm_h=h), note="")


@module("Total variation", "Image",
        weight=number(0.1, 0, 10, 0.01, "Weight, on grey levels / 255: larger gives flatter areas and removes more "
                                        "detail"))
def total_variation(state: State, weight: float) -> State:
    """The image nearest to this one that has the least total variation (Chambolle): flat areas with sharp edges
    between them, the noise gone. Faint texture goes with the noise, and slopes turn into steps. The slowest denoiser
    (some seconds for 2048 x 2048 px)."""
    return replace(state, image=prepare.denoised_image(state.image, "total variation", tv_weight=weight), note="")


@module("Equalize histogram", "Image")
def equalize_histogram(state: State) -> State:
    """Spread the grey levels so that each is used by about as many pixels (histogram equalization over the whole
    image): shows a faint pattern, and makes images of different brightness and contrast alike before a Fixed
    threshold or Canny edges. Otsu's threshold after it divides the pixels into about equal shares, which fits only a
    pattern that covers about half the image. It amplifies the noise of flat areas, and the grey levels no longer
    follow the signal: put Edge image before it, so that the edges are measured on the image as it was."""
    return replace(state, image=cv2.equalizeHist(state.image), note="")


@module("CLAHE", "Image",
        clip_limit=number(2.0, 1, 40, 0.5, "How much a grey level may count in a tile's histogram, in times the "
                                           "average: larger gives more contrast, and more noise in flat tiles"),
        tiles=number(8, 1, 64, 1, "Tiles per side of the image"))
def clahe(state: State, clip_limit: float, tiles: int) -> State:
    """Contrast-limited adaptive histogram equalization: as Equalize histogram, but each tile of the image by its
    own histogram, blended between the tiles, so that a faint pattern stands out wherever it is in the image. As
    after Equalize histogram, Otsu's threshold then tends to divide the pixels into equal shares: look at the mask.
    The grey level of an edge differs from place to place: put Edge image before it, so that the edges are measured
    on the image as it was."""
    image = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles)).apply(state.image)
    return replace(state, image=image, note="")


@module("Stretch contrast", "Image",
        low_percent=number(0.5, 0, 49, 0.1, "This share of the pixels, the darkest, becomes black (0)"),
        high_percent=number(99.5, 51, 100, 0.1, "The pixels above this share, the brightest, become white (255)"))
def stretch_contrast(state: State, low_percent: float, high_percent: float) -> State:
    """Stretch the grey levels in a straight line, so that the darkest pixels become 0 and the brightest 255: uses
    the whole range, and makes images of different brightness and contrast alike before a Fixed threshold or Canny
    edges. A straight line keeps every slope in proportion: the edges are where they were."""
    low, high = np.percentile(state.image, (low_percent, high_percent))
    if high <= low:
        return replace(state, note="one grey level: nothing to stretch")
    stretched = (state.image.astype(np.float32) - low) * (255 / (high - low))
    return replace(state, image=np.clip(np.round(stretched), 0, 255).astype(np.uint8),
                   note=f"grey levels {low:.0f} to {high:.0f}")


@module("Invert", "Image")
def invert(state: State) -> State:
    """The negative of the image: dark becomes bright. (The threshold modules find dark or bright regions without
    it: their option dark.)"""
    if state.edge_image is not None:
        raise WorkflowError("Invert after Edge image would find the pattern in the negative of the image its edges "
                            "are measured on: put Invert before Edge image")
    return replace(state, image=255 - state.image, dark=None if state.dark is None else not state.dark,
                   threshold=255 - state.threshold, note="")


@module("Upscale", "Image",
        factor=number(2, 1, 8, 1, "Times as many pixels per side; 1: none"))
def upscale(state: State, factor: int) -> State:
    """Interpolate the image (bicubic) to more pixels, so that the mask's outline is finer: for features only a few
    pixels across. Denoise before it, not after: interpolation spreads each noisy pixel over several. The options in
    pixels of the modules after it stay pixels of the image as loaded, and the contours are given in those. Time and
    memory grow with the square of the factor."""
    before_the_mask(state, "Upscale")
    image, kept = resized(state, lambda image: prepare.upscaled_image(image, factor))
    return replace(state, image=image, edge_image=kept, scale=state.scale * factor,
                   unscaled=state.measured if state.scale == 1 else state.unscaled,
                   note=f"{image.shape[1]} x {image.shape[0]} px")


@module("Edge image", "Image")
def edge_image(state: State) -> State:
    """Keep the image as it is here as the one the edges are measured on (Refine edges, the contrast of Filter
    contours). The image modules after this one then only help to find the pattern: strong denoising or equalization
    there cannot move the edges or bend their slopes. Without this module the edges are measured on the image as it
    is when they are refined."""
    return replace(state, edge_image=state.image, note="the edges are measured on this image")


# --- Mask modules -----------------------------------------------------------------------------------------------------

DARK = "The regions are the darker side (holes, trenches); off: the brighter side"


def with_mask(state: State, mask: np.ndarray, note: str, **changed) -> State:
    """The state with a new mask, and so without the contours of the mask before it; its note ends with the share of
    the image that is mask."""
    share = f"{100 * float(mask.mean()):.0f} % of the image"
    return replace(state, mask=mask, contours=None, labels=None, note=f"{note}, {share}" if note else share, **changed)


def own_level(state: State, level: float) -> float:
    """A threshold's grey level, if it is one of the image the edges are measured on (NaN: it is not: the image was
    changed after Edge image)."""
    return float(level) if state.edge_image is None or state.edge_image is state.image else np.nan


@module("Otsu threshold", "Mask",
        method=choice("otsu", detect.METHODS,
                      "otsu: two classes of grey level, the threshold between them. otsu3: three classes (the pattern | "
                      "the background | the bright band at its edge), the threshold between the first two: when wide "
                      "bands make plain Otsu divide background from band. band: the mask is everything but the bright "
                      "band, so the regions a band encloses are found however grey they are; pixels that touch only by "
                      "a corner are then separate regions, so that none leaks through a thin band"),
        dark=flag(True, DARK),
        blur_px=number(1.5, 0, 50, 0.1, "Gaussian blur (sigma, px) of the image for this threshold only: the mask comes "
                                        "out smooth, and the image stays as it is for the edges; 0: none"))
def otsu_threshold(state: State, method: str, dark: bool, blur_px: float) -> State:
    """Make the mask by Otsu's threshold: the grey level that best divides the image's pixels into classes, found
    for each image from its own histogram. For an even background: if the brightness varies across the image, put
    Flatten background before it."""
    mask, level = detect.contact_mask(state.image, dark, blur_px * state.scale, method)
    return with_mask(state, mask, f"grey level {level:.0f}", dark=dark, connectivity=4 if method == "band" else 8,
                     threshold=own_level(state, level))


@module("Fixed threshold", "Mask",
        level=number(128, 0, 255, 1, "The grey level: the darker side is at or below it"),
        dark=flag(True, DARK))
def fixed_threshold(state: State, level: int, dark: bool) -> State:
    """Make the mask at a grey level you choose (the status bar shows the grey level under the cursor). The same
    level for every image: for images of the same brightness and contrast (or put Stretch contrast before it)."""
    mask = (state.image <= level) if dark else (state.image > level)
    return with_mask(state, mask.astype(np.uint8), "", dark=dark, connectivity=8, threshold=own_level(state, level))


@module("Adaptive threshold", "Mask",
        block_px=number(51, 3, 2001, 2, "Width of the neighbourhood, in pixels (odd): wider than a feature of the "
                                        "pattern"),
        offset=number(5.0, -100, 100, 0.5, "How many grey levels darker than its neighbourhood (bright regions: "
                                           "brighter) a pixel must be: more than the noise"),
        dark=flag(True, DARK))
def adaptive_threshold(state: State, block_px: int, offset: float, dark: bool) -> State:
    """Make the mask by comparing each pixel with the mean of its neighbourhood (Gaussian weights): a threshold that
    follows brightness changing across the image. In flat areas wider than the neighbourhood it cuts regions out of
    the noise. It has no single grey level, so Refine edges cannot use its threshold rule."""
    block = max(3, int(round(block_px * state.scale)) | 1)
    mode = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY  # at or below mean - offset | above mean + offset
    mask = cv2.adaptiveThreshold(state.image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, block, offset if dark else -offset)
    return with_mask(state, mask, "", dark=dark, connectivity=8, threshold=np.nan)


@module("Canny edges", "Mask",
        blur_px=number(2.0, 0, 50, 0.1, "Gaussian blur (sigma, px) before the gradient is taken: larger finds less "
                                        "noise, and merges edges closer together than about twice it"),
        high=number(0.0, 0, 127, 0.5, "A line of edge pixels needs a pixel where the brightness changes by more than "
                                      f"this, in grey levels per pixel; 0: {detect.CANNY_TOP_SHARE:g} x the steepest "
                                      "change in the image"),
        low_ratio=number(0.5, 0.05, 1, 0.05, "The other pixels of the line need this share of `high`: lower follows "
                                             "an edge further into the noise"))
def canny_edges(state: State, blur_px: float, high: float, low_ratio: float) -> State:
    """Make a mask of the edges themselves: the pixels where the brightness changes fastest, as lines one pixel wide
    (Canny). No grey level has to divide the pattern from the background, so it also finds a pattern of the
    background's own grey that is known by its bright edges; a bright band gives two lines, its inner and its outer
    slope. Put Trace edges after it."""
    f = state.scale
    mask, used = detect.canny_edges(state.image, blur_px * f, high / f, low_ratio)
    return with_mask(state, mask, f"above {used * f:.1f} grey levels / px", dark=None, connectivity=8,
                     threshold=np.nan, edge_blur=blur_px * f)


MORPHOLOGY = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE, "dilate": cv2.MORPH_DILATE, "erode": cv2.MORPH_ERODE}


@module("Morphology", "Mask",
        operation=choice("open", tuple(MORPHOLOGY),
                         "open: remove what is narrower than the disk (specks, thin bridges between regions). close: "
                         "fill what is narrower (pinholes, gaps in a line of edge pixels). dilate: grow every region by "
                         "the disk's radius. erode: shrink it"),
        size_px=number(3, 1, 999, 2, "Width of the disk, in pixels (odd; an even width is taken one wider)"))
def morphology(state: State, operation: str, size_px: int) -> State:
    """Change the mask with a disk before its contours are traced: clean it (open, close), or grow and shrink its
    regions (dilate, erode). Open and close hardly move a smooth outline; dilate and erode move it by the disk's
    radius."""
    side = int(round(size_px * state.scale)) | 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    return with_mask(state, cv2.morphologyEx(need_mask(state, "Morphology"), MORPHOLOGY[operation], disk), "")


@module("Fill holes", "Mask")
def fill_holes(state: State) -> State:
    """Fill every hole in the mask (the pixels a region encloses on all sides): a region then has an outer contour
    only, and closed lines of edge pixels (Canny edges) become filled regions."""
    filled = ndimage.binary_fill_holes(need_mask(state, "Fill holes")).astype(np.uint8)
    return with_mask(state, filled, "")


# --- Contours modules -------------------------------------------------------------------------------------------------

LEFT_OUT = {"border": "at the image border", "small": "too small", "large": "too large", "short": "too short",
            "open": "open", "not solid": "not solid", "low contrast": "of low contrast"}  # the reasons, in words
NORMAL_SPAN = ("The direction each point's edge faces is taken across the contour points this many before and after "
               "it: smooths the pixel staircase")


def described(contours: Contours) -> str:
    """How many contours there are, closed and open, and how many were left out and why, in words."""
    closed = int(contours.closed.sum())
    text = f"{len(contours.closed)} contours ({closed} closed, {len(contours.closed) - closed} open)"
    left = [f"{count} {LEFT_OUT.get(why, why)}" for why, count in contours.left_out.items() if count]
    return text + (f"; left out: {', '.join(left)}" if left else "")


@module("Trace regions", "Contours",
        min_area_px=number(20, 0, 10 ** 8, 5, "Regions of fewer pixels (px²) are left out: noise"),
        max_area_px=number(0, 0, 10 ** 8, 100, "Regions of more pixels (px²) are left out; 0: no limit"),
        holes=flag(True, "The boundaries of the holes in a region too (a ring gives two contours); off: its outer "
                         "boundary only"),
        open_at_border=flag(True, "A region that runs out of the image gives open contours, ending at the image "
                                  "border (lines, features cut off); off: it is left out (whole features only)"),
        normal_span=number(3, 1, 50, 1, NORMAL_SPAN))
def trace_regions(state: State, min_area_px: int, max_area_px: int, holes: bool, open_at_border: bool,
                  normal_span: int) -> State:
    """Make contours of the mask's regions: the boundary of every connected region, pixel by pixel, each point with
    the direction its edge faces (out of the region), which Refine edges moves it along. A contour is closed where
    its region lies inside the image."""
    mask, f = need_mask(state, "Trace regions"), state.scale
    contours, labels = detect.mask_contours(mask, min_area_px * f ** 2, max_area_px * f ** 2 or None,
                                            state.connectivity, normal_span * f, holes, open_at_border, state.threshold)
    dark = state.dark
    if dark is None:  # a mask of edges, filled: its regions are the darker side if their pixels are the darker ones
        inside = mask > 0
        dark = bool(inside.any() and not inside.all()
                    and np.median(state.measured[inside]) <= np.median(state.measured[~inside]))
    return replace(state, contours=contours, labels=labels, dark=dark, note=described(contours))


@module("Trace edges", "Contours",
        min_length_px=number(10, 2, 100000, 1, "Lines of fewer pixels are left out: noise"),
        normal_span=number(3, 1, 50, 1, NORMAL_SPAN))
def trace_edges(state: State, min_length_px: int, normal_span: int) -> State:
    """Make contours of a mask of thin lines (Canny edges): each line followed pixel by pixel from one end to the
    other, closed if it closes on itself; a line that branches gives several. Each point is put where the gradient
    peaks between the pixels (of the image blurred as Canny edges blurred it) and faces the brighter side, so the
    contours need no Refine edges; it could move them by the steepest gradient only, as the lines bound no region."""
    f = state.scale
    contours = detect.edge_contours(need_mask(state, "Trace edges"), state.measured, min_length_px * f, normal_span * f,
                                    state.edge_blur)
    return replace(state, contours=contours, labels=None, dark=True, note=described(contours))


@module("Filter contours", "Contours",
        closed_only=flag(False, "Leave out the open contours (features cut by the image border, broken edges)"),
        min_solidity=number(0.0, 0, 1, 0.01, "Leave out the closed contours less solid than this (area / area of the "
                                             "convex hull): a convex one is 1, background pinched off between nearly "
                                             "touching bands 0.7-0.88; 0: off"),
        min_contrast=number(0.0, 0, 100, 0.5, "Leave out the closed contours whose mean grey level differs from a 3 px "
                                              "ring around them by less than this many noise sigma: regions a "
                                              "threshold cuts out of plain noise (an image without pattern); 0: off. "
                                              "Noisy images give real features only a few sigma"))
def filter_contours(state: State, closed_only: bool, min_solidity: float, min_contrast: float) -> State:
    """Leave out the contours that are not the pattern: open ones, dented ones, ones that hardly differ from what is
    around them. Each one left out is counted under the first of these reasons that applies."""
    contours, f = need_contours(state, "Filter contours"), state.scale
    closed = contours.closed
    why = np.zeros(len(closed), int)  # 0: kept
    if closed_only:
        why[~closed] = 1
    if min_solidity > 0:
        dented = [whole and check.contour_solidity(points) < min_solidity
                  for points, whole in zip(contours.points, closed)]
        why[(why == 0) & np.array(dented, bool)] = 2
    if min_contrast > 0 and closed.any():  # on the image not upscaled (its own noise), as extract_contours
        image = state.measured if f == 1 else state.unscaled
        rings = [points if f == 1 else (points + 0.5) / f - 0.5 for points, whole in zip(contours.points, closed) if whole]
        faint = np.zeros(len(closed), bool)
        faint[closed] = check.contact_contrast(image, DetectedContacts(np.empty((0, 2)), np.empty(0), rings,
                                                                       contours.threshold)) < min_contrast
        why[(why == 0) & faint] = 3
    counts = {reason: int((why == k).sum()) for k, reason in enumerate(("open", "not solid", "low contrast"), 1)}
    kept = check.keep_contours(contours, why == 0, counts)
    return replace(state, contours=kept, note=described(kept))


@module("Refine edges", "Contours",
        rule=choice("mid-level", refine.RULES,
                    "Where the edge is. mid-level: where the smoothed image is half-way between the region's interior "
                    "and the bright band at its edge, each region with its own two levels: follows contrast that "
                    "changes across the image. steepest gradient: where the brightness changes fastest: noisier "
                    "(single points jump to noise), and up to half a pixel off on an edge along the pixel rows or "
                    "columns; the only rule after Trace edges, which needs none. threshold: where the image "
                    "crosses the grey level the mask was cut at, one level for the whole image: for images without a "
                    "bright band, on an even background"),
        search_px=number(5.0, 0.5, 100, 0.5, "Each point is searched this many pixels in and out, along the direction "
                                             "its edge faces"),
        step_px=number(0.25, 0.01, 10, 0.05, "The image is sampled along that direction every this many pixels"),
        blur_px=number(1.0, 0, 50, 0.1, "Gaussian blur (sigma, px) of the image the edges are found on, for this "
                                        "module only; 0: none"))
def refine_edges(state: State, rule: str, search_px: float, step_px: float, blur_px: float) -> State:
    """Move every contour point along the direction its edge faces to its edge in the grey-level image, between the
    pixels: the mask's outline is a staircase of whole pixels, the edge is where the rule says. A point without an
    edge, or far off its neighbours', takes its neighbours' place. The edges are measured on the image as it is here,
    or on the one Edge image kept."""
    contours, f = need_contours(state, "Refine edges"), state.scale
    if rule == "mid-level" and state.labels is None:
        raise WorkflowError("Refine edges: mid-level needs the regions of Trace regions; the lines of Trace edges "
                            "bound none: use steepest gradient")
    if rule == "threshold" and not np.isfinite(contours.threshold):
        raise WorkflowError("Refine edges: threshold needs the grey level of an Otsu or Fixed threshold, found in the "
                            "image the edges are measured on: use mid-level or steepest gradient")
    refined = refine.refine_contours(state.measured, contours, state.labels, state.dark, search_px * f, step_px * f,
                                     blur_px * f, rule)
    return replace(state, contours=refined, note=rule)


@module("Smooth contours", "Contours",
        window=number(9, 5, 999, 2, "Contour points each cubic is fitted to (odd; an even number is taken one more): "
                                    "scatter over fewer points than about half of it goes"))
def smooth_contours(state: State, window: int) -> State:
    """Take the point-to-point scatter (noise) out of the contours: each point is put on a cubic fitted to the
    points around it along its contour (Savitzky-Golay), which follows a curved edge without cutting it short, so
    closed contours do not shrink. It smooths real roughness shorter than the window too: not for measuring
    roughness."""
    smooth = refine.smoothed_contours(need_contours(state, "Smooth contours"), (window * state.scale) | 1)
    return replace(state, contours=smooth, note="")
