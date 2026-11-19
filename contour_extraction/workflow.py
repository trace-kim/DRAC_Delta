"""A workflow: a list of modules (modules.py), each with its options, run in turn on an image to get its contours.

    steps = [step("median_filter"), step("otsu_threshold", blur_px=2.0), step("trace_regions"), step("refine_edges")]
    contours = workflow_contours(image, steps)  # contacts.Contours, in the image's pixels
    save_workflow("my.json", steps)             # and load_workflow("my.json") another day

A workflow is designed by trying (viewer.py): run_workflow() gives the State after every step, for a window that
shows each, and takes up an earlier run where the steps begin alike, so that changing a late step does not run the
early ones again. PRESETS are workflows to start from.
"""

import copy
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .contacts import Contours
from .modules import MODULES, State, WorkflowError, image_contours

FORMAT, VERSION = "contour_extraction workflow", 1  # of a workflow file


@dataclass
class Step:
    """One module of a workflow, with its options."""
    module: str                                  # its name in MODULES
    options: dict = field(default_factory=dict)  # every option of the module, by name
    enabled: bool = True                         # False: skipped, what the steps before it made goes on as it is


def step(module: str, enabled: bool = True, **options) -> Step:
    """A step of this module with these options, the others at their defaults. A ValueError names a module or an
    option that does not exist, or a value the option cannot have."""
    if module not in MODULES:
        raise ValueError(f"there is no module {module!r}")
    known = MODULES[module].options
    for name in options:
        if name not in known:
            raise ValueError(f"{module} has no option {name!r}")
    return Step(module, {name: option.checked(options.get(name, option.default), f"{module}.{name}")
                         for name, option in known.items()}, enabled)


@dataclass
class Run:
    """What a workflow made of an image, step by step."""
    image: np.ndarray               # the image it was given
    steps: list[Step]               # its steps, as they were when it ran
    states: list[State]             # states[0]: the image as given; states[k]: after step k (the first is 1)
    seconds: list[float]            # the time each step took
    error: Exception | None = None  # why step len(states) failed; the steps after it did not run

    @property
    def final(self) -> State:
        """What the last step that ran made."""
        return self.states[-1]

    @property
    def failure(self) -> str:
        """Which step failed and why, in words ("": none)."""
        if self.error is None:
            return ""
        number, name = len(self.states), self.steps[len(self.states) - 1].module
        title = MODULES[name].title if name in MODULES else name
        reason = str(self.error) if isinstance(self.error, WorkflowError) else f"{type(self.error).__name__}: {self.error}"
        return f"Step {number} ({title}) failed: {reason}"


def run_workflow(image: np.ndarray, steps: list[Step], previous: Run | None = None) -> Run:
    """Run the steps in turn on an image (2-D uint8, as load_sem_image gives it). A step that fails ends the run
    (Run.error); a step that is not enabled passes on what it was given.

    previous: an earlier run. If it was of this very image, the states of the steps the two begin with alike (module,
    options, enabled) are taken from it instead of being made again.
    """
    steps = copy.deepcopy(list(steps))
    alike = previous is not None and previous.image is image
    states, seconds = [previous.states[0] if alike else State(image)], []
    for number, this in enumerate(steps):
        alike = alike and number < len(previous.states) - 1 and previous.steps[number] == this
        if alike:
            states.append(previous.states[number + 1])
            seconds.append(previous.seconds[number])
            continue
        start = time.perf_counter()
        try:
            if this.enabled:
                state = MODULES[this.module].function(states[-1], **this.options)
            else:
                state = replace(states[-1], note="off")
        except Exception as error:  # told by Run.failure
            return Run(image, steps, states, seconds, error)
        states.append(state)
        seconds.append(time.perf_counter() - start)
    return Run(image, steps, states, seconds)


def workflow_contours(image: np.ndarray, steps: list[Step]) -> Contours:
    """The contours a workflow finds in an image, in the image's own pixels. A WorkflowError says which step failed
    and why, or that the workflow makes no contours."""
    run = run_workflow(image, steps)
    if run.error is not None:
        raise WorkflowError(run.failure) from run.error
    if run.final.contours is None:
        raise WorkflowError("the workflow makes no contours: it needs Trace regions or Trace edges")
    return image_contours(run.final)


# --- Workflow files (JSON) --------------------------------------------------------------------------------------------

def workflow_data(steps: list[Step]) -> dict:
    """A workflow as the data of its file."""
    return {"format": FORMAT, "version": VERSION,
            "steps": [{"module": s.module, "enabled": s.enabled, "options": dict(s.options)} for s in steps]}


def workflow_steps(data) -> list[Step]:
    """The workflow in the data of a workflow file; a ValueError says what is wrong with it. An option the file does
    not give has its default (a file written before the module had it)."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("not a workflow file")
    if not isinstance(data.get("version"), int) or data["version"] > VERSION:
        raise ValueError(f"a workflow file of version {data.get('version')!r}: this program reads up to {VERSION}")
    steps = []
    for number, entry in enumerate(data.get("steps", []), 1):
        try:
            steps.append(step(entry["module"], entry.get("enabled", True), **entry.get("options", {})))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"step {number}: {error}") from error
    return steps


def save_workflow(path: str | Path, steps: list[Step]):
    Path(path).write_text(json.dumps(workflow_data(steps), indent=2) + "\n", encoding="utf-8")


def load_workflow(path: str | Path) -> list[Step]:
    """The workflow of a file save_workflow wrote; a ValueError says what is wrong with the file."""
    return workflow_steps(json.loads(Path(path).read_text(encoding="utf-8")))


# --- Workflows to start from ------------------------------------------------------------------------------------------

PRESETS: dict[str, tuple[str, list[Step]]] = {  # name: (what it is for, its steps); preset() gives a copy to change
    "Contact holes": (
        "Holes darker than the background, whole ones only: Otsu's threshold, each region's outer boundary, moved to "
        "the edge half-way up to its bright band. What extract_contours does by default: the place to start.",
        [step("otsu_threshold"), step("trace_regions", holes=False, open_at_border=False), step("refine_edges")]),
    "Contact holes by their bright band": (
        "Holes no darker than the background, known by the bright band around them: the mask is everything but the "
        "band, and the dented regions (background pinched off between bands) are left out.",
        [step("otsu_threshold", method="band"), step("trace_regions", holes=False, open_at_border=False),
         step("filter_contours", min_solidity=0.9), step("refine_edges")]),
    "Holes without a bright band": (
        "Dark holes on a plain background, as wafer ADI images show them: with no band the mid-level rule would take "
        "its upper level from noise, so the edge is put where the image crosses the threshold's own grey level. One "
        "level for the whole image: if the brightness varies, put Flatten background first.",
        [step("otsu_threshold"), step("trace_regions", holes=False, open_at_border=False),
         step("refine_edges", rule="threshold")]),
    "Any pattern": (
        "Lines, spaces, pads, anything: every region's whole boundary, its holes' too, open where the region runs out "
        "of the image. What extract_any_contours does by default.",
        [step("otsu_threshold"), step("trace_regions"), step("refine_edges")]),
    "Small features (upscaled)": (
        "Features only a few pixels across: the image is interpolated to 4 times as many pixels per side first, which "
        "gives the threshold a finer outline to start the edges from. Its gain shows on real images rather than on "
        "synthetic ones: compare with the step unticked. The edge rule is the threshold's (features without a band); "
        "with a bright band choose mid-level.",
        [step("upscale", factor=4), step("otsu_threshold"), step("trace_regions", min_area_px=5),
         step("refine_edges", rule="threshold")]),
    "Noisy image": (
        "Few frames, low dose: the pattern is found in a strongly denoised image (median, then non-local means), which "
        "keeps noise out of the mask; its edges are still measured on the image as loaded (Edge image), so that the "
        "denoising cannot move them, and the scatter left in the contours is smoothed out.",
        [step("edge_image"), step("median_filter"), step("non_local_means", h=15.0), step("otsu_threshold"),
         step("trace_regions"), step("refine_edges"), step("smooth_contours")]),
    "Uneven background": (
        "Brightness that varies across the image (charging, shading): the background is flattened first, so that one "
        "threshold fits the whole image, and an edge is measured between the same two levels on all its sides.",
        [step("flatten_background"), step("otsu_threshold"), step("trace_regions"), step("refine_edges")]),
    "Edges without a threshold (Canny)": (
        "No grey level divides the pattern from the background (the same grey, known by its edges): the pixels of "
        "steepest change are followed along each edge, and each is put where the gradient peaks between the pixels "
        "(the Canny/Devernay detector). A bright band gives two contours: its inner and its outer slope.",
        [step("canny_edges"), step("trace_edges"), step("smooth_contours")]),
}


def preset(name: str) -> list[Step]:
    """The steps of a workflow to start from (PRESETS), yours to change."""
    return copy.deepcopy(PRESETS[name][1])
