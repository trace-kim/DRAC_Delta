# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Contour extraction of contact holes in SEM images (`contour_extraction` package). It was extracted from
`E:\PythonProjects\AffineRANSAC` (features/contact.py, features/edges.py, io/sem_image.py, pipeline.find_contacts),
and at extraction (commit 6bb7ed8) its output was bit-identical to the original for every option. The contours of any
pattern (`extract_any_contours`) were ported later from AffineRANSAC's D155 (features/contours.py,
pipeline.find_contours), also bit-identical. The reasons behind each algorithmic choice are in AffineRANSAC's
`docs/SPEC.md`, stage S2 (contours: its *Contours of any pattern* paragraph and decision D155): read that before
changing how a step works. On top of the two fixed pipelines there are workflows (`modules.py`, `workflow.py`): the
same steps as single modules in a list the user puts together in the viewer; they are this repository's own, not
AffineRANSAC's.

## Commands
Windows, Python 3.13, a venv in `.venv` (gitignored). Packaging and pytest settings are in `pyproject.toml`; the
library needs no Qt (`pip install .`), the viewer's packages are the `viewer` extra, and `dev` adds pytest to them:
```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest                                      # all tests (~20 s)
.venv\Scripts\python -m pytest tests/test_refine.py                 # one step
.venv\Scripts\python -m pytest tests/test_refine.py -k mid_level    # tests whose name matches
.venv\Scripts\python viewer.py [image] [workflow.json]              # the viewer: design a workflow on a real image
```
No linter or formatter is configured; code is wrapped at about 120 columns.

## Architecture
One file per step, run in this order by `pipeline.extract_contours`:
load → prepare (flatten background, denoise, upscale) → detect (threshold, connected regions, filters, outer
contours) → check (contrast against a ring, optional) → refine (sub-pixel edges along outward normals).
- `extract_contours` takes an already loaded image: the caller runs `crop_databar(load_sem_image(path), rows)` first.
  It returns `(detected, refined, dropped)`, where `dropped` is how many detected contacts the contrast check removed.
- Every step takes and returns `contacts.DetectedContacts` (centers, areas, contours, threshold, left_out). The
  step modules import only from `contacts.py`, never from each other; `pipeline.py` (and `modules.py`, below) are
  the only modules that combine them. Keep it that way.
- `pipeline.extract_any_contours` runs the same steps, without the check, for the contours of any pattern
  (`contacts.Contours`: points, normals, closed, region per contour): `detect.detect_contours` gives every region's
  whole boundary (holes too), cut into open contours where it reaches the image's outermost rows and columns;
  `refine.refine_contours` moves each point along the normal the detection gave it; `prepare.in_image_contours` maps
  them back. It returns `(detected, refined)`. Its options are a subset of `extract_contours`' with the same defaults.
- Upscaling: detect and refine run on the image upscaled by `upscale`. `extract_contours` therefore multiplies the
  pixel options by f and the area options by f², and maps the results back with `prepare.in_image_pixels`
  (`(xy + 0.5) / f - 0.5`, areas / f²). The contrast check runs on the image that was not upscaled, so that it
  measures that image's own noise.
- Coordinates are in the pixel frame: (x, y) = (column, row), y pointing down, pixel centres at integers. Option
  names ending in `_px` are in pixels of the image as given; `*_area_px` options are in px².
- Every function and class in a step module must also be re-exported from `contour_extraction/__init__.py`, in step
  order; `tests/test_package.py` fails otherwise. (Of `modules.py` and `workflow.py` only the names a caller needs
  are exported; a module's function is reached through `MODULES`.)
- Workflows. `modules.py`: a module is a function `name(state, **options) -> State` registered with `@module(title,
  stage, option=number(...) | choice(...) | flag(...))`; its name is its key in `MODULES` and in workflow files, its
  docstring and the options' `help` are what the viewer shows. Stages: Image (changes `State.image`), Mask (makes or
  changes `State.mask`: a threshold's regions or Canny's edge pixels), Contours (makes or changes `State.contours`,
  a `contacts.Contours` in the pixels of `State.image`). A module only fits a step function to the `State`; an
  algorithm belongs in a step file, with its test there. Rules a module must keep:
  - It never changes the arrays it is given (States share them: `workflow.run_workflow` takes up an earlier run
    where the steps begin alike).
  - `_px` options are pixels of the image as loaded: multiply them by `State.scale` (areas by its square), so that
    Upscale put into a workflow needs no other option changed.
  - The edges are measured on `State.measured` (the image kept by the module Edge image, else `State.image`), so
    that denoising or equalization done only to find the pattern cannot move them (SPEC D167's rule).
  - A module that changes the image's size (crop, upscale) comes before the mask (`before_the_mask`, `resized`).
  - What is missing is said by a `WorkflowError` in the user's words (`need_mask`, `need_contours`).
- `workflow.py`: `Step` (module, options, enabled), `step()` (checks them), `run_workflow` → `Run` (the `State` after
  every step, a failed step's error), `workflow_contours`, `save_workflow` / `load_workflow` (JSON, `FORMAT` and
  `VERSION`; an option a file lacks has its default, so adding an option does not break saved files; renaming a
  module or an option does), and `PRESETS`, the workflows to start from. The preset *Contact holes* finds what
  `extract_contours` finds and *Any pattern* is `extract_any_contours` to the last bit (`tests/test_workflow.py`):
  keep them so. A preset must be shown to work on a synthetic image there; equalization before Otsu did not (it
  makes Otsu divide the pixels into equal shares), which is why no preset uses it.
- `detect.edge_contours` is the Canny/Devernay detector: `canny_edges` → `thin` → `edge_chains` (the pixels of each
  line in order, from `cv2.findContours`' walk round it: `walk_chains`) → `gradient_peaks` (each pixel to the peak
  of the gradient between the pixels). It is independent of the threshold path and of refine's edge rules.
- `viewer.py` (repo root, outside the package, which stays free of Qt): a PySide6/pyqtgraph window to design a
  workflow on real images. It builds its menus and editors from `MODULES` and `PRESETS`, so a new module or preset
  needs no change in it. It runs the workflow in a background thread for the newest steps only (Auto process, or the
  Process button), and shows the state after the last step or after the step selected. Its tests run Qt on the
  `offscreen` platform (which has no fonts: to look at the window, grab it with `QT_QPA_PLATFORM=windows` and
  `WA_DontShowOnScreen`).

## Tests
One test file per step (`tests/test_<step>.py`), each importing from its submodule; `tests/test_modules.py` runs
every module with its defaults, and `tests/test_workflow.py` every preset. `tests/synthetic.py` holds the
synthetic SEM-like images with known contact centres (`write_sem_like`, `exact_ellipse_image`, `banded_image`, ...)
and `centre_errors`; the test files import it as `from synthetic import ...` (there is no `tests/__init__.py`).

## Working rule
The user's basic rule: implement step by step with small changes and run the tests after each step, but at a brisk
pace: group related changes into one step, and keep the commentary between steps short.

## Git
Commit straight to `main`; never create a branch. When a task is done (tests passing), commit and push all of its
work to `origin/main` without being asked. Why: everything is run and tried on a remote PC that only pulls `main`
and uses it, so work that is not pushed cannot be tested there, and other branches only cause confusion.
