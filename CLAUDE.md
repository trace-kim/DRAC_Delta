# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Contour extraction of contact holes in SEM images (`contour_extraction` package). It was extracted from
`E:\PythonProjects\AffineRANSAC` (features/contact.py, features/edges.py, io/sem_image.py, pipeline.find_contacts),
and at extraction (commit 6bb7ed8) its output was bit-identical to the original for every option. The reasons
behind each algorithmic choice are in AffineRANSAC's `docs/SPEC.md`, stage S2: read that before changing how a step
works.

## Commands
Windows, Python 3.13, a venv in `.venv` (gitignored). Packaging and pytest settings are in `pyproject.toml`; the
library needs no Qt (`pip install .`), the viewer's packages are the `viewer` extra, and `dev` adds pytest to them:
```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest                                      # all tests (~20 s)
.venv\Scripts\python -m pytest tests/test_refine.py                 # one step
.venv\Scripts\python -m pytest tests/test_refine.py -k mid_level    # tests whose name matches
.venv\Scripts\python viewer.py [image]                              # the viewer: try the options on a real image
```
No linter or formatter is configured; code is wrapped at about 120 columns.

## Architecture
One file per step, run in this order by `pipeline.extract_contours`:
load → prepare (flatten background, denoise, upscale) → detect (threshold, connected regions, filters, outer
contours) → check (contrast against a ring, optional) → refine (sub-pixel edges along outward normals).
- `extract_contours` takes an already loaded image: the caller runs `crop_databar(load_sem_image(path), rows)` first.
  It returns `(detected, refined, dropped)`, where `dropped` is how many detected contacts the contrast check removed.
- Every step takes and returns `contacts.DetectedContacts` (centers, areas, contours, threshold, left_out). The
  step modules import only from `contacts.py`, never from each other; `pipeline.py` is the only module that combines
  them. Keep it that way.
- Upscaling: detect and refine run on the image upscaled by `upscale`. `extract_contours` therefore multiplies the
  pixel options by f and the area options by f², and maps the results back with `prepare.in_image_pixels`
  (`(xy + 0.5) / f - 0.5`, areas / f²). The contrast check runs on the image that was not upscaled, so that it
  measures that image's own noise.
- Coordinates are in the pixel frame: (x, y) = (column, row), y pointing down, pixel centres at integers. Option
  names ending in `_px` are in pixels of the image as given; `*_area_px` options are in px².
- Every function and class in a step module must also be re-exported from `contour_extraction/__init__.py`, in step
  order; `tests/test_package.py` fails otherwise.
- `viewer.py` (repo root, outside the package, which stays free of Qt): a PySide6/pyqtgraph window to try
  `extract_contours` on real images, modelled on AffineRANSAC's `release/contour_dialog.py`. Its defaults come from
  `extract_contours`' signature; a new pipeline option also needs an entry in its `GROUPS`, `HELP` and (numbers)
  `LIMITS`, or `tests/test_viewer.py` fails. Its tests run Qt on the `offscreen` platform.

## Tests
One test file per step (`tests/test_<step>.py`), each importing from its submodule. `tests/synthetic.py` holds the
synthetic SEM-like images with known contact centres (`write_sem_like`, `exact_ellipse_image`, `banded_image`, ...)
and `centre_errors`; the test files import it as `from synthetic import ...` (there is no `tests/__init__.py`).

## Working rule
The user's basic rule: implement step by step with small changes and run the tests after each step, but at a brisk
pace: group related changes into one step, and keep the commentary between steps short.
