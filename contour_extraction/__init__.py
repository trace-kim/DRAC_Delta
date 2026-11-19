"""Contour extraction of contact holes, and of any pattern, in SEM images.

Extracted from AffineRANSAC (features/contact.py, features/edges.py, features/contours.py, io/sem_image.py,
pipeline.find_contacts and find_contours); the reasons behind each choice are in its docs/SPEC.md, stage S2.

One file per step, in the order extract_contours (pipeline.py) runs them:
  1. load.py     the image as grayscale uint8, the data bar cropped off its bottom.
  2. prepare.py  optionally flatten the background, denoise, upscale.
  3. detect.py   blur, threshold, connected regions, drop the regions that are not contacts; outer contours.
  4. check.py    optionally drop the contacts with too little contrast against a ring around them.
  5. refine.py   move every contour point along its outward normal to its sub-pixel edge.
Every step takes and returns contacts as contacts.DetectedContacts.

extract_any_contours (pipeline.py) runs the same steps but the check for the contours of any pattern
(contacts.Contours): every region's whole boundary, closed, or open where the region runs out of the image
(detect_contours), each point moved along the direction its edge faces (refine_contours).

A workflow (workflow.py) is the same steps, and more, as single modules (modules.py: Gaussian blur, non-local means,
Otsu threshold, Canny edges, Trace regions, Refine edges, ...) in a list of the user's making, designed in the viewer
(viewer.py), kept in a JSON file, and run with no window:

    contours = workflow_contours(load_sem_image(path), load_workflow("my workflow.json"))

Coordinates are in the pixel frame: (x, y) = (column, row) in pixels, y pointing DOWN, pixel centres at integer
values, (0, 0) the centre of the top-left pixel.

Every name is importable from the package itself (from contour_extraction import extract_contours).
"""

from .contacts import Contours, DetectedContacts, polygon_area, polygon_centroid
from .load import crop_databar, load_sem_image
from .prepare import (DENOISERS, NLM_PATCH_PX, NLM_SEARCH_PX, denoised_image, flatten_background, in_image_contours,
                      in_image_pixels, prepared_image, upscaled_image)
from .detect import (BAND_MIN_SOLIDITY, CANNY_TOP_SHARE, CONTOUR_LEFT_OUT, LEFT_OUT, METHODS, canny_edges, contact_mask,
                     detect_contacts, detect_contours, edge_chains, edge_contours, gradient_peaks, inside_image,
                     mask_contours, out_of_polygon, outer_contour, right_of_tangent, sampled, solidity, walk_chains)
from .check import contact_contrast, contour_solidity, keep_contacts, keep_contours, noise_sigma
from .refine import (RULES, crossings, edge_offsets, interior_level, level_offsets, mid_level_shifts, mid_levels,
                     outward_normals, refine_contours, refine_edges, sample_profiles, smoothed_contours,
                     without_isolated)
from .pipeline import REFINEMENTS, extract_any_contours, extract_contours
from .modules import MODULES, STAGES, Module, Option, State, WorkflowError, image_contours
from .workflow import (PRESETS, Run, Step, load_workflow, preset, run_workflow, save_workflow, step, workflow_contours,
                       workflow_data, workflow_steps)
