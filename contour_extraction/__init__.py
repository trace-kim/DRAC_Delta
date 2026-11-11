"""Contour extraction of contact holes in SEM images.

Extracted from AffineRANSAC (features/contact.py, features/edges.py, io/sem_image.py, pipeline.find_contacts);
the reasons behind each choice are in its docs/SPEC.md, stage S2.

One file per step, in the order extract_contours (pipeline.py) runs them:
  1. load.py     the image as grayscale uint8, the data bar cropped off its bottom.
  2. prepare.py  optionally flatten the background, denoise, upscale.
  3. detect.py   blur, threshold, connected regions, drop the regions that are not contacts; outer contours.
  4. check.py    optionally drop the contacts with too little contrast against a ring around them.
  5. refine.py   move every contour point along its outward normal to its sub-pixel edge.
Every step takes and returns contacts as contacts.DetectedContacts.

Coordinates are in the pixel frame: (x, y) = (column, row) in pixels, y pointing DOWN, pixel centres at integer
values, (0, 0) the centre of the top-left pixel.

Every name is importable from the package itself (from contour_extraction import extract_contours).
"""

from .contacts import DetectedContacts, polygon_area, polygon_centroid
from .load import crop_databar, load_sem_image
from .prepare import (DENOISERS, NLM_PATCH_PX, NLM_SEARCH_PX, denoised_image, flatten_background, in_image_pixels,
                      prepared_image, upscaled_image)
from .detect import BAND_MIN_SOLIDITY, LEFT_OUT, METHODS, contact_mask, detect_contacts, outer_contour, solidity
from .check import contact_contrast, keep_contacts, noise_sigma
from .refine import (RULES, crossings, edge_offsets, interior_level, level_offsets, mid_level_shifts, outward_normals,
                     refine_edges, sample_profiles, without_isolated)
from .pipeline import REFINEMENTS, extract_contours
