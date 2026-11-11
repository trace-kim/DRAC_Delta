"""All steps: the contacts of one SEM image, from the image to its refined contours (extract_contours)."""

import numpy as np

from .check import contact_contrast, keep_contacts
from .contacts import DetectedContacts
from .detect import BAND_MIN_SOLIDITY, detect_contacts
from .prepare import in_image_pixels, prepared_image, upscaled_image
from .refine import RULES, refine_edges

REFINEMENTS = RULES + ("none",)  # extract_contours refinement: an edge rule, or none (the detected contours)


def extract_contours(
    image: np.ndarray,
    *,  # by name only
    background_px: float = 0.0,
    denoise: str = "none",
    median_px: int = 3,
    nlm_h: float = 10.0,
    tv_weight: float = 0.1,
    upscale: int = 1,
    detection: str = "otsu",
    dark_contacts: bool = True,
    detection_blur_px: float = 1.5,
    min_area_px: int = 20,
    max_area_px: int = 0,
    min_solidity: float = BAND_MIN_SOLIDITY,
    min_contrast: float = 0.0,
    refinement: str = "mid-level",
    search_px: float = 5.0,
    edge_step_px: float = 0.25,
    edge_blur_px: float = 1.0,
    normal_span: int = 3,
) -> tuple[DetectedContacts, DetectedContacts, int]:
    """The contacts of one SEM image (uint8, data bar cropped): (detected, refined, how many detected contacts the
    contrast check dropped). AffineRANSAC's pipeline.find_contacts, with the same options.

    Every step works on prepared_image (background_px > 0: the background flattened first; denoise: then denoised;
    upscale > 1: then upscaled that many times). The options in pixels, and normal_span (contour points), are those
    of the image as given: in the upscaled image they are `upscale` times larger (areas its square), and the contacts
    are returned in the image's own pixels.
    detected: detect_contacts with method `detection` ("otsu", "otsu3" or "band"), blur, area limits (max_area_px 0:
    no limit) and, for "band" only, min_solidity; then, if min_contrast > 0, without the contacts whose contrast
    against a ring around them is below this many noise sigma (contact_contrast, on the image not upscaled: its own
    noise). refined: refine_edges with edge rule `refinement` (RULES), or with "none" the detected contacts themselves.
    """
    image = prepared_image(image, background_px, dark_contacts, denoise, median_px, nlm_h, tv_weight)
    f = upscale
    big = upscaled_image(image, f)
    detected = detect_contacts(big, dark_contacts, detection_blur_px * f, min_area_px * f ** 2,
                               max_area_px * f ** 2 or None, detection, min_solidity if detection == "band" else None)
    found = len(detected.centers)
    if min_contrast > 0:
        detected = keep_contacts(detected, contact_contrast(image, in_image_pixels(detected, f)) >= min_contrast)
    refined = detected if refinement == "none" else refine_edges(
        big, detected, dark_contacts, search_px * f, edge_step_px * f, edge_blur_px * f, normal_span * f, refinement)
    return in_image_pixels(detected, f), in_image_pixels(refined, f), found - len(detected.centers)
