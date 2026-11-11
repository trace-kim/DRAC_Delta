"""Step 1, load: an SEM image as grayscale uint8, the data bar cropped off its bottom.

The array is indexed image[y, x]: x is the column (to the right), y the row (pointing DOWN), in pixels; (0, 0) is the
centre of the top-left pixel.
"""

from pathlib import Path

import cv2
import numpy as np


def load_sem_image(path: str | Path) -> np.ndarray:
    """Load an image file as a 2-D uint8 grayscale array, shape (rows, cols), indexed image[y, x].

    Colour images are converted to grayscale. Values are kept exactly as stored in the file (uint8).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # cv2.imread fails silently on Windows for non-ASCII paths (e.g. Korean folder names), so read the raw bytes
    # with numpy and decode them in memory instead.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def crop_databar(image: np.ndarray, databar_rows: int) -> np.ndarray:
    """The image without the SEM info/data bar along its bottom (databar_rows rows; 0: the image itself).

    Removing BOTTOM rows does not change the (x, y) pixel coordinates of the pixels that remain, so results on the
    cropped image need no correction.
    """
    if databar_rows <= 0:
        return image
    return image[: image.shape[0] - databar_rows]
