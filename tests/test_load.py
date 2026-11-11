import numpy as np
import pytest

from contour_extraction.load import crop_databar, load_sem_image
from synthetic import write_sem_like


def test_load_sem_image_reads_grayscale_from_a_non_ascii_path(tmp_path):
    path = tmp_path / "폴더" / "tile.jpg"
    path.parent.mkdir()
    write_sem_like(path, rows=200, cols=300, databar_px=20)
    image = load_sem_image(path)
    assert image.shape == (200, 300) and image.dtype == np.uint8


def test_load_sem_image_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sem_image(tmp_path / "missing.jpg")


def test_crop_databar_keeps_coordinates():
    image = np.arange(50, dtype=np.uint8).reshape(10, 5)
    cropped = crop_databar(image, 3)
    assert cropped.shape == (7, 5)
    assert cropped[2, 4] == image[2, 4]  # same (x, y) -> same pixel
    assert crop_databar(image, 0) is image
