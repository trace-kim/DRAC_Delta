import numpy as np

from contour_extraction.contacts import polygon_area, polygon_centroid


def test_polygon_area_and_centroid_of_a_rectangle_for_either_order():
    rectangle = np.array([[1.0, 2.0], [5.0, 2.0], [5.0, 4.0], [1.0, 4.0]])  # 4 x 2, centre (3, 3)
    for vertices in (rectangle, rectangle[::-1]):
        assert polygon_area(vertices) == 8.0
        np.testing.assert_allclose(polygon_centroid(vertices), [3.0, 3.0])
