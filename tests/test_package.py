import contour_extraction
from contour_extraction import check, contacts, detect, load, pipeline, prepare, refine


def test_every_function_of_every_step_is_importable_from_the_package():
    """A function added to a step's file must also be added to __init__.py."""
    for module in (contacts, load, prepare, detect, check, refine, pipeline):
        for name, value in vars(module).items():
            if callable(value) and getattr(value, "__module__", None) == module.__name__:
                assert getattr(contour_extraction, name, None) is value, f"{module.__name__}.{name} not in __init__.py"
