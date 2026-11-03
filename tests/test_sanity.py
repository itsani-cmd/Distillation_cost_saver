"""Trivial import/sanity checks so CI is green from the first commit."""

import src
import utils  # noqa: F401


def test_package_version():
    assert isinstance(src.__version__, str)


def test_modules_importable():
    import src.config  # noqa: F401
    import src.losses  # noqa: F401
    import utils.colab_utils  # noqa: F401
