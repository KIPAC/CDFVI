from ._version import __version__

from . import utility_functions as funcs
from .fitting import PolyFit

__all__ = [
    "__version__",
    "funcs",
    "PolyFit",
]
