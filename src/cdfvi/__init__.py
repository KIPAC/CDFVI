"""
The :py:mod:`cdfvi` package collects a set of tools to do
visualizaiton, fitting and inference using measurements
of comoving distance
"""


from . import utility_functions as funcs
from ._version import __version__
from .fitting import SNCosmoChi2, SNPolyChi2, TDCosmoChi2, TDPolyChi2

__all__ = [
    "__version__",
    "funcs",
    "SNCosmoChi2",
    "SNPolyChi2",
    "TDPolyChi2",
    "TDCosmoChi2",
]
