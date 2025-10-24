from ._version import __version__

from . import utility_functions as funcs
from .fitting import SNCosmoChi2, SNPolyChi2, TDPolyChi2, TDCosmoChi2

__all__ = [
    "__version__",
    "funcs",
    "SNCosmoChi2",
    "SNPolyChi2",    
    "TDPolyChi2",
    "TDCosmoChi2",
]
