"""PREM-like Earth models"""

from .earth_model import OneDModel
from .const import R_EARTH
from .PREM import PREM

__all__ = ["OneDModel", "R_EARTH", "PREM"]