"""独立下坡；共享上坡的参数、难度与几何实现。"""

from .slope_up import LEVELS, PARAMETERS, SlopeUp


PRESET = "slope_down"
PARAMETER_KEY = "slope_down"


class SlopeDown(SlopeUp):
    descending = True


TERRAIN_CLASS = SlopeDown
