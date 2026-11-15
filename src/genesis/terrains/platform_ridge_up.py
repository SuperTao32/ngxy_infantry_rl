"""上行平台凸台：从地面沿 +x 登上第一平台、越过窄凸台，落到第二平台。"""

from .platform_ridge_down import PlatformRidgeDown


PRESET = "platform_ridge_up"
PARAMETER_KEY = "platform_ridge_up"
PARAMETERS = dict(first_height=0.20, first_length=0.80, ridge_height=0.35,
                  ridge_width=0.15, second_height=0.30, second_length=1.2,
                  approach_length=1.3, base_thickness=0.10)
LEVELS = {
    0: dict(first_height=0.10, ridge_height=0.15, second_height=0.12),
    1: dict(first_height=0.125, ridge_height=0.20, second_height=0.16),
    2: dict(first_height=0.15, ridge_height=0.25, second_height=0.20),
    3: dict(first_height=0.175, ridge_height=0.30, second_height=0.25),
    4: dict(first_height=0.20, ridge_height=0.35, second_height=0.30),
}


class PlatformRidgeUp(PlatformRidgeDown):
    preset = PRESET
    ascending = True


TERRAIN_CLASS = PlatformRidgeUp
