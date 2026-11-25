"""上行平台凸台：从地面沿 +x 直接越过窄凸台，落到后方平台。"""

from .platform_ridge_down import PlatformRidgeDown


PRESET = "platform_ridge_up"
PARAMETER_KEY = "platform_ridge_up"
PARAMETERS = dict(ridge_height=0.35, ridge_width=0.15, second_height=0.30, second_length=1.2,
                  approach_length=2.1, base_thickness=0.10)  # 原第一平台区域改为平地，凸台位置不变。
LEVELS = {
    0: dict(ridge_height=0.15, second_height=0.13),
    1: dict(ridge_height=0.15, second_height=0.10),
}


class PlatformRidgeUp(PlatformRidgeDown):
    preset = PRESET
    ascending = True


TERRAIN_CLASS = PlatformRidgeUp
