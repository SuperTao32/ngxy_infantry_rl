"""越障结果与各地形专属奖励；地形采样仍由 TerrainManager 负责。"""

from ...terrains.registry import TERRAIN_REGISTRY


def get_cfg():
    return {
        "finish_distance_m": 0.3,  # 双轮越过几何终点后再前进的距离。
        "success_hold_s": 0.2,     # 双轮支撑、无机身接触且正立的连续时间。
        "max_tilt_deg": 30.0,
        "terrains": {
            name: {
                "reset": {
                    "distance_range_m": [0.7, 1.2],  # 基座到障碍起点的 x 向距离。
                    "yaw_range_deg": [-5.0, 5.0],    # 相对地块 +x 的朝向。
                    "base_height_m": 0.22,
                    "standing_tof_margin_m": 0.02,  # 停车出生时，前向 ToF 光心和落点距地形边缘的余量。
                },
                "commands": {
                    "lin_vel_range": [0.5, 2.0],
                    "ang_vel_range": [-0.1, 0.1],
                    "base_height_range": [0.22, 0.28],
                    "standing_probability": None,  # None 跟随全局课程；特殊地形仅在 reset 抽停车回合。
                },
                "tracking_gate": {
                    "height_full_error": 0.05,
                    "height_zero_error": 0.08,
                    "attitude_full_angle_deg": 5.0,
                    "attitude_zero_angle_deg": 30.0,
                    "floor": 0.30,
                },
                "reward_scales": {
                    "progress": 5.0,
                    "success": 50.0,
                    "failure": -10.0,
                    "task_lateral": -1.0,  # 相对出生航向参考线的横向距离平方，m²。
                    "task_heading": -0.5,  # 与出生航向的偏差：1 - cos(yaw_error)。
                },
            }
            for name in TERRAIN_REGISTRY if name != "plane"
        },
    }
