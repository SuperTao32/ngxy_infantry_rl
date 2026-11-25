"""越障结果与各地形专属奖励；地形采样仍由 TerrainManager 负责。"""

from ...terrains.registry import TERRAIN_REGISTRY

# 迁自旧版 multi_terrain；仅保留当前已有同名实现的奖励。
# stairs_down 对应当前 stairs；未列出的共享项继续继承 locomotion。
TERRAIN_REWARD_OVERRIDES = {
    "standing": {
        # 所有特殊地形的静止命令共用；未列出的权重实时继承 locomotion（含课程）。
        "base_height": 0.0,
        "height_gate": 0.0,
    },
    "stairs_up": {
        "stairs_up_wheel_support": 5.0,  # 双轮均在台阶面上承重时为 1；按 dt 积分，静止时不生效。
        "stairs_up_wheel_airborne": 5.0,  # 先有轮子支撑再双轮离地，每回合只奖励一次；不乘 dt。
        "stairs_up_leg_retraction": 200.0,  # 入台后的平均缩腿量，仅奖励新纪录，累计上限为一级台阶高度；不乘 dt。
        "stairs_up_wheel_wall_force": -0.05,  # 双轮无地面支撑时，轮子撞压各级立面的法向力之和，N；按 dt 积分。
        "task_lateral": 0,
        "task_heading": 0,
        "base_height": 0.0,
        "height_gate": 0.0,
        "base_balance": -100.0,
        "base_contact": 0.0,
        "wheel_airborne": 0.0,
    },
    "stairs": {
        "stairs_down_landing_stability": -15.0,
        "base_height": 0.0,
        "height_gate": 0.0,
        "wheel_airborne": 0.0,
        "base_balance": -100.0,
        "base_contact": -100.0,
    },
    "slope_up": {
        "base_height": 0.0,
        "height_gate": 0.0,
        "base_balance": -100.0,
        "base_contact": -100.0,
    },
    "slope_down": {
        "base_height": 0.0,
        "height_gate": 0.0,
        "base_balance": -100.0,
        "base_contact": -100.0,
    },
    "square_wave": {
        "square_wave_wheel_wall_force": -1.0,  # 两轮与小台阶立面的法向力之和，N；按 dt 积分。
        "task_lateral": 0,
        "task_heading": 0,
        "base_height": 0.0,
        "height_gate": 0.0,
        "wheel_airborne": 0.0,
        "base_balance": -100.0,
        "base_contact": -100.0,
    },
    "platform_ridge_down": {
        "task_lateral": 0,
        "task_heading": 0,
        "base_height": 0.0,
        "height_gate": 0.0,
        "base_balance": -100.0,
        "base_contact": -100.0,
        "wheel_airborne": 0.0,
    },
    "platform_ridge_up": {
        "platform_ridge_up_wheel_support": 5.0,  # 双轮在平台或凸台顶面承重；按 dt 积分。
        "platform_ridge_up_wheel_airborne": 5.0,  # 支撑后首次双轮离地；每回合一次，不乘 dt。
        "platform_ridge_up_leg_retraction": 200.0,  # 入台后的缩腿新纪录，上限为凸台高度；不乘 dt。
        "platform_ridge_up_wheel_wall_force": -0.05,  # 双轮无支撑时撞压外露立面的法向力，N；按 dt 积分。
        "task_lateral": 0,
        "task_heading": 0,
        "base_height": 0.0,
        "height_gate": 0.0,
        "base_balance": -100.0,
        "base_contact": 0.0,
        "wheel_airborne": 0.0,
    },
}


def get_cfg():
    config = {
        "standing_reward_scales": dict(TERRAIN_REWARD_OVERRIDES["standing"]),
        "finish_distance_m": 0.3,  # 双轮越过几何终点后再前进的距离。
        "success_hold_s": 0.2,  # 双轮支撑、无机身接触且正立的连续时间。
        "max_tilt_deg": 30.0,
        "terrains": {
            name: {
                "reset": {
                    "distance_range_m": [0.7, 1.2],  # 基座到障碍起点的 x 向距离。
                    "yaw_range_deg": [-5.0, 5.0],  # 相对地块 +x 的朝向。
                    "base_height_m": 0.22,
                    "standing_tof_margin_m": 0.02,  # 停车出生时，前向 ToF 光心和落点距地形边缘的余量。
                },
                "commands": {
                    "lin_vel_range": [0.5, 2.5],
                    "ang_vel_range": [-0.1, 0.1],
                    "base_height_range": [0.22, 0.28],
                    "standing_probability": None,  # None 跟随全局课程；特殊地形仅在 reset 抽停车回合。
                },
                "tracking_gate": {
                    "height_full_error": 0.05,
                    "height_zero_error": 0.10,
                    "attitude_full_angle_deg": 1.0,
                    "attitude_zero_angle_deg": 5.0,
                    "floor": 0.20,
                },
                # 非静止命令下，同名 locomotion 项覆盖全局权重；其余项作为越障附加奖励。
                "reward_scales": {
                    "progress": 5.0,
                    "success": 50.0,
                    "failure": -10.0,
                    "task_lateral": -1.0,  # 相对出生航向参考线的横向距离平方，m²。
                    "task_heading": -0.5,  # 与出生航向的偏差：1 - cos(yaw_error)。
                    **TERRAIN_REWARD_OVERRIDES.get(name, {}),
                },
            }
            for name in TERRAIN_REGISTRY
            if name != "plane"
        },
    }

    config["terrains"]["stairs"]["landing_stability"] = {
        "duration_s": 0.3,  # 首次触地下方地面起，独立于越障通关状态计时。
        "ang_vel_weight": 0.25,  # vz² + 系数 × (wx² + wy²)。
        "surface_tolerance_m": 0.08,  # 轮底与下方地面高度差的容差，仍须轮子有接触。
    }
    config["terrains"]["square_wave"]["wheel_wall_force"] = {
        "normal_x_min": 0.5,  # 接触单位法向的 |nx| 下限，包含入口棱边，排除顶部和侧面。
        "surface_tolerance_m": 0.01,  # 接触点距入口立面及高度范围的容差。
    }

    # stairs_up 和 platform_ridge_up 专属
    for name in ("stairs_up", "platform_ridge_up"):
        config["terrains"][name]["wheel_wall_force"] = {
            "normal_x_min": 0.5,  # 立面接触单位法向的 |nx| 下限。
            "surface_tolerance_m": 0.01,  # 接触点距外露立面及其高度范围的容差，也用于台面支撑判定。
        }
        config["terrains"][name]["commands"].update(
            {
                "lin_vel_range": [1.5, 2.5],
                "ang_vel_range": [0.0, 0.0],
                "base_height_range": [0.38, 0.38],
                "standing_probability": None,
            }
        )
    return config
