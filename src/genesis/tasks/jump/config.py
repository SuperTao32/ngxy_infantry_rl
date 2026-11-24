"""命令重采样时可插入的跳跃输入；不创建独立场景或任务配额。"""

PHASE_NAMES = ("takeoff", "flight", "landing")
MODE_NAMES = ("JUMP_20", "JUMP_40")
MODE_HEIGHTS = (0.20, 0.40)


def get_cfg():
    return {
        "probability": 0.0,
        "mode_probabilities": [0.5, 0.5],
        "jump_phase_durations_s": {"takeoff": 0.20, "flight": 0.40, "landing": 0.50},
        "jump_input_duration_s": [0.60, 0.80],
        "jump_height_reference": {"min_distance_m": 0.22, "max_distance_m": 0.40, "retract_duration_s": 0.10},
        # 目标均相对本次起跳处的地面高度；沿用旧平地跳的模式阈值。
        "clearance_targets_m": [0.45, 0.63],
        "ground_success_wheel_clearance_m": [0.22, 0.42],
        "ground_success_base_height_m": [0.43, 0.63],
        "ground_base_height_excess_buffer_m": 0.03,
        "jump_max_tilt_deg": 20.0,
        "takeoff_min_airborne_time_s": 0.06,
        "takeoff_min_wheel_clearance_m": 0.01,
        "landing_height_tolerance_m": 0.025,
        **get_reward_cfg(),
    }


def get_reward_cfg():
    """奖励形状参数、共享奖励及各阶段权重。"""
    return {
        "height_sigma": 0.02,  # 腾空 base 高度误差，m²。
        "tilt_sigma": 0.04,  # 飞行/落地共用的机身倾斜误差尺度。
        "lin_vel_sigma": 0.2,  # 前向速度跟踪误差，(m/s)²。
        "vertical_vel_sigma": 0.05,  # 落地冲击与竖直速度误差，(m/s)²。
        "ang_vel_sigma": 0.02,  # yaw 跟踪与落地 roll/pitch 角速度误差，(rad/s)²。
        "flight_leg_angle_limit_deg": 3.0,  # 相对机身的腾空虚拟腿摆角容差。
        "takeoff_wheel_push_force_reference_n": 400.0,  # 两轮向上法向支撑力之和的归一化参考，单位 N。
        "reward_scales": {
            "base_balance": -10.0,
            "leg_symmetry": -10.0,
            "leg_symmetry_bonus": 10.0,
            "tracking_lin_vel": 10.0,
            "tracking_ang_vel": 10.0,
            "joint_vel": -0.005,
            "airborne_wheel_vel": -0.05,
            "wheel_action_rate": -1.0,
            "leg_action_rate": -0.1,
            "leg_extension_at_limit": -10.0,
            "base_contact": -150.0,
            "jump_invalid": -10.0,
            "takeoff_event": 100.0,
            "flight_height_shortfall": -300.0,
            "flight_height_excess": -50000.0,  # 新增超高 / 目标高度；超过缓冲后抵消峰值增益。
            # 持续项按每秒计，避免起跳速度和超高收益压过成功奖励。
            "takeoff_upward_velocity": 150.0,
            "takeoff_wheel_push": 100.0,  # 起跳接触期按 Fz / 参考力线性给分
            "takeoff_vertical_velocity": 2000.0,
            # 腾空
            "flight_airtime": 2.0,
            "flight_balance": 15.0,
            "flight_height_progress": 1200.0,
            "flight_height_tracking": 100.0,
            # 当前按平地跳权重结算。
            "flight_wheel_clearance": 100.0,
            "flight_leg_angle": -100.0,
            # 落地
            "landing_vertical_velocity": 0.0,
            "landing_base_acceleration": -0.01,  # 三轴加速度平方和；100 m/s² 持续一拍扣 1。
            "soft_landing": 0.0,
            "landing_stability": 1000.0,
            "landing_airborne": -10.0,  # 首次触地后，任一轮仍未着地即逐步惩罚。
            # 任务结果
            "death": -100.0,
            "task_success": 10000.0,
            "task_failure": -500.0,
        },
    }
