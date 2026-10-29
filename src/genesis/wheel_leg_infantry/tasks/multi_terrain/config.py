"""独立配置各任务的环境占比、几何、命令、奖励和成功率课程。"""

from copy import deepcopy

from .intermittent import DEFAULTS as INTERMITTENT_DEFAULTS
from ...terrains.registry import level_catalog

TASK_NAMES = ("stairs_down", "stairs_up", "slope_up", "slope_down", "square_wave", "platform_ridge_down", "platform_ridge_up")


def _intermittent_cfg(probability, **overrides):
    """各地形按回合抽样；距离、时长等可在对应 task 中覆盖。"""
    return {
        **deepcopy(INTERMITTENT_DEFAULTS),
        "probability": probability,
        "predictive_braking": True,
        "max_braking_distance_m": 1.1,
        **overrides,
    }


def _get_curriculum_cfg():
    """按阶段配置子任务权重与各地形命令范围。"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": "slope",
                "start_iteration": 0,
                "targets": {
                    "task_weights": {
                        "platform_ridge_down": 0.0,
                        "platform_ridge_up": 0.0,
                        "stairs_up": 0.0,
                    },
                },
            },
            {
                "name": "ridge'",
                "start_iteration": 1000,
                "targets": {
                    "task_weights": {
                        "platform_ridge_down": 1.0,
                        "stairs_down": 0.5,
                        "slope_down": 0.5,
                    },
                },
            },
            {
                "name": "stairsup",
                "start_iteration": 2000,
                "targets": {
                    "task_weights": {
                        "stairs_up": 1.0,
                    },
                    "task_command_ranges": {
                        "stairs_down": {
                            "base_height_range": [0.22, 0.26],
                        },
                        "slope_up": {
                            "base_height_range": [0.22, 0.26],
                        },
                        "slope_down": {
                            "base_height_range": [0.22, 0.26],
                        },
                        "platform_ridge_down": {
                            "base_height_range": [0.22, 0.26],
                        },
                    },
                },
            },
            {
                "name": "all",
                "start_iteration": 3000,
                "targets": {
                    "task_weights": {
                        "platform_ridge_up": 1.0,
                    },
                    "task_command_ranges": {
                        "stairs_down": {
                            "base_height_range": [0.22, 0.24],
                        },
                        "slope_up": {
                            "base_height_range": [0.22, 0.24],
                        },
                        "slope_down": {
                            "base_height_range": [0.22, 0.24],
                        },
                        "platform_ridge_down": {
                            "base_height_range": [0.22, 0.24],
                        },
                    },
                },
            },
        ],
    }


def get_cfgs():
    env = {}
    reward = {
        "standing_tracking_sigma": 0.02,
        "tracking_gate": {
            "height_full_error": 0.01,
            "height_zero_error": 0.04,
            "attitude_full_angle_deg": 0.5,
            "attitude_zero_angle_deg": 2.0,
            "floor": 0.05,
        },
    }
    commands = {
        "lin_vel_range": [-3.5, 3.5],
        "ang_vel_range": [-6.0, 6.0],
        "base_height_range": [0.2, 0.38],
        "high_speed_ang_vel": {
            "lin_vel_threshold": 1.5,
            "max_abs_ang_vel": 2.0,
        },
        "high_base_height_velocity_limits": {
            "base_height_threshold": 0.32,
            "max_abs_lin_vel": 2.0,
            "max_abs_ang_vel": 3.0,
        },
        "standing_probability": 0.25,
    }
    # 每个任务集中配置；地形几何和难度参数统一由 terrains 提供。
    # 所有地形保留线速度/偏航角速度跟踪，并加大机身倾斜惩罚。
    tasks = {}

    # 上台阶
    tasks["stairs_up"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[0.5, 1.0],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.38, 0.38],
            standing_probability=0.0,
        ),
        rewards=dict(
            standing_drift=-3.0,
            intermittent_stop=3.0,  # 每次连续停稳达标只奖励一次。
            task_progress=5.0,
            task_success=30.0,
            leg_symmetry_bonus=5.0,
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            stairs_up_support=2.0,
            stairs_up_retract=2.0,
            stairs_up_takeoff=2.0,  # 双轮首次离开台下地面，一回合只奖励一次。
            stairs_up_landing_distance=2.0,  # 首轮触台距离奖励，单位为奖励/m，仅首次触台发放。
            stairs_up_wheel_wall_force=-0.1,  # 轮子撞立面的法向力总和（N），按 dt 积分。
            stairs_up_approach_lift=1.0,
            stairs_up_backswing=4.0,
            stairs_up_forward_swing=3.0,  # 轮子抬够后向前摆腿的新增进度，与伸腿同时引导。
            stairs_up_extend=5.0,  # 轮子位于台面上方时，新增伸腿长度的进度奖励。
            stairs_up_wheel_support=2.0,  # 落台阶段双轮首次真实接触上台面支撑，一回合一次。
            stairs_up_wheel_forward_action=1.0,  # 双轮落台接地后、机身前向速度不超过 command 时奖励正转 action，按 dt 积分。
            base_balance=-50.0,
            death=-10.0,
        ),
        resampling_time_s=3.0,
        terrain=dict(preset="stairs_up", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        intermittent=_intermittent_cfg(probability=0.3),
        termination=dict(
            episode_length_s=4.0,
            termination_if_roll_greater_than=90.0,
            termination_if_pitch_greater_than=90.0,
            tilt_termination_duration_s=3.0,
            base_contact_termination_duration_s=10.0,
            corridor_half_width=1.0,
            stall_timeout_s=2.0,
            progress_epsilon=0.01,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        ascent=dict(
            # 接近台阶时的抬升与后摆引导。
            approach_distance_m=0.00,  # 非负距离（m）；抬升仅在沿前，触台前后摆允许沿前后，触台后后摆不受此限制。
            approach_clearance_m=0.01,  # 抬升目标：底盘碰撞体最低点高出台阶顶面的余量（m）。
            backswing_angle_deg=10.0,  # 虚拟腿后摆奖励的目标角度（°）；正值为后摆，达到后进度饱和。
            # 搭接奖励与收腿奖励共用；支撑中断后重新累计稳定时间。
            support_duration_s=0.10,  # 深度、支撑力和速度门限须连续满足的时间（s）。
            support_force_n=1.0,  # 满足搭接深度的底盘接触点，向上支撑力之和的下限（N）。
            support_normal_z=0.9,  # 台面接触法向 |nz| 的下限（无量纲），用于排除侧壁接触。
            support_surface_tolerance_m=0.015,  # 底盘或轮子接触点与台面高度的最大偏差（m）。
            support_margin_m=0.1,  # 计入支撑力的底盘接触点须深入台面至少此距离（m）。
            retract_shortening_m=0.05,  # 相对首次稳定支撑时的腿长，每腿可获奖励的最大缩短量（m）。
            # 机身已搭台、双轮轮底达到台面高度后启用；前摆和伸腿没有角度先后门槛。
            forward_swing_angle_deg=10.0,  # 前摆奖励的饱和角度（°），对应负腿角；不作为伸腿条件。
            landing_extension_m=0.05,  # 相对进入落台阶段时的腿长，每腿可获奖励的最大伸长量（m）。
            # 双轮落台判定及落台后的跌落终止。
            wheel_edge_margin_m=0.01,  # 落台时轮心须越过台阶沿的最小水平距离（m）。
            wheel_bottom_tolerance_m=0.005,  # 轮底贴地/贴台的高度容差，亦作为双轮离地的高度门限（m）。
            fall_height_tolerance_m=0.03,  # 双轮曾落台后，任一轮底低于台面超过此距离即失败（m）。
            # 轮子撞击台阶立面的接触力惩罚。
            wall_normal_z_max=0.3,  # 立面接触法向 |nz| 的上限（无量纲），要求法向近似水平。
            wall_surface_tolerance_m=0.015,  # 立面接触点沿前进方向及高度方向的范围容差（m）。
            # 到达终点后的成功确认，以及支撑/站立共用的速度门限。
            success_duration_s=0.20,  # 到达终点并满足站姿、接触、高度及速度条件的连续时间（s）。
            standing_height_m=0.20,  # 成功站立时，机体原点相对脚下台面的最低高度（m）。
            max_vertical_speed_m_s=0.3,  # 支撑和成功站立判稳时，竖直速度绝对值的上限（m/s）。
            max_roll_pitch_speed_rad_s=1.5,  # 支撑和站立判稳时，机体系横滚/俯仰角速度合范数上限（rad/s）。
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.9,
            demote_threshold=0.2,
            # 对应 stairs_up.LEVELS 的等级；机身目标离地高度（m），两端相同表示固定高度。
            overrides={
                0: dict(commands=dict(base_height_range=[0.30, 0.32])),
                1: dict(commands=dict(base_height_range=[0.32, 0.34])),
                2: dict(commands=dict(base_height_range=[0.34, 0.38])),
                3: dict(commands=dict(base_height_range=[0.38, 0.38])),
                4: dict(commands=dict(base_height_range=[0.38, 0.38])),
            },
        ),
    )

    # 下台阶
    tasks["stairs_down"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            standing_drift=-3.0,
            intermittent_stop=3.0,  # 每次连续停稳达标只奖励一次。
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-1.0,
            task_heading=-0.5,
            base_balance=-20.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.05,
            leg_symmetry_bonus=2.0,
            base_contact=-10.0,
            stairs_down_landing_stability=-15.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="stairs", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        intermittent=_intermittent_cfg(probability=0.3),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=90.0,
            termination_if_pitch_greater_than=90.0,
            tilt_termination_duration_s=3.0,
            base_contact_termination_duration_s=5.0,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=1.0,
            max_pitch_deg=2.0,
            wheel_clearance_m=0.1,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
        # 首次落到台阶下方后，短时抑制竖直振荡及 roll/pitch 角速度。
        landing_stability=dict(duration_s=0.3, ang_vel_weight=0.25),
    )

    # 上坡
    tasks["slope_up"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-10.0,
            tracking_ang_vel=-10.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            standing_drift=-3.0,
            intermittent_stop=3.0,  # 每次连续停稳达标只奖励一次。
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-5.0,
            task_heading=-5.0,
            base_balance=-20.0,
            wheel_action_rate=-0.05,
            wheel_airborne=-10.0,
            joint_vel=-0.001,
            leg_action_rate=-0.02,
            leg_symmetry=-5.0,
            leg_symmetry_bonus=3.0,
            base_contact=-30.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="slope_up", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=2.0),
        intermittent=_intermittent_cfg(probability=0.3),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=90.0,
            termination_if_pitch_greater_than=90.0,
            tilt_termination_duration_s=3.0,
            base_contact_termination_duration_s=5.0,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=2.0,
            max_pitch_deg=2.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    # 下坡
    tasks["slope_down"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-10.0,
            tracking_ang_vel=-10.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            standing_drift=-3.0,
            intermittent_stop=3.0,  # 每次连续停稳达标只奖励一次。
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-2.0,
            task_heading=-0.5,
            base_balance=-20.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            leg_symmetry=-5.0,
            leg_symmetry_bonus=2.0,
            base_contact=-30.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="slope_down", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        intermittent=_intermittent_cfg(probability=0.3),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=90.0,
            termination_if_pitch_greater_than=90.0,
            tilt_termination_duration_s=3.0,
            base_contact_termination_duration_s=5.0,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=2.0,
            max_pitch_deg=2.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    # 过小凸台
    tasks["square_wave"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[0.5, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            standing_drift=-3.0,
            intermittent_stop=3.0,  # 每次连续停稳达标只奖励一次。
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-1.0,
            task_heading=-0.5,
            base_balance=-20.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            base_contact=-3.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="square_wave", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        intermittent=_intermittent_cfg(probability=0.3),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=90.0,
            termination_if_pitch_greater_than=90.0,
            tilt_termination_duration_s=3.0,
            base_contact_termination_duration_s=5.0,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    # 下行凸台：从出生点沿 +x 越过窄凸台，再经第一平台下到地面。
    # enabled 控制是否构建；全局课程 task_weights 控制何时分配训练环境。
    tasks["platform_ridge_down"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[0.5, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            standing_drift=-3.0,
            intermittent_stop=3.0,  # 每次连续停稳达标只奖励一次。
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-2.0,
            task_heading=-0.5,
            base_balance=-2.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            base_contact=-1.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="platform_ridge_down", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        intermittent=_intermittent_cfg(probability=0.3),
        termination=dict(
            episode_length_s=15.0,
            termination_if_roll_greater_than=90.0,
            termination_if_pitch_greater_than=90.0,
            tilt_termination_duration_s=3.0,
            base_contact_termination_duration_s=5.0,
            corridor_half_width=1.0,
            stall_timeout_s=5.0,
            progress_epsilon=0.01,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    # 上行凸台：从地面沿 +x 登上第一平台，越过窄凸台后在第二平台结束。
    tasks["platform_ridge_up"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[0.5, 1.0],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.38, 0.38],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=2.0,
            standing_drift=-3.0,
            intermittent_stop=3.0,  # 每次连续停稳达标只奖励一次。
            task_progress=5.0,
            task_success=20.0,
            base_balance=-30.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            leg_symmetry_bonus=3.0,
            # base_contact=-1.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="platform_ridge_up", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        intermittent=_intermittent_cfg(probability=0.3),
        termination=dict(
            episode_length_s=15.0,
            termination_if_roll_greater_than=90.0,
            termination_if_pitch_greater_than=90.0,
            tilt_termination_duration_s=3.0,
            base_contact_termination_duration_s=5.0,
            corridor_half_width=1.0,
            stall_timeout_s=5.0,
            progress_epsilon=0.01,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.9,
            demote_threshold=0.2,
            overrides={
                0: dict(commands=dict(base_height_range=[0.30, 0.32])),
                1: dict(commands=dict(base_height_range=[0.32, 0.34])),
                2: dict(commands=dict(base_height_range=[0.34, 0.38])),
                3: dict(commands=dict(base_height_range=[0.36, 0.38])),
                4: dict(commands=dict(base_height_range=[0.37, 0.38])),
            },
        ),
    )

    env["multi_terrain"] = dict(
        tasks=tasks,
        level_catalog=level_catalog(),
        parking_curriculum=dict(
            enabled=True,
            window_episodes=100,
            success_threshold=0.9,
            replay_probability=0.2,
            required_speed_m_s=0.2,
        ),
    )
    env.update(
        tilt_termination_duration_s=3.0, base_contact_termination_duration_s=5.0,
        episode_length_s=25.0,
        resampling_time_s=0.01,
        base_init_pos_range=[[-0.3, 0.3], [0.0, 0.0], [0.22, 0.22]],
        # 地形保持固定方向；出生 yaw 决定本回合斜向跨越的参考线。
        base_init_rpy_offset_range_deg=[[0.0, 0.0], [0.0, 0.0], [-10.0, 10.0]],
    )
    # 仅注册奖励函数，实际系数按各环境本回合所处的任务/课程阶段查表。
    reward["reward_scales"] = dict.fromkeys(set().union(*(task["rewards"] for task in tasks.values())), 0.0)
    return dict(env_cfg=env, reward_cfg=reward, command_cfg=commands, curriculum_cfg=_get_curriculum_cfg())
