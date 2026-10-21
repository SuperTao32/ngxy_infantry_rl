"""独立配置各任务的环境占比、几何、命令、奖励和成功率课程。"""

from copy import deepcopy

from .intermittent import DEFAULTS as INTERMITTENT_DEFAULTS
from ..locomotion.config_locomotion import get_cfgs as locomotion_cfgs
from ..locomotion.config_common import get_final_command_cfg
from ...terrains.registry import level_catalog

TASK_NAMES = ("locomotion", "stairs_down", "stairs_up", "slope_up", "slope_down", "square_wave", "platform_ridge_down", "platform_ridge_up")


def _intermittent_cfg(probability, **overrides):
    """各地形按回合抽样；距离、时长等可在对应 task 中覆盖。"""
    return {**deepcopy(INTERMITTENT_DEFAULTS), "probability": probability, **overrides}


def _get_curriculum_cfg():
    """按阶段配置随机化、传感器噪声、任务权重与各地形命令范围。"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": "slope",
                "start_iteration": 0,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "wheel_slip": {"strength": 0.0},
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
                    "domain_rand": {"strength": 0.4},
                    "sensor_noise": {"strength": 0.4},
                    "wheel_slip": {"strength": 0.4},
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
                    "domain_rand": {"strength": 0.5},
                    "sensor_noise": {"strength": 0.5},
                    "wheel_slip": {"strength": 0.5},
                    "task_weights": {
                        "locomotion": 5.0,
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
                    "domain_rand": {"strength": 0.6},
                    "sensor_noise": {"strength": 0.6},
                    "wheel_slip": {"strength": 0.6},
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
    env, obs, reward, commands, flat_course = locomotion_cfgs()
    commands = get_final_command_cfg(commands, flat_course)
    # 合并原平地课程的全局奖励参数；各 task 的可调配置独立列在下方。
    for stage in flat_course["stages"]:
        targets = stage["targets"]
        reward["reward_scales"].update(targets.get("reward_scales", {}))
        reward["tracking_gate"].update(targets.get("tracking_gate", {}))
        reward.update({"standing_" + k: v for k, v in targets.get("standing_reward", {}).items()})
        env.update(targets.get("termination_limits", {}))
    # 每个任务集中配置；地形几何和难度参数统一由 terrains 提供。
    # 所有地形保留线速度/偏航角速度跟踪，并加大机身倾斜惩罚。
    tasks = {}

    # 平地运动
    tasks["locomotion"] = dict(
        enabled=True,
        weight=4.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[-3.2, 3.2],
            ang_vel_range=[-5.0, 5.0],
            base_height_range=[0.2, 0.38],
            high_speed_ang_vel=dict(lin_vel_threshold=1.5, max_abs_ang_vel=2.0),
            high_base_height_velocity_limits=dict(base_height_threshold=0.34, max_abs_lin_vel=1.5, max_abs_ang_vel=2.0),
            standing_probability=0.3,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            standing_drift=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            base_balance=-30.0,
            leg_symmetry=-10.0,
            leg_angle_limits=-5.0,
            leg_symmetry_bonus=2.0,
            base_height=-10.0,
            height_gate=10.0,
            joint_vel=-0.02,
            wheel_action_rate=-2.0,
            leg_action_rate=-0.15,
            landing_base_oscillation=-0.3,
            landing_joint_vel=-0.01,
            wheel_airborne=-10.0,
            base_contact=-15.0,
            alive=5.0,
            death=-100.0,
        ),
        terrain=dict(preset="plane", tile_size=[80.0, 80.0]),
        termination=dict(
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.1,
            episode_length_s=10.0,
        ),
        resampling_time_s=5.0,
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=False,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.8,
            demote_threshold=0.3,
        ),
        # 平地按完整回合的跟踪误差统计成功率。
        tracking_success=dict(
            lin_vel_rmse=0.5,
            ang_vel_rmse=1.0,
            height_rmse=0.1,
        ),
    )

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
            termination_if_roll_greater_than=30.0,
            termination_if_pitch_greater_than=30.0,
            tilt_termination_duration_s=0.3,
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
            leg_angle_limits=-5.0,
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
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
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
            leg_angle_limits=-5.0,
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
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.1,
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
            leg_angle_limits=-5.0,
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
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
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
            leg_angle_limits=-5.0,
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
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
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
            termination_if_roll_greater_than=45.0,
            termination_if_pitch_greater_than=45.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
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
        # 末端平台仅长 1.2 m，停车和重新起步使用更紧凑的距离。
        intermittent=_intermittent_cfg(
            probability=0.3,
            post_stop_distance_range_m=[0.25, 0.35],
            max_braking_distance_m=0.2,
            restart_distance_m=0.2,
        ),
        termination=dict(
            episode_length_s=15.0,
            termination_if_roll_greater_than=45.0,
            termination_if_pitch_greater_than=45.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
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

    env["multi_terrain"] = dict(tasks=tasks, level_catalog=level_catalog())
    obs["critic_task_one_hot"] = True
    env.update(
        episode_length_s=25.0,
        resampling_time_s=0.01,
        base_init_pos_range=[[-0.3, 0.3], [0.0, 0.0], [0.22, 0.22]],
        # 地形保持固定方向；出生 yaw 决定本回合斜向跨越的参考线。
        base_init_rpy_offset_range_deg=[[0.0, 0.0], [0.0, 0.0], [-10.0, 10.0]],
    )
    # 仅注册奖励函数，实际系数按各环境本回合所处的任务/课程阶段查表。
    reward["reward_scales"] = dict.fromkeys(set().union(*(task["rewards"] for task in tasks.values())), 0.0)
    return env, obs, reward, commands, _get_curriculum_cfg()
