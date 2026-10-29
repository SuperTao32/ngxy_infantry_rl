"""可由统一评估调用的平地制动测量与汇总函数。"""

import math
from itertools import product
from statistics import mean, pstdev

WHEEL_SIDES = ("left", "right")
WHEEL_METRICS = ("left_wheel_max_forward_distance_m", "right_wheel_max_forward_distance_m",
                 "wheel_max_forward_distance_m", "wheel_max_forward_from_base_m")


def _state(env):
    # 真值用于计量，不改策略输入；轮速不能代表打滑时的实际车速。
    position = env.base_pos[0].detach().cpu().tolist()
    velocity = env.base_lin_vel[0].detach().cpu().tolist()
    angular = env.base_ang_vel[0].detach().cpu().tolist()
    euler = env.base_euler[0].detach().cpu().tolist()
    # wheel link 原点位于轮轴；使用与 base_pos 相同的坐标系，不从轮速积分。
    wheels = env.robot.get_links_pos(links_idx_local=env.wheel_links_idx)[0].detach().cpu().tolist()
    return dict(x=position[0], y=position[1], z=position[2], vx=velocity[0], vy=velocity[1],
                speed=math.hypot(*velocity), angular_speed=math.hypot(*angular),
                tilt=max(abs(euler[0]), abs(euler[1])), yaw=euler[2],
                base_contact=float(env.base_contact[0].item()),
                **{f"{side}_wheel_{axis}": value for side, point in zip(WHEEL_SIDES, wheels)
                   for axis, value in zip("xyz", point)})


def _healthy(state, args):
    return state["tilt"] <= args.max_tilt and state["base_contact"] < 0.5


class StopMeasurement:
    """累积世界水平面路程；低速过零或反弹会重新开始停稳确认。"""

    def __init__(self, state, target_speed, dt, args):
        self.origin = self.previous = (state["x"], state["y"])
        yaw = math.radians(state["yaw"])
        sign = -1 if target_speed < 0 else 1
        self.direction = (sign * math.cos(yaw), sign * math.sin(yaw))
        self.wheel_origins = {side: (state[f"{side}_wheel_x"], state[f"{side}_wheel_y"]) for side in WHEEL_SIDES}
        self.wheel_max_forward = dict.fromkeys(WHEEL_SIDES, 0.0)
        # 同时提供相对于制动起点 base 的最远坐标，包含轮子已有的前后偏置。
        # 初始帧也参与最大值：制动后立即回退时不能丢掉起点的前伸量。
        self.wheel_max_from_base = max(self._project(point, self.origin) for point in self.wheel_origins.values())
        self.path = self.max_forward = self.max_excursion = 0.0
        self.displacement = self.forward = 0.0
        self.steps = self.stable_steps = 0
        self.dt, self.args = dt, args
        self.required_steps = math.ceil(args.stop_hold_time / dt)

    def _project(self, point, origin):
        return sum((p - o) * d for p, o, d in zip(point, origin, self.direction))

    def update(self, state):
        point = (state["x"], state["y"])
        self.path += math.dist(self.previous, point)
        self.previous = point
        self.displacement = math.dist(self.origin, point)
        self.forward = self._project(point, self.origin)
        self.max_forward = max(self.max_forward, self.forward)
        self.max_excursion = max(self.max_excursion, self.displacement)
        for side in WHEEL_SIDES:
            wheel = (state[f"{side}_wheel_x"], state[f"{side}_wheel_y"])
            self.wheel_max_forward[side] = max(self.wheel_max_forward[side], self._project(wheel, self.wheel_origins[side]))
            self.wheel_max_from_base = max(self.wheel_max_from_base, self._project(wheel, self.origin))
        self.steps += 1
        stable = (_healthy(state, self.args) and state["speed"] <= self.args.stop_speed
                  and state["angular_speed"] <= self.args.stop_angular_speed)
        self.stable_steps = self.stable_steps + 1 if stable else 0
        return self.stable_steps >= self.required_steps

    def result(self):
        return dict(brake_elapsed_s=self.steps * self.dt, path_distance_m=self.path,
                    displacement_m=self.displacement, forward_distance_m=self.forward,
                    max_forward_distance_m=self.max_forward, max_excursion_m=self.max_excursion,
                    left_wheel_max_forward_distance_m=self.wheel_max_forward["left"],
                    right_wheel_max_forward_distance_m=self.wheel_max_forward["right"],
                    wheel_max_forward_distance_m=max(self.wheel_max_forward.values()),
                    wheel_max_forward_from_base_m=self.wheel_max_from_base)



def run_trial(env, policy, speed, trial, height, args, trace):
    """单次试验；done 后环境已自动 reset，必须在读取位置前退出。"""
    env.reset()
    if callable(getattr(policy, "reset", None)):
        policy.reset()
    result = dict(target_height_m=height, target_speed_mps=speed, trial=trial, status="", actual_brake_vx_mps=None,
                  actual_brake_speed_mps=None, brake_elapsed_s=None, stop_time_s=None,
                  path_distance_m=None, displacement_m=None, forward_distance_m=None,
                  max_forward_distance_m=None, max_excursion_m=None,
                  **dict.fromkeys(WHEEL_METRICS))
    total_steps = 0

    def step(command, phase, phase_step):
        nonlocal total_steps
        env.commands[:, 0] = command
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = height
        env._update_observations()
        actions = policy(env.get_observations())
        _, _, dones, infos = env.step(actions)
        total_steps += 1
        if bool(dones[0].item()):
            timed_out = bool(infos["time_outs"][0].item())
            return None, f"{phase}_{'env_timeout' if timed_out else 'terminated'}"
        state = _state(env)
        if not all(math.isfinite(value) for value in state.values()):
            return None, f"{phase}_nonfinite"
        trace.writerow(dict(target_height_m=height, target_speed_mps=speed, trial=trial, phase=phase,
                            time_s=total_steps * env.dt, phase_time_s=phase_step * env.dt,
                            command_vx_mps=command, **state))
        return state, None

    for i in range(math.ceil(args.settle_time / env.dt)):
        state, failure = step(0.0, "settle", i + 1)
        if failure:
            result["status"] = failure
            return result

    held = 0
    for i in range(math.ceil(args.accel_timeout / env.dt)):
        state, failure = step(speed, "accel", i + 1)
        if failure:
            result["status"] = failure
            return result
        tracking = (abs(state["vx"] - speed) <= args.speed_tolerance
                    and abs(state["vy"]) <= args.speed_tolerance
                    and state["angular_speed"] <= args.stop_angular_speed and _healthy(state, args))
        held = held + 1 if tracking else 0
        if held >= math.ceil(args.speed_hold_time / env.dt):
            break
    else:
        result["status"] = "speed_not_reached"
        return result

    result.update(actual_brake_vx_mps=state["vx"], actual_brake_speed_mps=state["speed"])
    measurement = StopMeasurement(state, speed, env.dt, args)
    # t=0 就在下一次推理前：保留控制器原有动作延迟及所有策略/传感器历史。
    trace.writerow(dict(target_height_m=height, target_speed_mps=speed, trial=trial, phase="brake", time_s=total_steps * env.dt,
                        phase_time_s=0.0, command_vx_mps=0.0, **state))
    for i in range(math.ceil(args.stop_timeout / env.dt)):
        state, failure = step(0.0, "brake", i + 1)
        if failure:
            result["status"] = failure
            # 失败不报告不完整的路程为刹停距离。
            return result
        if measurement.update(state):
            result.update(status="stopped", stop_time_s=measurement.steps * env.dt, **measurement.result())
            return result
    result.update(status="stop_timeout", **measurement.result())
    return result


def summarize(results, speeds, heights):
    summaries = []
    for height, speed in product(heights, speeds):
        rows = [row for row in results if row["target_height_m"] == height and row["target_speed_mps"] == speed]
        stopped = [row for row in rows if row["status"] == "stopped"]
        summary = dict(target_height_m=height, target_speed_mps=speed, trials=len(rows),
                       stopped=len(stopped), failed=len(rows) - len(stopped))
        for name in ("actual_brake_vx_mps", "path_distance_m", "max_forward_distance_m", "stop_time_s", *WHEEL_METRICS):
            values = [row[name] for row in stopped]
            for stat, fn in (("mean", mean), ("std", pstdev), ("max", max)):
                summary[f"{name}_{stat}"] = fn(values) if values else None
        summaries.append(summary)
    return summaries
