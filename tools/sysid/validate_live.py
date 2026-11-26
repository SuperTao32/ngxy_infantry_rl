"""读取 SysID 串口数据，实时比较实车、原始 MJCF 和辨识后的模型响应。"""

import argparse
from collections import deque
import importlib
import math
from pathlib import Path
from threading import Event, Lock, Thread
import time

if __package__:
    from .record_serial import DEFAULT_BAUD, DEFAULT_DATA_LEN, DEFAULT_PORT, SAMPLE_RATE_HZ, TARGET_CHANNELS, FrameDecoder
else:
    from record_serial import DEFAULT_BAUD, DEFAULT_DATA_LEN, DEFAULT_PORT, SAMPLE_RATE_HZ, TARGET_CHANNELS, FrameDecoder


class ModelResponse:
    def __init__(self, model, joint_names, actuator_names, initial_state):
        import mujoco

        self.model = model
        self.data = mujoco.MjData(model)
        self.dt = model.opt.timestep
        self.q_indices = [model.joint(name).qposadr[0] for name in joint_names]
        self.dq_indices = [model.joint(name).dofadr[0] for name in joint_names]
        self.inputs = [model.actuator(name).id for name in actuator_names]
        self.initial_state = initial_state

    def reset(self, measured_angles):
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        if self.initial_state is None:
            self.data.qpos[self.q_indices] = measured_angles
        else:
            mujoco.mj_setState(self.model, self.data, self.initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    def advance(self, seconds):
        import mujoco

        # 用上一帧的力矩推进到当前 MCU 时间。抖动时仅调整最后一个子步。
        whole_steps = math.floor((seconds + 1e-12) / self.dt)
        remainder = seconds - whole_steps * self.dt
        if whole_steps:
            mujoco.mj_step(self.model, self.data, nstep=whole_steps)
        if remainder > 1e-12:
            self.model.opt.timestep = remainder
            try:
                mujoco.mj_step(self.model, self.data)
            finally:
                self.model.opt.timestep = self.dt
        if self.data.warning.number.any():
            raise RuntimeError("MuJoCo reported an invalid state or solver warning")

    def sample(self, torque):
        self.data.ctrl[self.inputs] = torque
        return (*self.data.qpos[self.q_indices], *self.data.qvel[self.dq_indices])


def load_models(target, path):
    import mujoco
    from mujoco import sysid
    import numpy as np

    prefix = f"{__package__}." if __package__ else ""
    module = importlib.import_module(f"{prefix}{target}.sysid_{target}")
    cfg = module.CFG
    channels = TARGET_CHANNELS[target]
    path = path if path is not None else cfg.out / "identified_params.npz"
    if target == "wheels":
        spec = module.load_wheel_spec(cfg)
        joints, actuators = module.WHEEL_JOINTS, module.WHEEL_ACTUATORS
    else:
        spec = mujoco.MjSpec.from_file(str(cfg.model.resolve()))
        module.configure_torque_inputs(spec)
        module.configure_gas_spring(spec, module.SPRING_JOINT)
        joints, actuators = module.ACTIVE_JOINTS, module.ACTIVE_ACTUATORS

    params = module.build_params(cfg)
    with np.load(path, allow_pickle=False) as saved:
        for name, parameter in params.parameters.items():
            if name not in saved:
                raise ValueError(f"{path}: missing parameter {name}")
            value = np.asarray(saved[name], dtype=float)
            if value.size != 1 or not np.isfinite(value.item()) or value.item() < 0:
                raise ValueError(f"{path}: {name} must be a finite nonnegative scalar")
            parameter.value = value.reshape(parameter.value.shape)

    original = spec.compile()
    identified = sysid.apply_param_modifiers_spec(params, spec).compile()
    if target != "wheels":
        from tools.sysid.gas_spring import enable_gas_springs

        enable_gas_springs(original)
        enable_gas_springs(identified)
    state = None if target == "wheels" else module.make_initial_state(original, cfg.initial_state)
    responses = tuple(
        ModelResponse(model, [joints[c] for c in channels], [actuators[c] for c in channels], state)
        for model in (identified, original)
    )
    signs = tuple(getattr(cfg, f"{channel}_torque_sign") for channel in channels)
    return responses, signs, path


class LiveComparison:
    def __init__(self, responses, signs, window, max_gap_ms):
        self.responses = responses
        self.signs = signs
        self.window = window
        self.max_gap_ms = max_gap_ms
        self.decoder = FrameDecoder(self.on_frame)
        self.history = deque(maxlen=math.ceil(window * 1000) + 1)
        self.lock = Lock()
        self.segment = 0
        self.last_ms = None
        self.elapsed_ms = 0
        self.last_receive = None
        self.failure = None

    def on_frame(self, time_ms, values):
        measured = tuple(values[2:])
        torque = tuple(value * sign for value, sign in zip(values[:2], self.signs, strict=True))
        delta = None if self.last_ms is None else (time_ms - self.last_ms) & 0xFFFFFFFF
        self.last_ms = time_ms
        restart = delta is None or delta > self.max_gap_ms
        if restart:
            for response in self.responses:
                response.reset(measured[:2])
            self.elapsed_ms = 0
        elif delta:
            for response in self.responses:
                response.advance(delta / 1000)
            self.elapsed_ms += delta

        elapsed = self.elapsed_ms / 1000
        predicted = tuple(response.sample(torque) for response in self.responses)
        with self.lock:
            if restart:
                self.segment += 1
                self.history.clear()
            elif delta == 0:
                self.history.pop()
            self.history.append((elapsed, torque, (measured, *predicted)))
            self.last_receive = time.monotonic()
            while self.history[0][0] < elapsed - self.window:
                self.history.popleft()

    def receive(self, port, stop):
        try:
            while not stop.is_set():
                self.decoder.accept(port.read(min(4096, port.in_waiting or 1)))
        except (OSError, RuntimeError) as exc:
            with self.lock:
                self.failure = exc

    def snapshot(self):
        with self.lock:
            return tuple(self.history), self.segment, self.last_receive, self.failure


def plot_live(port, comparison, target, plt):
    import numpy as np

    channels = TARGET_CHANNELS[target]
    fig, axes = plt.subplots(3, len(channels), figsize=(12, 8), sharex=True, layout="constrained")
    fig.canvas.manager.set_window_title(f"SysID live validation — {target}")
    status = fig.suptitle(f"{target}  |  Waiting for serial frames")
    styles = (("Measured", "#20252b", "-"), ("Identified", "#0072b2", "--"), ("Original MJCF", "#d55e00", ":"))
    response_lines = []
    torque_lines = []
    for index, channel in enumerate(channels):
        axes[0, index].set_title(channel.capitalize())
        response_lines.append([])
        for row, unit in enumerate(("Angle [rad]", "Speed [rad/s]")):
            axis = axes[row, index]
            axis.set_ylabel(unit)
            response_lines[index].append([
                axis.plot([], [], label=label, color=color, linestyle=style)[0]
                for label, color, style in styles
            ])
        torque_axis = axes[2, index]
        torque_axis.set_ylabel("Torque input [Nm]")
        torque_axis.set_xlabel("MCU time in current segment [s]")
        torque_lines.append(torque_axis.plot([], [], color="#5b6875")[0])
    axes[0, 0].legend(loc="upper left", fontsize="small")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlim(0, comparison.window)

    def refresh():
        history, segment, last_receive, failure = comparison.snapshot()
        if failure is not None:
            link = f"Receive stopped: {failure}"
        elif last_receive is None:
            link = "Waiting for serial frames"
        elif time.monotonic() - last_receive > 1:
            link = "No recent serial frames"
        else:
            link = "Receiving"
        decoder = comparison.decoder
        status.set_text(
            f"{target}  |  {link}  |  segment {segment}  |  {decoder.rows} frames  "
            f"|  CRC {decoder.crc_errors}  |  trailer {decoder.trailer_errors}"
        )

        if history:
            times, torques, responses = map(np.asarray, zip(*history))
            for channel, line in enumerate(torque_lines):
                line.set_data(times, torques[:, channel])
                for quantity, lines in enumerate(response_lines[channel]):
                    column = quantity * len(channels) + channel
                    for response, line in enumerate(lines):
                        line.set_data(times, responses[:, response, column])
            for axis in axes.flat:
                axis.relim()
                axis.autoscale_view(scalex=False)
            end = max(comparison.window, times[-1])
            axes[2, 0].set_xlim(end - comparison.window, end)
        fig.canvas.draw_idle()

    stop = Event()
    worker = Thread(target=comparison.receive, args=(port, stop), name="sysid-serial")
    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(refresh)
    worker.start()
    timer.start()
    try:
        plt.show(block=True)
    finally:
        timer.stop()
        stop.set()
        worker.join()
        plt.close(fig)
    if comparison.failure is not None:
        raise comparison.failure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGET_CHANNELS, required=True, help="固件当前发送的数据对象")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--params", type=Path, help="辨识脚本生成的 identified_params.npz")
    parser.add_argument("--window", type=float, default=8, help="曲线的滚动窗口，单位秒")
    parser.add_argument("--max-gap-ms", type=int, default=100, help="力矩可保持的最大帧间隔，超出后开始新段")
    args = parser.parse_args(argv)
    if not math.isfinite(args.window) or args.window <= 0:
        parser.error("--window 必须为有限正数")
    if not 2 <= args.max_gap_ms <= 0x7FFFFFFF:
        parser.error("--max-gap-ms 必须在 2 到 2147483647 之间")
    minimum_baud = (DEFAULT_DATA_LEN + 5) * SAMPLE_RATE_HZ * 10
    if args.baud < minimum_baud:
        parser.error(f"500 Hz 的 SysID 数据至少需要 {minimum_baud} 波特率")

    try:
        import serial
        import matplotlib.pyplot as plt
        from matplotlib.backends import backend_registry
    except ImportError as exc:
        parser.exit(1, f"缺少依赖：{exc}。请在项目根目录运行 uv sync。\n")
    try:
        if backend_registry.resolve_backend(plt.get_backend())[1] is None:
            raise RuntimeError("实时绘图需要可交互的 Matplotlib 后端和显示窗口")
        responses, signs, path = load_models(args.target, args.params)
        comparison = LiveComparison(responses, signs, args.window, args.max_gap_ms)
        with serial.Serial(args.port, args.baud, timeout=0.02) as port:
            print(f"接收 {args.port} @ {args.baud}，辨识参数 {path}", flush=True)
            plot_live(port, comparison, args.target, plt)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"实时验证停止：{exc}\n")


if __name__ == "__main__":
    main()
