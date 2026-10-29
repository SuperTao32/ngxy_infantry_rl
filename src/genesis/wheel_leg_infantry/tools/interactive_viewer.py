"""统一 eval 的键盘控制和 Viewer 实时诊断组件。"""

import time

from genesis.ext.pyrender.constants import FONT_SIZE, TEXT_PADDING, TextAlign
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.vis.viewer_plugins import ViewerPlugin

from .tof_viewer import tof_overlay_lines


class DiagnosticsOverlay(ViewerPlugin):
    """Draw persistent evaluation diagnostics as separate lines."""

    font_size = 25

    def __init__(self):
        super().__init__()
        self._lines = ()

    def update(self, lines, *, colors=None):
        # The simulation and viewer run in different threads. Replacing an
        # immutable tuple keeps the viewer from observing a partially updated list.
        colors = colors or {}
        self._lines = tuple((line, colors.get(index)) for index, line in enumerate(lines))

    def on_draw(self):
        viewer = self.viewer
        lines = self._lines
        if viewer is None or viewer._renderer is None or not lines:
            return

        x = viewer._viewport_size[0] - TEXT_PADDING
        # Leave the first row free for Genesis' keyboard-help prompt.
        y = viewer._viewport_size[1] - TEXT_PADDING - int(FONT_SIZE * 1.2)
        line_height = int(self.font_size * 1.2)
        for index, (line, color) in enumerate(lines):
            viewer._renderer.render_text(
                line,
                x,
                y - index * line_height,
                font_name="SpaceMono-Regular",
                font_pt=self.font_size,
                color=viewer._font_color if color is None else color,
                align=TextAlign.TOP_RIGHT,
            )


class KeyboardCommand:
    """保存键盘指令；满足一个 Viewer 线程和一个仿真线程的使用场景。"""

    def __init__(self, env):
        self.env = env
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.height_target = float(env.commands[0, 2].item())
        self.lin_step = 0.5
        self.ang_step = 0.5
        self.height_step = 0.05
        self._last_slip_seen = [float("-inf"), float("-inf")]
        self._last_slip_ratio = [1.0, 1.0]

        viewer = env.scene.viewer
        self.viewer = viewer
        self.diagnostics_overlay = viewer.add_plugin(DiagnosticsOverlay())
        viewer.register_keybinds(
            self._keybind("command_forward", Key.UP, self._change_lin, self.lin_step),
            self._keybind("command_backward", Key.DOWN, self._change_lin, -self.lin_step),
            self._keybind("command_turn_left", Key.LEFT, self._change_ang, self.ang_step),
            self._keybind("command_turn_right", Key.RIGHT, self._change_ang, -self.ang_step),
            self._keybind("command_raise_body", Key.PAGEUP, self._change_height, self.height_step),
            self._keybind("command_lower_body", Key.PAGEDOWN, self._change_height, -self.height_step),
            Keybind(
                "command_stop",
                Key.SPACE,
                key_action=KeyAction.PRESS,
                callback=self.stop,
            ),
        )

    @staticmethod
    def _keybind(name, key, callback, amount):
        return Keybind(
            name,
            key,
            key_action=KeyAction.PRESS,
            callback=callback,
            args=(amount,),
        )

    def _change_lin(self, amount):
        lower, upper = self.env.command_cfg["lin_vel_range"]
        self.lin_vel = min(max(self.lin_vel + amount, lower), upper)

    def _change_ang(self, amount):
        lower, upper = self.env.command_cfg["ang_vel_range"]
        self.ang_vel = min(max(self.ang_vel + amount, lower), upper)

    def _change_height(self, amount):
        lower, upper = self.env.command_cfg["base_height_range"]
        self.height_target = min(max(self.height_target + amount, lower), upper)

    def stop(self):
        self.lin_vel = 0.0
        self.ang_vel = 0.0

    def write_to_env(self):
        self.env.commands[:, 0] = self.lin_vel
        self.env.commands[:, 1] = self.ang_vel
        self.env.commands[:, 2] = self.height_target

    def _wheel_slip_caption(self):
        """显示实际施加的低摩擦事件；结束后保留两秒提示，避免短事件漏看。"""
        slip = self.env.wheel_slip
        if not slip.enabled:
            return "wheel slip OFF | enable with --wheel-slip", None
        diagnostics = slip.diagnostics()
        if diagnostics["wheel_slip_episode_strength"].item() <= 0:
            return "wheel slip OFF | episode strength=0", None

        now = time.monotonic()
        active = diagnostics["wheel_slip_active"].cpu().tolist()
        ratios = diagnostics["wheel_slip_ratio"].cpu().tolist()
        remaining = diagnostics["wheel_slip_steps_remaining"].cpu().tolist()
        next_event = diagnostics["wheel_slip_steps_until_next"].cpu().tolist()
        wheels = []
        recent = False
        for index, side in enumerate(("L", "R")):
            if active[index]:
                self._last_slip_seen[index] = now
                self._last_slip_ratio[index] = ratios[index]
                wheels.append(f"{side}=ACTIVE x{ratios[index]:.2f} {remaining[index] * self.env.dt:.2f}s")
            elif now - self._last_slip_seen[index] < 2.0:
                recent = True
                wheels.append(f"{side}=ENDED (last x{self._last_slip_ratio[index]:.2f})")
            else:
                wheels.append(f"{side}=READY in {next_event[index] * self.env.dt:.1f}s")
        color = (1.0, 0.25, 0.15, 1.0) if any(active) else ((1.0, 0.7, 0.15, 1.0) if recent else None)
        return "wheel slip | " + "  ".join(wheels), color

    def update_caption(self):
        env = self.env
        obs = env.obs_buf[0]
        pd = env.get_pd_diagnostics()
        slip_text, slip_color = self._wheel_slip_caption()
        terrain_type = env.terrain.tile_type(env.terrain_tile_index[0].item())
        text = (
            f"command  vx={self.lin_vel:+.2f} m/s  wz={self.ang_vel:+.2f} rad/s  " f"height={self.height_target:.3f} m",
            f"state    vx_true={env.base_lin_vel[0, 0].item():+.2f} m/s  "
            f"wz={env.base_ang_vel[0, 2].item():+.2f} rad/s",
            slip_text,
            f"pose     z={env.base_pos[0, 2].item():.3f} m  ground={env.terrain_height[0].item():.3f} m  "
            f"height={env.base_height[0].item():.3f} m",
            f"terrain  preset={env.terrain.preset}  patch={terrain_type}  "
            f"roll={env.base_euler[0, 0].item():+.1f} deg  pitch={env.base_euler[0, 1].item():+.1f} deg",
            f"legs     left={env.leg_length[0, 0].item():.3f} m  right={env.leg_length[0, 1].item():.3f} m",
            f"obs      n={obs.numel()}  min={obs.min().item():+.2f}  max={obs.max().item():+.2f}  " f"mean={obs.mean().item():+.2f}",
            *tof_overlay_lines(env),
            f"joint Kd       {format_tensor(pd['joint_kd'], precision=2)}",
            f"joint -Kd*qdot {format_tensor(pd['joint_kd_damping'], precision=2)}",
        )
        self.diagnostics_overlay.update(text, colors={2: slip_color})


def format_tensor(tensor, precision=3):
    values = tensor.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{value:+.{precision}f}" for value in values) + "]"
