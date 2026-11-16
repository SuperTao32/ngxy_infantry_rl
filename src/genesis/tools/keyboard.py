"""评估的 locomotion、jump 和 recovery 输入；Viewer 只入队，仿真线程写 buffer。"""

from queue import Empty, SimpleQueue

from genesis.vis.keybindings import Key, KeyAction, Keybind

from ..tasks.jump.config import MODE_NAMES


class KeyboardCommand:
    def __init__(self, env):
        self.env = env
        self.requests = SimpleQueue()
        self.selected_jump_mode = 0
        self.command = [0.0, 0.0, float(env.commands[0, 2])]
        self.ranges = [list(env.command_cfg[name]) for name in ("lin_vel_range", "ang_vel_range", "base_height_range")]
        for limits in self.ranges[:2]:
            limits[0], limits[1] = min(0.0, limits[0]), max(0.0, limits[1])
        bindings = []
        for key, axis, amount in (
            (Key.UP, 0, 0.5), (Key.DOWN, 0, -0.5), (Key.LEFT, 1, 0.5), (Key.RIGHT, 1, -0.5),
            (Key.PAGEUP, 2, 0.05), (Key.PAGEDOWN, 2, -0.05),
        ):
            bindings.append(Keybind(
                f"command_{key.name.lower()}", key, key_action=KeyAction.PRESS,
                callback=self.requests.put, args=((axis, amount),), allow_overload=False,
            ))
        for name, key, amount in (
            ("stop", Key.SPACE, 0), ("reset", Key.BACKSPACE, 0),
            ("jump", Key.J, 0), ("recovery", Key.R, 0),
            ("jump_mode", Key._1, 0), ("jump_mode", Key._2, 1),
        ):
            bindings.append(Keybind(
                f"command_{name}_{key.name.lower()}", key, key_action=KeyAction.PRESS,
                callback=self.requests.put, args=((name, amount),), allow_overload=False,
            ))
        env.scene.viewer.register_keybinds(*bindings, overwrite=True)

    def update(self):
        reset = False
        while True:
            try:
                axis, amount = self.requests.get_nowait()
            except Empty:
                break
            if axis in ("reset", "recovery"):
                if axis == "recovery" and self.env.recovery.reset_sampler is None:
                    print("[eval] recovery unavailable: no configured recovery pose pool")
                    continue
                self.env.reset(recovery=axis == "recovery")
                self.env.focus_viewer()
                reset = True
                self.command = [0.0, 0.0, float(self.env.commands[0, 2])]
                if axis == "recovery":
                    sampler = self.env.recovery.reset_sampler
                    pose_id = int(sampler.pose_ids[0])
                    print(f"[eval] recovery: {sampler.pose_names[pose_id]}" if pose_id >= 0 else "[eval] terrain reset: default pose")
            elif axis == "jump_mode":
                self.selected_jump_mode = amount
                print(f"[eval] jump mode: {MODE_NAMES[amount]}")
            elif axis == "jump":
                self._write_command()
                selected = self.env.jump.trigger(~self.env.recovery.buffers.active, self.selected_jump_mode)
                if selected.any():
                    print(f"[eval] jump: {MODE_NAMES[self.selected_jump_mode]}")
                else:
                    print("[eval] jump ignored: recovery/jump is active or terrain commands take priority")
            elif axis == "stop":
                self.command[:2] = [0.0, 0.0]
            else:
                lower, upper = self.ranges[axis]
                self.command[axis] = min(max(self.command[axis] + amount, lower), upper)
        self._write_command()
        return reset

    def _write_command(self):
        selected = ~self.env.recovery.buffers.active & ~self.env.jump.buffers.active & ~self.env.terrain_cross.assigned
        self.env.commands[selected] = self.env.commands.new_tensor(self.command)
