"""评估的 locomotion 键盘输入；Viewer 线程只入队，仿真线程写 buffer。"""

from queue import Empty, SimpleQueue

from genesis.vis.keybindings import Key, KeyAction, Keybind


class KeyboardCommand:
    def __init__(self, env):
        self.env = env
        self.requests = SimpleQueue()
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
        for name, key in (("stop", Key.SPACE), ("reset", Key.BACKSPACE)):
            bindings.append(Keybind(
                f"command_{name}", key, key_action=KeyAction.PRESS,
                callback=self.requests.put, args=((name, 0.0),), allow_overload=False,
            ))
        env.scene.viewer.register_keybinds(*bindings, overwrite=True)

    def update(self):
        reset = False
        while True:
            try:
                axis, amount = self.requests.get_nowait()
            except Empty:
                break
            if axis == "reset":
                self.env.reset()
                self.env.focus_viewer()
                reset = True
                self.command = [0.0, 0.0, float(self.env.commands[0, 2])]
            elif axis == "stop":
                self.command[:2] = [0.0, 0.0]
            else:
                lower, upper = self.ranges[axis]
                self.command[axis] = min(max(self.command[axis] + amount, lower), upper)
        selected = ~self.env.recovery.buffers.active & ~self.env.jump.buffers.active
        self.env.commands[selected] = self.env.commands.new_tensor(self.command)
        return reset
