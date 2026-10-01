"""可滚动散落球：分层随机采样防止初始穿插，按环境独立复位。"""

import math

import torch

class LooseSpheres:
    def __init__(self, parameters, tile_size):
        self.config = parameters
        self.count = parameters["count"]
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("loose_spheres.count must be a positive integer")
        for name in ("diameter", "mass", "friction", "ground_friction"):
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"loose_spheres.{name} must be positive and finite")
        for name in ("friction", "ground_friction"):
            if not 0.01 <= parameters[name] <= 5.0:
                raise ValueError(f"loose_spheres.{name} must be in [0.01, 5.0]")
        if "density" in parameters:
            raise ValueError("loose_spheres.density is derived from mass and diameter; configure mass in kg instead")
        shore_a = float(parameters["shore_a"])
        if not math.isfinite(shore_a) or not 0 <= shore_a <= 100:
            raise ValueError("loose_spheres.shore_a must be finite and in [0, 100]")
        self.radius = float(parameters["diameter"]) / 2
        self.mass = float(parameters["mass"])
        self.density = self.mass / ((4.0 / 3.0) * math.pi * self.radius**3)
        size = tuple(float(v) for v in parameters["scatter_size"])
        if len(size) != 2 or any(not math.isfinite(v) or v <= 0 for v in size):
            raise ValueError("loose_spheres.scatter_size must contain two positive finite values")
        if any(v > tile for v, tile in zip(size, tile_size)):
            raise ValueError("loose_spheres.scatter_size must fit inside terrain.tile_size")
        clearance = float(parameters["spawn_clearance"])
        if not math.isfinite(clearance) or clearance < 0 or 2 * clearance >= min(size):
            raise ValueError("loose_spheres.spawn_clearance must be nonnegative and smaller than half the scatter size")

        # 随机抽取无重复格后在格内抖动；球心距格边至少一个半径，防止初始重叠。
        available_area = size[0] * size[1] - (2 * clearance) ** 2
        spacing = math.sqrt(available_area / (2 * self.count))
        nx, ny = (max(1, math.ceil(v / spacing)) for v in size)
        dx, dy = size[0] / nx, size[1] / ny
        if min(dx, dy) <= 2 * self.radius:
            raise ValueError("loose_spheres.count is too large for scatter_size and diameter")
        cells = []
        for ix in range(nx):
            for iy in range(ny):
                x, y = -size[0] / 2 + (ix + .5) * dx, -size[1] / 2 + (iy + .5) * dy
                if abs(x) - dx / 2 >= clearance or abs(y) - dy / 2 >= clearance:
                    cells.append((x, y))
        if len(cells) < self.count:
            raise ValueError("not enough loose_spheres cells outside spawn_clearance; reduce count or clearance")
        self.cells = torch.tensor(cells, dtype=torch.float64)
        self.jitter = (dx - 2 * self.radius, dy - 2 * self.radius)
        self.entities = ()

    def sample_positions(self, num_envs, *, device, dtype):
        cells = self.cells.to(device=device, dtype=dtype)
        indices = torch.rand((num_envs, len(cells)), device=device).topk(self.count, dim=1).indices
        xy = cells[indices]
        jitter = torch.tensor(self.jitter, device=device, dtype=dtype)
        xy = xy + (torch.rand(xy.shape, device=device, dtype=dtype) - .5) * jitter
        z = torch.full((*xy.shape[:-1], 1), self.radius, device=device, dtype=dtype)
        return torch.cat((xy, z), dim=-1)

    def add_to_scene(self, scene):
        import genesis as gs

        ground = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=self.config["ground_friction"]))
        positions = self.sample_positions(1, device="cpu", dtype=torch.float32)[0].tolist()
        self.entities = tuple(
            scene.add_entity(
                gs.morphs.Sphere(pos=tuple(pos), radius=self.radius, fixed=False),
                material=gs.materials.Rigid(rho=self.density, friction=self.config["friction"]),
                surface=gs.surfaces.Default(color=(0.65, 0.70, 0.78, 1.0)),
            )
            for pos in positions
        )
        return (ground, *self.entities)

    def reset(self, env_ids, *, dtype):
        if not env_ids.numel():
            return
        positions = self.sample_positions(env_ids.numel(), device=env_ids.device, dtype=dtype)
        qpos = torch.zeros((env_ids.numel(), self.count, 7), device=env_ids.device, dtype=dtype)
        qpos[:, :, :3] = positions
        qpos[:, :, 3] = 1.0
        for index, entity in enumerate(self.entities):
            entity.set_qpos(qpos[:, index], envs_idx=env_ids, zero_velocity=True, skip_forward=True)
