"""MuJoCo 单点 ToF：使用碰撞几何，和 Genesis Raycaster 保持一致。"""

import mujoco
import numpy as np
import torch

from experiments.genesis.wheel_leg_infantry.core.tof import sanitize_tof_distances, tof_site_quat


def add_tof_sites(spec, cfg):
    if not cfg["enabled"]:
        return
    body = spec.body(cfg["link_name"])
    for sensor in cfg["sensors"]:
        site = spec.site(sensor["name"] + "_site")
        if site is None:
            site = body.add_site(name=sensor["name"] + "_site", size=[0.008, 0.008, 0.008])
        elif site.parent.name != body.name:
            raise ValueError(f"{site.name} is attached to a different ToF link")
        site.pos = sensor["pos_offset"]
        site.quat = tof_site_quat(sensor)
        site.rgba = [0.2, 0.9, 0.3, 1]


class MujocoToF:
    def __init__(self, model, data, cfg):
        self.model, self.data, self.cfg = model, data, cfg
        self.site_ids = [model.site(sensor["name"] + "_site").id for sensor in cfg["sensors"]]
        self.geomgroup = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)
        self.collision_ids = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
        self.geomid = np.empty(1, dtype=np.int32)

    def read(self):
        # mj_ray 忽略 alpha=0 的几何。查询期间临时启用不可见碰撞盒，
        # 并排除 group=2 的展示网格；finally 恢复 alpha，不改变渲染或动力学。
        alphas = self.model.geom_rgba[self.collision_ids, 3].copy()
        self.model.geom_rgba[self.collision_ids, 3] = 1
        try:
            distances = [mujoco.mj_ray(
                self.model, self.data, self.data.site_xpos[site_id],
                self.data.site_xmat[site_id].reshape(3, 3)[:, 2].copy(),
                self.geomgroup, True, -1, self.geomid,
            ) for site_id in self.site_ids]
        finally:
            self.model.geom_rgba[self.collision_ids, 3] = alphas
        return sanitize_tof_distances(torch.tensor(distances, dtype=torch.float32), self.cfg)
