"""复用 RSL-RL checkpoint 的 infos 保存地形成功率窗口与等级。"""


def terrain_runner_class(base):
    class TerrainRunner(base):
        def save(self, path, infos=None):
            infos = dict(infos or {})
            infos["terrain_curriculum"] = self.env.terrain_state_dict()
            return super().save(path, infos)

        def load(self, path, load_cfg=None, strict=True, map_location=None):
            infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
            # --load-weights 明确重启课程；完整恢复时在 set_training_iteration/reset 应用。
            if load_cfg is None or load_cfg.get("iteration", True):
                self.env._pending_terrain_state = (infos or {}).get("terrain_curriculum")
            return infos

    return TerrainRunner
