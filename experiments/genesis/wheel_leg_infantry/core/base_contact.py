"""按实际 box 碰撞面的局部位置统计机身底部、前侧、后侧接触。"""

import torch
import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat


class BaseContactRegions:
    """只读接触统计；不改变碰撞几何或完整机身接触判定。"""

    def __init__(self, link, num_envs, *, device, dtype, force_threshold):
        self.force_threshold = force_threshold
        self.contact = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.normal_force = torch.zeros_like(self.contact)
        self.geometries = []
        for geom in link.geoms:
            if geom.type != gs.GEOM_TYPE.BOX:
                raise ValueError("critic_base_contact requires box base-link collision geometries")
            vertices = torch.as_tensor(geom.init_verts, device=device, dtype=dtype)
            lower, upper = vertices.amin(0), vertices.amax(0)
            pos = torch.as_tensor(geom.init_pos, device=device, dtype=dtype)
            quat = torch.as_tensor(geom.init_quat, device=device, dtype=dtype)
            # 碰撞体自身可能有偏移和旋转（当前模型的 box z 轴几乎翻转）。
            # 面的语义按机体系 -z / +x / -x 定义，而非直接按 box 轴命名。
            normals = torch.cat((-torch.eye(3), torch.eye(3))).to(self.contact)
            body_normals = transform_by_quat(normals, quat.expand(6, -1))
            axis = body_normals.abs().argmax(-1)
            face_regions = torch.full((6,), -1, device=device, dtype=torch.long)
            face_regions[(axis == 2) & (body_normals[:, 2] < 0)] = 0
            face_regions[(axis == 0) & (body_normals[:, 0] > 0)] = 1
            face_regions[(axis == 0) & (body_normals[:, 0] < 0)] = 2
            self.geometries.append((geom.idx, pos, inv_quat(quat), (lower + upper) / 2,
                                    (upper - lower) / 2, face_regions))
        if not self.geometries:
            raise ValueError("critic_base_contact requires a base-link collision geometry")

    def update(self, contacts, base_pos, base_quat):
        self.normal_force.zero_()
        body_inv = inv_quat(base_quat)[:, None, :]
        body_position = transform_by_quat(contacts["position"] - base_pos[:, None, :], body_inv)
        body_normal = transform_by_quat(contacts["normal"], body_inv)
        normal_length = contacts["normal"].norm(dim=-1)
        for geom_idx, pos, quat_inv, center, half_size, face_regions in self.geometries:
            is_a = contacts["geom_a"] == geom_idx
            valid = (contacts["valid_mask"] & (is_a | (contacts["geom_b"] == geom_idx))
                     & (normal_length > 1e-12))
            local_position = transform_by_quat(body_position - pos, quat_inv) - center
            local_normal = transform_by_quat(body_normal, quat_inv)
            # 最近的实际 box 面决定部位；棱边 1 mm 内的并列面由法线方向消歧。
            # 每个接触点只归属一个面，避免棱角接触重复计数。
            distance = (local_position.abs() - half_size).abs()
            nearest = distance <= distance.amin(-1, keepdim=True) + .001
            axis = torch.where(nearest, local_normal.abs(), -1.).argmax(-1)
            positive = local_position.gather(-1, axis[..., None]).squeeze(-1) >= 0
            region = face_regions[axis + positive.long() * 3]
            force = torch.where(is_a[..., None], contacts["force_a"], contacts["force_b"])
            magnitude = (force * contacts["normal"]).sum(-1).abs() / normal_length.clamp_min(1e-12)
            for index in range(3):
                self.normal_force[:, index].add_(torch.where(valid & (region == index), magnitude, 0.).sum(1))
        self.contact.copy_(self.normal_force > self.force_threshold)

    def reset(self, env_mask=None):
        if env_mask is None:
            self.contact.zero_()
            self.normal_force.zero_()
        else:
            self.contact.masked_fill_(env_mask[:, None], 0.)
            self.normal_force.masked_fill_(env_mask[:, None], 0.)
