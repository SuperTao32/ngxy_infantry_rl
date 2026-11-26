"""训练与悬挂调试共用的气弹簧力曲线。"""


def gas_spring_force(position, velocity, *, preload_force, stiffness, quadratic_stiffness, damping, max_compression):
    q = position.clamp(0.0, max_compression)
    return preload_force + stiffness * q + quadratic_stiffness * q.square() - damping * velocity
