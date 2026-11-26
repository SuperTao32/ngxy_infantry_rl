"""Nominal gas spring parameters for the wheelbipeV14_2 robot."""

GAS_SPRING_PRELOAD_FORCE = 400.0  # N
GAS_SPRING_STIFFNESS = 368.573  # N/m
GAS_SPRING_QUADRATIC_STIFFNESS = 39671.64  # N/m²
GAS_SPRING_DAMPING = 0.0  # N·s/m
GAS_SPRING_STROKE_AT_ZERO_POSITION = 0.043  # m
GAS_SPRING_MAX_COMPRESSION = 0.0675  # m


def gas_spring_force_at_compression(compression, velocity, *, preload_force, stiffness, quadratic_stiffness, damping):
    return preload_force + stiffness * compression + quadratic_stiffness * compression**2 - damping * velocity
