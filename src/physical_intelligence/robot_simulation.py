"""A simple 2D mobile-robot benchmark with linear velocity drag.

This deliberately differs from the ideal vehicle benchmark: robot motion
includes a small drag term. It is a simplified point-mass model, not a
differential-drive or hardware-accurate robot simulator.
"""
import numpy as np


ROBOT_DRAG = 0.25


def robot_step(state, action, dt: float, drag: float = ROBOT_DRAG):
    """Return [next_x, next_y, next_vx, next_vy]."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if drag < 0:
        raise ValueError("drag cannot be negative")

    x, y, vx, vy = state
    ax, ay = action
    effective_ax = ax - drag * vx
    effective_ay = ay - drag * vy

    return np.array([
        x + vx * dt + 0.5 * effective_ax * dt**2,
        y + vy * dt + 0.5 * effective_ay * dt**2,
        vx + effective_ax * dt,
        vy + effective_ay * dt,
    ], dtype=float)


def generate_robot_dataset(samples: int, seed: int, drag: float = ROBOT_DRAG):
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    features, targets = [], []

    for _ in range(samples):
        state = np.array([
            rng.uniform(-10, 10),
            rng.uniform(-10, 10),
            rng.uniform(-3, 3),
            rng.uniform(-3, 3),
        ])
        action = np.array([
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
        ])
        dt = rng.uniform(0.05, 0.2)
        next_state = robot_step(state, action, dt, drag)

        features.append([*state, *action, dt])
        targets.append(next_state)

    return np.asarray(features), np.asarray(targets)


def robot_physics_predict(features: np.ndarray, drag: float = ROBOT_DRAG):
    x, y, vx, vy, ax, ay, dt = features.T
    effective_ax = ax - drag * vx
    effective_ay = ay - drag * vy
    return np.column_stack([
        x + vx * dt + 0.5 * effective_ax * dt**2,
        y + vy * dt + 0.5 * effective_ay * dt**2,
        vx + effective_ax * dt,
        vy + effective_ay * dt,
    ])
