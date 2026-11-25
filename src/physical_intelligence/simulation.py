
from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleState:
    """Simple 2D vehicle state. Positions are metres,
    velocities are metres/second, and time is seconds."""

    x: float
    y: float
    vx: float
    vy: float
    time_s: float = 0.0


@dataclass
class VehicleAction:
    """Acceleration command in metres/second squared."""

    ax: float
    ay: float


def step(
    state: VehicleState,
    action: VehicleAction,
    dt: float = 0.1,
) -> VehicleState:
    """Advance the simulated vehicle by one time step.

    This is a simplified point-mass model, not a realistic
    vehicle dynamics or steering model.
    """

    if dt <= 0:
        raise ValueError("dt must be positive")

    new_x = state.x + state.vx * dt + 0.5 * action.ax * dt**2
    new_y = state.y + state.vy * dt + 0.5 * action.ay * dt**2

    new_vx = state.vx + action.ax * dt
    new_vy = state.vy + action.ay * dt

    return VehicleState(
        x=new_x,
        y=new_y,
        vx=new_vx,
        vy=new_vy,
        time_s=state.time_s + dt,
    )


def state_to_array(state: VehicleState) -> np.ndarray:
    """Convert state to a consistent numerical representation."""

    return np.array(
        [state.x, state.y, state.vx, state.vy],
        dtype=float,
    )


def generate_dataset(
    samples: int,
    seed: int,
):
    """Generate reproducible simulated transitions."""

    rng = np.random.default_rng(seed)

    features = []
    targets = []

    for _ in range(samples):
        state = VehicleState(
            x=rng.uniform(-10, 10),
            y=rng.uniform(-10, 10),
            vx=rng.uniform(-3, 3),
            vy=rng.uniform(-3, 3),
        )

        action = VehicleAction(
            ax=rng.uniform(-1, 1),
            ay=rng.uniform(-1, 1),
        )

        dt = rng.uniform(0.05, 0.2)

        next_state = step(state, action, dt)

        # Inputs: current state, acceleration, and time interval.
        x = [
            state.x,
            state.y,
            state.vx,
            state.vy,
            action.ax,
            action.ay,
            dt,
        ]

        # Target: next state.
        y = state_to_array(next_state)

        features.append(x)
        targets.append(y)

    return np.asarray(features), np.asarray(targets)
