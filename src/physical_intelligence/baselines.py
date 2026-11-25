
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def physics_predict(features: np.ndarray) -> np.ndarray:
    """Predict next state using the known motion equations."""

    x = features[:, 0]
    y = features[:, 1]
    vx = features[:, 2]
    vy = features[:, 3]
    ax = features[:, 4]
    ay = features[:, 5]
    dt = features[:, 6]

    next_x = x + vx * dt + 0.5 * ax * dt**2
    next_y = y + vy * dt + 0.5 * ay * dt**2

    next_vx = vx + ax * dt
    next_vy = vy + ay * dt

    return np.column_stack(
        [next_x, next_y, next_vx, next_vy]
    )


def train_existing_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
):
    """Train an established machine-learning model."""

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train_features, train_targets)
    return model


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calculate separate position and velocity errors."""

    errors = actual - predicted

    position_error = np.mean(errors[:, :2] ** 2)
    velocity_error = np.mean(errors[:, 2:] ** 2)

    return {
        "position_mse": float(position_error),
        "velocity_mse": float(velocity_error),
        "overall_mse": float(np.mean(errors**2)),
    }
