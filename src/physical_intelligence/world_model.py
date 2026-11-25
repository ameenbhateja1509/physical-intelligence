import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class WorldModel:
    """Learn a one-step transition model from simulated observations.

    Inputs are [x, y, vx, vy, ax, ay, dt].
    Outputs are [next_x, next_y, next_vx, next_vy].

    This is a learned dynamics predictor, not a general physical
    reasoning system. Its limits must be evaluated on held-out data.
    """

    def __init__(self, random_state: int = 42):
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=random_state,
            ),
        )
        self.is_trained = False

    def train(
        self,
        observations: np.ndarray,
        next_states: np.ndarray,
    ):
        if observations.ndim != 2 or observations.shape[1] != 7:
            raise ValueError("observations must have shape (n, 7)")
        if next_states.ndim != 2 or next_states.shape[1] != 4:
            raise ValueError("next_states must have shape (n, 4)")
        if len(observations) != len(next_states) or len(observations) == 0:
            raise ValueError("training inputs and targets must have equal nonzero length")
        if not np.isfinite(observations).all() or not np.isfinite(next_states).all():
            raise ValueError("training data must contain only finite values")

        self.model.fit(observations, next_states)
        self.is_trained = True
        return self

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("WorldModel must be trained before prediction")
        if observations.ndim != 2 or observations.shape[1] != 7:
            raise ValueError("observations must have shape (n, 7)")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")
        return self.model.predict(observations)
