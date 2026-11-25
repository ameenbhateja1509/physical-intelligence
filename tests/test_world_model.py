import numpy as np
import pytest

from physical_intelligence.world_model import WorldModel


def test_world_model_requires_training_before_prediction():
    model = WorldModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 7)))


def test_world_model_rejects_wrong_input_shape():
    model = WorldModel()
    with pytest.raises(ValueError):
        model.train(np.zeros((5, 6)), np.zeros((5, 4)))


def test_world_model_trains_and_predicts_expected_shape():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(160, 7))
    y = rng.normal(size=(160, 4))

    model = WorldModel(random_state=7)
    model.train(x, y)
    predictions = model.predict(x[:8])

    assert predictions.shape == (8, 4)
    assert np.isfinite(predictions).all()
