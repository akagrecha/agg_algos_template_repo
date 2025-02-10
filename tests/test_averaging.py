import pytest
import torch

from editable_proj import Averaging

# Fixtures to create common test data
@pytest.fixture
def sample_data():
    return torch.tensor([
        [0.2, 0.4, torch.nan],
        [0.8, 0.5, 0.3],
        [0.1, torch.nan, 0.1]
    ])

# Test for 'mean' algorithm
def test_mean_prediction(sample_data):
    policy = Averaging()
    prediction = policy.predict(sample_data)
    expected = torch.nanmean(sample_data, dim=1)
    torch.testing.assert_close(prediction, expected, msg="Prediction does not match what was expected.")

# Test for regularization and extremization effect
def test_regularization_effect(sample_data):
    reg_param = 0.5
    prior_mean = 0.3
    policy = Averaging(prior_mean=prior_mean, reg=reg_param)
    expected = torch.nanmean(sample_data, dim=1)
    expected = reg_param * prior_mean + (1 - reg_param) * expected
    prediction = policy.predict(sample_data)
    torch.testing.assert_close(prediction, expected, msg="Prediction does not match what was expected.")
