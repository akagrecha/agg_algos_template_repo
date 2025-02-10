import pytest
import torch
from editable_proj import TwoLayerMLP

@pytest.fixture
def sample_input():
    return torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)

@pytest.fixture
def mlp_model():
    return TwoLayerMLP(seed=42, input_size=3, hidden_size=5, output_size=1, dropout_prob=0.5)

def test_forward_pass(mlp_model, sample_input):
    output = mlp_model(sample_input)
    assert output.shape == (2, 1), "Output shape is incorrect"

def test_weight_initialization(mlp_model):
    generator = torch.Generator()
    generator.manual_seed(42)
    expected_fc1_weight = torch.empty(mlp_model.fc1.weight.shape)
    expected_fc2_weight = torch.empty(mlp_model.fc2.weight.shape)
    torch.nn.init.kaiming_normal_(expected_fc1_weight, nonlinearity='relu', generator=generator)
    torch.nn.init.kaiming_normal_(expected_fc2_weight, generator=generator)
    assert torch.allclose(mlp_model.fc1.weight, expected_fc1_weight), "fc1 weights are not initialized correctly"
    assert torch.allclose(mlp_model.fc2.weight, expected_fc2_weight), "fc2 weights are not initialized correctly"

def test_dropout_effect(mlp_model, sample_input):
    mlp_model.train()
    output_train = mlp_model(sample_input)
    mlp_model.eval()
    output_eval = mlp_model(sample_input)
    assert not torch.allclose(output_train, output_eval), "Dropout is not applied correctly during training"