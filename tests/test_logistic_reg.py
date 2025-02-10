import pytest
import torch
from editable_proj import LogisticRegModel

@pytest.fixture
def sample_input():
    return torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])

@pytest.fixture
def sample_output():
    return torch.tensor([[1.0], [0.0], [1.0]])

@pytest.fixture
def logistic_reg_model():
    seed = 42
    input_size = 2
    return LogisticRegModel(seed, input_size)

def test_logistic_reg_model_forward(logistic_reg_model, sample_input):
    output = logistic_reg_model(sample_input)
    assert output.shape == (3, 1), "Output shape is incorrect"

def test_logistic_reg_model_initialization(logistic_reg_model):
    assert logistic_reg_model.linear.weight is not None, "Weights are not initialized"
    assert logistic_reg_model.linear.bias is not None, "Bias is not initialized"

def test_logistic_reg_model_training(logistic_reg_model, sample_input, sample_output):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(logistic_reg_model.parameters(), lr=0.01)
    
    logistic_reg_model.train()
    optimizer.zero_grad()
    output = logistic_reg_model(sample_input)
    loss = criterion(output, sample_output)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0, "Loss should be greater than zero after one training step"