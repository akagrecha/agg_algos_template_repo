import pytest
import torch
from editable_proj import ReplaceNanDataset

@pytest.fixture
def sample_data():
    x = torch.tensor([
        [0.2, 0.4, torch.nan],
        [0.8, torch.nan, 0.3],
        [torch.nan, 0.5, 0.1]
    ])
    y = torch.tensor([1, 0, 1])
    return x, y

def test_replace_nan_dataset_length(sample_data):
    x, y = sample_data
    dataset = ReplaceNanDataset(x, y)
    assert len(dataset) == len(y), "Dataset length does not match the length of the target tensor."

def test_replace_nan_dataset_getitem(sample_data):
    x, y = sample_data
    nan_replace = -1.0
    dataset = ReplaceNanDataset(x, y, nan_replace=nan_replace)
    
    for idx in range(len(dataset)):
        x_with_mask, target = dataset[idx]
        assert target == y[idx], f"Target at index {idx} does not match expected value."
        
        x_replaced = torch.nan_to_num(x[idx], nan=nan_replace)
        mask = (~torch.isnan(x[idx])).float()
        expected_x_with_mask = torch.cat((x_replaced, mask), dim=0)
        
        assert torch.equal(x_with_mask, expected_x_with_mask), f"Feature vector with mask at index {idx} does not match expected value."

def test_replace_nan_dataset_custom_nan_replace(sample_data):
    x, y = sample_data
    nan_replace = 0.0
    dataset = ReplaceNanDataset(x, y, nan_replace=nan_replace)
    
    for idx in range(len(dataset)):
        x_with_mask, target = dataset[idx]
        assert target == y[idx], f"Target at index {idx} does not match expected value."
        
        x_replaced = torch.nan_to_num(x[idx], nan=nan_replace)
        mask = (~torch.isnan(x[idx])).float()
        expected_x_with_mask = torch.cat((x_replaced, mask), dim=0)
        
        assert torch.equal(x_with_mask, expected_x_with_mask), f"Feature vector with mask at index {idx} does not match expected value."