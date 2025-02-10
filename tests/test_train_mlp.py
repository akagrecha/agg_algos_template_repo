import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from editable_proj import train_mlp

class DummyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DummyDataset(Dataset):
    def __init__(self, size=100, input_dim=10, output_dim=1):
        self.X = torch.randn(size, input_dim)
        self.y = torch.randint(0, 2, (size, output_dim)).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

@pytest.fixture
def setup_training_components():
    input_dim, hidden_dim, output_dim = 10, 20, 1
    batch_size = 16
    
    # Create model and datasets
    model = DummyMLP(input_dim, hidden_dim, output_dim)
    train_dataset = DummyDataset(size=100, input_dim=input_dim, output_dim=output_dim)
    val_dataset = DummyDataset(size=50, input_dim=input_dim, output_dim=output_dim)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    inf_train_loader = infinite_loader(train_loader)
    
    return model, inf_train_loader, val_loader

def test_basic_training(setup_training_components):
    """Test that training runs without errors and returns expected outputs"""
    model, inf_train_loader, val_loader = setup_training_components
    
    trained_model, stats = train_mlp(
        model, 
        inf_train_loader,
        val_loader,
        testing=True,
        max_grad_steps=100
    )
    
    # Check returned objects
    assert isinstance(trained_model, nn.Module)
    assert isinstance(stats, dict)
    assert 'best_step' in stats
    assert 'min_loss' in stats
    assert 'train_loss' in stats
    assert 'val_loss' in stats

def test_early_stopping(setup_training_components):
    """Test that early stopping works correctly"""
    model, inf_train_loader, val_loader = setup_training_components
    
    _, stats = train_mlp(
        model,
        inf_train_loader,
        val_loader,
        patience=2,
        max_grad_steps=1000,
        testing=True,
        eval_interval=10
    )
    
    # Check that training stopped before max_grad_steps
    assert stats['best_step'] < 1000
    # Verify that min_loss matches the corresponding validation loss
    assert np.isclose(
        stats['val_loss'][stats['best_step']//10-1],
        stats['min_loss'],
        rtol=1e-5
    )

def test_different_loss_functions(setup_training_components):
    """Test that different loss functions work correctly"""
    model, inf_train_loader, val_loader = setup_training_components
    
    loss_types = ['mse', 'bce_logit', 'bce']
    for loss_type in loss_types:
        # For BCE losses, ensure output is properly scaled
        if loss_type in ['bce', 'bce_logit']:
            model.network[-1].register_forward_hook(
                lambda m, inp, out: torch.sigmoid(out) if loss_type == 'bce' else out
            )
        
        _, stats = train_mlp(
            model,
            inf_train_loader,
            val_loader,
            loss_fn_type=loss_type,
            max_grad_steps=50,
            testing=True
        )
        
        assert stats['min_loss'] > 0
        assert not np.isnan(stats['min_loss'])

def test_invalid_loss_function(setup_training_components):
    """Test that invalid loss function raises appropriate error"""
    model, inf_train_loader, val_loader = setup_training_components
    
    with pytest.raises(ValueError, match="Unknown loss function type"):
        train_mlp(
            model,
            inf_train_loader,
            val_loader,
            loss_fn_type='invalid_loss'
        )

def test_optimization_parameters(setup_training_components):
    """Test that learning rate and weight decay affect training"""
    model, inf_train_loader, val_loader = setup_training_components
    
    # Train with different learning rates
    results = []
    for lr in [0.1, 0.01, 0.001]:
        _, stats = train_mlp(
            model,
            inf_train_loader,
            val_loader,
            lr=lr,
            max_grad_steps=50,
            testing=True
        )
        results.append(stats['min_loss'])
    
    # Check that different learning rates lead to different results
    assert len(set(results)) > 1

def test_model_state(setup_training_components):
    """Test that model state is properly managed during training"""
    model, inf_train_loader, val_loader = setup_training_components
    
    initial_state = copy.deepcopy(model.state_dict())
    trained_model, _ = train_mlp(
        model,
        inf_train_loader,
        val_loader,
        max_grad_steps=50
    )
    
    # Check that weights have been updated
    for (k1, v1), (k2, v2) in zip(
        initial_state.items(),
        trained_model.state_dict().items()
    ):
        assert not torch.allclose(v1, v2)

def test_inf_loader_behavior(setup_training_components):
    """Test that infinite loader works correctly"""
    model, inf_train_loader, val_loader = setup_training_components
    
    # Ensure we can draw more samples than dataset size
    n_steps = 200
    seen_batches = []
    for _ in range(n_steps):
        x, y = next(inf_train_loader)
        seen_batches.append(x.sum().item())  # Use sum as a proxy for batch content
    
    # Check that we've seen more batches than original dataset size
    assert len(seen_batches) > len(val_loader.dataset)