import copy

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW

def split_seed(seed: int, num_splits: int = 2):
    # Use torch to generate seeds deterministically
    rng = torch.Generator()
    rng.manual_seed(seed)
    seeds = [torch.randint(0, 2**31 - 1, (1,), generator=rng).item() for _ in range(num_splits)]
    return seeds

def compute_ci(bootstrap_means, confidence_level=0.95):
    alpha = (1 - confidence_level) / 2
    bootstrap_means = torch.tensor(bootstrap_means)
    lower, upper = torch.quantile(bootstrap_means, torch.tensor([alpha, 1 - alpha]))
    mean = bootstrap_means.mean().item()
    return mean, lower.item(), upper.item()

def find_bootstrap_mean(seed, arr: torch.Tensor):
    # Create a deterministic random generator with the seed
    rng = torch.Generator()
    rng.manual_seed(seed)
    n_samples = arr.size(0)
    indices = torch.randint(0, n_samples, (n_samples,), generator=rng)
    assert torch.all(indices < n_samples), "Indices are out of bounds"
    bootstrap_mean = arr[indices].mean().item()
    return bootstrap_mean

def find_bootstrap_cf(seed, 
                      arr: torch.Tensor, 
                      confidence_level: float,
                      num_bootstraps: int):
    seeds = split_seed(seed, num_bootstraps)
    bootstrap_means = [find_bootstrap_mean(seed, arr) for seed in seeds]
    mean, lower, upper = compute_ci(bootstrap_means, confidence_level)
    return mean, lower, upper


class TwoLayerMLP(nn.Module):
    def __init__(self, seed: int, input_size: int, hidden_size: int,
                 output_size: int=1, dropout_prob: float=0.0):
        super(TwoLayerMLP, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.dropout_prob = dropout_prob
        
        # Initialize layers with the generator
        # print(f"input_size: {input_size}, hidden_size: {hidden_size}, output_size: {output_size}")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with the generator
        self._initialize_weights()

    def _initialize_weights(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # Initialize weights using the generator
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu', generator=generator)
        nn.init.kaiming_normal_(self.fc2.weight, generator=generator)
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        # apply deterministic dropout
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # Apply deterministic dropout
        if self.training and self.dropout_prob > 0:
            dropout_mask = (torch.rand(x.shape, generator=generator) > self.dropout_prob).float()
            x = x * dropout_mask / (1 - self.dropout_prob)
        x = self.fc2(x)  # Output without activation for BCEWithLogitsLoss
        return x

class LogisticRegModel(nn.Module):
    def __init__(self, seed: int, input_size: int, output_size: int = 1):
        super(LogisticRegModel, self).__init__()
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed

        # Define a single linear layer
        self.linear = nn.Linear(input_size, output_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='linear', generator=generator)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)  # Output without activation for BCEWithLogitsLoss

def train_mlp(
    mlp,
    inf_train_loader,
    val_loader,
    lr: float=0.001, 
    weight_decay: float=1e-4,
    patience: int=4,
    max_grad_steps: int=2000,
    testing: bool=False,
    loss_fn_type: str='mse',
    eval_interval: int=50,
    print_stuff: bool=False
):
    """
    Train a Multi-Layer Perceptron model with early stopping.
    
    Args:
        mlp: The MLP model to train
        inf_train_loader: Infinite training data loader
        val_loader: Validation data loader
        lr: Learning rate
        weight_decay: L2 regularization factor
        patience: Early stopping patience
        max_grad_steps: Maximum number of gradient steps
        testing: If True, return detailed statistics
        loss_fn_type: Type of loss function ('mse', 'bce', 'bce_logit')
        eval_interval: Steps between evaluations
        print_stuff: Whether to print progress
        
    Returns:
        Tuple of (trained model, statistics dictionary)
    """
    if loss_fn_type == 'bce_logit':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fn_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn_type == 'bce':
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss function type: {loss_fn_type}")
    optimizer = AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    min_loss = float('inf')
    best_step = 0
    best_wts = None
    no_improve = 0
    stats_dict = {'train_loss': [], 'val_loss': []}
    for grad_step in range(max_grad_steps):
        mlp.train()
        x_train, y_train = next(inf_train_loader)
        # print(x_train.shape, y_train.shape)
        outputs = mlp(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (grad_step + 1) % eval_interval == 0:
            mlp.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    outputs_val = mlp(x_val)
                    loss_val = criterion(outputs_val, y_val)
                    val_losses.append(loss_val.item())
                avg_val_loss = np.mean(val_losses)
                stats_dict['val_loss'].append(avg_val_loss)
                stats_dict['train_loss'].append(loss.item())
                if print_stuff:
                    print(f"Step {grad_step}, Loss: {avg_val_loss}, train loss: {loss}")
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                best_wts = copy.deepcopy(mlp.state_dict())
                best_step = grad_step+1
                no_improve = 0
            else:
                no_improve += 1
            if no_improve == patience:
                if print_stuff:
                    print(f"Early stopping at step {grad_step+1}")
                break
    mlp.load_state_dict(best_wts, strict=True)
    mlp.eval()
    # with torch.no_grad():
    #     for x_val, y_val in val_loader:
    #         print(x_val.shape)
    #         outputs_val = mlp(x_val)
    #     loss_val = criterion(outputs_val, y_val)
    #     print("loss val after loading best weights: ", loss_val.item())
    #     print("min loss: ", min_loss)
    stats_dict['best_step'] = best_step
    stats_dict['min_loss'] = min_loss
    assert np.isclose(stats_dict['val_loss'][best_step//eval_interval-1], min_loss, rtol=1e-5), "Validation loss does not match min loss"
    if testing:
        return mlp, stats_dict
    else:
        return mlp, {'best_step': stats_dict['best_step'], 'min_loss': stats_dict['min_loss']}
