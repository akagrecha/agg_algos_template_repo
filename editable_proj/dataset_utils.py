import itertools

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ReplaceNanDataset(Dataset):
    """
    A PyTorch Dataset that replaces NaN values in the input tensor with a specified value
    and concatenates a mask indicating the presence of NaNs.
    Args:
        x (torch.Tensor): The input tensor containing features.
        y (torch.Tensor): The target tensor containing labels.
        nan_replace (float, optional): The value to replace NaNs with. Default is -1.0.
    Attributes:
        x (torch.Tensor): The input tensor with features.
        nan_replace (float): The value used to replace NaNs.
        y (torch.Tensor): The target tensor with labels.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the feature vector with NaNs replaced and concatenated
                          with a mask, along with the corresponding label.
    """
    def __init__(self, x: torch.Tensor, 
                 y: torch.Tensor, 
                 nan_replace: float = -1.0):
        self.x = x
        self.nan_replace = nan_replace
        if y.dim() == 1:
            y = y.unsqueeze(1)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Create mask: 1 where x is not NaN, 0 where x is NaN
        mask = (~torch.isnan(self.x[idx])).float()
        # Replace NaNs with nan_replace
        x_replaced = torch.nan_to_num(self.x[idx], nan=self.nan_replace)
        # Concatenate the feature vector with the mask
        x_with_mask = torch.cat((x_replaced, mask), dim=0)
        return x_with_mask, self.y[idx]


class InfIterator:
    """
    An infinite iterator for sampling batches from a dataset with shuffling.
    Attributes:
        rng (torch.Generator): Random number generator for shuffling.
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        batch_size (int): The number of samples per batch.
        dim (int): The dimensionality of the input samples.
        indices (torch.Tensor): Tensor containing indices of the dataset.
        index_cycle (itertools.cycle): Infinite cycle of shuffled indices.
        samples_seen (int): Counter for the number of samples seen so far.
    Methods:
        shuffle_indices():
            Shuffles the dataset indices and resets the index cycle and sample counter.
        __iter__():
            Returns the iterator object itself.
        __next__():
            Returns the next batch of samples from the dataset.
    """
    def __init__(self, 
                 seed: int, 
                 dataset: torch.utils.data.Dataset, 
                 batch_size: int,):
        self.rng = torch.Generator().manual_seed(seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.dim = dataset[0][0].size(0)
        self.indices = torch.arange(len(dataset))  # Create a list of indices
        self.shuffle_indices()  # Shuffle initially
        self.index_cycle = itertools.cycle(self.indices)  # Create an infinite cycle of shuffled indices
        self.samples_seen = 0  # Track the number of samples seen so far

    def shuffle_indices(self):
        """Shuffles the dataset indices."""
        shuffled_indices = self.indices[torch.randperm(len(self.indices), generator=self.rng)]
        self.index_cycle = itertools.cycle(shuffled_indices)  # Reset the cycle with shuffled indices
        self.samples_seen = 0  # Reset the counter when reshuffling happens

    def __iter__(self):
        return self

    def __next__(self):
        # Sample a batch of indices
        batch_indices = [next(self.index_cycle) for _ in range(self.batch_size)]
        self.samples_seen += len(batch_indices)  # Increment samples seen
        
        # Fetch the samples
        batch = [self.dataset[idx] for idx in batch_indices]
        x, y = zip(*batch)  # Extract inputs and targets
        # print(f"x.shape: {x.shape}")
        # print(f"y.shape: {y[0].shape}")
        x = torch.stack(x)
        if len(y[0]) == 1:
            y = torch.tensor(y).reshape(-1, 1)
        else:
            y = torch.stack(y)

        # Reshuffle when all dataset indices have been used once
        if self.samples_seen >= len(self.indices):
            self.shuffle_indices()

        return x, y
