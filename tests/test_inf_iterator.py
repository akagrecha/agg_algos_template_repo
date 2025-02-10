import pytest
import torch
from editable_proj import InfIterator, ReplaceNanDataset

@pytest.fixture
def sample_dataset():
    x = torch.tensor([
        [0.2, 0.4, torch.nan],
        [0.8, 0.5, 0.3],
        [0.1, torch.nan, 0.1],
        [0.3, 0.7, 0.2]
    ])
    y = torch.tensor([1, 0, 1, 0])
    return ReplaceNanDataset(x, y)

def test_inf_iterator_initialization(sample_dataset):
    seed = 42
    batch_size = 2
    iterator = InfIterator(seed, sample_dataset, batch_size)
    assert iterator.batch_size == batch_size
    assert iterator.dim == sample_dataset[0][0].size(0)
    assert len(iterator.indices) == len(sample_dataset)

def test_inf_iterator_next_batch(sample_dataset):
    seed = 42
    batch_size = 2
    iterator = InfIterator(seed, sample_dataset, batch_size)
    x, y = next(iterator)
    assert x.size(0) == batch_size
    assert y.size(0) == batch_size

def test_inf_iterator_shuffle(sample_dataset):
    seed = 42
    batch_size = 4
    iterator = InfIterator(seed, sample_dataset, batch_size)
    init_ids = []
    for i in range(4):
        init_ids.append(next(iterator.index_cycle))
    iterator.shuffle_indices()
    shuffled_ids = []
    for i in range(4):
        shuffled_ids.append(next(iterator.index_cycle))
    assert init_ids != shuffled_ids

def test_inf_iterator_resets_after_epoch(sample_dataset):
    seed = 42
    batch_size = 2
    iterator = InfIterator(seed, sample_dataset, batch_size)
    num_batches = len(sample_dataset) // batch_size
    for _ in range(num_batches):
        next(iterator)
    assert iterator.samples_seen == 0
    assert iterator.index_cycle is not None