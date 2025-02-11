from typing import Union

import torch
from torch.utils.data import DataLoader

from .utils import train_mlp
from .dataset_utils import ReplaceNanDataset, InfIterator

class Averaging:
    def __init__(self, reg: float=0.0, 
                 prior_mean: float=0.5,
                 dim: Union[float, None]=None) -> None:
        self.reg = reg
        self.prior_mean = prior_mean
    
    def fit(self, ests: torch.Tensor):
        pass

    def predict(self, ests: torch.Tensor):
        assert ests.dim() == 2
        # missing ests are nans
        return self.reg*self.prior_mean + (1-self.reg)*ests.nanmean(dim=1)

class SLAgg:
    def __init__(self,
                 mlp,
                 seed: int,
                 dim: int,
                 batch_size: int=50,
                 lr: float=0.001, 
                 weight_decay: float=1e-4,
                 patience: int=4,
                 max_grad_steps: int=2000,
                 testing: bool=False,
                 loss_fn_type: str='mse',
                 eval_interval: int=50,
                 print_stuff: bool=False,
                 nan_replace: float=-1.0,
                 ) -> None:
        self.seed = seed
        self.mlp = mlp
        self.dim = dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_grad_steps = max_grad_steps
        self.testing = testing
        self.loss_fn_type = loss_fn_type
        self.eval_interval = eval_interval
        self.print_stuff = print_stuff
        self.nan_replace = nan_replace

    def fit(self, 
            ests: torch.Tensor,
            outcomes: torch.Tensor,
            ests_val: torch.Tensor,
            outcomes_val: torch.Tensor,):
        # print("ests shape: ", ests.shape)
        # print("outcomes shape: ", outcomes.shape)
        train_dataset = ReplaceNanDataset(ests, outcomes)
        # print(train_dataset[0])
        if outcomes_val.dim() == 1:
            y=outcomes_val[:,None]
        inf_train_loader = InfIterator(seed=self.seed, 
                                    dataset=train_dataset, 
                                    batch_size=self.batch_size)
        val_dataset = ReplaceNanDataset(x=ests_val, y=outcomes_val)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        self.mlp, stats_dict = train_mlp(
                                    mlp=self.mlp,
                                    inf_train_loader=inf_train_loader,
                                    val_loader=val_loader,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    patience=self.patience,
                                    max_grad_steps=self.max_grad_steps,
                                    testing=self.testing,
                                    loss_fn_type=self.loss_fn_type,
                                    eval_interval=self.eval_interval,
                                    print_stuff=self.print_stuff,)
        return stats_dict

    def predict(self, x: torch.Tensor, concat_mask: bool=True):
        if concat_mask:
            mask = (~torch.isnan(x)).float()
            # Replace NaNs with nan_replace
            x = torch.nan_to_num(x, nan=self.nan_replace)
            # Concatenate the feature vector with the mask
            x = torch.cat((x, mask), dim=1)
        with torch.no_grad():
            preds = self.mlp(x)
        return preds
