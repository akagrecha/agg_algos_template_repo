from typing import Union
from datetime import datetime

import hydra
import wandb
from omegaconf import OmegaConf
import torch

import editable_proj

# load data
def get_data(cfg):
    # print(cfg.data_loader.name)
    data_constructor = editable_proj.__dict__[cfg.data_loader.name]
    df, outcomes = data_constructor(**cfg.data_loader.params).get_data()
    return df, outcomes

# cross validation splits -- modify main to use this
def get_cv_split(cfg, split: int, num_splits: int=5,):
    df, outcomes = get_data(cfg)
    ests = df.values
    # convert to torch tensor
    ests = torch.tensor(ests, dtype=torch.float)
    outcomes = torch.tensor(outcomes, dtype=torch.float)
    split_size = len(ests)//num_splits
    val_ids = list(range(split*split_size, (split+1)*split_size))
    train_ids = list(set(range(len(ests))) - set(val_ids))
    ests_train = ests[train_ids]
    ests_val = ests[val_ids]
    outcomes_train = outcomes[train_ids]
    outcomes_val = outcomes[val_ids]
    return {"ests_train": ests_train, "ests_val": ests_val,
            "outcomes_train": outcomes_train, "outcomes_val": outcomes_val}

# untrained MLP
def get_untrained_mlp(cfg, input_size: int):
    mlp_constructor = editable_proj.__dict__[cfg.mlp.name]
    mlp = mlp_constructor(input_size=input_size, **cfg.mlp.params)
    return mlp

# get policy
def get_policy(cfg, dim: int,):
    policy_constructor = editable_proj.__dict__[cfg.policy.name]
    if cfg.policy.name in ["SLAgg"]:
        # 2*dim because we add a mask to the input indicating if a value is missing
        # missing values are imputed with -1 and mask is 0 if value is missing
        mlp = get_untrained_mlp(cfg, 2*dim)
        policy = policy_constructor(**cfg.policy.params, 
                                    mlp=mlp,
                                    loss_fn_type=cfg.mlp.loss_fn_type,
                                    dim=dim)
    else:
        policy = policy_constructor(**cfg.policy.params, dim=dim)
    return policy

def eval_policy(cfg, policy, ests_val, outcomes_val):
    if cfg.mlp.loss_fn_type == "bce_logit":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.mlp.loss_fn_type == "mse":
        criterion = torch.nn.MSELoss()
    elif cfg.mlp.loss_fn_type == "bce":
        criterion = torch.nn.BCELoss()
    else:
        raise ValueError(f"loss_fn_type {cfg.mlp.loss_fn_type} not recognized")
    if cfg.policy.name in ['SLAgg']:
        with torch.no_grad():
            outputs_val = policy.predict(ests_val, concat_mask=True)
    else:
        outputs_val = policy.predict(ests_val)
        if outputs_val.dim() == 1:
            outputs_val = outputs_val[:,None]
        if cfg.mlp.loss_fn_type == "bce_logit":
            outputs_val = torch.log(outputs_val/(1-outputs_val))
    if outcomes_val.dim() == 1:
        outcomes_val = outcomes_val[:,None]
    # print(outputs_test.shape, oos_ests_test.shape)
    # breakpoint()
    loss = criterion(outputs_val, outcomes_val)
    return loss.item()

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    # wandb init
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert cfg.wandb.entity is not None or not cfg.main.use_wandb, "Please provide a wandb entity"
    if cfg.main.use_wandb:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            config=cfg_dict,
            **cfg.wandb,
            )
    metrics = {'loss': [], 'time_fit': []}
    num_splits = cfg.main.num_splits

    for split in range(num_splits):
        data = get_cv_split(cfg, split, num_splits=num_splits)
        ests_train, ests_val = data['ests_train'], data['ests_val']
        outcomes_train, outcomes_val = data['outcomes_train'], data['outcomes_val']
        dim = ests_train.shape[1]
        policy = get_policy(cfg, dim=dim)
        # fit policy
        if cfg.policy.name in ['SLAgg']:
            time_start = datetime.now()
            stats_dict = policy.fit(ests=ests_train, outcomes=outcomes_train,
                                    ests_val=ests_val, outcomes_val=outcomes_val)
            time_end = datetime.now()
            time_fit = (time_end - time_start).total_seconds()
            if split == 0:
                metrics['best_step'] = [stats_dict['best_step']]
            else:
                metrics['best_step'].append(stats_dict['best_step'])
        else:
            time_start = datetime.now()
            policy.fit(ests=ests_train,)
            time_end = datetime.now()
            time_fit = (time_end - time_start).total_seconds()
        metrics['time_fit'].append(time_fit)

        # evaluate policy
        loss = eval_policy(cfg, policy, ests_val, outcomes_val)
        metrics['loss'].append(loss)
        if cfg.mlp.loss_fn_type == "mse":
            if split==0:
                metrics['rmse'] = [torch.sqrt(loss)]
            else:
                metrics['rmse'].append(torch.sqrt(loss))

        if cfg.main.use_wandb:
            log_dict = {}
            for key in metrics:
                log_dict[key] = metrics[key][-1]
            wandb.log(log_dict, step=dim)

    if not cfg.main.use_wandb:
        for key in metrics:
            print(f"{key}:")
            print(metrics[key])
    
if __name__ == "__main__":
    main()
