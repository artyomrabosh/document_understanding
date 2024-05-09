import datetime
import wandb
from omegaconf import DictConfig
from ruamel.yaml import YAML
import torch
from torch.utils.data.dataloader import DataLoader

from os import path

from modelling.datasets import SpbuDataset, single_item_collate_fn
from modelling.predictors import SpbuPredictor
from modelling.training import set_training_parameters
from modelling.training import validate
from modelling.training import train_epoch

import warnings
warnings.filterwarnings('ignore')

def start_wandb(config):
    name = 'test-({})'.format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    wandb_config = config.copy()

    wandb_run = wandb.init(project='Document_Understanding',
                            name=name, config=wandb_config)
    wandb_run.log_code('.')
    return wandb_run

def get_current_lr(optim):
    return optim.param_groups[0]['lr']

def main():
    cfg = DictConfig(YAML(typ="safe").load(open("params.yaml")))
    device = cfg.training.device
    start_wandb(config=dict(cfg))
    match cfg.model.name:
        case 'vila':
            predictor = SpbuPredictor.from_pretrained(
                "allenai/hvila-block-layoutlm-finetuned-docbank"
                )

    predictor.model.to(device)
    wandb.watch(predictor.model)

    train_path = path.join('data', 'spbu', 'latex', 'train')
    val_path = path.join('data', 'spbu', 'latex', 'val')

    train_loader =DataLoader(SpbuDataset(train_path),
                             batch_size=1,
                             shuffle=True,
                             collate_fn=single_item_collate_fn)
    
    val_loader = SpbuDataset(val_path)

    set_training_parameters(predictor, cfg.model.trainable_params)
    match cfg.optimizer.name:
        case "adam":
            optim = torch.optim.Adam(predictor.model.parameters(), 
                                     lr=cfg.optimizer.lr)
            
    match cfg.scheduler.name:
        case "Exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
        case "Linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optim, 
                                                          start_factor=1.0, 
                                                          end_factor=0.33)
    
    for _ in range(cfg.training.num_epochs):
        train_epoch(predictor=predictor, 
                    dataloader=train_loader, 
                    optimizer=optim, 
                    device=device, 
                    accumulation_grad_steps=cfg.optimizer.accumulation_grad_steps)
        validate(predictor, val_loader, device) 
        scheduler.step()
        wandb.log({'lr': get_current_lr(optim)})




if __name__ == "__main__":
    main()


