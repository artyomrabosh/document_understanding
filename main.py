import datetime
import wandb
import hydra
from omegaconf import DictConfig
from ruamel.yaml import YAML
import torch

from os import path

from modelling.datasets import SpbuDataset
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

    train_loader = SpbuDataset(train_path)
    val_loader = SpbuDataset(val_path)

    set_training_parameters(predictor, cfg.model.trainable_params)
    match cfg.optimizer.name:
        case "adam":
            optim = torch.optim.Adam(predictor.model.parameters(), 
                                     lr=cfg.optimizer.lr)
    
    for _ in range(cfg.training.num_epochs):
        train_epoch(predictor=predictor, 
                    dataloader=train_loader, 
                    optimizer=optim, 
                    device=device, 
                    accumulation_grad_steps=cfg.optimizer.accumulation_grad_steps)
        validate(predictor, val_loader, device) 
        wandb.log({'lr': get_current_lr(optim)})




if __name__ == "__main__":
    main()


