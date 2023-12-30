import torch
import yaml
import types
import json

import pytorch_lightning as pl
# TODO: Install Matplotlib, wandb

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from models.lightning.rcGAN_mac import rcGAN
from data.lightning.MNISTDataModule import MNISTDataModule
from models.lightning.rcGAN_mac_w_pc_reg import rcGANWRegOptuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def load_object(dct):
    return types.SimpleNamespace(**dct)


def objective(trial):
    with open('configs/rcgan.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    args = create_arg_parser().parse_args()

    print(f"Experiment Name: {args.exp_name}")

    start_lr = trial.suggest_float("start_lr", 1e-4, 1e-3)
    beta_pca = trial.suggest_float('beta_pca', 1e-3, 1)
    patience = trial.suggest_int('lr_patience', 5, 20)
    lr_step = trial.suggest_float('lr_step', 0.5, 0.99)

    model = rcGANWRegOptuna(cfg, args.exp_name, start_lr, beta_pca, patience, lr_step)
    dm = MNISTDataModule()

    trainer = pl.Trainer(accelerator="gpu", strategy='ddp', devices=1, enable_checkpointing=False,
                         max_epochs=cfg.num_epochs,
                         num_sanity_val_steps=2, profiler="simple", logger=True, benchmark=False,
                         log_every_n_steps=10)

    trainer.fit(model, dm)

    return trainer.callback_metrics["cfid"].item()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    seed_everything(0, workers=True)


    with open('configs/rcgan.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))