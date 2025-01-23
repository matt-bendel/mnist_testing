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
from models.lightning.gan_ddb import rcGANDDB
from data.lightning.MNISTDataModule import MNISTDataModule

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")

    with open('configs/ddb.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    model = rcGANDDB(cfg, args.exp_name)

    dm = MNISTDataModule()

    wandb_logger = WandbLogger(
        project="pca_exps",
        name=args.exp_name,
        log_model="all",
    )
    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='checkpoint-{epoch}',
        save_top_k=50
    )

    trainer = pl.Trainer(accelerator="gpu", strategy='ddp', devices="auto",
                         max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback_epoch],
                         num_sanity_val_steps=2, profiler="simple", logger=wandb_logger, benchmark=False, log_every_n_steps=10)


    trainer.fit(model, dm)
