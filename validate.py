import numpy
import numpy as np
import torch
import yaml
import types
import json

from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN_mac import rcGAN, rcGANLatent, rcGANJoint
from models.lightning.rcGAN_mac_w_pc_reg import rcGANWReg, rcGANWRegLatent
from models.lightning.mnist_autoencoder import MNISTAutoencoder
from data.lightning.MNISTDataModule import MNISTDataModule
from metrics.cfid import CFIDMetric

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")

    with open('configs/rcgan.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    model = rcGAN(cfg, args.exp_name)

    dm = MNISTDataModule()
    dm.setup()

    best_epoch = 0
    best_cfid = 100000000

    embedding = MNISTAutoencoder.load_from_checkpoint('/storage/matt_models/mnist/autoencoder/best.ckpt').autoencoder
    embedding.eval().cuda()

    for epoch in range(75, 125):
        print(epoch)
        model = rcGANJoint.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={epoch}.ckpt')
        model.eval().cuda()

        cfid = CFIDMetric(model, dm.val_dataloader(), embedding, embedding, True)

        cfid_val, m_val, c_val = cfid.get_cfid_torch_pinv()
        print(cfid_val)

        if cfid_val < best_cfid:
            best_epoch = epoch
            best_cfid = cfid_val

    print(best_epoch)
    print(best_cfid)



