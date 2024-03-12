import torch
import yaml
import types
import json

import matplotlib.pyplot as plt
import numpy as np
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN_mac import rcGAN, rcGANLatent, rcGANJoint
from models.lightning.rcGAN_mac_w_pc_reg import rcGANWReg, rcGANWRegLatent, rcGANWRegJoint
from data.lightning.MNISTDataModule import MNISTDataModule
from matplotlib import gridspec
import sklearn.preprocessing
from data.lightning.MNISTDataModule import MNISTDataModule
from metrics.cfid import CFIDMetric
from models.lightning.mnist_autoencoder import MNISTAutoencoder

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")

    with open('configs/rcgan.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    model = rcGAN.load_from_checkpoint(cfg.checkpoint_dir + 'rcgan_denoising/best.ckpt').cuda()
    model.eval()

    model_lazy = rcGANWReg.load_from_checkpoint(cfg.checkpoint_dir + 'eigengan_denoising_k=5/best.ckpt').cuda()
    model_lazy.eval()
    # model = model_lazy

    dm = MNISTDataModule()
    dm.setup()
    test_loader = dm.test_dataloader()

    #
    # cfid = CFIDMetric(model_lazy, dm.val_dataloader(), embedding, embedding, True)
    #
    # cfid_val, m_val, c_val = cfid.get_cfid_torch_pinv() # 2.66, 10.60, 13.26
    # print(cfid_val)

    with torch.no_grad():
        l2s = []
        weird_l2s = []

        for i, data in enumerate(test_loader):
            print(f'{i}/{len(test_loader)}')
            x, _ = data
            x = x.cuda()
            y = x + torch.randn_like(x) * 1
            y = y.clamp(0, 1)

            gens = torch.zeros(size=(y.size(0), 128, 1, 28, 28), device=x.device)
            for z in range(128):
                gens[:, z, :, :, :] = model.forward(y)

            x = x
            y = y

            avg = torch.mean(gens, dim=1)

            gens_zm = gens - avg[:, None, :, :, :]
            gens_zm = gens_zm.flatten(2)

            err = (x - avg).flatten(1)
            err_norm = err.norm(dim=1)

            l2s.append(err_norm.mean().cpu().numpy())

            for n in range(x.shape[0]):
                _, S, Vh = torch.linalg.svd(gens_zm[n], full_matrices=False)
                print(Vh.shape)

                unsqueezed_err = torch.unsqueeze(err[n, :], dim=1)

                weird_l2 = torch.norm(unsqueezed_err - torch.matmul(torch.matmul(Vh.transpose(0, 1), Vh), unsqueezed_err), p=2).cpu().numpy()

                weird_l2s.append(weird_l2)

        print(f'L2: {np.mean(l2s)}')
        print(f'Weird L2: {np.mean(weird_l2s)}')