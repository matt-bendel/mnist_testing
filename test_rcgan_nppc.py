import torch
import yaml
import types
import json

import matplotlib.pyplot as plt
import numpy as np
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN_mac import rcGAN, rcGANLatent
from models.lightning.rcGAN_mac_w_pc_reg import rcGANWReg, rcGANWRegLatent
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

    model = rcGANLatent.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '/best.ckpt').cuda()
    model.eval()

    model_lazy = rcGANWRegLatent.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '_w_reg_k=5/best.ckpt').cuda()
    model_lazy.eval()

    dm = MNISTDataModule()
    dm.setup()
    test_loader = dm.test_dataloader()

    embedding = MNISTAutoencoder.load_from_checkpoint('/storage/matt_models/mnist/autoencoder/best.ckpt').autoencoder.cuda()
    embedding.eval()

    cfid = CFIDMetric(model, dm.val_dataloader(), embedding, embedding, True)

    cfid_val, m_val, c_val = cfid.get_cfid_torch_pinv() # 1.57, 12.89, 14.45
    print(cfid_val)

    cfid = CFIDMetric(model_lazy, dm.val_dataloader(), embedding, embedding, True)

    cfid_val, m_val, c_val = cfid.get_cfid_torch_pinv() # 2.66, 10.60, 13.26
    print(cfid_val)

    with torch.no_grad():
        l2s = []
        weird_l2s = []

        for i, data in enumerate(test_loader):
            x, _ = data
            x = x.cuda()
            mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
            mask[:, :, 0:21, :] = 0
            y = x * mask
            fig_count = 0
            x = (x - 0.1307) / 0.3081
            y = (y - 0.1307) / 0.3081

            gens = torch.zeros(size=(y.size(0), 784, 1, 28, 28), device=x.device)
            for z in range(784):
                gens[:, z, :, :, :] = model.forward(y) * 0.3081 + 0.1307

            x = x * 0.3081 + 0.1307
            y = y * 0.3081 + 0.1307

            avg = torch.mean(gens, dim=1)

            err = (x - avg).flatten(1)
            err_norm = err.norm(dim=1)

            l2s.append(err_norm.mean().cpu().numpy())

            for n in range(x.shape[0]):
                samps_np = gens[n, :, 0, :, :].cpu().numpy()
                avg_np = avg[n, 0, :, :].cpu().numpy()

                single_samps = samps_np - avg_np[None, :, :]

                cov_mat = np.zeros((784, avg_np.shape[-1] * avg_np.shape[-2]))

                for z in range(784):
                    cov_mat[z, :] = single_samps[z].flatten()

                u, s, vh = np.linalg.svd(cov_mat, full_matrices=False)
                v = vh.transpose()

                weird_l2s.append(np.linalg.norm(err.unsueeze(1).cpu().numpy() - v @ vh @ err[n, :].unsueeze(1).cpu.numpy()))

        print(f'L2: {np.mean(l2s)}')
        print(f'Weird L2: {np.mean(weird_l2s)}')