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

    model_lazy = rcGANWRegLatent.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '_w_reg_k=25/best.ckpt').cuda()
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
        for i, data in enumerate(test_loader):
            x, _ = data
            x = x.cuda()
            mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
            mask[:, :, 0:21, :] = 0
            y = x * mask
            print(y.device)
            fig_count = 0
            x = (x - 0.1307) / 0.3081
            y = (y - 0.1307) / 0.3081

            gens = torch.zeros(size=(y.size(0), 128, 1, 28, 28), device=x.device)
            for z in range(128):
                gens[:, z, :, :, :] = model_lazy.forward(y) * 0.3081 + 0.1307

            x = x * 0.3081 + 0.1307
            y = y * 0.3081 + 0.1307

            avg = torch.mean(gens, dim=1)

            if i <= 5:
                x_np = x[0, 0, :, :].cpu().numpy()
                x_hat_np = avg[0, 0, :, :].cpu().numpy()
                y_np = y[0, 0, :, :].cpu().numpy()

                nrow = 4
                ncol = 1

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(x_np, cmap='gray', vmin=0, vmax=1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('x')

                ax = plt.subplot(gs[1, 0])
                ax.imshow(y_np, cmap='gray', vmin=0, vmax=1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('y')

                ax = plt.subplot(gs[2, 0])
                ax.imshow(x_hat_np, cmap='gray', vmin=0, vmax=1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('x_hat')

                ax = plt.subplot(gs[3, 0])
                ax.imshow(x_np - x_hat_np, cmap='bwr')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('error')

                plt.savefig(f'test_ims_rcgan_lazy/x_y_x_hat_error_{i}.png')

                plt.close(fig)

                nrow = 5
                ncol = 6

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                samps_np = gens[0, :, 0, :, :].cpu().numpy()
                avg_np = avg[0, 0, :, :].cpu().numpy()

                single_samps = samps_np - avg_np[None, :, :]

                cov_mat = np.zeros((784, avg_np.shape[-1] * avg_np.shape[-2]))

                for z in range(784):
                    cov_mat[z, :] = single_samps[z].flatten()

                u, s, vh = np.linalg.svd(cov_mat, full_matrices=False)
                s = s.reshape((1, -1))[:, 0:5]
                print(s.shape)

                plt.figure()
                plt.scatter(range(5), s)
                plt.savefig(f'test_ims_rcgan_lazy/{args.exp_name}_pca_plot_sv_{i}.png')
                plt.close()

                for k in range(5):
                    pc_np = vh[k].reshape((28, 28))

                    ax = plt.subplot(gs[k, 0])
                    ax.imshow(pc_np, cmap='bwr')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylabel(f'i={k+1}')
                    if k == 0:
                        ax.set_title('w_i')

                    pc_np = pc_np * s[k]

                    ax = plt.subplot(gs[k, 1])
                    ax.imshow(x_hat_np - 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat - 3 sigma_i w_i')

                    ax = plt.subplot(gs[k, 2])
                    ax.imshow(x_hat_np - 2 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat - 2 sigma_i w_i')

                    ax = plt.subplot(gs[k, 3])
                    ax.imshow(x_hat_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat')

                    ax = plt.subplot(gs[k, 4])
                    ax.imshow(x_hat_np + 2 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat + 2 sigma_i w_i')

                    ax = plt.subplot(gs[k, 5])
                    ax.imshow(x_hat_np + 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat + 3 sigma_i w_i')

                plt.savefig(f'test_ims_rcgan_lazy/{args.exp_name}_pca_plot_{i}.png')

                plt.close(fig)
            else:
                exit()
