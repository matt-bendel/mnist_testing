import torch
import yaml
import types
import json

import matplotlib.pyplot as plt

from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.PCANET_mac_joint import PCANET
from data.lightning.MNISTDataModule import MNISTDataModule
from matplotlib import gridspec
import sklearn.preprocessing

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")

    with open('configs/pcanet.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    mps_device = torch.device("mps")
    model = PCANET.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '/best-pca.ckpt').to(mps_device)
    model.eval()
    model.pca_net.eval()

    dm = MNISTDataModule()
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, _ = data
            x = x.to(mps_device)
            mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
            mask[:, :, 0:21, :] = 0
            y = x * mask
            fig_count = 0
            x = (x - 0.1307) / 0.3081
            y = (y - 0.1307) / 0.3081

            x_hat = model.mean_net(y)
            x_hat = model.readd_measures(x_hat, y)

            x = x * 0.3081 + 0.1307
            y = y * 0.3081 + 0.1307
            x_hat = x_hat * 0.3081 + 0.1307

            directions = model.forward(y, x_hat)
            principle_components, diff_vals = model.gramm_schmidt(directions)
            sigma_k = torch.zeros(directions.shape[0], directions.shape[1]).to(directions.device)
            for k in range(directions.shape[1]):
                sigma_k[:, k] = torch.sum(diff_vals[:, k, :, :] * diff_vals[:, k, :, :], dim=(1, 2))

            if i <= 5:
                x_np = x[0, 0, :, :].cpu().numpy()
                x_hat_np = x_hat[0, 0, :, :].cpu().numpy()
                y_np = y[0, 0, :, :].cpu().numpy()
                sigma_vals_np = sigma_k[0, :].cpu().numpy()
                print(sigma_vals_np)

                plt.figure()
                plt.scatter(range(5), sigma_vals_np)
                plt.savefig(f'test_ims/pca_plot_sv_{i}.png')
                plt.close()

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

                plt.savefig(f'test_ims/x_y_x_hat_error_{i}.png')

                plt.close(fig)

                nrow = 5
                ncol = 6

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                for k in range(5):
                    pc_np = (principle_components[0, k, :, :] * (1 - mask[0, 0, :, :])).cpu().numpy()

                    ax = plt.subplot(gs[k, 0])
                    ax.imshow(pc_np, cmap='bwr')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylabel(f'i={k+1}')
                    if k == 0:
                        ax.set_title('w_i')

                    ax = plt.subplot(gs[k, 1])
                    ax.imshow(x_hat_np - 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat - 3 w_i')

                    ax = plt.subplot(gs[k, 2])
                    ax.imshow(x_hat_np - 2 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat - 2 w_i')

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
                        ax.set_title('x_hat + 2 w_i')

                    ax = plt.subplot(gs[k, 5])
                    ax.imshow(x_hat_np + 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 0:
                        ax.set_title('x_hat + 3 w_i')

                plt.savefig(f'test_ims/pca_plot_{i}.png')

                plt.close(fig)
            else:
                exit()
