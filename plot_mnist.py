import torch
import yaml
import types
import json

import matplotlib as mpl
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
class InceptionEmbedding:
    def __init__(self, parallel=False):
        # Expects inputs to be in range [-1, 1]
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model = WrapInception(inception_model.eval()).cuda()
        if parallel:
            inception_model = nn.DataParallel(inception_model)
        self.inception_model = inception_model

    def __call__(self, x):
        return self.inception_model(x)


# Wrapper for Inceptionv3, from Andrew Brock (modified slightly)
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)

    def forward(self, x):
        # Normalize x
        x = (x + 1.) / 2.0  # assume the input is normalized to [-1, 1], reset it to [0, 1]
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 80 or x.shape[3] != 80:
            x = F.interpolate(x, size=(80, 80), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.net.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.net.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.net.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.net.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.net.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.net.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.net.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.net.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.net.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6e(x)
        # 17 x 17 x 768
        # 17 x 17 x 768
        x = self.net.Mixed_7a(x)
        # 8 x 8 x 3840
        x = self.net.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.net.Mixed_7c(x)
        # 8 x 8 x 2048
        pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        # 1 x 1 x 2048
        return pool

def rect(pos):
    r = plt.Rectangle(pos - 0.5, 1, 1, facecolor="none", edgecolor="k", linewidth=2)
    plt.gca().add_patch(r)

def load_object(dct):
    return types.SimpleNamespace(**dct)

# TODO: PCANET in here
# TODO: Separate Eigenvectors
# TODO: Colored squares...
def scale_img(x):
    return x / torch.abs(x).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5

# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

evec1 = 0
evec2 = 3

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

    model_lazy = rcGANWReg.load_from_checkpoint(cfg.checkpoint_dir + 'eigengan_denoising_k=10/best.ckpt').cuda()
    model_lazy.eval()
    model = model_lazy

    dm = MNISTDataModule()
    dm.setup()
    test_loader = dm.test_dataloader()

    embedding = InceptionEmbedding()

    fig_count = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, _ = data
            x = x.cuda()
            y = x + torch.randn_like(x) * 1
            y = y.clamp(0, 1)
            fig_count = 0

            gens = torch.zeros(size=(y.size(0), 784, 1, 28, 28), device=x.device)
            for z in range(784):
                gens[:, z, :, :, :] = model.forward(y)

            avg = torch.mean(gens, dim=1)

            for j in range(x.shape[0]):
                if fig_count <= 10:
                    x_np = x[j, 0, :, :].cpu().numpy()
                    x_hat_np = avg[j, 0, :, :].cpu().numpy()
                    y_np = y[j, 0, :, :].cpu().numpy()

                    nrow = 1
                    ncol = 1

                    # fig = plt.figure(figsize=(ncol + 1, nrow + 1))
                    #
                    # gs = gridspec.GridSpec(nrow, ncol,
                    #                        wspace=0.05, hspace=0.05,
                    #                        top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                    #                        left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                    #
                    #
                    #
                    # ax = plt.subplot(gs[0, 0])
                    # ax.imshow(y_np, cmap='gray', vmin=0, vmax=1)
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    #
                    # plt.savefig(f'test_ims_rcgan/y_{j}.png', bbox_inches='tight', dpi=300)
                    #
                    # plt.close(fig)

                    plt.figure()
                    plt.imshow(y_np, cmap='gray', vmin=0, vmax=1)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    plt.savefig(f'test_ims_rcgan/y_{j}.png', bbox_inches='tight', dpi=300)
                    plt.close()

                    plt.figure()
                    plt.imshow(x_np, cmap='gray', vmin=0, vmax=1)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    plt.savefig(f'test_ims_rcgan/x_{j}.png', bbox_inches='tight', dpi=300)
                    plt.close()

                    nrow = 2
                    ncol = 1

                    fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                    gs = gridspec.GridSpec(nrow, ncol,
                                           wspace=0.0, hspace=0.0,
                                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                    ax = plt.subplot(gs[0, 0])
                    ax.imshow(x_hat_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax = plt.subplot(gs[1, 0])
                    ax.imshow(x_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # plt.savefig(f'test_ims_rcgan/mnist_left_bottom_eigengan_{j}.png', bbox_inches='tight', dpi=300)

                    plt.close(fig)

                    nrow = 1
                    ncol = 5

                    fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                    gs = gridspec.GridSpec(nrow, ncol,
                                           wspace=0.0, hspace=0.05,
                                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                    samps_np = gens[j, :, 0, :, :].cpu().numpy()
                    avg_np = avg[j, 0, :, :].cpu().numpy()

                    single_samps = samps_np - avg_np[None, :, :]

                    cov_mat = np.zeros((784, avg_np.shape[-1] * avg_np.shape[-2]))

                    for z in range(784):
                        cov_mat[z, :] = single_samps[z].flatten()

                    u, s, vh = np.linalg.svd(cov_mat, full_matrices=False)
                    s = s.reshape((1, -1))[:, 0:5]
                    print(s.shape)
                    cur_row = 1

                    vh = vh[0:5]

                    for k in range(5):
                        new_vh = vh.reshape((5, 28, 28))
                        new_vh = scale_img(torch.from_numpy(new_vh).unsqueeze(0).unsqueeze(2)).numpy()
                        new_vh = new_vh[0, :, 0, :, :]
                        pc_np = new_vh[k]

                        ax = plt.subplot(gs[0, k])
                        ax.imshow(samps_np[k], cmap='gray', vmin=0, vmax=1)
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # if k == evec1:
                        #     ax.patch.set_edgecolor('red')
                        #     ax.patch.set_linewidth(3)
                        # elif k == evec2:
                        #     ax.patch.set_edgecolor('blue')
                        #     ax.patch.set_linewidth(3)

                    plt.savefig(f'test_ims_rcgan/mnist_posterior_samps_{j}.png', bbox_inches='tight', dpi=300)

                    plt.close(fig)

                    nrow = 2
                    ncol = 5

                    fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                    gs = gridspec.GridSpec(nrow, ncol,
                                           wspace=0.05, hspace=0.05,
                                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                    cur_row = 0

                    for k in range(5):
                        pc_np = vh[k].reshape((28, 28))

                        if k == evec1 or k == evec2:
                            ax = plt.subplot(gs[cur_row, 0])
                            ax.imshow(x_hat_np - 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            if k == evec1:
                                ax.patch.set_edgecolor('red')
                                ax.patch.set_linewidth(3)
                            else:
                                ax.patch.set_edgecolor('blue')
                                ax.patch.set_linewidth(3)

                            ax = plt.subplot(gs[cur_row, 1])
                            ax.imshow(x_hat_np - 2 * pc_np, cmap='gray', vmin=0, vmax=1)
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            if k == evec1:
                                ax.patch.set_edgecolor('red')
                                ax.patch.set_linewidth(3)
                            else:
                                ax.patch.set_edgecolor('blue')
                                ax.patch.set_linewidth(3)

                            ax = plt.subplot(gs[cur_row, 2])
                            ax.imshow(x_hat_np, cmap='gray', vmin=0, vmax=1)
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            if k == evec1:
                                ax.patch.set_edgecolor('red')
                                ax.patch.set_linewidth(3)
                            else:
                                ax.patch.set_edgecolor('blue')
                                ax.patch.set_linewidth(3)

                            ax = plt.subplot(gs[cur_row, 3])
                            ax.imshow(x_hat_np + 2 * pc_np, cmap='gray', vmin=0, vmax=1)
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            if k == evec1:
                                ax.patch.set_edgecolor('red')
                                ax.patch.set_linewidth(3)
                            else:
                                ax.patch.set_edgecolor('blue')
                                ax.patch.set_linewidth(3)

                            ax = plt.subplot(gs[cur_row, 4])
                            ax.imshow(x_hat_np + 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            if k == evec1:
                                ax.patch.set_edgecolor('red')
                                ax.patch.set_linewidth(3)
                            else:
                                ax.patch.set_edgecolor('blue')
                                ax.patch.set_linewidth(3)

                            cur_row += 1

                    plt.savefig(f'test_ims_rcgan/mnist_right_bottom_eigengan_{j}.png', bbox_inches='tight', dpi=300)

                    plt.close(fig)

                    fig_count += 1
                else:
                    exit()
