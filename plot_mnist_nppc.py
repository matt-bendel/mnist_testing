import nppc
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import torchvision

restoration_net = nppc.RestorationModel(
    dataset='mnist',
    data_folder='/storage/matt_models/mnist/',
    distortion_type='denoising_1',
    net_type='unet',
    lr=1e-4,
    device='cuda:0',
)
restoration_net = restoration_net.load('./results/mnist_denoising/restoration/checkpoint.pt')
restoration_net.net.eval()

nppc_model = nppc.NPPCModel(
    restoration_model_folder='./results/mnist_denoising/restoration/',
    net_type='unet',
    n_dirs=5,
    lr=1e-4,
    device='cuda:0',
)
nppc_model = nppc_model.load('./results/mnist_denoising/nppc/checkpoint.pt')
nppc_model.net.eval()

dataloader = torch.utils.data.DataLoader(
    nppc_model.data_module.test_set,
    batch_size=4,
    shuffle=False,
    generator=torch.Generator().manual_seed(0),
)


def sample_to_width(x, width=1580, padding_size=2):
    n_samples = min((width - padding_size) // (x.shape[-1] + padding_size), x.shape[0])
    indices = np.linspace(0, x.shape[0] - 1, n_samples).astype(int)
    return x[indices]

def imgs_to_grid(imgs, nrows=None, **make_grid_args):
    imgs = imgs.detach().cpu()
    if imgs.ndim == 5:
        nrow = imgs.shape[1]
        imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
    elif nrows is None:
        nrow = int(np.ceil(imgs.shape[0] ** 0.5))

    make_grid_args2 = dict(value_range=(0, 1), pad_value=1.)
    make_grid_args2.update(make_grid_args)
    img = torchvision.utils.make_grid(imgs, nrow=nrow, **make_grid_args2).clamp(0, 1)
    return img

def scale_img(x):
    return x / torch.abs(x).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5

def tensor_img_to_numpy(x):
    return x.detach().permute(-2, -1, -3).cpu().numpy()

def imshow(img, scale=1, **kwargs):
    if isinstance(img, torch.Tensor):
        img = tensor_img_to_numpy(img)
    img = img.clip(0, 1)

    fig = px.imshow(img, **kwargs).update_layout(
        height=img.shape[0] * scale,
        width=img.shape[1] * scale,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
    )
    return fig

class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

fig_count = 0
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        x_org, x_distorted, x_restored = nppc_model.process_batch(batch)
        # plt.figure()
        # plt.imshow(x_org[0, 0, :, :].cpu().numpy(), cmap='gray')
        # plt.savefig('gt.png')
        # plt.close()
        #
        # plt.figure()
        # plt.imshow(x_distorted[0, 0, :, :].cpu().numpy(), cmap='gray')
        # plt.savefig('distorted.png')
        # plt.close()
        #
        # plt.figure()
        # plt.imshow(x_restored[0, 0, :, :].cpu().numpy(), cmap='gray')
        # plt.savefig('restored.png')
        # plt.close()
        # exit()

        w_mat = nppc_model.get_dirs(x_distorted, x_restored, use_best=True, use_ddp=False)

        for i in range(x_org.shape[0]):
            nrow = 2
            ncol = 1

            fig = plt.figure(figsize=(ncol + 1, nrow + 1))

            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.05, hspace=0.05,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
            ax = plt.subplot(gs[0, 0])
            ax.imshow(x_restored[i, 0, :, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(gs[1, 0])
            ax.imshow(x_org[i, 0, :, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

            plt.savefig(f'/home/bendel.8/Git_Repos/mnist_testing/test_ims_rcgan/mnist_left_bottom_nppc_{fig_count}.png',
                        bbox_inches='tight', dpi=300)

            plt.close(fig)

            nrow = 1
            ncol = 5

            fig = plt.figure(figsize=(ncol + 1, nrow + 1))

            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.05, hspace=0.05,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

            for k in range(5):
                pc_np = scale_img(w_mat)[i, k, 0].cpu().numpy()

                ax = plt.subplot(gs[0, k])
                ax.imshow(pc_np, cmap='bwr', norm=MidpointNormalize(np.min(pc_np), np.max(pc_np), pc_np[0, 0]))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                if k == 1:
                    ax.patch.set_edgecolor('red')
                    ax.patch.set_linewidth(3)
                elif k == 4:
                    ax.patch.set_edgecolor('blue')
                    ax.patch.set_linewidth(3)

            plt.savefig(f'/home/bendel.8/Git_Repos/mnist_testing/test_ims_rcgan/mnist_right_top_nppc_{i}.png',
                        bbox_inches='tight', dpi=300)

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
                pc_np = w_mat[i, k, 0].cpu().numpy()

                if k == 1 or k == 4:
                    ax = plt.subplot(gs[cur_row, 0])
                    ax.imshow(x_restored[i, 0, :, :].cpu().numpy() - 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 1:
                        ax.patch.set_edgecolor('red')
                        ax.patch.set_linewidth(3)
                    else:
                        ax.patch.set_edgecolor('blue')
                        ax.patch.set_linewidth(3)

                    ax = plt.subplot(gs[cur_row, 1])
                    ax.imshow(x_restored[i, 0, :, :].cpu().numpy() - 2 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 1:
                        ax.patch.set_edgecolor('red')
                        ax.patch.set_linewidth(3)
                    else:
                        ax.patch.set_edgecolor('blue')
                        ax.patch.set_linewidth(3)

                    ax = plt.subplot(gs[cur_row, 2])
                    ax.imshow(x_restored[i, 0, :, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 1:
                        ax.patch.set_edgecolor('red')
                        ax.patch.set_linewidth(3)
                    else:
                        ax.patch.set_edgecolor('blue')
                        ax.patch.set_linewidth(3)

                    ax = plt.subplot(gs[cur_row, 3])
                    ax.imshow(x_restored[i, 0, :, :].cpu().numpy() + 2 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 1:
                        ax.patch.set_edgecolor('red')
                        ax.patch.set_linewidth(3)
                    else:
                        ax.patch.set_edgecolor('blue')
                        ax.patch.set_linewidth(3)

                    ax = plt.subplot(gs[cur_row, 4])
                    ax.imshow(x_restored[i, 0, :, :].cpu().numpy() + 3 * pc_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 1:
                        ax.patch.set_edgecolor('red')
                        ax.patch.set_linewidth(3)
                    else:
                        ax.patch.set_edgecolor('blue')
                        ax.patch.set_linewidth(3)

                    cur_row += 1

            plt.savefig(f'/home/bendel.8/Git_Repos/mnist_testing/test_ims_rcgan/mnist_right_bottom_nppc_{i}.png',
                        bbox_inches='tight', dpi=300)

            plt.close(fig)

            fig_count += 1

            if (fig_count >= 5):
                exit()
