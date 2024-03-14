import nppc
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

restoration_net = nppc.RestorationModel(
    dataset='mnist',
    data_folder='/storage/matt_models/mnist/',
    distortion_type='inpainting_1',
    net_type='unet',
    lr=1e-4,
    device='cuda:0',
)
restoration_net.load('./results/mnist_inpainting/restoration/checkpoint.pt')

nppc_model = nppc.NPPCModel(
    restoration_model_folder='./results/mnist_inpainting/restoration/',
    net_type='unet',
    n_dirs=5,
    lr=1e-4,
    device='cuda:0',
)
nppc_model.load('./results/mnist_inpainting/nppc/checkpoint.pt')

dataloader = torch.utils.data.DataLoader(
    nppc_model.data_module.test_set,
    batch_size=256,
    shuffle=False,
    generator=torch.Generator().manual_seed(0),
)

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

        w_mat = nppc_model.get_dirs(x_distorted, x_restored, use_best=False, use_ddp=False)

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

            plt.savefig(f'/home/bendel.8/Git_Repos/mnist_testing/test_ims_rcgan/mnist_left_bottom_nppc_{fig_count}.png', bbox_inches='tight', dpi=300)

            plt.close(fig)


            nrow = 1
            ncol = 5

            fig = plt.figure(figsize=(ncol + 1, nrow + 1))

            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.05, hspace=0.05,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


            for k in range(5):
                pc_np = w_mat[i, k].cpu().numpy()

                ax = plt.subplot(gs[0, k])
                ax.imshow(pc_np, cmap='bwr')
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

            plt.savefig(f'/home/bendel.8/Git_Repos/mnist_testing/test_ims_rcgan/mnist_right_top_nppc_{i}.png', bbox_inches='tight', dpi=300)

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
                pc_np = w_mat[i, k].cpu().numpy()

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

            plt.savefig(f'/home/bendel.8/Git_Repos/mnist_testing/test_ims_rcgan/mnist_right_bottom_nppc_{i}.png', bbox_inches='tight', dpi=300)

            plt.close(fig)



            fig_count += 1

            if (fig_count >= 1):
                exit()

