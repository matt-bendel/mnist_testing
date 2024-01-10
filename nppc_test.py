import nppc
import torch
import numpy as np
import matplotlib.pyplot as plt

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

l2s = []
weird_l2s = []
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        x_org, x_distorted = restoration_net.process_batch(batch)
        plt.imshow(x_org[0, 0, :, :].cpu().numpy(), cmap='gray')
        plt.savefig('gt.png')
        plt.close()

        plt.imshow(x_distorted[0, 0, :, :].cpu().numpy(), cmap='gray')
        plt.savefig('gt.png')
        plt.close()

        x_restored = restoration_net.restore(x_distorted)

        plt.imshow(x_restored[0, 0, :, :].cpu().numpy(), cmap='gray')
        plt.savefig('gt.png')
        plt.close()
        exit()

        w_mat = nppc_model.get_dirs(x_distorted, x_restored, use_best=False, use_ddp=False)

        w_mat = w_mat.flatten(2)

        err = (x_org - x_restored).flatten(1)
        l2s.append(err.norm(dim=1).mean().cpu().numpy())

        for n in range(x_org.shape[0]):
            unsqueezed_err = torch.unsqueeze(err[n, :], dim=1)
            weird_l2 = torch.norm(unsqueezed_err - torch.matmul(torch.matmul(w_mat[n, :, :].transpose(0, 1), w_mat[n, :, :]), unsqueezed_err),
                                  p=2).cpu().numpy()

            weird_l2s.append(weird_l2)

    print(f'L2: {np.mean(l2s)}')
    print(f'Weird L2: {np.mean(weird_l2s)}')
