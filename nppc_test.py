import nppc
import torch

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

for i, batch in enumerate(dataloader):
    x_org, x_distorted = restoration_net.process_batch(batch)
    x_restored = restoration_net.restore(x_distorted)
    w_mat = nppc_model.get_dirs(x_distorted, x_restored, use_best=False, use_ddp=False)
    print(w_mat.shape)
    exit()
    # nppc_model
