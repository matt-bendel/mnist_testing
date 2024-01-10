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
restoration_net.eval()

nppc_model = nppc.NPPCModel(
    restoration_model_folder='./results/mnist_inpainting/restoration/',
    net_type='unet',
    n_dirs=5,
    lr=1e-4,
    device='cuda:0',
)
nppc_model.load('./results/mnist_inpainting/nppc/checkpoint.pt')
nppc_model.eval()

dataloader = torch.utils.data.DataLoader(
    nppc_model.data_module.test_set,
    batch_size=256,
    shuffle=False,
    generator=torch.Generator().manual_seed(0),
)

for i, batch in enumerate(dataloader):
    x_org, x_distorted = batch
    print(x_org.shape)
    print(x_distorted.shape)
    # nppc_model
