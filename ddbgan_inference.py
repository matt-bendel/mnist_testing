import torch
import yaml
import types
import json

import matplotlib.pyplot as plt
import numpy as np
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.gan_ddb import rcGANDDB
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

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")

    with open('configs/ddb.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    model = rcGANDDB.load_from_checkpoint(cfg.checkpoint_dir + 'ddb_rcgan/best.ckpt').cuda()
    model.eval()

    dm = MNISTDataModule()
    dm.setup()
    test_loader = dm.test_dataloader()

    N = 20
    delta = 1 / N
    num_samps = 4
    t_steps = (torch.arange(N) + 1) / N
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, _ = data
            y = x + torch.randn_like(x) * 1
            y = y.clamp(0, 1).repeat(4, 1, 1, 1).cuda()
            x_t = y

            for i in reversed(range(N)):
                t = t_steps[i]
                x_0_hat = model.forward(x_t, t.unsqueeze(0).repeat(num_samps).cuda())
                x_t = delta / t * x_0_hat + (1 - delta / t) * x_t

            x_np = x[0, 0, :, :].cpu().numpy()

            plt.figure()
            plt.imshow(x_np, cmap='gray')
            plt.savefig(f'test_x_ddbgan.png')
            plt.close()

            y_np = y[0, 0, :, :].cpu().numpy()

            plt.figure()
            plt.imshow(y_np, cmap='gray')
            plt.savefig(f'test_y_ddbgan.png')
            plt.close()

            for j in range(num_samps):
                x_hat_np = x_t[j, 0, :, :].cpu().numpy()

                plt.figure()
                plt.imshow(x_hat_np, cmap='gray')
                plt.savefig(f'test_recon_ddbgan_{j}.png')
                plt.close()

            exit()

        # embedding = MNISTAutoencoder.load_from_checkpoint('/storage/matt_models/mnist/autoencoder/best.ckpt').autoencoder.cuda()
    # embedding.eval()

    embedding = InceptionEmbedding()
    cfid = CFIDMetric(model, dm.val_dataloader(), embedding, embedding, True)

    cfid_val_r, m_val_r, c_val_r = cfid.get_cfid_torch_pinv() # 1.57, 12.89, 14.45
    #
    # cfid = CFIDMetric(model_lazy, dm.val_dataloader(), embedding, embedding, True)
    #
    # cfid_val, m_val, c_val = cfid.get_cfid_torch_pinv() # 2.66, 10.60, 13.26
    #
    # print('rcGAN:')
    # print(f'CFID: {cfid_val_r}')
    # print(f'M: {m_val_r}')
    # print(f'C: {c_val_r}')
    # print('EigenGAN:')
    # print(f'CFID: {cfid_val}')
    # print(f'M: {m_val}')
    # print(f'C: {c_val}')
    # exit()
