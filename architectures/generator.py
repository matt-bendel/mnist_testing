import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Generator(nn.Module):
    """Implementation of the U-Net architecture
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, in_chans, out_chans, batch_norm=True, instance_norm=False):
        """
        """
        self.name = 'UNet'
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.filter_sizes = [32, 64, 128, 256]
        self.filter_sizes_rev = [128, 64, 32]
        self.n_block = len(self.filter_sizes)
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm
        # self.num_layers = [2, 2, 2, 2, 2]

        super(Generator, self).__init__()
        self.contract_blocks = self.contract()
        self.expand_blocks = self.expand()
        self.segment = nn.Sequential(
            nn.Conv2d(
                self.filter_sizes[0],
                self.out_chans,
                kernel_size=1
            )
        )

    def forward(self, x):
        """Performs a forward pass through the network
        """
        xs = []
        for block in self.contract_blocks:
            new_x = block['convs'](x)
            xs.append(new_x)
            x = block['resize'](new_x)

        for i, block in enumerate(self.expand_blocks):
            k = self.n_block - i - 2
            x = block['resize'](x)
            x = F.interpolate(x, size=(xs[k].shape[-2], xs[k].shape[-1]), mode='nearest')
            x = torch.concat([xs[k], x], dim=1)
            x = block['conv'](x)

        y_pred = self.segment(x)

        return y_pred

    def contract(self):
        """Define contraction block in U-Net
        """
        blocks = []
        old = self.in_chans
        for i, size in enumerate(self.filter_sizes):
            mpool = nn.MaxPool2d(kernel_size=2)
            conv1 = nn.Conv2d(old, size, kernel_size=3, padding=1, stride=1)
            conv2 = nn.Conv2d(size, size, kernel_size=3, padding=1, stride=1)
            relu = nn.LeakyReLU(negative_slope=0.1)
            convs = [conv1, relu, conv2, relu]
            if self.batch_norm:
                b_norm = nn.BatchNorm2d(size)
                convs = [conv1, b_norm, relu, conv2, b_norm, relu]
            elif self.instance_norm:
                i_norm = nn.InstanceNorm2d(size)
                convs = [conv1, i_norm, relu, conv2, i_norm, relu]

            block = nn.Sequential(*convs)
            blocks.append({'convs': block, 'resize': mpool})
            old = size
            self.add_module(f'contract{i + 1}', block)
        return blocks

    def expand(self):
        """Define expansion block in U-Net
        """
        blocks = []
        for i, size in enumerate(self.filter_sizes_rev):
            resize = nn.Conv2d(size * 2, size, kernel_size=3, stride=1, padding=1)
            self.add_module(f'up{i + 1}', resize)
            conv1 = nn.Conv2d(size * 2, size, kernel_size=3, stride=1, padding=1)
            conv2 = nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1)
            relu = nn.LeakyReLU(negative_slope=0.1)
            convs = [conv1, relu, conv2, relu]
            if self.batch_norm:
                b_norm = nn.BatchNorm2d(size)
                convs = [conv1, b_norm, relu, conv2, b_norm, relu]
            elif self.instance_norm:
                i_norm = nn.InstanceNorm2d(size)
                convs = [conv1, i_norm, relu, conv2, i_norm, relu]
            convs = nn.Sequential(*convs)
            self.add_module(f'deconv{i + 1}', convs)
            blocks.append({'resize': resize, 'conv': convs})

        return blocks
