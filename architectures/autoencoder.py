import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Fix compression
# TODO: SwAV

class Autoencoder(nn.Module):
    """ Autoencoder adapted from [1]

    [1]:https://github.com/Kaixhin/Autoencoders/blob/master/models/ConvAE.lua
    """

    def __init__(self, width_factor=1):
        super().__init__()
        width = int(width_factor * 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(2, stride=2, padding=1),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(2, stride=2, padding=0)
        )

        modules = []
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.decoder_3 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, features=False):
        x = self.encoder(x)

        if features:
            return x

        x = self.decoder_1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.decoder_2(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.decoder_3(x)

        return x