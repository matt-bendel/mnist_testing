import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)

class Discriminator(nn.Module):
    def __init__(self, in_chans, chans=32, num_pool_layers=4):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, 0)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, 0))
            ch *= 2

        self.fc = nn.Linear(256, 1)

    def forward(self, x, y):
        output = torch.cat([x, y], dim=1)

        for layer in self.down_sample_layers:
            output = layer(output)
            output = self.avg_pool(output)

        img_flat = output.view(output.size(0), -1)

        return self.fc(img_flat)
