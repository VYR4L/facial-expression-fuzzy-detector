import torch
from torch import nn
from blocks.conv import ConvBlock
from blocks.bottleneck import BottleneckBlock


class C2PSABlock(nn.Module):
    """
    Bloco C2PSA (Cross-Covariance Pooling with Spatial Attention)
    Este bloco implementa uma combinação de pooling de covariância cruzada e atenção espacial.
    """
    def __init__(self, in_channels, out_channels, bottleneck_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBlock(hidden_channels, out_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.Sigmoid()
        ) 
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1)
        self.bottleneck = nn.ModuleList([
            BottleneckBlock(out_channels, out_channels, shortcut=shortcut, expansion=1.0)
            for _ in range(bottleneck_channels)
        ])

    def forward(self, x):
        """
        Passa o tensor de entrada através do bloco C2PSA.
        """
        x = self.conv1(x)
        attn = self.attention(x)
        x = x * attn
        x = self.conv2(x)
        for bottleneck in self.bottleneck:
            x = bottleneck(x)
        x = self.conv3(x)
        return x

