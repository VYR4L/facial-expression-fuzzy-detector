import torch
from torch import nn
from blocks.conv import ConvBlock


class SPFFBlock(nn.Module):
    """
    Bloco SPFF (Spatial Pyramid Feature Fusion)
    Este bloco implementa uma fusão de características em pirâmide espacial.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.convs = nn.ModuleList([
            ConvBlock(hidden_channels, hidden_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.conv2 = ConvBlock(hidden_channels * (len(kernel_sizes) + 1), out_channels, kernel_size=1)

    def forward(self, x):
        """
        Passa o tensor de entrada através do bloco SPFF.
        """
        x = self.conv1(x)
        features = [x] + [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.conv2(x)
        return x