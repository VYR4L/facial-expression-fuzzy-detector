import torch
from torch import nn
from blocks.conv import ConvBlock
from blocks.bottleneck import BottleneckBlock


class C3K2Block(nn.Module):
    """
    Bloco C3K2
    Este bloco implementa uma combinação de convoluções com Bottleneck blocks e conexão de atalho opcional.
    Segue a arquitetura do YOLOv11.
    """
    def __init__(self, in_channels, out_channels, bottleneck_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBlock(hidden_channels, out_channels, kernel_size=1)
        
        self.bottleneck = nn.ModuleList([
            BottleneckBlock(hidden_channels, hidden_channels, shortcut=False, expansion=1.0)
            for _ in range(bottleneck_channels)
        ])
        
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Passa o tensor de entrada através do bloco C3K2.
        Se a conexão de atalho for usada, a entrada é somada à saída das convoluções.
        """
        identity = x
        x = self.conv1(x)
        
        # Passa por cada Bottleneck block
        for bottleneck in self.bottleneck:
            x = bottleneck(x)
        
        x = self.conv2(x)
        
        if self.use_shortcut:
            x = x + identity
        return x

