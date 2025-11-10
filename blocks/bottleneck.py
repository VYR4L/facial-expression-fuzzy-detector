import torch
from torch import nn
from blocks.conv import ConvBlock


class BottleneckBlock(nn.Module):
    """
    Bloco Bottleneck
    Este bloco implementa um bloco residual com duas convoluções.
    O bloco pode incluir uma conexão de atalho (skip connection) se o número de canais de entrada
    for igual ao número de canais de saída e o parâmetro shortcut for True.
    """
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Passa o tensor de entrada através do bloco Bottleneck.
        Se a conexão de atalho for usada, a entrada é somada à saída das convoluções.
        """
        y = self.conv2(self.conv1(x))
        if self.use_shortcut:
            y = y + x
        return y
