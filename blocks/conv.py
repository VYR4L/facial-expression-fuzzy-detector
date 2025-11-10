import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Bloco de convolução padrão
    Este bloco realiza uma convolução 2D seguida de normalização em lote e
    uma função de ativação SiLU (Swish).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Passa o tensor de entrada através do bloco de convolução.
        """       
        return self.act(self.bn(self.conv(x)))
    
