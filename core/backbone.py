import torch
from torch import nn
from blocks.conv import ConvBlock
from blocks.c3k2 import C3K2Block
from blocks.spff import SPFFBlock


class YOLOv11Backbone(nn.Module):
    """
    Backbone do YOLOv11 para extração de features
    
    Esta implementação segue a arquitetura do YOLOv11, com estágios progressivos
    de downsampling e extração de features em múltiplas escalas.
    
    Saídas:
        - P3: features em 1/8 da resolução original (escala pequena)
        - P4: features em 1/16 da resolução original (escala média)
        - P5: features em 1/32 da resolução original (escala grande)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Stem - entrada inicial
        self.stem = ConvBlock(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        
        # Stage 1: 640x640 -> 320x320
        self.stage1 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 2, base_channels * 2, bottleneck_channels=1, shortcut=True)
        )
        
        # Stage 2: 320x320 -> 160x160
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 4, base_channels * 4, bottleneck_channels=2, shortcut=True)
        )
        
        # Stage 3: 160x160 -> 80x80 (P3 - saída para neck)
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 8, base_channels * 8, bottleneck_channels=2, shortcut=True)
        )
        
        # Stage 4: 80x80 -> 40x40 (P4 - saída para neck)
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 16, base_channels * 16, bottleneck_channels=1, shortcut=True)
        )
        
        # Stage 5: 40x40 -> 20x20 (P5 - saída para neck)
        self.stage5 = nn.Sequential(
            ConvBlock(base_channels * 16, base_channels * 16, kernel_size=3, stride=2, padding=1),
            C3K2Block(base_channels * 16, base_channels * 16, bottleneck_channels=1, shortcut=True),
            SPFFBlock(base_channels * 16, base_channels * 16, kernel_sizes=[5, 9, 13])
        )

    def forward(self, x):
        """
        Forward pass através da backbone
        
        Args:
            x: Tensor de entrada (B, 3, H, W)
            
        Returns:
            Tupla com três tensores de features em diferentes escalas (P3, P4, P5)
        """
        x = self.stem(x)      # /2
        x = self.stage1(x)    # /4
        x = self.stage2(x)    # /8
        
        p3 = self.stage3(x)   # /16 - features de escala pequena
        p4 = self.stage4(p3)  # /32 - features de escala média
        p5 = self.stage5(p4)  # /64 - features de escala grande
        
        return p3, p4, p5
