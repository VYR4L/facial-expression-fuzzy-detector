import torch
from torch import nn
from blocks.conv import ConvBlock


class LandmarkDetectionHead(nn.Module):
    """
    Head para detecção de landmarks faciais (68 pontos)
    
    Detecta os 68 landmarks do padrão 300W/iBUG para análise de expressões faciais.
    Para cada landmark, prediz:
        - (x, y): coordenadas normalizadas
        - confidence: confiança da detecção
    
    Mapeia as features do neck para as coordenadas finais dos landmarks.
    """
    def __init__(self, in_channels=[512, 1024, 1024], num_landmarks=68):
        super().__init__()
        self.num_landmarks = num_landmarks
        
        # Head para cada escala (N3, N4, N5)
        # Cada landmark tem: x, y, confidence = 3 valores
        self.head_n3 = self._make_head(in_channels[0])
        self.head_n4 = self._make_head(in_channels[1])
        self.head_n5 = self._make_head(in_channels[2])
        
        # Camada de fusão para combinar predições multi-escala
        self.fusion = nn.Sequential(
            nn.Conv2d(num_landmarks * 3 * 3, num_landmarks * 3, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Camada final para predições dos landmarks
        # Saída: (num_landmarks * 3) -> para cada landmark: x, y, confidence
        self.output = nn.Conv2d(num_landmarks * 3, num_landmarks * 3, kernel_size=1)

    def _make_head(self, in_channels):
        """
        Cria um head de detecção para uma escala específica
        
        Args:
            in_channels: número de canais de entrada
            
        Returns:
            Sequential com camadas de detecção
        """
        return nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, kernel_size=3, padding=1),
            ConvBlock(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 4, self.num_landmarks * 3, kernel_size=1)
        )

    def forward(self, features):
        """
        Forward pass através do head
        
        Args:
            features: Tupla (N3, N4, N5) do neck
            
        Returns:
            Tensor (B, num_landmarks, 3) com:
                - [:, :, 0]: coordenadas x normalizadas [0, 1]
                - [:, :, 1]: coordenadas y normalizadas [0, 1]
                - [:, :, 2]: confidence scores [0, 1]
        """
        n3, n4, n5 = features
        batch_size = n3.size(0)
        
        # Predições em cada escala
        pred_n3 = self.head_n3(n3)  # (B, num_landmarks*3, H/8, W/8)
        pred_n4 = self.head_n4(n4)  # (B, num_landmarks*3, H/16, W/16)
        pred_n5 = self.head_n5(n5)  # (B, num_landmarks*3, H/32, W/32)
        
        # Redimensionar todas para a mesma resolução (N3)
        target_size = pred_n3.shape[2:]
        pred_n4_up = nn.functional.interpolate(pred_n4, size=target_size, mode='bilinear', align_corners=False)
        pred_n5_up = nn.functional.interpolate(pred_n5, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenar predições multi-escala
        multi_scale = torch.cat([pred_n3, pred_n4_up, pred_n5_up], dim=1)
        
        # Fusão das predições
        fused = self.fusion(multi_scale)
        
        # Predição final
        output = self.output(fused)  # (B, num_landmarks*3, H, W)
        
        # Global Average Pooling para reduzir dimensões espaciais
        output = nn.functional.adaptive_avg_pool2d(output, (1, 1))  # (B, num_landmarks*3, 1, 1)
        output = output.view(batch_size, self.num_landmarks, 3)  # (B, num_landmarks, 3)
        
        # Aplicar ativações apropriadas
        # Sigmoid para normalizar coordenadas e confidence em [0, 1]
        output = torch.sigmoid(output)
        
        return output


class YOLOv11LandmarkDetector(nn.Module):
    """
    Modelo completo YOLOv11 para detecção de landmarks faciais
    
    Combina Backbone + Neck + Head para detectar 68 landmarks faciais
    conforme descrito no pipeline do README.
    
    Uso:
        model = YOLOv11LandmarkDetector()
        landmarks = model(image)  # (B, 68, 3) -> x, y, confidence
    """
    def __init__(self, in_channels=3, base_channels=64, num_landmarks=68):
        super().__init__()
        
        # Importar aqui para evitar imports circulares
        from core.backbone import YOLOv11Backbone
        from core.neck import YOLOv11Neck
        
        self.backbone = YOLOv11Backbone(in_channels=in_channels, base_channels=base_channels)
        
        # Canais da backbone: P3, P4, P5
        channels = [base_channels * 8, base_channels * 16, base_channels * 16]
        self.neck = YOLOv11Neck(channels=channels)
        
        self.head = LandmarkDetectionHead(in_channels=channels, num_landmarks=num_landmarks)

    def forward(self, x):
        """
        Forward pass completo
        
        Args:
            x: Tensor de entrada (B, 3, H, W) - imagem RGB
            
        Returns:
            Tensor (B, 68, 3) com landmarks detectados:
                - [:, :, 0]: coordenadas x
                - [:, :, 1]: coordenadas y
                - [:, :, 2]: confidence
        """
        # Extração de features (backbone)
        p3, p4, p5 = self.backbone(x)
        
        # Agregação multi-escala (neck)
        n3, n4, n5 = self.neck((p3, p4, p5))
        
        # Detecção de landmarks (head)
        landmarks = self.head((n3, n4, n5))
        
        return landmarks

    def predict_landmarks(self, x, confidence_threshold=0.5):
        """
        Predição com filtragem por confidence
        
        Args:
            x: Tensor de entrada (B, 3, H, W)
            confidence_threshold: threshold mínimo de confiança
            
        Returns:
            landmarks: (B, 68, 2) coordenadas x, y
            confidence: (B, 68) scores de confiança
            mask: (B, 68) máscara booleana de landmarks válidos
        """
        output = self.forward(x)
        
        landmarks = output[:, :, :2]  # (B, 68, 2) - x, y
        confidence = output[:, :, 2]  # (B, 68) - confidence
        
        # Máscara de landmarks com confiança suficiente
        mask = confidence > confidence_threshold
        
        return landmarks, confidence, mask
