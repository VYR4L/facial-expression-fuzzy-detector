import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    """
    Wing Loss para regressão de landmarks
    """
    def __init__(self, omega=10.0, epsilon=2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1 + self.omega / self.epsilon))

    def forward(self, pred, target):
        """
        Calcula a Wing Loss entre as previsões e os valores verdadeiros.

        Args:
            pred (torch.Tensor): Tensor de previsões.
            target (torch.Tensor): Tensor de valores verdadeiros.
        Returns:
            torch.Tensor: Valor da Wing Loss.
        """
        delta = (target - pred).abs()

        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        return loss.mean()
    

class LandmarkLoss(nn.Module):
    """
    Loss completa para detecção de landmarks faciais
    
    Combina:
    1. Localization Loss (Wing Loss para coordenadas)
    2. Confidence Loss (BCE para visibilidade)
    3. Weighted Loss (pesos diferentes por região facial)
    """
    def __init__(self, wing_omega=10.0, wing_epsilon=2.0, lambda_coord=5.0, lambda_conf=1.0, use_weights=True):
        super().__init__()
        self.wing_loss = WingLoss(omega=wing_omega, epsilon=wing_epsilon)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf
        self.use_weights = use_weights

        # Pesos para diferentes regiões faciais 
        self.region_weights =  self._create_region_weights()

    def _create_region_weights(self):
        weights = torch.ones(68)
        
        # Contorno facial (1-17)
        weights[0:17] = 1.0
        
        # Sobrancelhas (18-27)
        weights[17:27] = 1.5
        
        # Nariz (28-36)
        weights[27:36] = 1.0
        
        # Olhos (37-48)
        weights[36:48] = 2.0
        
        # Boca (49-68)
        weights[48:68] = 2.0
        
        return weights
    
    def forward(self, predictions, targets, masks=None):
        """
        Calcula a perda total para detecção de landmarks faciais.

        Args:
            predictions: (B, 68, 3) - [x, y, confidence_logit]
            targets: dict com:
                - 'landmarks': (B, 68, 2) - coordenadas ground truth
                - 'visibility': (B, 68) - 1 se visível, 0 se ocluído
            masks: (B, 68) - opcional, máscara de landmarks válidos
        
        Returns:
            dict com losses individuais e total
        """
        pred_coords = predictions[:, :, :2]
        pred_conf_logits = predictions[:, :, 2]

        target_coords = targets['landmarks']  # (B, 68, 2)
        target_visibility = targets['visibility'].float()  # (B, 68)
        
        # 1. LOCALIZATION LOSS (Wing Loss para coordenadas)
        loc_loss = self.wing_loss(pred_coords, target_coords)
        
        # Aplicar pesos por região se habilitado
        if self.use_weights:
            weights = self.region_weights.to(pred_coords.device)
            # Expandir para (B, 68, 2)
            weights = weights.unsqueeze(0).unsqueeze(-1).expand_as(pred_coords)
            loc_loss = (loc_loss * weights).mean()
        
        # Aplicar máscara de visibilidade (apenas landmarks visíveis)
        if masks is not None:
            mask_expanded = masks.unsqueeze(-1).expand_as(pred_coords)
            loc_loss = (loc_loss * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
        
        # 2. CONFIDENCE LOSS (BCE para visibilidade)
        conf_loss = self.bce_loss(pred_conf_logits, target_visibility)
        
        # Aplicar máscara se fornecida
        if masks is not None:
            conf_loss = (conf_loss * masks).sum() / masks.sum().clamp(min=1)
        else:
            conf_loss = conf_loss.mean()
        
        # 3. LOSS TOTAL (combinação ponderada)
        total_loss = (
            self.lambda_coord * loc_loss +
            self.lambda_conf * conf_loss
        )
        
        return {
            'loss': total_loss,
            'loc_loss': loc_loss.detach(),
            'conf_loss': conf_loss.detach()
        }
    

class AdaptativeWingLoss(nn.Module):
    """
    Adaptative Wing Loss para regressão de landmarks
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, 2) - coordenadas preditas
            target: (B, N, 2) - coordenadas ground truth
        """
        delta = (target - pred).abs()
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target)))
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )
        
        return loss.mean()
