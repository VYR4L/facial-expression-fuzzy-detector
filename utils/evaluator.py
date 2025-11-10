"""
Evaluator - Avaliação e Testes do Modelo

Responsável por:
- Testes do modelo
- Métricas de avaliação
- Inferência
"""

import torch
from tqdm import tqdm


class Evaluator:
    """
    Classe para avaliar o modelo
    """
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Avalia o modelo no dataset"""
        self.model.eval()
        
        total_loss = 0
        total_loc_loss = 0
        total_conf_loss = 0
        
        print("\n🔍 Avaliando modelo...")
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            visibility = batch['visibility'].to(self.device)
            
            predictions = self.model(images)
            
            targets = {
                'landmarks': landmarks,
                'visibility': visibility
            }
            losses = self.criterion(predictions, targets)
            
            total_loss += losses['loss'].item()
            total_loc_loss += losses['loc_loss'].item()
            total_conf_loss += losses['conf_loss'].item()
        
        avg_loss = total_loss / len(dataloader)
        avg_loc_loss = total_loc_loss / len(dataloader)
        avg_conf_loss = total_conf_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'loc_loss': avg_loc_loss,
            'conf_loss': avg_conf_loss
        }

    def print_results(self, results):
        """Imprime resultados formatados"""
        print("\n" + "=" * 70)
        print("📊 Resultados da Avaliação:")
        print(f"   Total Loss: {results['loss']:.4f}")
        print(f"   Localization Loss: {results['loc_loss']:.4f}")
        print(f"   Confidence Loss: {results['conf_loss']:.4f}")
        print("=" * 70)
