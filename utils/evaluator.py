"""
Evaluator - Avaliação e Testes do Modelo

Responsável por:
- Testes do modelo
- Métricas de avaliação
- Inferência
"""

import torch
from tqdm import tqdm
from pathlib import Path
from utils.metrics import LandmarkMetrics


class Evaluator:
    """
    Classe para avaliar o modelo
    """
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.metrics = LandmarkMetrics()

    @torch.no_grad()
    def evaluate(self, test_loader, save_visualizations=True, save_dir='results'):
        """
        Avalia o modelo no dataset de teste
        
        Args:
            test_loader: DataLoader com dados de teste
            save_visualizations: Se True, salva gráficos
            save_dir: Diretório para salvar resultados
        
        Returns:
            dict com todas as métricas
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0
        pbar = tqdm(test_loader, desc='Avaliando')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            visibility = batch['visibility'].to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            
            # Extrair predições
            pred_landmarks = predictions[:, :, :2]  # (B, 68, 2)
            pred_confidence = torch.sigmoid(predictions[:, :, 2])  # (B, 68)
            
            # Calcular loss
            targets = {
                'landmarks': landmarks,
                'visibility': visibility
            }
            losses = self.criterion(predictions, targets)
            total_loss += losses['loss'].item()
            
            # Atualizar métricas
            self.metrics.update(pred_landmarks, landmarks, pred_confidence)
            
            pbar.set_postfix({'loss': losses['loss'].item()})
        
        avg_loss = total_loss / len(test_loader)
        
        # Calcular todas as métricas
        nme = self.metrics.compute_nme()
        auc = self.metrics.compute_auc()
        fr = self.metrics.compute_failure_rate()
        pr = self.metrics.compute_precision_recall()
        
        # Salvar visualizações
        if save_visualizations:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            
            print(f"\n📊 Gerando visualizações...")
            self.metrics.plot_error_distribution(save_path / 'error_distribution.png')
            self.metrics.plot_confusion_heatmap(save_path / 'region_correlation.png')
            print(f"✅ Visualizações salvas em: {save_path}")
        
        return {
            'loss': avg_loss,
            'nme': nme,
            'auc': auc,
            'failure_rate': fr,
            'precision_recall': pr
        }

    def print_results(self, results):
        """Imprime relatório de resultados"""
        report = self.metrics.generate_report()
        print(report)
        
        # Salvar relatório em arquivo
        save_path = Path('results/evaluation_report.txt')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Relatório salvo em: {save_path}")
