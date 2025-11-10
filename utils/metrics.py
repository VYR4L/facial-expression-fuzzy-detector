import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from pathlib import Path
from config.settings import REGION_LANDMARK_INDICES


class LandmarkMetrics:
    """
    Classe para calcular métricas de avaliação de landmarks
    """

    def __init__(self):
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.inter_ocular_distances = []

    def reset(self):
        """Reseta as métricas"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.inter_ocular_distances = []

    def update(
        self,
        pred_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor,
        pred_confidences: torch.Tensor
    ):
        """
        Atualiza as métricas com um novo batch de dados
        Args:
            pred_landmarks (torch.Tensor): Previsões do modelo (B, 68, 2)
            target_landmarks (torch.Tensor): Landmarks verdadeiros (B, 68, 2)
            pred_confidences (torch.Tensor): Confianças previstas (B, 68)
        """
        pred_landmarks = pred_landmarks.cpu().numpy()
        target_landmarks = target_landmarks.cpu().numpy()

        if pred_confidences is not None:
            pred_confidences = pred_confidences.cpu().numpy()
        else:
            pred_confidences = np.ones((pred_landmarks.shape[0], 68))

        inter_ocular = np.linalg.norm(
            target_landmarks[:, 36] - target_landmarks[:, 45],
            axis=1
        )

        self.predictions.append(pred_landmarks)
        self.targets.append(target_landmarks)
        self.confidences.append(pred_confidences)
        self.inter_ocular_distances.append(inter_ocular)

    def compute_nme(self) -> Dict[str, float]:
        """
        Calcula o Normalized Mean Error (NME)
        
        NME = (1/N) * Σ ||pred - target|| / inter_ocular_distance

        Returns:
            Dict[str, float]: NME médio e por região facial
        """
        preds = np.concatenate(self.predictions, axis=0)  # (N, 68, 2)
        targets = np.concatenate(self.targets, axis=0)  # (N, 68, 2)
        inter_ocular_distances = np.concatenate(self.inter_ocular_distances)  # (N, 68)
        
        # Erro por landmark
        errors = np.linalg.norm(preds - targets, axis=2)  # (N, 68)

        # Normalizar pelo distância inter-ocular
        inter_ocular_distances = inter_ocular_distances[:, np.newaxis]  # (N, 1)
        normalized_errors = errors / inter_ocular_distances  # (N, 68)

        # MME Global
        nme_global = np.mean(normalized_errors) * 100  # em porcentagem

        # NME por região facial
        nme_regions = {}
        for region_name, indexes in REGION_LANDMARK_INDICES.items():
            region_error = normalized_errors[:, indexes]
            nme_regions[region_name] = np.mean(region_error) * 100

        return {
            'NME_Global': nme_global,
            **{f'NME_{k}': v for k, v in nme_regions.items()}
        }
    
    def compute_auc(self, threshold_max: float = 0.8) -> float:
        """
        Calcula a AUC (Area Under the Curve) do NME

        AUC mede a proporção de imagens com NME abaixo de diferentes thresholds
        
        Args:
            threshold_max: Threshold máximo para calcular AUC (padrão: 0.08 = 8%)
        
        Returns:
            AUC score [0, 1]
        """
        preds = np.concatenate(self.predictions, axis=0) 
        targets = np.concatenate(self.targets, axis=0) 
        inter_ocular_distances = np.concatenate(self.inter_ocular_distances)

        errors = np.linalg.norm(preds - targets, axis=2)  
        nme_per_sample = np.mean(errors / inter_ocular_distances[:, np.newaxis], axis=1)

        thresholds = np.linspace(0, threshold_max, 100)
        proportions = [np.mean(nme_per_sample < t) for t in thresholds]
        auc = np.trapezoid(proportions, thresholds) / threshold_max

        return auc

    def compute_failure_rate(self, threshold: float = 0.8) -> Dict[str, float]:
        """"
        Calcula Failure Rate (FR)

        FR = proporção de imagens com NME acima do threshold

        Args:
            threshold: Threshold para definir falha (padrão: 0.08 = 8%)

        Returns:
            Dict[str, float]: FR global e por região facial
        """
        preds = np.concatenate(self.predictions, axis=0) 
        targets = np.concatenate(self.targets, axis=0) 
        inter_ocular_distances = np.concatenate(self.inter_ocular_distances)

        errors = np.linalg.norm(preds - targets, axis=2)
        nme_per_sample = np.mean(errors / inter_ocular_distances[:, np.newaxis], axis=1)

        fr_global = np.mean(nme_per_sample > threshold) * 100  # em porcentagem

        fr_regions = {}
        for region_name, indexes in REGION_LANDMARK_INDICES.items():
            region_errors = errors[:, indexes] / inter_ocular_distances[:, np.newaxis]
            region_nme = np.mean(region_errors, axis=1)
            fr_regions[region_name] = np.mean(region_nme > threshold) * 100

        return {
            'FR_Global': fr_global,
            **{f'FR_{k}': v for k, v in fr_regions.items()}
        }
    
    def compute_precision_recall(
        self,
        distance_threshold: float = 0.05
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula Precision/Recall por região facial

        Considera um landmark como "correto" se a distância normalizada
        for menor que o threshold.

        Args:
            distance_threshold: Threshold de distância (padrão: 0.05 = 5%)
        
        Returns:
            Dict com precision/recall por região
        """
        preds = np.concatenate(self.predictions, axis=0) 
        targets = np.concatenate(self.targets, axis=0) 
        confs = np.concatenate(self.confidences, axis=0)
        inter_ocular_distances = np.concatenate(self.inter_ocular_distances)

        errors = np.linalg.norm(preds - targets, axis=2)  
        normalized_errors = errors / inter_ocular_distances[:, np.newaxis]  

        metrics_by_region = {}

        for region_name, indexes in REGION_LANDMARK_INDICES.items():
            region_errors = normalized_errors[:, indexes]
            region_confs = confs[:, indexes]

            # Landmarks corretos
            correct = region_errors < distance_threshold

            # Landmarks detectados com confiança alta (TP + FP)
            detected = region_confs > 0.5

            # Positivos verdadeiros (TP)
            tp = np.sum(correct & detected)

            # Falsos positivos (FP)
            fp = np.sum(~correct & detected)

            # Falsos negativos (FN)
            fn = np.sum(correct & ~detected)

            # Precision e Recall
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            # F1-Score
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

            metrics_by_region[region_name] = {
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100,
                'Correct': int(np.sum(correct)),
                'Total': int(correct.size)
            }

        return metrics_by_region
    
    def plot_error_distribution(self, save_path: str = None):
        """Plota distribuição de erros por região"""
        preds = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        iod = np.concatenate(self.inter_ocular_distances)
        
        errors = np.linalg.norm(preds - targets, axis=2)
        normalized_errors = (errors / iod[:, np.newaxis]) * 100  # Em percentual
        
        # Criar figura
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, (region_name, indices) in enumerate(REGION_LANDMARK_INDICES.items()):
            region_errors = normalized_errors[:, indices].flatten()
            
            axes[idx].hist(region_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].axvline(np.mean(region_errors), color='red', linestyle='--', 
                            label=f'Média: {np.mean(region_errors):.2f}%')
            axes[idx].set_title(region_name.replace('_', ' '))
            axes[idx].set_xlabel('NME (%)')
            axes[idx].set_ylabel('Frequência')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico salvo em: {save_path}")
        else:
            plt.show()

    def plot_confusion_heatmap(self, save_path: str = None):
        """
        Plota heatmap de confusão entre regiões
        
        Mostra quais regiões têm erros correlacionados
        """
        preds = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        iod = np.concatenate(self.inter_ocular_distances)
        
        errors = np.linalg.norm(preds - targets, axis=2)
        normalized_errors = errors / iod[:, np.newaxis]
        
        region_nmes = {}
        for region_name, indices in REGION_LANDMARK_INDICES.items():
            region_nmes[region_name] = np.mean(normalized_errors[:, indices], axis=1)
        
        # Criar matriz de correlação
        region_names = list(region_nmes.keys())
        correlation_matrix = np.zeros((len(region_names), len(region_names)))
        
        for i, name1 in enumerate(region_names):
            for j, name2 in enumerate(region_names):
                correlation_matrix[i, j] = np.corrcoef(
                    region_nmes[name1],
                    region_nmes[name2]
                )[0, 1]
        
        # Plotar heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            xticklabels=[n.replace('_', ' ') for n in region_names],
            yticklabels=[n.replace('_', ' ') for n in region_names],
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Correlação de Erros Entre Regiões Faciais')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap salvo em: {save_path}")
        else:
            plt.show()

    def generate_report(self) -> str:
        """Gera relatório completo de métricas"""
        nme = self.compute_nme()
        auc = self.compute_auc()
        fr = self.compute_failure_rate()
        pr = self.compute_precision_recall()
        
        report = []
        report.append("=" * 70)
        report.append("RELATÓRIO DE AVALIAÇÃO - DETECÇÃO DE LANDMARKS FACIAIS")
        report.append("=" * 70)
        
        # NME Global
        report.append(f"\n📊 NME (Normalized Mean Error):")
        report.append(f"   Global: {nme['NME_Global']:.4f}%")
        report.append(f"\n   Por Região:")
        for region in REGION_LANDMARK_INDICES.keys():
            report.append(f"      {region.replace('_', ' ')}: {nme[f'NME_{region}']:.4f}%")
        
        # AUC
        report.append(f"\n📈 AUC (Area Under Curve): {auc:.4f}")
        
        # Failure Rate
        report.append(f"\n❌ Failure Rate (NME > 8%):")
        report.append(f"   Global: {fr['FR_Global']:.2f}%")
        report.append(f"\n   Por Região:")
        for region in REGION_LANDMARK_INDICES.keys():
            report.append(f"      {region.replace('_', ' ')}: {fr[f'FR_{region}']:.2f}%")
        
        # Precision/Recall
        report.append(f"\n🎯 Precision, Recall e F1-Score por Região:")
        for region, metrics in pr.items():
            report.append(f"\n   {region.replace('_', ' ')}:")
            report.append(f"      Precision: {metrics['Precision']:.2f}%")
            report.append(f"      Recall:    {metrics['Recall']:.2f}%")
            report.append(f"      F1-Score:  {metrics['F1-Score']:.2f}%")
            report.append(f"      Corretos:  {metrics['Correct']}/{metrics['Total']}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)