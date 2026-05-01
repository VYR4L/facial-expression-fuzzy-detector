"""
metrics.py — Métricas para avaliação de Action Units.

Métricas implementadas:
    - F1-Score por AU (binary)
    - F1 médio (macro e weighted)
    - MAE de intensidade por AU
    - MAE médio
    - Relatório de classificação por AU
"""
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
)

from config.settings import AU_NAMES


class AUMetrics:
    """
    Acumula predições e rótulos ao longo de um epoch e calcula métricas AU.

    Uso:
        metrics = AUMetrics()
        for batch in loader:
            out = model(batch['image'])
            metrics.update(out, batch)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._binary_pred:     list[np.ndarray] = []  # lista de (B, 12) bool
        self._binary_true:     list[np.ndarray] = []  # lista de (B, 12) 0/1
        self._intensity_pred:  list[np.ndarray] = []  # lista de (B, 12) float
        self._intensity_true:  list[np.ndarray] = []  # lista de (B, 12) float
        self._logits:          list[np.ndarray] = []  # lista de (B, 12) float (para mAP)

    # ── Acumulação ────────────────────────────────────────────────────────────
    def update(self, predictions: dict, targets: dict, threshold: float = 0.5):
        """
        Registra um batch de predições e rótulos.

        Args:
            predictions: dict com 'binary_logits' e 'intensity'
            targets:     dict com 'binary' e 'intensity'
            threshold:   limiar para converter logits em binário
        """
        logits = predictions['binary_logits'].detach().cpu()
        probs  = torch.sigmoid(logits).numpy()
        binary_pred = (probs >= threshold).astype(np.float32)
        binary_true = targets['binary'].detach().cpu().numpy().astype(np.float32)
        intensity_pred = predictions['intensity'].detach().cpu().numpy()
        intensity_true = targets['intensity'].detach().cpu().numpy()

        self._logits.append(probs)
        self._binary_pred.append(binary_pred)
        self._binary_true.append(binary_true)
        self._intensity_pred.append(intensity_pred)
        self._intensity_true.append(intensity_true)

    # ── Cálculo ───────────────────────────────────────────────────────────────
    def compute(self) -> dict:
        """
        Calcula e retorna todas as métricas acumuladas.

        Returns:
            dict com:
                'f1_per_au':        ndarray (12,) — F1 por AU
                'f1_macro':         float          — média simples de F1
                'f1_weighted':      float          — F1 ponderado por suporte
                'mae_per_au':       ndarray (12,)  — MAE de intensidade por AU
                'mae_mean':         float          — MAE médio
                'map':              float          — mAP multi-label
                'classification_report': str       — relatório detalhado
        """
        binary_pred  = np.concatenate(self._binary_pred,    axis=0)   # (N, 12)
        binary_true  = np.concatenate(self._binary_true,    axis=0)
        intens_pred  = np.concatenate(self._intensity_pred, axis=0)
        intens_true  = np.concatenate(self._intensity_true, axis=0)
        logits       = np.concatenate(self._logits,         axis=0)

        # F1 por AU
        f1_per_au = np.array([
            f1_score(binary_true[:, i], binary_pred[:, i], zero_division=0)
            for i in range(len(AU_NAMES))
        ])

        # F1 macro e weighted (sklearn flattened multi-label)
        f1_macro    = f1_score(binary_true, binary_pred, average='macro',    zero_division=0)
        f1_weighted = f1_score(binary_true, binary_pred, average='weighted', zero_division=0)

        # MAE de intensidade (só frames com AU ativa no rótulo)
        mae_per_au = np.zeros(len(AU_NAMES), dtype=np.float32)
        for i in range(len(AU_NAMES)):
            mask = binary_true[:, i] > 0
            if mask.any():
                mae_per_au[i] = np.abs(intens_pred[mask, i] - intens_true[mask, i]).mean()

        # mAP multi-label (média da AP por AU)
        ap_per_au = np.array([
            average_precision_score(binary_true[:, i], logits[:, i], pos_label=1)
            if binary_true[:, i].sum() > 0 else 0.0
            for i in range(len(AU_NAMES))
        ])
        map_score = float(ap_per_au.mean())

        # Relatório textual
        report = classification_report(
            binary_true,
            binary_pred,
            target_names=AU_NAMES,
            zero_division=0,
        )

        return {
            'f1_per_au':             f1_per_au,
            'f1_macro':              float(f1_macro),
            'f1_weighted':           float(f1_weighted),
            'mae_per_au':            mae_per_au,
            'mae_mean':              float(mae_per_au.mean()),
            'map':                   map_score,
            'ap_per_au':             ap_per_au,
            'classification_report': report,
        }

    # ── Utilidades ────────────────────────────────────────────────────────────
    @staticmethod
    def format_summary(results: dict) -> str:
        """Formata um resumo legível das métricas."""
        lines = [
            f"{'AU':<8} {'F1':>6} {'MAE':>6} {'AP':>6}",
            "-" * 30,
        ]
        for i, au in enumerate(AU_NAMES):
            lines.append(
                f"{au:<8} {results['f1_per_au'][i]:>6.4f} "
                f"{results['mae_per_au'][i]:>6.4f} "
                f"{results['ap_per_au'][i]:>6.4f}"
            )
        lines += [
            "-" * 30,
            f"{'Macro F1':<15} {results['f1_macro']:.4f}",
            f"{'Weighted F1':<15} {results['f1_weighted']:.4f}",
            f"{'Mean MAE':<15} {results['mae_mean']:.4f}",
            f"{'mAP':<15} {results['map']:.4f}",
        ]
        return "\n".join(lines)
