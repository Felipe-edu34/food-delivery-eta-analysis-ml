"""
src/metrics.py
Funções para cálculo das métricas de avaliação de modelos de Regressão e Classificação
conforme a Aula 08 (Prof. Dr. Thommas K. S. Flores):
- Regressão: MAE, MSE, RMSE, MAPE (%), R2, MedAE
- Classificação: Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix (TN, FP, FN, TP)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)


def compute_regression_metrics(y_true, y_pred) -> dict:
    """
    Calcula todas as métricas obrigatórias de regressão da Aula 08 (Seção 3):
    MAE, MSE, RMSE, MAPE, R2 e MedAE.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))

    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4),
        'MAPE_%': round(mape * 100.0, 2),
        'R2': round(r2, 4),
        'MedAE': round(medae, 4)
    }


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Calcula todas as métricas obrigatórias de classificação da Aula 08 (Seção 4):
    Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC e Matriz de Confusão (TN, FP, FN, TP).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    roc_auc = np.nan
    pr_auc = np.nan

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall_vals, precision_vals))
        except Exception:
            pass

    return {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        'PR-AUC': round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'Confusion Matrix': cm
    }
