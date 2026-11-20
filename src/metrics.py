"""
Módulo de métricas de avaliação para Regressão e Classificação (Taxonomia da Aula 08),
incluindo otimização de limiar de decisão sobre o conjunto de validação.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula as 6 métricas obrigatórias de regressão na escala original do target (minutos):
    MAE, MSE, RMSE, MAPE (%), R2 e MedAE.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_t - y_p) / np.clip(np.abs(y_t), 1e-8, None))) * 100.0)
    r2 = float(r2_score(y_t, y_p))
    medae = float(median_absolute_error(y_t, y_p))

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2, 4),
        "MedAE": round(medae, 4),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    threshold: float = None,
) -> Dict[str, Any]:
    """
    Calcula as métricas obrigatórias de classificação binária da Aula 08:
    Matriz de Confusão, Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC e PR-AUC.
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    if threshold is not None and y_prob is None:
        y_prob = np.asarray(y_pred, dtype=np.float64).ravel()
        y_p = (y_prob >= float(threshold)).astype(np.int64)
    else:
        y_p = np.asarray(y_pred, dtype=np.int64).ravel()

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=np.float64).ravel()
        roc_auc = float(roc_auc_score(y_t, y_pr))
        pr_auc = float(average_precision_score(y_t, y_pr))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Specificity": round(spec, 4),
        "F1-score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "PR-AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Confusion Matrix": cm,
    }


def evaluate_thresholds_on_validation(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Avalia diferentes limiares de decisão exclusivamente no conjunto de validação
    para maximizar o F1-score e demonstrar o trade-off entre Precision e Recall.
    """
    if thresholds is None:
        thresholds = np.linspace(0.15, 0.85, 71)

    y_t = np.asarray(y_val, dtype=np.int64).ravel()
    y_pr = np.asarray(y_prob_val, dtype=np.float64).ravel()

    rows = []
    for thr in thresholds:
        y_p = (y_pr >= thr).astype(np.int64)
        rows.append({
            "limiar": round(float(thr), 3),
            "Accuracy": round(float(accuracy_score(y_t, y_p)), 4),
            "Precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
            "F1-score": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
        })

    df_thr = pd.DataFrame(rows)
    best_idx = df_thr["F1-score"].idxmax()
    best_thr = float(df_thr.loc[best_idx, "limiar"])
    return best_thr, df_thr
