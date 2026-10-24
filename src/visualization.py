"""
src/visualization.py
Funções para geração e salvamento de gráficos para análise exploratória e métricas de modelos (Aula 08).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.tree import plot_tree

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _get_save_path(filename):
    return os.path.join(FIGURES_DIR, filename)


def plot_outliers_boxplot(df, filename="outliers_boxplot.png"):
    plt.figure(figsize=(12, 6))
    df_num = df.select_dtypes(include=['number'])
    sns.boxplot(data=df_num, orient='h')
    plt.title("Boxplot para Detecção de Outliers")
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()


def plot_missing_values(df, filename="missing_values.png"):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Valores Ausentes no Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()


def plot_correlation_matrix(df, filename="correlation_matrix.png"):
    plt.figure(figsize=(18, 14))
    df_num = df.select_dtypes(include=['number'])
    sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.5)
    plt.title("Matriz de Correlação", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()


def plot_confusion_matrix_fig(y_true, y_pred, model_name="Modelo", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=["No Prazo (0)", "Atrasado (1)"],
        yticklabels=["No Prazo (0)", "Atrasado (1)"]
    )
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.title(f"Matriz de Confusão - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()


def plot_all_confusion_matrices(preds_dict, y_true, filename="confusion_matrices_all.png"):
    """Plota um painel 2x2 com as Matrizes de Confusão dos 4 modelos lado a lado."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (name, y_pred) in zip(axes.flatten(), preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([
            [f"TN = {tn:,}\n({tn/len(y_true)*100:.1f}%)", f"FP = {fp:,}\n({fp/len(y_true)*100:.1f}%)"],
            [f"FN = {fn:,}\n({fn/len(y_true)*100:.1f}%)", f"TP = {tp:,}\n({tp/len(y_true)*100:.1f}%)"]
        ])
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Prev: No Prazo (0)", "Prev: Atrasado (1)"],
            yticklabels=["Real: No Prazo (0)", "Real: Atrasado (1)"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"Matriz de Confusão: {name}", fontsize=12, fontweight="bold")
    plt.suptitle("Comparação Lado a Lado das Matrizes de Confusão no Conjunto de Teste (N = 9.827)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models_dict, X_test, y_test, filename="roc_curves.png"):
    plt.figure(figsize=(8.5, 6))
    colors = ["#8d99ae", "#457b9d", "#2a9d8f", "#e63946"]
    for (name, (model, y_prob)), col in zip(models_dict.items(), colors):
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            score = auc(fpr, tpr)
            ls = "--" if "Especialista" in name else "-"
            plt.plot(fpr, tpr, linestyle=ls, lw=2.3, color=col, label=f"{name} (ROC-AUC = {score:.4f})")
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5, label="Classificador Aleatório (AUC = 0.5000)")
    plt.xlabel("Taxa de Falsos Positivos - FPR = FP / (FP + TN)", fontsize=11)
    plt.ylabel("Taxa de Verdadeiros Positivos - TPR (Recall)", fontsize=11)
    plt.title("Curvas ROC Comparativas no Conjunto de Teste (Aula 08)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()


def plot_precision_recall_curves(models_dict, X_test, y_test, filename="pr_curves.png"):
    plt.figure(figsize=(8.5, 6))
    colors = ["#8d99ae", "#457b9d", "#2a9d8f", "#e63946"]
    baseline = float(np.mean(y_test))
    for (name, (model, y_prob)), col in zip(models_dict.items(), colors):
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            score = auc(recall, precision)
            ls = "--" if "Especialista" in name else "-"
            plt.plot(recall, precision, linestyle=ls, lw=2.3, color=col, label=f"{name} (PR-AUC = {score:.4f})")
    plt.axhline(baseline, color='k', linestyle=':', lw=1.5, label=f"Proporção Positiva Base ({baseline:.4f})")
    plt.xlabel("Recall (Sensibilidade = TP / (TP + FN))", fontsize=11)
    plt.ylabel("Precision (Precisão = TP / (TP + FP))", fontsize=11)
    plt.title("Curvas Precision-Recall (PR) Comparativas no Conjunto de Teste (Aula 08)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()


def plot_residuals(y_true, y_pred, model_name="Modelo", filename="residuals_plot.png"):
    residuos = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuos, alpha=0.35, color='teal', s=20)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel("Valores Previstos (minutos)")
    plt.ylabel("Resíduos e_i = y_i - y_hat_i (minutos)")
    plt.title(f"Gráfico de Resíduos - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()


def plot_decision_tree_structure(model, feature_names, max_depth=3, filename="decision_tree.png"):
    plt.figure(figsize=(16, 8))
    plot_tree(model, max_depth=max_depth, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
    plt.title(f"Árvore de Decisão (Profundidade Máx = {max_depth})")
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=300)
    plt.close()