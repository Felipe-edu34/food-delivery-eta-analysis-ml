"""
Módulo de visualizações científicas em alta resolução (300 DPI) para o estudo de
Redes Neurais Artificiais (MLP em PyTorch): curvas de perda, impacto de hiperparâmetros,
mapas de calor de interação, limiar de decisão, ROC, PR, resíduos e eficiência.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _get_save_path(filename: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return os.path.join(fig_dir, filename)


def plot_learning_curves_pair(
    hist_base: Dict[str, Any],
    hist_final: Dict[str, Any],
    task_title: str,
    loss_label: str,
    filename: str,
) -> None:
    """
    Plota lado a lado as curvas de aprendizado (perda de treino e validação por época)
    da configuração Baseline vs. Configuração Final, destacando o ponto ótimo do early stopping.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, h, sub_t in [
        (axes[0], hist_base, "(a) Configuração de Referência (Baseline)"),
        (axes[1], hist_final, "(b) Configuração Final Otimizada"),
    ]:
        ep = np.arange(1, len(h["treino"]) + 1)
        ax.plot(ep, h["treino"], label="Perda no Treino", color="#2a9d8f", lw=2.2)
        ax.plot(ep, h["val"], label="Perda na Validação", color="#e63946", lw=2.2)
        best_ep = h["melhor_epoca"]
        best_val = h["val"][best_ep - 1]
        ax.axvline(best_ep, color="#1d3557", linestyle="--", lw=1.8, label=f"Melhor Época = {best_ep} (Val={best_val:.4f})")
        ax.scatter([best_ep], [best_val], color="#1d3557", s=65, zorder=5)
        ax.set_title(sub_t, fontsize=11, fontweight="bold")
        ax.set_xlabel("Época")
        ax.set_ylabel(loss_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Curvas de Aprendizado e Early Stopping — {task_title}", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=150, bbox_inches="tight")
    plt.show()


def plot_hyperparameter_impact(
    df_reg: pd.DataFrame,
    curves_reg: Dict[str, Dict[str, Any]],
    df_clf: pd.DataFrame,
    curves_clf: Dict[str, Dict[str, Any]],
    param_title: str,
    filename: str,
) -> None:
    """
    Gera um painel 2x2 completo para um hiperparâmetro avaliado:
    - (0,0): Regressão - RMSE (Treino e Validação com barras de desvio entre sementes)
    - (0,1): Regressão - Curvas de perda de validação por época para cada valor testado
    - (1,0): Classificação - F1-Score (Treino e Validação com barras de desvio entre sementes)
    - (1,1): Classificação - Curvas de perda de validação por época para cada valor testado
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    palette = sns.color_palette("tab10", max(len(df_reg), len(df_clf)))

    # (a) Regressão: RMSE Treino vs Validação com barras de desvio
    x_r = np.arange(len(df_reg))
    axes[0, 0].errorbar(
        x_r, df_reg["treino_media"], yerr=df_reg["treino_dp"],
        fmt="o--", color="#2a9d8f", lw=2, capsize=5, label="Treino (média ± DP)"
    )
    axes[0, 0].errorbar(
        x_r, df_reg["val_media"], yerr=df_reg["val_dp"],
        fmt="s-", color="#e63946", lw=2.2, capsize=5, label="Validação (média ± DP)"
    )
    axes[0, 0].set_xticks(x_r)
    axes[0, 0].set_xticklabels(df_reg["valor_testado"].astype(str), fontweight="bold", rotation=15)
    axes[0, 0].set_ylabel("RMSE em Minutos (menor é melhor)")
    axes[0, 0].set_title(f"(a) Regressão: RMSE vs. {param_title}", fontsize=11, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (b) Regressão: Curvas de perda na validação por época
    for idx, (val_lbl, h) in enumerate(curves_reg.items()):
        ep = np.arange(1, len(h["val"]) + 1)
        axes[0, 1].plot(ep, np.clip(h["val"], 0, 2.5), lw=1.9, color=palette[idx % len(palette)], label=f"{val_lbl}")
    axes[0, 1].set_title(f"(b) Regressão: Evolução da Perda de Validação ({param_title})", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Época")
    axes[0, 1].set_ylabel("Perda na Validação (Escala Padronizada)")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, alpha=0.3)

    # (c) Classificação: F1-Score Treino vs Validação com barras de desvio
    x_c = np.arange(len(df_clf))
    axes[1, 0].errorbar(
        x_c, df_clf["treino_media"], yerr=df_clf["treino_dp"],
        fmt="o--", color="#2a9d8f", lw=2, capsize=5, label="Treino (média ± DP)"
    )
    axes[1, 0].errorbar(
        x_c, df_clf["val_media"], yerr=df_clf["val_dp"],
        fmt="s-", color="#1d3557", lw=2.2, capsize=5, label="Validação (média ± DP)"
    )
    axes[1, 0].set_xticks(x_c)
    axes[1, 0].set_xticklabels(df_clf["valor_testado"].astype(str), fontweight="bold", rotation=15)
    axes[1, 0].set_ylabel("F1-Score (maior é melhor)")
    axes[1, 0].set_title(f"(c) Classificação: F1-Score vs. {param_title}", fontsize=11, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Classificação: Curvas de perda na validação por época
    for idx, (val_lbl, h) in enumerate(curves_clf.items()):
        ep = np.arange(1, len(h["val"]) + 1)
        axes[1, 1].plot(ep, np.clip(h["val"], 0, 1.5), lw=1.9, color=palette[idx % len(palette)], label=f"{val_lbl}")
    axes[1, 1].set_title(f"(d) Classificação: Evolução da Perda BCE na Validação ({param_title})", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Época")
    axes[1, 1].set_ylabel("BCEWithLogitsLoss na Validação")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Impacto do Hiperparâmetro: {param_title} (Média ± Desvio entre Sementes)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(_get_save_path(filename), dpi=150, bbox_inches="tight")
    plt.show()