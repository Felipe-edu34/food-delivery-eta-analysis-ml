import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Garante que a pasta exigida no PDF exista
os.makedirs('../reports/figures', exist_ok=True)

# ==========================================
# 1. FUNÇÕES PARA O NOTEBOOK DE EDA (01_eda)
# ==========================================

def plot_missing_values(df, filename="missing_values.png"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Valores Ausentes no Dataset")
    plt.tight_layout()
    plt.savefig(f'../reports/figures/{filename}')
    plt.show()

def plot_outliers_boxplot(df, filename="outliers_boxplot.png"):
    plt.figure(figsize=(12, 6))
    # Seleciona apenas as numéricas para não dar erro no gráfico
    df_num = df.select_dtypes(include=['number'])
    sns.boxplot(data=df_num, orient='h')
    plt.title("Boxplot para Detecção de Outliers")
    plt.tight_layout()
    plt.savefig(f'../reports/figures/{filename}')
    plt.show()

def plot_correlation_matrix(df, filename="correlation_matrix.png"):
    plt.figure(figsize=(10, 8))
    df_num = df.select_dtypes(include=['number'])
    sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(f'../reports/figures/{filename}')
    plt.show()

# ==========================================
# 2. FUNÇÕES PARA OS NOTEBOOKS DE MODELAGEM
# ==========================================

def plot_regression_results(y_true, y_pred, filename="regression_scatter.png"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel("Tempo Real de Entrega (min)")
    plt.ylabel("Tempo Previsto (min)")
    plt.title("Regressão Linear: Real vs Previsto")
    plt.tight_layout()
    plt.savefig(f'../reports/figures/{filename}')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - Classificação")
    plt.tight_layout()
    plt.savefig(f'../reports/figures/{filename}')
    plt.show()