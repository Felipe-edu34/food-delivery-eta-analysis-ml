"""
src/eda.py
Funções utilitárias para Análise Exploratória de Dados (EDA):
- Inspeção de tipos e dimensões
- Estatísticas descritivas completas (incluindo mediana, IQR, assimetria e curtose)
- Mapeamento quantitativo e percentual de valores ausentes
- Detecção sistemática de outliers pelo método do Intervalo Interquartil (IQR)
- Extração de pares de variáveis com correlação significativa
"""

import pandas as pd
import numpy as np


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a quantidade, porcentagem e tipo de dado das colunas com valores nulos."""
    total = len(df)
    null_counts = df.isnull().sum()
    missing = pd.DataFrame({
        'Tipo': df.dtypes,
        'Valores_Ausentes': null_counts,
        'Valores_Preenchidos': total - null_counts,
        'Porcentagem_Ausente (%)': ((null_counts / total) * 100).round(2)
    })
    return missing[missing['Valores_Ausentes'] > 0].sort_values(
        by='Porcentagem_Ausente (%)', ascending=False
    )


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna estatísticas descritivas detalhadas das colunas numéricas:
    contagem, média, desvio padrão, mínimo, Q1 (25%), mediana (50%), Q3 (75%), máximo, IQR e assimetria.
    """
    num_df = df.select_dtypes(include=[np.number])
    desc = num_df.describe().T
    desc['IQR'] = desc['75%'] - desc['25%']
    desc['skewness'] = num_df.skew()
    cols_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'IQR', 'skewness']
    return desc[cols_order].round(2)


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o método IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR) em todas as variáveis numéricas contínuas/discretas
    (ignorando variáveis binárias 0/1) e retorna uma tabela com limites, contagens e percentuais.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    records = []
    for col in num_cols:
        series = df[col].dropna()
        if series.nunique() <= 2:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_low = (series < lower_bound).sum()
        outliers_high = (series > upper_bound).sum()
        total_outliers = outliers_low + outliers_high
        pct_outliers = (total_outliers / len(series)) * 100

        records.append({
            'Variavel': col,
            'Q1 (25%)': round(q1, 2),
            'Mediana (50%)': round(series.median(), 2),
            'Q3 (75%)': round(q3, 2),
            'IQR': round(iqr, 2),
            'Limite_Inferior': round(lower_bound, 2),
            'Limite_Superior': round(upper_bound, 2),
            'Outliers_Abaixo': int(outliers_low),
            'Outliers_Acima': int(outliers_high),
            'Total_Outliers': int(total_outliers),
            'Porcentagem (%)': round(pct_outliers, 2)
        })
    return pd.DataFrame(records).sort_values(by='Total_Outliers', ascending=False).reset_index(drop=True)


def get_high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Identifica todos os pares de variáveis numéricas cuja correlação linear de Pearson
    em módulo seja maior ou igual ao limiar especificado (|r| >= threshold).
    """
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i):
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) >= threshold:
                pairs.append({
                    'Variavel_1': cols[i],
                    'Variavel_2': cols[j],
                    'Correlacao_Pearson (r)': round(float(r_val), 4),
                    'Correlacao_Absoluta (|r|)': round(abs(float(r_val)), 4)
                })
    return pd.DataFrame(pairs).sort_values(by='Correlacao_Absoluta (|r|)', ascending=False).reset_index(drop=True)