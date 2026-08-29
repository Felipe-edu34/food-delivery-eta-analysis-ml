import pandas as pd

def check_missing_values(df):
    """Retorna a quantidade e porcentagem de valores nulos por coluna."""
    missing = pd.DataFrame({
        'Valores Nulos': df.isnull().sum(),
        'Porcentagem (%)': (df.isnull().mean() * 100).round(2)
    })
    return missing[missing['Valores Nulos'] > 0]

def summary_statistics(df):
    """Retorna estatísticas descritivas básicas das colunas numéricas."""
    return df.describe()