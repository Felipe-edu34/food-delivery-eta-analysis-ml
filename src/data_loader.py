import pandas as pd

def load_data(filepath):
    """Carrega o dataset a partir de um arquivo CSV."""
    return pd.read_csv(filepath)