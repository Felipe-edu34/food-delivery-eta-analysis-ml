import pandas as pd

def clean_data(df):
    """Realiza a limpeza básica e seleção de variáveis numéricas."""
    colunas_texto_ids = [
        'order_id', 'customer_id', 'restaurant_id', 'driver_id', 
        'order_timestamp', 'order_date', 'city', 'delivery_area', 
        'restaurant_type', 'restaurant_primary_category', 'payment_method'
    ]
    df_clean = df.drop(columns=colunas_texto_ids, errors='ignore')
    return df_clean