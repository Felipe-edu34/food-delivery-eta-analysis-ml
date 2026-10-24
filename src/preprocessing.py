"""
src/preprocessing.py
Funções de limpeza, tratamento de outliers (Winsorization/Capping), remoção de vazamento de dados (data leakage),
eliminação de multicolinearidade financeira, codificação One-Hot e seleção estatística de features
para os problemas de Regressão (actual_delivery_time_minutes) e Classificação (late_delivery).
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression, f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Colunas de identificação e metadados textuais de alta cardinalidade / sem poder preditivo direto
ID_AND_TEXT_COLS = [
    'order_id', 'customer_id', 'restaurant_id', 'driver_id',
    'order_timestamp', 'order_date', 'city', 'delivery_area',
    'restaurant_type', 'restaurant_primary_category', 'payment_method'
]

# Colunas pós-evento que configuram vazamento de dados (Data Leakage) no momento do pedido
POST_DELIVERY_LEAKAGE_COLS = [
    'cancellation_reason',
    'order_status',
    'customer_rating',
    'tip_amount'
]

# Colunas financeiras redundantes por multicolinearidade severa com order_total (|r| > 0.94)
COLLINEAR_FINANCIAL_COLS = [
    'subtotal',
    'tax_amount',
    'service_fee'
]

# Colunas categóricas nominais para One-Hot Encoding
CATEGORICAL_ENCODE_COLS = [
    'weather',
    'traffic_level',
    'customer_type',
    'day_of_week'
]


def clean_raw_dataset(df_raw: pd.DataFrame, max_delivery_minutes: float = 120.0) -> pd.DataFrame:
    """
    Realiza a limpeza fundamentada do dataset bruto:
    1. Remove a coluna cancellation_reason (98,26% de nulos e variável pós-evento).
    2. Remove os 868 registros de pedidos cancelados (order_status != 'Completed'), que concentram
       100% dos valores ausentes em actual_delivery_time_minutes, late_delivery, tip_amount e customer_rating.
    3. Aplica Winsorization (capping superior em 120,0 min) sobre actual_delivery_time_minutes
       para estabilizar a cauda extrema superior sem descartar entregas reais.
    4. Garante tipagem inteira (0/1) para o alvo de classificação late_delivery.
    """
    df = df_raw.copy()
    if 'cancellation_reason' in df.columns:
        df = df.drop(columns=['cancellation_reason'])

    # Filtra apenas pedidos efetivamente concluídos (remove os 868 cancelados com alvos nulos)
    df = df.dropna(subset=['actual_delivery_time_minutes', 'late_delivery']).copy()

    # Capping (Winsorization) no teto operacional de 120 minutos
    df['actual_delivery_time_minutes'] = np.where(
        df['actual_delivery_time_minutes'] > max_delivery_minutes,
        max_delivery_minutes,
        df['actual_delivery_time_minutes']
    )

    df['late_delivery'] = df['late_delivery'].astype(int)
    return df


def prepare_modeling_datasets(df_clean: pd.DataFrame):
    """
    Prepara os datasets de modelagem (em unidades originais para evitar data leakage antes do split):
    - Remove IDs, variáveis pós-entrega (customer_rating, tip_amount, order_status) e variáveis
      financeiras colineares (subtotal, tax_amount, service_fee), mantendo apenas order_total.
    - Aplica One-Hot Encoding (drop_first=True) nas variáveis categóricas nominais.
    - Retorna (df_encoded, df_regression, df_classification).
    """
    cols_to_remove = ID_AND_TEXT_COLS + POST_DELIVERY_LEAKAGE_COLS + COLLINEAR_FINANCIAL_COLS
    df_base = df_clean.drop(columns=[c for c in cols_to_remove if c in df_clean.columns]).copy()

    # One-Hot Encoding com drop_first=True para evitar a armadilha da variável dummy
    df_encoded = pd.get_dummies(
        df_base,
        columns=[c for c in CATEGORICAL_ENCODE_COLS if c in df_base.columns],
        drop_first=True,
        dtype=int
    )

    # Garante que todas as colunas booleanas sejam int (0 ou 1)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)

    # Dataset de Regressão: todas as features preditivas + alvo actual_delivery_time_minutes (sem late_delivery)
    df_regression = df_encoded.drop(columns=['late_delivery'], errors='ignore').copy()

    # Dataset de Classificação: todas as features preditivas + alvo late_delivery (sem actual_delivery_time_minutes)
    df_classification = df_encoded.drop(columns=['actual_delivery_time_minutes'], errors='ignore').copy()

    return df_encoded, df_regression, df_classification


def compute_regression_feature_stats(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Calcula a correlação de Pearson (r) e o teste F de regressão linear (f_regression e p-valor)
    exclusivamente sobre o conjunto de treino.
    """
    f_vals, p_vals = f_regression(X_train, y_train)
    corrs = [X_train[col].corr(y_train) for col in X_train.columns]
    stats_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Correlacao_Pearson (r)': np.round(corrs, 4),
        'Correlacao_Absoluta (|r|)': np.round(np.abs(corrs), 4),
        'Estatistica_F (f_regression)': np.round(f_vals, 2),
        'p_valor': p_vals
    }).sort_values(by='Estatistica_F (f_regression)', ascending=False).reset_index(drop=True)
    return stats_df


def compute_classification_feature_stats(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Calcula o teste ANOVA F (f_classif) para todas as features e o teste Qui-Quadrado (chi2)
    para as features binárias/não-negativas em relação ao alvo binário late_delivery (no treino).
    """
    f_vals, p_vals_f = f_classif(X_train, y_train)
    # Chi2 exige valores não-negativos
    X_non_neg = X_train.clip(lower=0)
    chi2_vals, p_vals_chi2 = chi2(X_non_neg, y_train)

    stats_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Tipo_Variavel': ['Binaria (Dummy/Flag)' if X_train[c].nunique() <= 2 else 'Continua/Discreta' for c in X_train.columns],
        'ANOVA_F (f_classif)': np.round(f_vals, 2),
        'p_valor_ANOVA': p_vals_f,
        'Qui_Quadrado (chi2)': np.round(chi2_vals, 2),
        'p_valor_Chi2': p_vals_chi2
    }).sort_values(by='ANOVA_F (f_classif)', ascending=False).reset_index(drop=True)
    return stats_df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Retorna um Pipeline do scikit-learn com imputação mediana (segurança para produção)
    e padronização Z-score (StandardScaler), garantindo ajuste exclusivo no conjunto de treino.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])