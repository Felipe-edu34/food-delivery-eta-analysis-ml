"""
src/models.py
Construção, treinamento, validação cruzada e busca de hiperparâmetros para
Árvores de Decisão (Aula 06) e Métodos de Ensemble (Aula 07).
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    BaggingClassifier,
    BaggingRegressor
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold


def get_tree_complexity(tree_model) -> dict:
    """Retorna a profundidade máxima, número total de nós e número de folhas de uma Árvore de Decisão."""
    return {
        "profundidade": int(tree_model.get_depth()),
        "n_nos": int(tree_model.tree_.node_count),
        "n_folhas": int(tree_model.get_n_leaves())
    }


def get_ensemble_complexity(ensemble_model) -> dict:
    """
    Calcula o número total de estimadores (árvores B), total de nós somados em toda a floresta/ensemble,
    total de folhas e profundidade média/máxima das árvores (Aula 07 / Seção 5.3).
    """
    estimators = getattr(ensemble_model, "estimators_", [])
    # Em GradientBoosting, estimators_ é um array 2D (n_estimators, 1)
    flat_trees = []
    for est in estimators:
        if isinstance(est, (list, np.ndarray)):
            flat_trees.extend(est)
        else:
            flat_trees.append(est)

    n_trees = len(flat_trees)
    total_nodes = sum(int(t.tree_.node_count) for t in flat_trees)
    total_leaves = sum(int(t.get_n_leaves()) for t in flat_trees)
    depths = [int(t.get_depth()) for t in flat_trees]

    return {
        "n_arvores": n_trees,
        "nos_totais": total_nodes,
        "folhas_totais": total_leaves,
        "profundidade_media": round(float(np.mean(depths)), 2) if depths else 0,
        "profundidade_max": int(np.max(depths)) if depths else 0
    }


def compute_node_entropy_gini(y: np.ndarray) -> dict:
    """
    Calcula manualmente a Entropia E(S) e o Índice de Gini de um conjunto de rótulos y,
    exatamente conforme as fórmulas da Aula 06 (Seções 4.1 e 5):
      E(S) = - sum_{i=1}^c p_i log_2(p_i)
      Gini(S) = 1 - sum_{k=1}^K p_k^2
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
    gini = 1.0 - np.sum(probs ** 2)
    return {
        "n_amostras": int(len(y)),
        "proporcoes": [round(float(p), 4) for p in probs],
        "entropia": round(float(entropy), 4),
        "gini": round(float(gini), 4)
    }


def compute_split_gain(y_parent: np.ndarray, mask_left: np.ndarray) -> dict:
    """
    Calcula o Ganho de Informação G(S, A) e a Redução de Gini (Delta Gini) de uma partição binária,
    exatamente conforme as Seções 4.2 e 5 da Aula 06.
    """
    parent_stats = compute_node_entropy_gini(y_parent)
    y_left = y_parent[mask_left]
    y_right = y_parent[~mask_left]
    left_stats = compute_node_entropy_gini(y_left)
    right_stats = compute_node_entropy_gini(y_right)

    n_total = len(y_parent)
    w_left = len(y_left) / n_total
    w_right = len(y_right) / n_total

    entropy_split = w_left * left_stats["entropia"] + w_right * right_stats["entropia"]
    info_gain = parent_stats["entropia"] - entropy_split

    gini_split = w_left * left_stats["gini"] + w_right * right_stats["gini"]
    delta_gini = parent_stats["gini"] - gini_split
    pct_reduction_gini = (delta_gini / parent_stats["gini"]) * 100.0 if parent_stats["gini"] > 0 else 0.0

    return {
        "entropia_pai": parent_stats["entropia"],
        "entropia_ponderada_split": round(float(entropy_split), 4),
        "ganho_informacao_G": round(float(info_gain), 4),
        "gini_pai": parent_stats["gini"],
        "gini_ponderado_split": round(float(gini_split), 4),
        "reducao_gini_delta": round(float(delta_gini), 4),
        "reducao_gini_pct": round(float(pct_reduction_gini), 2),
        "n_esq": int(len(y_left)),
        "n_dir": int(len(y_right)),
        "gini_esq": left_stats["gini"],
        "gini_dir": right_stats["gini"],
        "entropia_esq": left_stats["entropia"],
        "entropia_dir": right_stats["entropia"]
    }


def build_decision_tree_classifier(criterion='gini', max_depth=None, min_samples_leaf=1, ccp_alpha=0.0, random_state=42):
    """Retorna uma instância de DecisionTreeClassifier."""
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )


def build_decision_tree_regressor(criterion='squared_error', max_depth=None, min_samples_leaf=1, ccp_alpha=0.0, random_state=42):
    """Retorna uma instância de DecisionTreeRegressor."""
    return DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )


def tune_decision_tree_classifier(X_train, y_train, cv=5, random_state=42):
    """
    Ajusta hiperparâmetros da Árvore de Decisão de Classificação usando GridSearchCV estratificado,
    comparando os critérios 'gini' e 'entropy' (Aula 06).
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 6, 8, 10, 12],
        'min_samples_leaf': [5, 15, 30, 50],
        'ccp_alpha': [0.0, 0.0005, 0.001, 0.005]
    }
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, pd.DataFrame(grid.cv_results_)


def tune_decision_tree_regressor(X_train, y_train, cv=5, random_state=42):
    """
    Ajusta hiperparâmetros da Árvore de Decisão de Regressão usando GridSearchCV,
    comparando 'squared_error' (MSE/RSS - Aula 06, Seção 7) e 'poisson'.
    """
    param_grid = {
        'criterion': ['squared_error', 'poisson'],
        'max_depth': [4, 6, 8, 10, 12],
        'min_samples_leaf': [5, 15, 30, 50],
        'ccp_alpha': [0.0, 0.001, 0.01, 0.05]
    }
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=random_state),
        param_grid,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, pd.DataFrame(grid.cv_results_)


def tune_random_forest_classifier(X_train, y_train, cv=5, random_state=42):
    """Ajusta hiperparâmetros do RandomForestClassifier via RandomizedSearchCV estratificado (K=5)."""
    param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [12, 16, 20],
        'max_features': ['sqrt', 0.5, 0.7],
        'min_samples_leaf': [5, 10]
    }
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=8,
        cv=skf,
        scoring='f1',
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_random_forest_regressor(X_train, y_train, cv=5, random_state=42):
    """Ajusta hiperparâmetros do RandomForestRegressor via RandomizedSearchCV (K=5)."""
    param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [12, 16, 20],
        'max_features': [0.5, 0.7, 1.0],
        'min_samples_leaf': [5, 10]
    }
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=8,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_gradient_boosting_classifier(X_train, y_train, cv=5, random_state=42):
    """Ajusta hiperparâmetros do GradientBoostingClassifier via RandomizedSearchCV estratificado (K=5)."""
    param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [10, 20]
    }
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        param_distributions=param_dist,
        n_iter=6,
        cv=skf,
        scoring='f1',
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_gradient_boosting_regressor(X_train, y_train, cv=5, random_state=42):
    """Ajusta hiperparâmetros do GradientBoostingRegressor via RandomizedSearchCV (K=5)."""
    param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [10, 20]
    }
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=random_state),
        param_distributions=param_dist,
        n_iter=6,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
