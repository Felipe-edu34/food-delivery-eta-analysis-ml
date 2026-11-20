"""
Módulo de preparação de tensores/DataLoaders, funções de perda, otimizadores,
laço de treinamento com early stopping e avaliação multi-sementes em PyTorch.
"""

import copy
import random
import time
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models import MLP, count_trainable_parameters
from src.metrics import compute_regression_metrics, compute_classification_metrics


class FastTensorDataLoader(DataLoader):
    """
    Subclasse de DataLoader para TensorDataset que realiza fatiamento vetorizado de tensores
    em memória com torch.randperm, mantendo 100% da interface padrão do PyTorch DataLoader
    (incluindo .dataset = TensorDataset(X, y)) e eliminando o overhead de collate item a item.
    """

    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int = 128,
        shuffle: bool = False,
        generator: torch.Generator = None,
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
        self.tensors = dataset.tensors
        self.n_samples = self.tensors[0].size(0)
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._generator = generator

    def __iter__(self):
        if self._shuffle:
            perm = torch.randperm(self.n_samples, generator=self._generator)
            tensors_iter = tuple(t[perm] for t in self.tensors)
        else:
            tensors_iter = self.tensors

        for i in range(0, self.n_samples, self._batch_size):
            yield tuple(t[i : i + self._batch_size] for t in tensors_iter)

    def __len__(self) -> int:
        return (self.n_samples + self._batch_size - 1) // self._batch_size


def set_seeds(seed: int = 42) -> None:
    """Fixa sementes de random, numpy e PyTorch para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_regression_data(
    csv_path: str,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Carrega dataset_regression.csv, aplica exatamente a mesma divisão treino/teste (80/20, random_state=42),
    separa 20% do treino para validação, padroniza features (StandardScaler) e target (z-score)
    usando apenas estatísticas do conjunto de treino.
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "actual_delivery_time_minutes"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["actual_delivery_time_minutes"].to_numpy(dtype=np.float32)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    scaler_sub = StandardScaler()
    if scale_features:
        X_tr_s = scaler_sub.fit_transform(X_tr).astype(np.float32)
        X_val_s = scaler_sub.transform(X_val).astype(np.float32)
        X_test_s = scaler_sub.transform(X_test).astype(np.float32)
    else:
        scaler_sub.fit(X_tr)
        X_tr_s = X_tr.copy()
        X_val_s = X_val.copy()
        X_test_s = X_test.copy()

    y_mean = float(np.mean(y_tr))
    y_std = float(np.std(y_tr))
    if y_std < 1e-8:
        y_std = 1.0

    y_tr_s = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_val_s = ((y_val - y_mean) / y_std).astype(np.float32)

    return {
        "feature_cols": feature_cols,
        "in_dim": len(feature_cols),
        "X_tr": X_tr_s,
        "y_tr_scaled": y_tr_s,
        "y_tr_orig": y_tr,
        "X_val": X_val_s,
        "y_val_scaled": y_val_s,
        "y_val_orig": y_val,
        "X_test": X_test_s,
        "y_test_orig": y_test,
        "X_train_full_raw": X_train_full,
        "y_train_full_orig": y_train_full,
        "X_test_raw": X_test,
        "scaler": scaler_sub,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def prepare_classification_data(
    csv_path: str,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Carrega dataset_classification.csv, aplica exatamente a mesma divisão treino/teste estratificada
    (80/20, random_state=42), separa 20% do treino para validação estratificada e padroniza features
    usando apenas o conjunto de treino.
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "late_delivery"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["late_delivery"].to_numpy(dtype=np.float32)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
    )

    scaler_sub = StandardScaler()
    if scale_features:
        X_tr_s = scaler_sub.fit_transform(X_tr).astype(np.float32)
        X_val_s = scaler_sub.transform(X_val).astype(np.float32)
        X_test_s = scaler_sub.transform(X_test).astype(np.float32)
    else:
        scaler_sub.fit(X_tr)
        X_tr_s = X_tr.copy()
        X_val_s = X_val.copy()
        X_test_s = X_test.copy()

    n_neg = float(np.sum(y_tr == 0))
    n_pos = float(np.sum(y_tr == 1))
    pos_weight_val = n_neg / max(n_pos, 1.0)

    return {
        "feature_cols": feature_cols,
        "in_dim": len(feature_cols),
        "X_tr": X_tr_s,
        "y_tr": y_tr,
        "X_val": X_val_s,
        "y_val": y_val,
        "X_test": X_test_s,
        "y_test": y_test,
        "X_train_full_raw": X_train_full,
        "y_train_full": y_train_full,
        "X_test_raw": X_test,
        "scaler": scaler_sub,
        "pos_weight": float(pos_weight_val),
    }


def make_dataloaders(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: Union[int, str] = 128,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Converte arrays para TensorDataset e DataLoader com gerador determinístico."""
    ds_tr = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    ds_val = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )

    if isinstance(batch_size, str) and batch_size.lower() == "total":
        bs_tr = len(ds_tr)
    else:
        bs_tr = int(batch_size)

    g = torch.Generator()
    g.manual_seed(seed)

    dl_tr = FastTensorDataLoader(ds_tr, batch_size=bs_tr, shuffle=True, generator=g)
    dl_val = FastTensorDataLoader(ds_val, batch_size=len(ds_val), shuffle=False)
    return dl_tr, dl_val


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "Adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Instancia o otimizador escolhido."""
    opt = optimizer_name.strip()
    if opt == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt in ("SGD_momentum", "SGD com momentum"):
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if opt == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Otimizador desconhecido: {optimizer_name}")


def build_loss_fn(
    task: str = "regression",
    loss_name: str = "default",
    pos_weight: float = None,
) -> nn.Module:
    """Instancia a função de perda adequada para regressão ou classificação binária."""
    if task == "regression":
        ln = loss_name.upper()
        if ln in ("DEFAULT", "MSE", "MSELOSS"):
            return nn.MSELoss()
        if ln in ("MAE", "L1", "L1LOSS"):
            return nn.L1Loss()
        if ln in ("HUBER", "HUBERLOSS"):
            return nn.HuberLoss(delta=1.0)
        raise ValueError(f"Loss de regressão desconhecida: {loss_name}")
    else:
        if loss_name == "BCE_pos_weight" and pos_weight is not None:
            pw = torch.tensor([float(pos_weight)], dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        return nn.BCEWithLogitsLoss()


def treinar(
    modelo: nn.Module,
    dl_tr: DataLoader,
    dl_val: DataLoader,
    loss_fn: nn.Module,
    otimizador: torch.optim.Optimizer,
    n_epocas: int = 100,
    paciencia: int = 15,
) -> Dict[str, Any]:
    """
    Laço de treinamento em PyTorch com validação e early stopping (Listing 2 do edital).
    Restaura automaticamente o state_dict da melhor época de validação.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    hist = {"treino": [], "val": []}
    melhor = float("inf")
    espera = 0
    melhor_epoca = 1
    melhor_estado = copy.deepcopy(modelo.state_dict())

    t0 = time.perf_counter()
    ep_realizadas = 0

    try:
        for ep in range(1, n_epocas + 1):
            ep_realizadas = ep
            modelo.train()
            soma = 0.0
            for xb, yb in dl_tr:
                otimizador.zero_grad()
                saida = modelo(xb)
                perda = loss_fn(saida, yb)
                if torch.isnan(perda) or torch.isinf(perda):
                    soma = float("nan")
                    break
                perda.backward()
                otimizador.step()
                soma += perda.item() * len(xb)

            perda_tr = soma / len(dl_tr.dataset)
            hist["treino"].append(float(perda_tr))

            modelo.eval()
            with torch.no_grad():
                val = sum(
                    loss_fn(modelo(xb), yb).item() * len(xb) for xb, yb in dl_val
                ) / len(dl_val.dataset)
            hist["val"].append(float(val))

            if np.isnan(val) or np.isinf(val) or np.isnan(perda_tr):
                break

            if val < melhor - 1e-6:
                melhor = val
                espera = 0
                melhor_epoca = ep
                melhor_estado = copy.deepcopy(modelo.state_dict())
            else:
                espera += 1
                if espera >= paciencia:
                    break
    finally:
        torch.set_num_threads(prev_threads)

    tempo_total = time.perf_counter() - t0
    modelo.load_state_dict(melhor_estado)

    return {
        "treino": hist["treino"],
        "val": hist["val"],
        "melhor_val_loss": float(melhor),
        "melhor_epoca": int(melhor_epoca),
        "epoca_parada": int(ep_realizadas),
        "tempo_treino_s": float(tempo_total),
    }


def predict_regression(
    modelo: nn.Module,
    X_array: np.ndarray,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    """Realiza predição em modo eval/inference_mode e reverte para a escala original (minutos)."""
    modelo.eval()
    x_t = torch.tensor(X_array, dtype=torch.float32)
    with torch.inference_mode():
        pred_scaled = modelo(x_t).squeeze(1).cpu().numpy()
    pred_scaled = np.nan_to_num(pred_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
    return pred_scaled * y_std + y_mean


def predict_classification(
    modelo: nn.Module,
    X_array: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Realiza predição em modo eval/inference_mode aplicando torch.sigmoid aos logits."""
    modelo.eval()
    x_t = torch.tensor(X_array, dtype=torch.float32)
    with torch.inference_mode():
        logits = modelo(x_t).squeeze(1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(np.int64)
    return probs, preds


def evaluate_config_seeds(
    task: str,
    data_dict: Dict[str, Any],
    config: Dict[str, Any],
    seeds: Tuple[int, ...] = (42, 100, 2026),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Treina e avalia uma configuração de hiperparâmetros com múltiplas sementes.
    Retorna:
    - resumo agregado (média, desvio padrão, gap, parâmetros, tempo);
    - lista detalhada de cada semente (para salvar em logs/hiperparametros.csv);
    - histórico de curvas de perda da primeira semente e o melhor modelo treinado.
    """
    hidden = config.get("hidden", (32, 32))
    activation = config.get("activation", "ReLU")
    dropout = float(config.get("dropout", 0.0))
    use_bn = bool(config.get("use_batchnorm", False))
    weight_init = config.get("weight_init", "default")
    optimizer_name = config.get("optimizer", "Adam")
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    batch_size = config.get("batch_size", 128)
    n_epocas = int(config.get("n_epocas", 60))
    paciencia = int(config.get("paciencia", 10))
    loss_name = config.get("loss_name", "default")

    seed_rows = []
    first_hist = None
    best_model = None
    best_score = float("inf") if task == "regression" else float("-inf")

    for s in seeds:
        set_seeds(s)
        modelo = MLP(
            in_dim=data_dict["in_dim"],
            out_dim=1,
            hidden=hidden,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_bn,
            weight_init=weight_init,
        )
        n_params = count_trainable_parameters(modelo)

        y_tr_target = data_dict["y_tr_scaled"] if task == "regression" else data_dict["y_tr"]
        y_val_target = data_dict["y_val_scaled"] if task == "regression" else data_dict["y_val"]

        dl_tr, dl_val = make_dataloaders(
            data_dict["X_tr"], y_tr_target,
            data_dict["X_val"], y_val_target,
            batch_size=batch_size, seed=s,
        )

        loss_fn = build_loss_fn(
            task=task,
            loss_name=loss_name,
            pos_weight=data_dict.get("pos_weight"),
        )
        otim = build_optimizer(modelo, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)

        res_tr = treinar(
            modelo, dl_tr, dl_val, loss_fn, otim,
            n_epocas=n_epocas, paciencia=paciencia,
        )
        if first_hist is None:
            first_hist = res_tr

        if task == "regression":
            pred_tr = predict_regression(modelo, data_dict["X_tr"], data_dict["y_mean"], data_dict["y_std"])
            pred_val = predict_regression(modelo, data_dict["X_val"], data_dict["y_mean"], data_dict["y_std"])
            m_tr = compute_regression_metrics(data_dict["y_tr_orig"], pred_tr)
            m_val = compute_regression_metrics(data_dict["y_val_orig"], pred_val)
            score_tr = m_tr["RMSE"]
            score_val = m_val["RMSE"]
            sec_val = m_val["MAE"]
            r2_val = m_val["R2"]
            gap = score_tr - score_val
            if score_val < best_score:
                best_score = score_val
                best_model = copy.deepcopy(modelo)
        else:
            prob_tr, pred_tr = predict_classification(modelo, data_dict["X_tr"], threshold=0.5)
            prob_val, pred_val = predict_classification(modelo, data_dict["X_val"], threshold=0.5)
            m_tr = compute_classification_metrics(data_dict["y_tr"], pred_tr, prob_tr)
            m_val = compute_classification_metrics(data_dict["y_val"], pred_val, prob_val)
            score_tr = m_tr["F1-score"]
            score_val = m_val["F1-score"]
            sec_val = m_val["ROC-AUC"]
            r2_val = m_val["PR-AUC"]
            gap = score_tr - score_val
            if score_val > best_score:
                best_score = score_val
                best_model = copy.deepcopy(modelo)

        seed_rows.append({
            "problema": task,
            "experimento": config.get("experimento", "baseline"),
            "valor_testado": str(config.get("valor_testado", "padrao")),
            "seed": int(s),
            "hidden": str(tuple(hidden) if not isinstance(hidden, int) else (hidden,)),
            "activation": activation,
            "optimizer": optimizer_name,
            "lr": lr,
            "batch_size": str(batch_size),
            "dropout": dropout,
            "weight_decay": weight_decay,
            "loss_name": loss_name,
            "weight_init": weight_init,
            "use_batchnorm": use_bn,
            "n_parametros": int(n_params),
            "metrica_principal": "RMSE" if task == "regression" else "F1-score",
            "score_treino": round(float(score_tr), 4),
            "score_val": round(float(score_val), 4),
            "score_secundario_val": round(float(sec_val), 4),
            "score_terciario_val": round(float(r2_val), 4),
            "gap_treino_val": round(float(gap), 4),
            "melhor_epoca": int(res_tr["melhor_epoca"]),
            "epoca_parada": int(res_tr["epoca_parada"]),
            "tempo_treino_s": round(float(res_tr["tempo_treino_s"]), 4),
        })

    df_s = pd.DataFrame(seed_rows)
    summary = {
        "problema": task,
        "experimento": config.get("experimento", "baseline"),
        "valor_testado": str(config.get("valor_testado", "padrao")),
        "n_parametros": int(df_s["n_parametros"].iloc[0]),
        "metrica": "RMSE (min)" if task == "regression" else "F1-score",
        "treino_media": round(float(df_s["score_treino"].mean()), 4),
        "treino_dp": round(float(df_s["score_treino"].std(ddof=1)) if len(df_s) > 1 else 0.0, 4),
        "val_media": round(float(df_s["score_val"].mean()), 4),
        "val_dp": round(float(df_s["score_val"].std(ddof=1)) if len(df_s) > 1 else 0.0, 4),
        "secundaria_val_media": round(float(df_s["score_secundario_val"].mean()), 4),
        "terciaria_val_media": round(float(df_s["score_terciario_val"].mean()), 4),
        "gap_media": round(float(df_s["gap_treino_val"].mean()), 4),
        "epoca_parada_media": round(float(df_s["epoca_parada"].mean()), 1),
        "tempo_treino_medio_s": round(float(df_s["tempo_treino_s"].mean()), 3),
        "tempo_treino_dp_s": round(float(df_s["tempo_treino_s"].std(ddof=1)) if len(df_s) > 1 else 0.0, 3),
    }

    return summary, seed_rows, {"hist": first_hist, "best_model": best_model}
