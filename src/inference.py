"""
Script independente e módulo de persistência e inferência em ambiente limpo para modelos PyTorch (.pt).

Uso via linha de comando (conforme Seção 4 e Listing 6 do edital):
    python src/inference.py --modelo models/modelo_classificacao.pt --entrada data/processed/dataset_classification.csv --n 10
    python src/inference.py --modelo models/modelo_regressao.pt --entrada data/processed/dataset_regression.csv --n 10
"""

import argparse
import os
import sys
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import torch

# Garante que a raiz do projeto esteja no sys.path mesmo quando executado diretamente
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import MLP


def salvar_checkpoint_completo(
    caminho_pt: str,
    modelo: MLP,
    scaler,
    colunas_features: list,
    task: str = "classification",
    limiar: float = 0.5,
    classes: list = None,
    y_media: float = None,
    y_desvio: float = None,
    hiperparametros_treino: dict = None,
) -> Dict[str, Any]:
    """
    Salva um checkpoint autocontido em arquivo PyTorch (.pt) contendo:
    - state_dict (pesos e vieses);
    - config (hiperparâmetros da arquitetura para recriar o MLP);
    - x_media e x_desvio do StandardScaler ajustado no treino;
    - features (nomes das variáveis na ordem exata de treinamento);
    - y_media e y_desvio (para regressão) ou classes e limiar (para classificação).
    """
    os.makedirs(os.path.dirname(os.path.abspath(caminho_pt)), exist_ok=True)

    checkpoint = {
        "task": task,
        "state_dict": modelo.state_dict(),
        "config": modelo.get_config(),
        "x_media": torch.tensor(scaler.mean_, dtype=torch.float32),
        "x_desvio": torch.tensor(scaler.scale_, dtype=torch.float32),
        "features": list(colunas_features),
        "hiperparametros_treino": hiperparametros_treino or {},
    }

    if task == "regression":
        checkpoint["y_media"] = float(y_media if y_media is not None else 0.0)
        checkpoint["y_desvio"] = float(y_desvio if y_desvio is not None else 1.0)
    else:
        checkpoint["limiar"] = float(limiar)
        checkpoint["classes"] = classes or ["No Prazo (0)", "Atrasado (1)"]

    torch.save(checkpoint, caminho_pt)
    return checkpoint


def carregar_checkpoint_completo(
    caminho_pt: str,
    device: str = "cpu",
) -> tuple:
    """
    Carrega o checkpoint salvo com map_location='cpu' e weights_only=True,
    recria a arquitetura MLP e coloca o modelo em modo de avaliação (modelo.eval()).
    """
    ckpt = torch.load(caminho_pt, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    modelo = MLP(
        in_dim=cfg["in_dim"],
        out_dim=cfg["out_dim"],
        hidden=tuple(cfg["hidden"]),
        activation=cfg["activation"],
        dropout=cfg.get("dropout", 0.0),
        use_batchnorm=cfg.get("use_batchnorm", False),
        weight_init=cfg.get("weight_init", "default"),
    )
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.to(device)
    modelo.eval()
    return modelo, ckpt


def prever_com_checkpoint(
    caminho_pt: str,
    dados_entrada: Union[str, pd.DataFrame, Dict[str, Any], np.ndarray],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Executa inferência completa em ambiente limpo a partir do arquivo .pt:
    1. Carrega o checkpoint e recria o modelo em modo eval();
    2. Ordena as features na mesma sequência do treinamento;
    3. Aplica a mesma padronização (x - x_media) / x_desvio;
    4. Executa o forward pass sob torch.inference_mode();
    5. Converte a saída para a escala original (minutos na regressão) ou probabilidades/classes (na classificação).
    """
    modelo, ckpt = carregar_checkpoint_completo(caminho_pt, device=device)
    features = ckpt["features"]

    if isinstance(dados_entrada, str):
        df_in = pd.read_csv(dados_entrada)
        X_arr = df_in[features].to_numpy(dtype=np.float32)
    elif isinstance(dados_entrada, pd.DataFrame):
        X_arr = dados_entrada[features].to_numpy(dtype=np.float32)
    elif isinstance(dados_entrada, dict):
        df_in = pd.DataFrame([dados_entrada]) if not isinstance(next(iter(dados_entrada.values())), (list, np.ndarray)) else pd.DataFrame(dados_entrada)
        X_arr = df_in[features].to_numpy(dtype=np.float32)
    else:
        X_arr = np.asarray(dados_entrada, dtype=np.float32)

    x_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
    x_media = ckpt["x_media"].to(device)
    x_desvio = ckpt["x_desvio"].to(device)
    x_norm = (x_t - x_media) / x_desvio

    with torch.inference_mode():
        saida = modelo(x_norm).squeeze(1)

        if ckpt["task"] == "regression":
            y_pred_min = (saida * ckpt["y_desvio"] + ckpt["y_media"]).cpu().numpy()
            return {
                "task": "regression",
                "predicoes_minutos": y_pred_min,
            }
        else:
            prob = torch.sigmoid(saida).cpu().numpy()
            limiar = float(ckpt["limiar"])
            classes_idx = (prob >= limiar).astype(np.int64)
            nomes_classes = [ckpt["classes"][i] for i in classes_idx]
            return {
                "task": "classification",
                "probabilidades": prob,
                "classes_preditas": classes_idx,
                "rotulos": nomes_classes,
                "limiar_usado": limiar,
            }


def main():
    parser = argparse.ArgumentParser(
        description="Script independente de inferência em PyTorch a partir de checkpoint .pt"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        required=True,
        help="Caminho para o arquivo .pt (ex: models/modelo_classificacao.pt ou models/modelo_regressao.pt)",
    )
    parser.add_argument(
        "--entrada",
        type=str,
        required=True,
        help="Caminho para o arquivo CSV com amostras de entrada",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Número de primeiras linhas do CSV para exibir no terminal (padrão: 5)",
    )
    parser.add_argument(
        "--saida",
        type=str,
        default=None,
        help="Caminho opcional para salvar as previsões em CSV",
    )
    args = parser.parse_args()

    df_raw = pd.read_csv(args.entrada)
    if args.n is not None and args.n > 0 and args.saida is None:
        df_proc = df_raw.head(args.n).copy()
    else:
        df_proc = df_raw.copy()

    res = prever_com_checkpoint(args.modelo, df_proc)
    print(f"=== INFERÊNCIA EM AMBIENTE LIMPO ({args.modelo}) ===")
    print(f"Amostras processadas: {len(df_proc)}")

    if res["task"] == "regression":
        df_out = pd.DataFrame({
            "amostra_id": df_proc.index,
            "tempo_previsto_min": np.round(res["predicoes_minutos"], 2),
        })
        if "actual_delivery_time_minutes" in df_proc.columns:
            df_out["tempo_real_min"] = df_proc["actual_delivery_time_minutes"].values
            df_out["erro_absoluto_min"] = np.round(np.abs(df_out["tempo_real_min"] - df_out["tempo_previsto_min"]), 2)
    else:
        df_out = pd.DataFrame({
            "amostra_id": df_proc.index,
            "probabilidade_atraso": np.round(res["probabilidades"], 4),
            "classe_predita": res["classes_preditas"],
            "rotulo_predito": res["rotulos"],
        })
        if "late_delivery" in df_proc.columns:
            df_out["classe_real"] = df_proc["late_delivery"].astype(int).values

    print(df_out.head(args.n).to_string(index=False))

    if args.saida:
        df_out.to_csv(args.saida, index=False)
        print(f"\nPrevisões salvas em: {args.saida}")


if __name__ == "__main__":
    main()
