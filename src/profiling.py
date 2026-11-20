"""
Módulo de profiling computacional para Redes Neurais em PyTorch (Seção 5.3 e Listing 4):
- Medição de tempo (time.perf_counter) com warm-up e 30 repetições;
- Medição de memória residente nativa via psutil (Process().memory_info().rss);
- Comparação de inferência em lote (batch) versus amostra individual (1 por vez);
- Estimativa de energia (kWh) e emissão de carbono (45,8 gCO2/kWh - SIN Brasil);
- Exportação TorchScript (torch.jit.trace) e Quantização Dinâmica (qint8).
"""

import os
import time
from typing import Dict, Any
import numpy as np
import psutil
import torch
import torch.nn as nn


def medir_inferencia_pytorch(
    modelo: nn.Module,
    X: Any,
    n_rep: int = 30,
    n_aquecimento: int = 5,
) -> Dict[str, float]:
    """
    Mede tempo e memória residente (RSS via psutil) na etapa de inferência em lote
    de uma rede PyTorch, realizando execuções prévias de warm-up (Listing 4).
    """
    if not isinstance(X, torch.Tensor):
        X_t = torch.tensor(X, dtype=torch.float32)
    else:
        X_t = X

    modelo.eval()
    proc = psutil.Process(os.getpid())

    with torch.inference_mode():
        for _ in range(n_aquecimento):
            _ = modelo(X_t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        mem_antes = proc.memory_info().rss
        tempos = []
        pico_rss = mem_antes

        for _ in range(n_rep):
            inicio = time.perf_counter()
            _ = modelo(X_t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tempos.append(time.perf_counter() - inicio)
            rss_atual = proc.memory_info().rss
            if rss_atual > pico_rss:
                pico_rss = rss_atual

        mem_depois = proc.memory_info().rss

    media = float(np.mean(tempos))
    desvio = float(np.std(tempos, ddof=1)) if len(tempos) > 1 else 0.0
    n_amostras = len(X_t)

    return {
        "tempo_medio_s": media,
        "tempo_desvio_s": desvio,
        "tempo_por_amostra_ms": (1000.0 * media) / n_amostras,
        "vazao_amostras_s": n_amostras / max(media, 1e-12),
        "rss_antes_mb": mem_antes / (1024.0 * 1024.0),
        "rss_pico_mb": pico_rss / (1024.0 * 1024.0),
        "delta_memoria_kb": max(0.0, (pico_rss - mem_antes) / 1024.0),
        "num_threads_cpu": int(torch.get_num_threads()),
    }


def medir_inferencia_individual_vs_lote(
    modelo: nn.Module,
    X: np.ndarray,
    n_amostras_unitarias: int = 500,
    n_rep_lote: int = 30,
) -> Dict[str, Any]:
    """
    Compara a latência e a vazão da inferência em lote (todas as amostras do teste de uma vez)
    versus inferência individual (1 amostra por vez, simulando requisição online unitária).
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    res_lote = medir_inferencia_pytorch(modelo, X_t, n_rep=n_rep_lote, n_aquecimento=5)

    modelo.eval()
    n_unit = min(n_amostras_unitarias, len(X_t))
    with torch.inference_mode():
        # Warm-up unitário
        for i in range(min(20, n_unit)):
            _ = modelo(X_t[i : i + 1])

        tempos_unit = []
        for i in range(n_unit):
            t0 = time.perf_counter()
            _ = modelo(X_t[i : i + 1])
            tempos_unit.append(time.perf_counter() - t0)

    media_unit_s = float(np.mean(tempos_unit))
    dp_unit_s = float(np.std(tempos_unit, ddof=1))

    return {
        "lote_tempo_total_ms": round(res_lote["tempo_medio_s"] * 1000.0, 4),
        "lote_tempo_dp_ms": round(res_lote["tempo_desvio_s"] * 1000.0, 4),
        "lote_latencia_por_amostra_ms": round(res_lote["tempo_por_amostra_ms"], 6),
        "lote_vazao_amostras_s": round(res_lote["vazao_amostras_s"], 1),
        "individual_latencia_ms": round(media_unit_s * 1000.0, 4),
        "individual_dp_ms": round(dp_unit_s * 1000.0, 4),
        "individual_vazao_amostras_s": round(1.0 / max(media_unit_s, 1e-12), 1),
        "fator_aceleracao_vazao": round(res_lote["vazao_amostras_s"] / (1.0 / max(media_unit_s, 1e-12)), 1),
    }


def estimar_energia_co2(
    tempo_segundos: float,
    potencia_watts: float = 45.0,
    fator_co2_g_kwh: float = 45.8,
) -> Dict[str, float]:
    """
    Estima o consumo energético (kWh e uWh) e a emissão de CO2 (gCO2 e mgCO2)
    usando a potência média do processador (45 W) e o fator do SIN Brasil (45,8 gCO2/kWh).
    """
    energia_kwh = (potencia_watts * tempo_segundos) / 3_600_000.0
    emissao_g = energia_kwh * fator_co2_g_kwh
    return {
        "energia_kwh": float(energia_kwh),
        "energia_uwh": float(energia_kwh * 1e9),
        "emissao_co2_g": float(emissao_g),
        "emissao_co2_mg": float(emissao_g * 1000.0),
    }
