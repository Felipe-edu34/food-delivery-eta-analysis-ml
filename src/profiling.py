"""
src/profiling.py
Medição de Tempo de Inferência, Pico de Memória (RAM via tracemalloc e RSS via psutil),
Consumo de Energia (kWh e uWh) e Emissões de CO2 (gCO2 e mgCO2) conforme a Aula 08 e o notebook
de referência do Prof. Dr. Thommas K. S. Flores.
"""

import os
import time
import tracemalloc
import psutil
import numpy as np


def formatar_tempo(segundos: float) -> str:
    """Formata tempos em ms, s ou min exatamente como no notebook de exemplo do professor."""
    if segundos < 1.0:
        return f"{segundos * 1000.0:.2f} ms"
    elif segundos < 60.0:
        return f"{segundos:.3f} s"
    else:
        return f"{segundos / 60.0:.2f} min"


def get_memory_usage() -> float:
    """Retorna o uso atual de memória RSS do processo em MB via psutil (conforme exemplo do professor)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def medir_inferencia(funcao_predicao, X, n_rep: int = 30) -> dict:
    """
    Mede tempo e pico de memória apenas da etapa de inferência conforme a Listing 1
    do documento da atividade prática (30 repetições com time.perf_counter e tracemalloc).
    """
    tempos = []
    tracemalloc.start()
    for _ in range(n_rep):
        inicio = time.perf_counter()
        funcao_predicao(X)
        tempos.append(time.perf_counter() - inicio)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    media = float(np.mean(tempos))
    desvio = float(np.std(tempos))
    n_samples = len(X) if hasattr(X, '__len__') else 1

    return {
        "tempo_medio_s": media,
        "tempo_desvio_s": desvio,
        "tempo_medio_ms": media * 1000.0,
        "tempo_por_amostra_ms": 1000.0 * media / n_samples,
        "vazao_amostras_s": n_samples / media if media > 0 else 0.0,
        "pico_memoria_kb": pico / 1024.0,
        "pico_memoria_mb": pico / (1024.0 * 1024.0)
    }


def estimar_energia_co2(tempo_segundos: float, potencia_watts: float = 15.0, fator_co2_g_kwh: float = 45.8) -> dict:
    """
    Estima a energia consumida e as emissões de CO2 usando a fórmula da Aula 08 e do notebook do professor:
      energia_kwh = (potencia_w * tempo_total) / (3600 * 1000)
      co2_kg      = energia_kwh * 0.0458  (equivale a 45,8 gCO2/kWh)
    Retorna valores em kWh, uWh (micro-watt-hora), gCO2, kgCO2 e mgCO2.
    """
    energia_kwh = (potencia_watts * tempo_segundos) / (3600.0 * 1000.0)
    energia_uwh = energia_kwh * 1e6
    co2_g = energia_kwh * fator_co2_g_kwh
    co2_kg = co2_g / 1000.0
    co2_mg = co2_kg * 1e6

    return {
        "potencia_watts": potencia_watts,
        "energia_kwh": energia_kwh,
        "energia_uwh": energia_uwh,
        "emissao_co2_g": co2_g,
        "emissao_co2_kg": co2_kg,
        "emissao_co2_mg": co2_mg
    }
