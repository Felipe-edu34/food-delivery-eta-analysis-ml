"""
src/expert_system.py
Implementação do Sistema Especialista baseado em Regras SE-ENTÃO (Aula 05):
- Arquitetura com separação explícita entre:
  1. Base de Conhecimento (Regras SE-ENTÃO tipadas em Relação, Recomendação, Diretriz, Estratégia e Heurística, sem mistura de E e OU)
  2. Banco de Dados de Fatos (Memória de Trabalho)
  3. Motor de Inferência com Encadeamento para Frente (Forward Chaining: Correspondência -> Disparo -> Atualização de Fatos)
  4. Módulo de Explicação (rastreamento completo das regras acionadas para cada instância)
"""

import os
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class Rule:
    """Representa uma regra SE-ENTÃO estruturada na Base de Conhecimento."""
    rule_id: str
    rule_type: str  # Relação, Recomendação, Diretriz, Estratégia ou Heurística
    antecedent_desc: str
    consequent_desc: str
    condition_fn: Callable[[Dict[str, Any]], bool]
    action_fn: Callable[[Dict[str, Any]], None]


class ForwardChainingEngine:
    """
    Motor de Inferência genérico com Encadeamento para Frente (Forward Chaining - Aula 05, Seção 8.1).
    Executa ciclos sucessivos de:
      Correspondência (Match) -> Disparo (Fire) -> Atualização do Banco de Dados de Fatos
    independentemente do domínio, operando sobre uma Base de Conhecimento fornecida.
    """
    def __init__(self, knowledge_base: List[Rule], stop_on_first_goal: bool = False, goal_key: str = None):
        self.knowledge_base = knowledge_base
        self.stop_on_first_goal = stop_on_first_goal
        self.goal_key = goal_key

    def infer_single(self, initial_facts: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Executa o ciclo de inferência sobre os fatos iniciais de uma instância.
        Retorna a memória de trabalho final (fatos deduzidos) e a trilha de explicação (regras disparadas).
        """
        facts = dict(initial_facts)
        fired_rules: List[str] = []

        for rule in self.knowledge_base:
            if rule.condition_fn(facts):
                rule.action_fn(facts)
                fired_rules.append(f"[{rule.rule_id} - {rule.rule_type}] SE {rule.antecedent_desc} -> ENTAO {rule.consequent_desc}")
                if self.stop_on_first_goal and self.goal_key and self.goal_key in facts:
                    break

        return facts, fired_rules


# ==============================================================================
# FUNÇÕES DE CONDIÇÃO E AÇÃO DE NÍVEL DE MÓDULO (PARA SERIALIZAÇÃO / PICKLE)
# ==============================================================================

def _is_rain(f: Dict[str, Any]) -> bool:
    return f.get('weather_Rain', 0) == 1 or f.get('weather', '') == 'Rain'

def _is_storm(f: Dict[str, Any]) -> bool:
    return f.get('weather_Storm', 0) == 1 or f.get('weather', '') == 'Storm'

def _is_cloudy(f: Dict[str, Any]) -> bool:
    return f.get('weather_Cloudy', 0) == 1 or f.get('weather', '') == 'Cloudy'

def _is_high_traffic(f: Dict[str, Any]) -> bool:
    if 'traffic_level' in f:
        return f['traffic_level'] == 'High'
    return f.get('traffic_level_Low', 0) == 0 and f.get('traffic_level_Medium', 0) == 0


# --- Regras de Classificação ---
def _cond_clf_1(f: Dict[str, Any]) -> bool:
    return _is_storm(f)

def _act_clf_1(f: Dict[str, Any]) -> None:
    f['late_delivery_pred'] = 1
    f['confidence_score'] = 0.90

def _cond_clf_2(f: Dict[str, Any]) -> bool:
    return _is_rain(f) and float(f.get('distance_km', 0.0)) > 4.0

def _act_clf_2(f: Dict[str, Any]) -> None:
    f['late_delivery_pred'] = 1
    f['confidence_score'] = 0.78

def _cond_clf_3(f: Dict[str, Any]) -> bool:
    return _is_rain(f) and float(f.get('restaurant_preparation_time_minutes', 0.0)) > 11.0

def _act_clf_3(f: Dict[str, Any]) -> None:
    f['late_delivery_pred'] = 1
    f['confidence_score'] = 0.72

def _cond_clf_4(f: Dict[str, Any]) -> bool:
    return _is_high_traffic(f) and float(f.get('restaurant_preparation_time_minutes', 0.0)) > 14.0

def _act_clf_4(f: Dict[str, Any]) -> None:
    f['late_delivery_pred'] = 1
    f['confidence_score'] = 0.65

def _cond_clf_5(f: Dict[str, Any]) -> bool:
    est = float(f.get('estimated_delivery_time_minutes', 30.0))
    prep = float(f.get('restaurant_preparation_time_minutes', 15.0))
    return (est - prep) <= 14.0 and prep > 13.0

def _act_clf_5(f: Dict[str, Any]) -> None:
    f['late_delivery_pred'] = 1
    f['confidence_score'] = 0.68

def _cond_clf_6(f: Dict[str, Any]) -> bool:
    return 'late_delivery_pred' not in f

def _act_clf_6(f: Dict[str, Any]) -> None:
    f['late_delivery_pred'] = 0
    f['confidence_score'] = 0.22


# --- Regras de Regressão ---
def _cond_reg_1(f: Dict[str, Any]) -> bool:
    return float(f.get('distance_km', 1.0)) > 0.0 and float(f.get('restaurant_preparation_time_minutes', 1.0)) > 0.0

def _act_reg_1(f: Dict[str, Any]) -> None:
    est = float(f.get('estimated_delivery_time_minutes', 0.0))
    if est > 0:
        f['tempo_calculado'] = est + 2.5
    else:
        prep = float(f.get('restaurant_preparation_time_minutes', 15.0))
        dist = float(f.get('distance_km', 4.0))
        f['tempo_calculado'] = 10.0 + prep + (dist * 3.2)

def _cond_reg_2(f: Dict[str, Any]) -> bool:
    return _is_storm(f)

def _act_reg_2(f: Dict[str, Any]) -> None:
    f['tempo_calculado'] = float(f.get('tempo_calculado', 30.0)) + 12.5

def _cond_reg_3(f: Dict[str, Any]) -> bool:
    return _is_rain(f) and not _is_storm(f)

def _act_reg_3(f: Dict[str, Any]) -> None:
    f['tempo_calculado'] = float(f.get('tempo_calculado', 30.0)) + 5.8

def _cond_reg_4(f: Dict[str, Any]) -> bool:
    return _is_cloudy(f) and float(f.get('distance_km', 0.0)) > 4.0

def _act_reg_4(f: Dict[str, Any]) -> None:
    f['tempo_calculado'] = float(f.get('tempo_calculado', 30.0)) + 1.8

def _cond_reg_5(f: Dict[str, Any]) -> bool:
    return _is_high_traffic(f) and float(f.get('distance_km', 0.0)) > 3.5

def _act_reg_5(f: Dict[str, Any]) -> None:
    f['tempo_calculado'] = float(f.get('tempo_calculado', 30.0)) + 3.2

def _cond_reg_6(f: Dict[str, Any]) -> bool:
    return float(f.get('tempo_calculado', 0.0)) > 120.0

def _act_reg_6(f: Dict[str, Any]) -> None:
    f['tempo_calculado'] = min(float(f.get('tempo_calculado', 30.0)), 120.0)


def build_classification_knowledge_base() -> List[Rule]:
    """
    Constrói a Base de Conhecimento para Classificação de Atrasos (late_delivery: 1 = Atraso, 0 = No Prazo).
    Todas as regras usam exclusivamente conjunções (E) no antecedente, sem misturar E e OU.
    """
    return [
        Rule(
            rule_id="REGRA_CLF_1",
            rule_type="Relação",
            antecedent_desc="weather_Storm == 1",
            consequent_desc="late_delivery = 1 (Fator de Certeza = 0.90)",
            condition_fn=_cond_clf_1,
            action_fn=_act_clf_1
        ),
        Rule(
            rule_id="REGRA_CLF_2",
            rule_type="Estratégia",
            antecedent_desc="weather_Rain == 1 E distance_km > 4.0",
            consequent_desc="late_delivery = 1 (Fator de Certeza = 0.78)",
            condition_fn=_cond_clf_2,
            action_fn=_act_clf_2
        ),
        Rule(
            rule_id="REGRA_CLF_3",
            rule_type="Recomendação",
            antecedent_desc="weather_Rain == 1 E restaurant_preparation_time_minutes > 11.0",
            consequent_desc="late_delivery = 1 (Fator de Certeza = 0.72)",
            condition_fn=_cond_clf_3,
            action_fn=_act_clf_3
        ),
        Rule(
            rule_id="REGRA_CLF_4",
            rule_type="Diretriz",
            antecedent_desc="traffic_level_Low == 0 E traffic_level_Medium == 0 E restaurant_preparation_time_minutes > 14.0",
            consequent_desc="late_delivery = 1 (Fator de Certeza = 0.65)",
            condition_fn=_cond_clf_4,
            action_fn=_act_clf_4
        ),
        Rule(
            rule_id="REGRA_CLF_5",
            rule_type="Heurística",
            antecedent_desc="(estimated_delivery_time_minutes - restaurant_preparation_time_minutes) <= 14.0 E restaurant_preparation_time_minutes > 13.0",
            consequent_desc="late_delivery = 1 (Fator de Certeza = 0.68)",
            condition_fn=_cond_clf_5,
            action_fn=_act_clf_5
        ),
        Rule(
            rule_id="REGRA_CLF_6",
            rule_type="Diretriz",
            antecedent_desc="nenhuma_regra_anterior_disparada == 1",
            consequent_desc="late_delivery = 0 (Fator de Certeza = 0.22)",
            condition_fn=_cond_clf_6,
            action_fn=_act_clf_6
        )
    ]


def build_regression_knowledge_base() -> List[Rule]:
    """
    Constrói a Base de Conhecimento para Regressão do Tempo de Entrega (actual_delivery_time_minutes).
    Todas as regras usam exclusivamente conjunções (E) no antecedente, sem misturar E e OU.
    """
    return [
        Rule(
            rule_id="REGRA_REG_1",
            rule_type="Heurística",
            antecedent_desc="distance_km > 0.0 E restaurant_preparation_time_minutes > 0.0",
            consequent_desc="tempo_calculado = estimated_delivery_time_minutes + 2.5 min",
            condition_fn=_cond_reg_1,
            action_fn=_act_reg_1
        ),
        Rule(
            rule_id="REGRA_REG_2",
            rule_type="Relação",
            antecedent_desc="weather_Storm == 1",
            consequent_desc="tempo_calculado += 12.5 min",
            condition_fn=_cond_reg_2,
            action_fn=_act_reg_2
        ),
        Rule(
            rule_id="REGRA_REG_3",
            rule_type="Estratégia",
            antecedent_desc="weather_Rain == 1 E weather_Storm == 0",
            consequent_desc="tempo_calculado += 5.8 min",
            condition_fn=_cond_reg_3,
            action_fn=_act_reg_3
        ),
        Rule(
            rule_id="REGRA_REG_4",
            rule_type="Recomendação",
            antecedent_desc="weather_Cloudy == 1 E distance_km > 4.0",
            consequent_desc="tempo_calculado += 1.8 min",
            condition_fn=_cond_reg_4,
            action_fn=_act_reg_4
        ),
        Rule(
            rule_id="REGRA_REG_5",
            rule_type="Diretriz",
            antecedent_desc="traffic_level_Low == 0 E traffic_level_Medium == 0 E distance_km > 3.5",
            consequent_desc="tempo_calculado += 3.2 min",
            condition_fn=_cond_reg_5,
            action_fn=_act_reg_5
        ),
        Rule(
            rule_id="REGRA_REG_6",
            rule_type="Recomendação",
            antecedent_desc="tempo_calculado > 120.0",
            consequent_desc="tempo_calculado = 120.0 min (Teto Operacional)",
            condition_fn=_cond_reg_6,
            action_fn=_act_reg_6
        )
    ]


class ExpertSystemClassifier:
    """
    Sistema Especialista baseado em Regras SE-ENTÃO para Classificação (late_delivery).
    Separa a Base de Conhecimento (build_classification_knowledge_base + arquivo .txt)
    do Motor de Inferência (ForwardChainingEngine) e provê Módulo de Explicação.
    """
    def __init__(self, rules_file: str = None):
        self.rules_file = rules_file
        self.raw_rules_text: List[str] = []
        if rules_file and os.path.exists(rules_file):
            self.load_rules(rules_file)
        self.knowledge_base = build_classification_knowledge_base()
        self.engine = ForwardChainingEngine(
            self.knowledge_base,
            stop_on_first_goal=True,
            goal_key='late_delivery_pred'
        )

    def load_rules(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.raw_rules_text = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    def explain_instance(self, row: pd.Series) -> Dict[str, Any]:
        """Módulo de Explicação: retorna o fato inferido e a lista exata de regras acionadas."""
        facts, fired = self.engine.infer_single(row.to_dict())
        return {
            'predicao': int(facts['late_delivery_pred']),
            'fator_certeza': float(facts['confidence_score']),
            'regras_acionadas': fired
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Executa a inferência sobre o lote X."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        w_storm = df['weather_Storm'].to_numpy() == 1 if 'weather_Storm' in df.columns else (df.get('weather', pd.Series(False, index=df.index)) == 'Storm').to_numpy()
        w_rain = df['weather_Rain'].to_numpy() == 1 if 'weather_Rain' in df.columns else (df.get('weather', pd.Series(False, index=df.index)) == 'Rain').to_numpy()
        if 'traffic_level_Low' in df.columns and 'traffic_level_Medium' in df.columns:
            t_high = (df['traffic_level_Low'].to_numpy() == 0) & (df['traffic_level_Medium'].to_numpy() == 0)
        elif 'traffic_level' in df.columns:
            t_high = (df['traffic_level'] == 'High').to_numpy()
        else:
            t_high = np.zeros(len(df), dtype=bool)

        dist = df['distance_km'].to_numpy(dtype=float)
        prep = df['restaurant_preparation_time_minutes'].to_numpy(dtype=float)
        est = df['estimated_delivery_time_minutes'].to_numpy(dtype=float)

        # Aplicação estritamente na ordem de prioridade do encadeamento para frente
        r1 = w_storm
        r2 = (~r1) & w_rain & (dist > 4.0)
        r3 = (~(r1 | r2)) & w_rain & (prep > 11.0)
        r4 = (~(r1 | r2 | r3)) & t_high & (prep > 14.0)
        r5 = (~(r1 | r2 | r3 | r4)) & ((est - prep) <= 14.0) & (prep > 13.0)

        preds = np.where(r1 | r2 | r3 | r4 | r5, 1, 0)
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna pseudo-probabilidades baseadas nos Fatores de Certeza (Certainty Factors)
        associados a cada regra na Base de Conhecimento (conforme exigido na Seção 5.1 do edital).
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        w_storm = df['weather_Storm'].to_numpy() == 1 if 'weather_Storm' in df.columns else (df.get('weather', pd.Series(False, index=df.index)) == 'Storm').to_numpy()
        w_rain = df['weather_Rain'].to_numpy() == 1 if 'weather_Rain' in df.columns else (df.get('weather', pd.Series(False, index=df.index)) == 'Rain').to_numpy()
        if 'traffic_level_Low' in df.columns and 'traffic_level_Medium' in df.columns:
            t_high = (df['traffic_level_Low'].to_numpy() == 0) & (df['traffic_level_Medium'].to_numpy() == 0)
        elif 'traffic_level' in df.columns:
            t_high = (df['traffic_level'] == 'High').to_numpy()
        else:
            t_high = np.zeros(len(df), dtype=bool)

        dist = df['distance_km'].to_numpy(dtype=float)
        prep = df['restaurant_preparation_time_minutes'].to_numpy(dtype=float)
        est = df['estimated_delivery_time_minutes'].to_numpy(dtype=float)

        scores = np.full(len(df), 0.22, dtype=float)
        r5 = ((est - prep) <= 14.0) & (prep > 13.0)
        r4 = t_high & (prep > 14.0)
        r3 = w_rain & (prep > 11.0)
        r2 = w_rain & (dist > 4.0)
        r1 = w_storm

        scores = np.where(r5, 0.68, scores)
        scores = np.where(r4, 0.65, scores)
        scores = np.where(r3, 0.72, scores)
        scores = np.where(r2, 0.78, scores)
        scores = np.where(r1, 0.90, scores)

        return np.column_stack([1.0 - scores, scores])


class ExpertSystemRegressor:
    """
    Sistema Especialista baseado em Regras SE-ENTÃO para Regressão (actual_delivery_time_minutes).
    Separa a Base de Conhecimento (build_regression_knowledge_base + arquivo .txt)
    do Motor de Inferência (ForwardChainingEngine) e provê Módulo de Explicação.
    """
    def __init__(self, rules_file: str = None):
        self.rules_file = rules_file
        self.raw_rules_text: List[str] = []
        if rules_file and os.path.exists(rules_file):
            self.load_rules(rules_file)
        self.knowledge_base = build_regression_knowledge_base()
        self.engine = ForwardChainingEngine(
            self.knowledge_base,
            stop_on_first_goal=False
        )

    def load_rules(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.raw_rules_text = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    def explain_instance(self, row: pd.Series) -> Dict[str, Any]:
        """Módulo de Explicação: retorna o valor contínuo inferido e a sequência de regras disparadas."""
        facts, fired = self.engine.infer_single(row.to_dict())
        return {
            'predicao_minutos': round(float(facts['tempo_calculado']), 2),
            'regras_acionadas': fired
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Executa o encadeamento para frente sobre todas as amostras de X."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        est = df['estimated_delivery_time_minutes'].to_numpy(dtype=float)
        dist = df['distance_km'].to_numpy(dtype=float)

        w_storm = df['weather_Storm'].to_numpy() == 1 if 'weather_Storm' in df.columns else (df.get('weather', pd.Series(False, index=df.index)) == 'Storm').to_numpy()
        w_rain = df['weather_Rain'].to_numpy() == 1 if 'weather_Rain' in df.columns else (df.get('weather', pd.Series(False, index=df.index)) == 'Rain').to_numpy()
        w_cloudy = df['weather_Cloudy'].to_numpy() == 1 if 'weather_Cloudy' in df.columns else (df.get('weather', pd.Series(False, index=df.index)) == 'Cloudy').to_numpy()

        if 'traffic_level_Low' in df.columns and 'traffic_level_Medium' in df.columns:
            t_high = (df['traffic_level_Low'].to_numpy() == 0) & (df['traffic_level_Medium'].to_numpy() == 0)
        elif 'traffic_level' in df.columns:
            t_high = (df['traffic_level'] == 'High').to_numpy()
        else:
            t_high = np.zeros(len(df), dtype=bool)

        # Ciclo 1: Regra 1 (Heurística de tempo base)
        tempo = est + 2.5
        # Ciclo 2: Regra 2 (Relação - Tempestade)
        tempo = np.where(w_storm, tempo + 12.5, tempo)
        # Ciclo 3: Regra 3 (Estratégia - Chuva sem tempestade)
        tempo = np.where(w_rain & (~w_storm), tempo + 5.8, tempo)
        # Ciclo 4: Regra 4 (Recomendação - Nublado e distância > 4km)
        tempo = np.where(w_cloudy & (dist > 4.0), tempo + 1.8, tempo)
        # Ciclo 5: Regra 5 (Diretriz - Trânsito intenso e distância > 3.5km)
        tempo = np.where(t_high & (dist > 3.5), tempo + 3.2, tempo)
        # Ciclo 6: Regra 6 (Recomendação - Teto operacional 120 min)
        tempo = np.minimum(tempo, 120.0)

        return tempo
