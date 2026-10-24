# Avaliação Comparativa de Modelos de ML: Sistema Especialista, Árvore de Decisão e Ensemble Learning

**Instituição:** Universidade Federal do Rio Grande do Norte (UFRN) — Departamento de Computação e Tecnologia (DCT / CERES)  
**Curso:** Bacharelado em Sistemas da Informação (BSI)  
**Disciplina:** DCT1401 - Inteligência Artificial (Período Letivo: 2026.2)  
**Professor:** Prof. Dr. Thommas K. S. Flores  
**Aluno:** Felipe Eduardo Silva Fernandes  
**Matrícula:** 20250037964  
**Dataset:** [Food Delivery Orders and ETA Logistics Dataset (Kaggle)](https://www.kaggle.com/datasets/razanihababdellatif/food-delivery-orders-and-eta-logistics-dataset)  
**Repositório GitHub:** [https://github.com/Felipe-edu34/food-delivery-eta-analysis-ml](https://github.com/Felipe-edu34/food-delivery-eta-analysis-ml)

---

## 📌 Descrição e Objetivos do Projeto

Este projeto conduz a análise exploratória, pré-processamento rigoroso sem vazamento de dados (*data leakage*) e avaliação comparativa entre três famílias de modelos de Inteligência Artificial e Aprendizado de Máquina aplicadas à logística de entregas de refeições (*food delivery*):

1. **Sistema Especialista Baseado em Regras SE-ENTÃO (Aula 05):** Base de Conhecimento com regras tipadas (*Relação*, *Recomendação*, *Diretriz*, *Estratégia* e *Heurística*) sem mistura de conjunções (`E`) e disjunções (`OU`), separada do Motor de Inferência por Encadeamento para Frente (*Forward Chaining*) e do Módulo de Explicação, com medição cronometrada do esforço humano de elaboração.
2. **Árvore de Decisão (Aula 06):** `DecisionTreeClassifier` e `DecisionTreeRegressor`, com cálculo analítico de Entropia, Ganho de Informação ($G$) e Índice de Gini ($\Delta\text{Gini}$), comparação de critérios de impureza, poda via `GridSearchCV` ($K=5$), contraste explícito com árvores sem restrição de profundidade (*overfitting*) e interpretação de regras extraídas.
3. **Ensemble Learning (Aula 07):** `RandomForest` (*Bagging*) e `GradientBoosting` (*Boosting*), com ajuste de hiperparâmetros via validação cruzada, curva de convergência do erro em função do número de árvores ($B \in [1, 200]$) fundamentada no *trade-off* Viés $\times$ Variância e análise de importância das variáveis (`feature_importances_`).

Os dois problemas investigados sobre os $49.132$ pedidos concluídos da base (divididos em $80\%$ treino $= 39.305$ e $20\%$ teste $= 9.827$, com `random_state=42`) são:
- **Regressão:** Predição do tempo real de entrega em minutos (`actual_delivery_time_minutes`), avaliado por $\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $\text{MAPE}$, $R^2$ e $\text{MedAE}$ (Aula 08).
- **Classificação:** Predição de ocorrência de atraso na entrega (`late_delivery`: $1 = \text{Atrasado}$ $[36{,}57\%]$, $0 = \text{No Prazo}$ $[63{,}43\%]$), avaliada por Matriz de Confusão, $\text{Accuracy}$, $\text{Precision}$, $\text{Recall}$, $\text{F1-score}$, $\text{ROC-AUC}$ e $\text{PR-AUC}$ (Aula 08).

---

## 📂 Estrutura de Diretórios do Repositório

```text
.
├── README.md                           # Descrição do projeto, instruções de uso e resumo dos resultados
├── requirements.txt                    # Dependências (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, psutil, codecarbon)
├── data/
│   ├── raw/                            # Dataset bruto original intacto (50.000 linhas x 37 colunas)
│   │   └── food_delivery_orders_dataset.csv
│   └── processed/                      # Datasets limpos e codificados (49.132 pedidos entregues)
│       ├── dataset_clean.csv           # Base limpa sem pedidos cancelados e com Winsorization (120 min)
│       ├── dataset_regression.csv      # Base para Regressão (23 features + actual_delivery_time_minutes)
│       ├── dataset_classification.csv  # Base para Classificação (23 features + late_delivery)
│       └── dataset_preprocessed.csv    # Base unificada codificada sem multicolinearidade ou leakage
├── notebooks/
│   ├── 01_eda.ipynb                    # Inspeção inicial, estatísticas, nulos, outliers (IQR), correlações e gráficos
│   ├── 02_preprocessing.ipynb          # Limpeza por coluna, Winsorization, One-Hot, ANOVA/Chi2/f_regression e Pipeline
│   ├── 03_expert_system.ipynb          # Sistema Especialista SE-ENTÃO, Forward Chaining, cronômetro e inferência
│   ├── 04_decision_tree.ipynb          # Árvore de Decisão, Ganho/Gini, GridSearchCV, overfitting e extração de regras
│   ├── 05_ensemble.ipynb               # Random Forest e Gradient Boosting, curva de erro vs. B e feature importances
│   ├── 06_evaluation_metrics.ipynb     # Avaliação lado a lado (Aula 08), Matrizes de Confusão, ROC, PR e Resíduos
│   ├── 07_generalization.ipynb         # Gap de generalização, Validação Cruzada K-Fold (K=5) e Viés x Variância
│   └── 08_computational_efficiency.ipynb # Tempo, memória RAM, energia (uWh), CO2 (mg), Edge vs. Nuvem e Seção 6
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Carregamento padronizado de datasets CSV
│   ├── eda.py                          # Estatísticas descritivas, nulos, contagem de outliers IQR e correlações
│   ├── preprocessing.py                # Limpeza, remoção de leakage/colinearidade, testes estatísticos e Pipeline
│   ├── expert_system.py                # Base de regras SE-ENTÃO, ForwardChainingEngine e Módulo de Explicação
│   ├── models.py                       # Cálculo de Entropia/Gini, tuning (GridSearch/RandomizedSearch) e complexidade
│   ├── metrics.py                      # Cálculo das métricas obrigatórias de regressão e classificação (Aula 08)
│   ├── profiling.py                    # Medição de inferência (30 repetições), RAM (tracemalloc/psutil) e energia/CO2
│   └── visualization.py                # Geração e salvamento de gráficos padronizados em alta resolução (300 DPI)
├── rules/
│   ├── regras_classificacao.txt        # Base de Conhecimento SE-ENTÃO para Classificação (6 regras tipadas)
│   └── regras_regressao.txt            # Base de Conhecimento SE-ENTÃO para Regressão (6 regras tipadas)
├── logs/
│   └── cronometro_regras.csv           # Registro das 6 sessões cronometradas de elaboração das regras
├── models/                             # Modelos treinados e pipelines serializados (.pkl)
├── reports/
│   ├── tables/                         # Tabelas CSV exportadas (incluindo resumo_final.csv)
│   └── figures/                        # Figuras PNG (300 DPI) geradas pelos notebooks 01 a 08
└── docs/
    └── relatorio_final.pdf             # Relatório técnico da atividade em formato PDF
```

---

## 🛠️ Instruções de Execução

### 1. Clonar o Repositório e Ativar o Ambiente Virtual
```bash
git clone https://github.com/Felipe-edu34/food-delivery-eta-analysis-ml.git
cd food-delivery-eta-analysis-ml
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 3. Executar os Notebooks na Ordem Sequencial (`01` a `08`)
Todos os notebooks possuem semente aleatória fixa (`random_state=42`) e podem ser executados interativamente no JupyterLab/VS Code ou via linha de comando na ordem abaixo:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_expert_system.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_decision_tree.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_ensemble.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_evaluation_metrics.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/07_generalization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/08_computational_efficiency.ipynb
```

---

## 📊 Resumo dos Principais Resultados Experimentais

### 1. Desempenho Preditivo e Generalização — Problema de Classificação (`late_delivery`)

| Família de Modelo | Algoritmo | Accuracy | Precision | Recall | F1-Score (Teste) | ROC-AUC | PR-AUC | Gap F1 ($\text{Tr}-\text{Te}$) | K-Fold ($K=5$) F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sistema Especialista** | Regras SE-ENTÃO (6 regras) | $0{,}6804$ | $0{,}5466$ | **$0{,}7398$** | $0{,}6287$ | $0{,}7370$ | $0{,}6713$ | **$0{,}0000$** | $0{,}6286 \pm 0{,}0066$ |
| **Árvore de Decisão** | `DecisionTreeClassifier` (Podada) | $0{,}8031$ | $0{,}7465$ | $0{,}6989$ | $0{,}7219$ | $0{,}8755$ | $0{,}8115$ | $+0{,}0094$ | $0{,}7205 \pm 0{,}0035$ |
| **Ensemble (Bagging)** | `RandomForestClassifier` ($B=150$) | $0{,}8287$ | $0{,}7831$ | $0{,}7354$ | $0{,}7585$ | $0{,}9006$ | $0{,}8533$ | $+0{,}1225$ | $0{,}7486 \pm 0{,}0077$ |
| **Ensemble (Boosting)** | **`GradientBoostingClassifier` ($B=150$)** | **$0{,}8335$** | **$0{,}7943$** | $0{,}7351$ | **$0{,}7636$** | **$0{,}9100$** | **$0{,}8673$** | $+0{,}0199$ | **$0{,}7562 \pm 0{,}0060$** |

---

### 2. Desempenho Preditivo e Generalização — Problema de Regressão (`actual_delivery_time_minutes`)

| Família de Modelo | Algoritmo | MAE (min) | MSE ($\text{min}^2$) | RMSE (min) | MAPE (%) | $R^2$ | MedAE (min) | Gap RMSE ($\text{Tr}-\text{Te}$) | K-Fold ($K=5$) RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sistema Especialista** | Regras SE-ENTÃO (6 regras) | $3{,}7587$ | $22{,}9555$ | $4{,}7912$ | $12{,}38\%$ | $0{,}8949$ | $3{,}3000$ | **$-0{,}0529$** | $4{,}7382 \pm 0{,}0251$ |
| **Árvore de Decisão** | `DecisionTreeRegressor` (Podada) | $2{,}7625$ | $12{,}4854$ | $3{,}5335$ | $8{,}78\%$ | $0{,}9428$ | $2{,}2500$ | $-0{,}6438$ | $3{,}5811 \pm 0{,}0488$ |
| **Ensemble (Bagging)** | `RandomForestRegressor` ($B=150$) | $2{,}3593$ | $8{,}9755$ | $2{,}9959$ | $7{,}63\%$ | $0{,}9589$ | $1{,}9820$ | $-0{,}9064$ | $3{,}0554 \pm 0{,}0348$ |
| **Ensemble (Boosting)** | **`GradientBoostingRegressor` ($B=150$)** | **$2{,}1563$** | **$7{,}2903$** | **$2{,}7001$** | **$7{,}11\%$** | **$0{,}9666$** | **$1{,}8192$** | $-0{,}1330$ | **$2{,}7164 \pm 0{,}0165$** |

---

### 3. Eficiência Computacional, Esforço Humano e Sustentabilidade (30 Repetições no Teste: $N = 9.827$)

| Problema | Algoritmo | Esforço Humano (Cronômetro) | Treino Computacional | Inferência Lote (30 rep) | Vazão (amostras/s) | Pico RAM Inf. (KB) | Tamanho Disco (KB) | Energia ($\mu\text{Wh}$) | Emissão $\text{CO}_2$ ($\text{mg}$) | Complexidade Estrutural |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classificação** | Regras SE-ENTÃO | **$110{,}0\text{ min}$** ($36{,}7 \pm 2{,}9\text{ min/sessão}$) | Não se aplica | **$0{,}37\text{ ms}$** | $26.612.536$ | $348{,}72$ | $5{,}09$ | $0{,}0015$ | $0{,}0001$ | 6 regras SE-ENTÃO |
| **Classificação** | `DecisionTreeClassifier` | Não se aplica | $0{,}2277\text{ s}$ | $1{,}99\text{ ms}$ | $4.941.013$ | $3.004{,}06$ | $17{,}92$ | $0{,}9570$ | $0{,}0438$ | Prof. 12 (203 nós / 102 folhas) |
| **Classificação** | `RandomForestClassifier` | Não se aplica | $3{,}5751\text{ s}$ | $100{,}10\text{ ms}$ | $98.167$ | $3.648{,}93$ | $41.478{,}31$ | $15{,}3133$ | $0{,}7014$ | 150 árvores ($530.144$ nós) |
| **Classificação** | `GradientBoostingClassifier` | Não se aplica | $9{,}4948\text{ s}$ | $22{,}82\text{ ms}$ | $430.657$ | $2.998{,}61$ | $363{,}87$ | $39{,}6568$ | $1{,}8163$ | 150 árvores rasas ($4.522$ nós) |
| **Regressão** | Regras SE-ENTÃO | **$90{,}0\text{ min}$** ($30{,}0 \pm 5{,}0\text{ min/sessão}$) | Não se aplica | **$0{,}35\text{ ms}$** | $28.032.062$ | $394{,}68$ | $4{,}30$ | $0{,}0015$ | $0{,}0001$ | 6 regras SE-ENTÃO |
| **Regressão** | `DecisionTreeRegressor` | Não se aplica | $0{,}1821\text{ s}$ | $2{,}25\text{ ms}$ | $4.358.530$ | $2.999{,}15$ | $189{,}03$ | $0{,}7681$ | $0{,}0352$ | Prof. 12 (2.661 nós / 1.331 folhas) |
| **Regressão** | `RandomForestRegressor` | Não se aplica | $5{,}1789\text{ s}$ | $119{,}62\text{ ms}$ | $82.155$ | $3.265{,}63$ | $70.662{,}64$ | $22{,}0772$ | $1{,}0111$ | 150 árvores ($1.004.300$ nós) |
| **Regressão** | `GradientBoostingRegressor` | Não se aplica | $9{,}1108\text{ s}$ | $22{,}00\text{ ms}$ | $446.681$ | $2.998{,}30$ | $358{,}88$ | $38{,}0533$ | $1{,}7428$ | 150 árvores rasas ($4.454$ nós) |

> **Ambiente Computacional Utilizado nas Medições:** Notebook **Lenovo** equipado com processador **12th Gen Intel® Core™ i7-1255U** (12 threads, TDP considerado de $45\text{ W}$), **12 GB de memória RAM** ($\approx 10\text{ GiB}$ disponíveis ao sistema operacional), armazenamento **SSD de 512 GB**, Sistema Operacional **Linux (Kernel 7.0.0-38-generic)**, **Python 3.14.4** e `scikit-learn 1.9.0`.
>
> **Nota:** A tabela consolidada completa em formato CSV encontra-se em [`reports/tables/resumo_final.csv`](file:///home/felipe/Documentos/food-delivery-eta-analysis-ml/reports/tables/resumo_final.csv), e o registro detalhado das sessões de cronometragem das regras encontra-se em [`logs/cronometro_regras.csv`](file:///home/felipe/Documentos/food-delivery-eta-analysis-ml/logs/cronometro_regras.csv).

---

## 📬 Contato

- **Aluno:** Felipe Eduardo Silva Fernandes
- **E-mail Institucional:** felipe.fernandes.710@ufrn.edu.br
