# Predição de ETA e Atrasos em Entregas de Comida: Redes Neurais Artificiais com PyTorch

[![Python 3.14](https://img.shields.io/badge/Python-3.14.4-blue.svg)](https://www.python.org/)
[![PyTorch 2.15](https://img.shields.io/badge/PyTorch-2.15.0%2Bcpu-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)

Repositório oficial da **Atividade Prática da Unidade 3 (DCT1401 — Inteligência Artificial, UFRN/CERES)** dedicada ao projeto, treinamento, estudo sistemático de hiperparâmetros, persistência em disco, avaliação de generalização e eficiência computacional de **Redes Neurais Artificiais (*Multilayer Perceptrons* — MLP)** implementadas em **PyTorch (`torch.nn.Module`)** para problemas de **Regressão** e **Classificação Binária**.

---

## 1. Identificação e Fonte dos Dados

* **Autor:** Felipe Eduardo Silva Fernandes (`Matrícula: 20250037964` | `felipe.fernandes.710@ufrn.edu.br`)
* **Docente:** Prof. Dr. Thommas K. S. Flores (Departamento de Computação e Tecnologia — DCT/CERES/UFRN)
* **Fonte do Conjunto de Dados:** [Food Delivery Orders and ETA Logistics Dataset (Kaggle)](https://www.kaggle.com/datasets/razanihababdellatif/food-delivery-orders-and-eta-logistics-dataset)
* **Dimensões e Escopo:** `50.000` registros brutos auditados $\rightarrow$ `49.132` pedidos concluídos (*Completed*) após filtragem de cancelamentos sem tempo real de entrega (`868` registros), estruturados em **26 *features* preditivas** (10 numéricas contínuas/discretas e 16 binárias oriundas de *One-Hot Encoding*) e 1 variável-alvo por tarefa:
  * **Problema de Regressão (`data/processed/dataset_regression.csv`):** Predição contínua do tempo real total de entrega em minutos (`actual_delivery_time_minutes`, média de $44{,}22\text{ min}$ e desvio padrão de $14{,}76\text{ min}$).
  * **Problema de Classificação Binária (`data/processed/dataset_classification.csv`):** Predição da ocorrência de atraso logístico severo (`late_delivery` $\in \{0, 1\}$, com $63{,}4\%$ de entregas no prazo e $36{,}6\%$ de entregas atrasadas).

---

## 2. Ambiente de Hardware e Software

Todos os experimentos de treinamento, estudo controlado de hiperparâmetros e *profiling* computacional foram executados localmente no seguinte ambiente:

| Especificação | Detalhamento |
| :--- | :--- |
| **Equipamento** | Notebook Lenovo |
| **Processador (CPU)** | 12th Gen Intel® Core™ i7-1255U (10 núcleos físicos / 12 *threads* lógicos, TDP considerado: $45\text{ W}$) |
| **Memória RAM** | 12 GB DDR4 físicos instalados (~$9{,}93\text{ GiB}$ disponíveis ao Sistema Operacional) |
| **Armazenamento** | SSD NVMe de 512 GB |
| **Dispositivo PyTorch** | CPU (`torch 2.15.0+cpu`, sem GPU dedicada CUDA) |
| **Sistema Operacional** | Linux Ubuntu / Kernel `7.0.0-38-generic` (`x86_64`) |
| **Linguagem e Bibliotecas** | `Python 3.14.4`, `torch 2.15.0+cpu`, `scikit-learn 1.9.0`, `pandas 2.3.3`, `numpy 2.3.5`, `psutil 7.1.3` |

---

## 3. Estrutura do Repositório

A organização do repositório segue estritamente a estrutura recomendada no **Listing 5** do edital da Unidade 3:

```text
food-delivery-eta-analysis-ml/
├── README.md                                  # Documentação principal e instruções de execução
├── requirements.txt                           # Dependências com versões fixadas (incluindo torch)
├── data/
│   ├── raw/                                   # Dataset original bruto (food_delivery_dataset.csv)
│   └── processed/                             # Datasets limpos e pré-processados (Unidade 1)
│       ├── dataset_regression.csv             # 49.132 linhas x 27 colunas (alvo: actual_delivery_time_minutes)
│       └── dataset_classification.csv         # 49.132 linhas x 27 colunas (alvo: late_delivery)
├── notebooks/
│   ├── 01_eda.ipynb                           # Análise Exploratória de Dados (Unidade 1)
│   ├── 02_preprocessing.ipynb                 # Limpeza, Winsorização e Engenharia de Atributos (Unidade 1)
│   ├── 03_mlp_regressao.ipynb                 # Rede de Regressão em PyTorch: baseline (5 sementes) e treino final
│   ├── 04_mlp_classificacao.ipynb             # Rede de Classificação em PyTorch: baseline, limiar ótimo e treino final
│   ├── 05_hiperparametros.ipynb               # Estudo sistemático de 9 hiperparâmetros, interação 2D e busca final
│   ├── 06_salvar_carregar_inferencia.ipynb    # Persistência (.pt), inferência limpa, TorchScript e Quantização INT8
│   ├── 07_avaliacao_generalizacao.ipynb       # Métricas Aula 08, ROC/PR, resíduos, gap, 5-Fold CV e comp. Unidade 2
│   └── 08_eficiencia_computacional.ipynb      # Tempo, memória (psutil), lote vs. individual, MACs, CO2 e resumo final
├── src/
│   ├── __init__.py
│   ├── eda.py                                 # Funções auxiliares de análise exploratória
│   ├── preprocessing.py                       # Pipeline de limpeza e codificação
│   ├── models.py                              # Definições parametrizadas de redes neurais (MLP / nn.Module), params e MACs
│   ├── train.py                               # FastTensorDataLoader, laço de treino, validação e early stopping
│   ├── metrics.py                             # Cálculo de todas as métricas de regressão e classificação (Aula 08)
│   ├── inference.py                           # Carregamento de checkpoint .pt e predição (importável e CLI)
│   ├── profiling.py                           # Medição de tempo, memória residente (psutil), lote vs. unitário e CO2
│   └── visualization.py                       # Gráficos de curvas de aprendizado e painéis de hiperparâmetros
├── models/
│   ├── modelo_regressao.pt                    # Checkpoint completo final de regressão (pesos + arquitetura + scaler)
│   ├── modelo_classificacao.pt                # Checkpoint completo final de classificação (pesos + scaler + limiar)
│   ├── modelo_regressao_torchscript.pt        # Modelo de regressão exportado via TorchScript (torch.jit.trace)
│   └── modelo_classificacao_torchscript.pt    # Modelo de classificação exportado via TorchScript (torch.jit.trace)
├── logs/
│   └── hiperparametros.csv                    # Registro tabular de todas as 352 execuções do estudo de hiperparâmetros
├── reports/
│   ├── figures/                               # Gráficos gerados em alta resolução (300 DPI)
│   └── tables/                                # Tabelas-resumo e de sensibilidade exportadas em CSV
│       ├── resumo_final.csv
│       ├── sensibilidade_hiperparametros.csv
│       ├── avaliacao_regressao_completa.csv
│       ├── avaliacao_classificacao_completa.csv
│       ├── kfold_mlp_estabilidade.csv
│       └── inferencia_lote_vs_individual.csv
└── docs/
    ├── relatorio_final.tex                    # Relatório técnico completo em LaTeX (formato article 12pt A4)
    └── imagens_relatorio.zip                  # Pacote de imagens (.jpg) e .tex para compilação rápida no Overleaf
```

---

## 4. Instruções de Instalação e Reprodutibilidade

### 4.1 Instalação das Dependências

```bash
python3 -m venv 1ia.venv
source 1ia.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Ordem de Execução dos Notebooks

Para reproduzir integralmente os experimentos a partir dos dados brutos:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_mlp_regressao.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_mlp_classificacao.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_hiperparametros.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_salvar_carregar_inferencia.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/07_avaliacao_generalizacao.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/08_eficiencia_computacional.ipynb
```

### 4.3 Como Carregar o Modelo Salvo e Fazer Predições via Linha de Comando (`src/inference.py`)

Os checkpoints `.pt` salvos em `models/` são totalmente autocontidos (armazenam `state_dict`, dicionário `config` da arquitetura, vetores `x_media` e `x_desvio` do `StandardScaler`, lista ordenada de `features`, parâmetros do *target* e limiar ótimo de decisão), não exigindo acesso ao conjunto de treino:

```bash
# Predição de Classificação Binária (Atraso de Entrega) em novos pedidos
python src/inference.py --modelo models/modelo_classificacao.pt --entrada data/processed/dataset_classification.csv --n 10

# Predição de Regressão (Tempo Real de Entrega em minutos) em novos pedidos
python src/inference.py --modelo models/modelo_regressao.pt --entrada data/processed/dataset_regression.csv --n 10
```

---

## 5. Resumo dos Resultados Principais

### 5.1 Comparativo de Desempenho no Conjunto de Teste ($N = 9.827$)

| Problema | Modelo / Configuração | Arquitetura | Parâmetros | Métrica Principal (Teste) | Métrica Secundária (Teste) | Gap (Treino vs. Teste) | Tempo Treino (s) | Latência Lote ($\mu\text{s}$/amostra) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão** | **MLP Final Otimizada (Unid. 3)** | `(64, 64)` GELU | `5.953` | **RMSE = 2,5506 min** | **$\text{MAE} = 2{,}0454\text{ min}$ / $R^2 = 0{,}9702$** | $+0{,}0154\text{ min}$ | $21{,}97\text{ s}$ | $0{,}241\,\mu\text{s}$ |
| **Regressão** | **MLP Referência Baseline (Unid. 3)** | `(32, 32)` ReLU | `1.953` | RMSE = 2,5958 min | $\text{MAE} = 2{,}0759\text{ min}$ / $R^2 = 0{,}9691$ | $+0{,}0813\text{ min}$ | $7{,}80\text{ s}$ | $0{,}058\,\mu\text{s}$ |
| **Regressão** | *Gradient Boosting* (Unid. 2) | 150 árvores | — | RMSE = 2,6866 min | $\text{MAE} = 2{,}1448\text{ min}$ / $R^2 = 0{,}9670$ | $+0{,}1761\text{ min}$ | $12{,}79\text{ s}$ | $1{,}820\,\mu\text{s}$ |
| **Regressão** | *Árvore de Decisão* (Unid. 2) | `max_depth=8` | — | RMSE = 4,0349 min | $\text{MAE} = 3{,}1386\text{ min}$ / $R^2 = 0{,}9255$ | $+0{,}1603\text{ min}$ | $0{,}23\text{ s}$ | $0{,}150\,\mu\text{s}$ |
| **Classificação** | **MLP Final Otimizada (Unid. 3, Limiar=0,43)** | `(64, 64)` GELU | `5.953` | **F1 = 0,7822** | **$\text{ROC-AUC} = 0{,}9182$ / $\text{PR-AUC} = 0{,}8796$** | $+0{,}0034$ | $18{,}83\text{ s}$ | $0{,}178\,\mu\text{s}$ |
| **Classificação** | **MLP Referência Baseline (Unid. 3, Limiar=0,50)** | `(32, 32)` ReLU | `1.953` | F1 = 0,7697 | $\text{ROC-AUC} = 0{,}9153$ / $\text{PR-AUC} = 0{,}8748$ | $+0{,}0114$ | $3{,}31\text{ s}$ | $0{,}069\,\mu\text{s}$ |
| **Classificação** | *Gradient Boosting* (Unid. 2) | 150 árvores | — | F1 = 0,7644 | $\text{ROC-AUC} = 0{,}9089$ / $\text{PR-AUC} = 0{,}8664$ | $+0{,}0106$ | $10{,}50\text{ s}$ | $1{,}230\,\mu\text{s}$ |
| **Classificação** | *Árvore de Decisão* (Unid. 2) | `max_depth=7` | — | F1 = 0,7060 | $\text{ROC-AUC} = 0{,}8612$ / $\text{PR-AUC} = 0{,}7884$ | $+0{,}0105$ | $0{,}18\text{ s}$ | $0{,}120\,\mu\text{s}$ |

### 5.2 Principais Conclusões sobre Sensibilidade de Hiperparâmetros

1. **Inicialização dos Pesos e Taxa de Aprendizado ($\eta$) são os fatores mais críticos:** Inicializar pesos com `zeros` impede a quebra de simetria ($\nabla_{W^{(l)}} L = 0$), colapsando o modelo ($R^2 = 0{,}0000$, $\text{F1} = 0{,}0000$). Taxas de aprendizado excessivas ($\eta = 1{,}0$) violam a condição de estabilidade $\eta < 2/\lambda_{\max}(H)$ e fazem o gradiente divergir na primeira época ($\Delta\text{RMSE} = 11{,}83\text{ min}$).
2. **Forte acoplamento entre Taxa de Aprendizado ($\eta$) e Tamanho do *Batch* ($B$):** *Batches* grandes ($B = 2048$ ou `total`) reduzem drasticamente o número de atualizações de pesos por época e sofrem subajuste severo sob $\eta \le 10^{-3}$, exigindo escalonamento proporcional da taxa de aprendizado ($\eta = 10^{-2}$).
3. **Superioridade de Ativações Não-Saturantes Suaves (`GELU`) e Otimizadores Adaptativos (`AdamW`):** A ativação `GELU` combinada ao `AdamW` ($\eta = 10^{-3}$, `weight_decay = 1e-4`) reduziu o RMSE de validação para $2{,}57\text{ min}$ e elevou o F1 para $0{,}7809$, superando inclusive os métodos de *ensemble* baseados em árvores da Unidade 2.
