# Análise Exploratória e Preparação de Dados para Machine Learning

**Aluno:** Felipe Eduardo Silva Fernandes  
**Matrícula:** 20250037964  
**Curso:** Bacharelado em Sistemas de Informação (BSI)  
**Disciplina:** DCT1401 - Inteligência Artificial (2026.2)  
**Professor:** Thommas K. S. Flores  
**Dataset:** https://www.kaggle.com/datasets/razanihababdellatif/food-delivery-orders-and-eta-logistics-dataset

---

## 📌 Descrição do Projeto

Este repositório contém o desenvolvimento prático da atividade avaliativa de Análise Exploratória de Dados (EDA) e Preparação de Dados para modelos de Aprendizado de Máquina. O projeto foi desenvolvido simulando a demanda de uma empresa de consultoria analítica, utilizando dados brutos de um aplicativo de delivery de comida.

O cliente deseja resolver dois tipos de problemas distintos a partir do mesmo conjunto de dados:
1. **Problema de Regressão:** Prever o tempo real de entrega em minutos.
2. **Problema de Classificação:** Prever se um pedido sofrerá atraso na entrega.

Todo o processo de higienização, tratamento de nulos, detecção de outliers e análise de correlação foi documentado passo a passo nos notebooks da pasta `notebooks/`.

---

## 📂 Estrutura do Repositório

A arquitetura de pastas e arquivos segue rigorosamente as diretrizes da atividade para garantir organização e reprodutibilidade:

```text
├── README.md                      # Documentação principal do projeto
├── requirements.txt               # Dependências Python necessárias
├── data/
│   ├── raw/                       # Dataset original bruto (dataset.csv)
│   └── processed/                 # Datasets após limpeza e pré-processamento
│       ├── dataset_regression.csv
│       └── dataset_classification.csv
├── notebooks/
│   ├── 01_eda.ipynb               # Análise exploratória de dados
│   ├── 02_preprocessing.ipynb     # Limpeza e pré-processamento
│   ├── 03_regression_analysis.ipynb   # Percepções para regressão
│   └── 04_classification_analysis.ipynb # Percepções para classificação
├── src/
│   ├── __init__.py                
│   ├── data_loader.py             # Funções para carregar dados
│   ├── eda.py                     # Funções para análise exploratória
│   ├── preprocessing.py            # Funções para limpeza e pré-processamento
│   └── visualization.py            # Funções para criação de gráficos
└── reports/
    ├── figures/                   # Gráficos gerados durante a análise
    │   ├── correlation_matrix.png
    │   ├── missing_values.png
    │   ├── outliers_boxplot.png
    │   ├── regression_scatter.png
    │   └── confusion_matrix.png
    └── relatorio_final.pdf        # Relatório técnico da atividade
```   

## 🛠️ Dependências do Projeto

O ambiente de execução necessita das bibliotecas Python listadas abaixo (presentes no arquivo requirements.txt na raiz do projeto):

pandas
numpy
matplotlib
seaborn
scikit-learn

## 🚀 Como Executar o Código

Para reproduzir esta análise do zero na sua máquina, siga rigorosamente as instruções abaixo:

Clone este repositório:


git clone [link-do-seu-repositorio-aqui]
cd [nome-da-pasta-do-repositorio]


Instale as dependências listadas no arquivo de requisitos:


pip install -r requirements.txt

Ordem de Execução dos Notebooks:
Abra a pasta notebooks/ em sua IDE (como VS Code ou Jupyter) e execute todas as células (Run All) estritamente nesta ordem para garantir que os dados fluam corretamente:
01_eda.ipynb: Lê os dados brutos, exibe estatísticas descritivas e gera os gráficos iniciais.
02_preprocessing.ipynb: Realiza a limpeza do dataset, aplica as técnicas de encoding e salva os arquivos .csv finais na pasta data/processed/.
03_regression_analysis.ipynb: Carrega os dados processados, treina o modelo de regressão linear, analisa os pesos das variáveis e gera o gráfico de dispersão.
04_classification_analysis.ipynb: Treina o modelo de classificação de atrasos e gera a matriz de confusão.
Observação: As imagens geradas pelas funções utilitárias da pasta src/ (Matriz de Correlação, Boxplots, Gráficos de Dispersão, etc.) serão salvas automaticamente no diretório reports/figures/ ao rodar os notebooks correspondentes.

## 📊 Principais Resultados e Percepções

Com base nas etapas de exploração e treinamento inicial, destacam-se os seguintes insights que compõem o relatório técnico:

Para o Problema de Regressão
Variável Alvo Escolhida: actual_delivery_time_minutes.
Justificativa: É uma variável numérica contínua, perfeitamente adequada para prevermos o tempo exato (em minutos) que um pedido leva até chegar ao cliente.
Features mais relevantes: distance_km (Distância) e restaurant_preparation_time_minutes (Tempo de Preparo).
Recomendações e Desafios: Detectou-se o risco de data leakage (vazamento de dados) em colunas que estimavam o tempo de entrega previamente. Estas foram removidas. Como ponto de partida para a modelagem, recomenda-se a Regressão Linear Múltipla, com sugestão de evolução para algoritmos como Random Forest Regressor para capturar relações não-lineares.
Para o Problema de Classificação
Variável Alvo Escolhida: late_delivery.
Justificativa: É uma variável categórica binária (0 = No Prazo, 1 = Atrasado), ideal para a criação de modelos de classificação que ajudam a empresa a acionar alertas de monitoramento.
Features mais relevantes: Picos de demanda (entregas em finais de semana) e eventos climáticos adversos (como weather_Storm).
Recomendações e Desafios: O maior desafio encontrado foi o desbalanceamento natural da base (há um volume consideravelmente maior de entregas no prazo). Devido a isso, a Matriz de Confusão mostrou-se mais eficaz que a Acurácia para avaliar o modelo. Algoritmos sugeridos para testes iniciais incluem a Regressão Logística e Árvores de Decisão (Decision Trees).

## 📬 Contato

Email: felipe.fernandes.710@ufrn.edu.br
