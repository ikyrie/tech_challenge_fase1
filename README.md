# Diagnóstico de câncer de mama (benigno e maligno) com Machine Learning

Este projeto foi desenvolvido como parte do **Tech Challenge – Fase 1**, cujo objetivo é criar uma solução inicial baseada em Inteligência Artificial para **suporte ao diagnóstico médico**.  
Utilizei o dataset público **Breast Cancer Wisconsin (Diagnostic)** para construir e avaliar modelos de classificação.

| O **relatório técnico e passo-a-passo** está no [Google Colab](https://colab.research.google.com/drive/10a7l8W8aCbOZOgRnD7w6hCH2GsJA5xLl?usp=sharing)

## Descrição
O sistema realiza:
- **Exploração de dados**: estatísticas descritivas e visualizações para entender o dataset.
- **Pré-processamento**: tratamento de valores ausentes, normalização e codificação.
- **Modelagem**: aplicação de algoritmos como KNN e SVM.
- **Avaliação**: métricas com *accuracy*.
- **Interpretabilidade**: uso de *feature importance*.

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Data Set**
📌 Disponível no [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

O data set está no código como `breast_cancer_dataset.csv`.

## 🛠 Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Manipulação de dados:** Pandas e NumPy
* **Visualização:** Matplotlib e Seaborn
* **Machine Learning:** Scikit-learn
* **Interpretabilidade:** Scikit-learn


## Como Rodar o Projeto
### Localmente

#### Criar e ativar um ambiente virtual

```bash
python -m venv .venv

# Ativar no Windows
.\.venv\Scripts\activate

# Ativar no Linux/macOS
source .venv/bin/activate
```

#### Instalar as dependências

```bash
pip install -r requirements.txt
```

#### Executar o código

```bash
python3 main.py
```
