# 🏥 Agente de Triagem Médica com Rede Bayesiana

O projeto tem o objetivo de criar um agente de saúde com IA para diagnosticar doenças a partir de sintomas informados pelo usuário. O agente faz perguntas estratégicas, atualiza suas crenças a cada resposta usando o **Teorema de Bayes** e converge para o diagnóstico mais provável entre 41 doenças.

---

## ⚙️ Pré-requisitos e Instalação

**Python 3.10+** instalado na máquina.

```bash
pip install pgmpy==0.1.25 pandas numpy matplotlib networkx scikit-learn
```

### Dataset

Baixe os 4 arquivos CSV no link abaixo e coloque todos dentro da pasta `datasets/`:

🔗 [Disease Symptom Prediction — Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

---

## 🗂️ Estrutura do Projeto

```
Agente_Saude_IA/
│
├── datasets/                        ← CSVs do Kaggle (não versionados)
│
├── notebooks/
│   └── 01_tratamento_dados.ipynb   ← Exploração, limpeza e construção da rede
│
├── scripts/
│   ├── 01_Motor_Inferencia.py      ← Motor de inferência
│   ├── 02_Agent.py                 ← Agente interativo
│   ├── 03_Validation.py            ← Métricas de avaliação
│   └── 04_Web_view.py              ← Visualização do grafo da rede
│
├── .gitignore
└── README.md
```

---

## ▶️ Ordem de Execução

**1. Notebook — Tratamento dos dados e construção da rede**

Abra e execute todas as células do notebook:

```
notebooks/01_tratamento_dados.ipynb
```

**2. Scripts — em ordem**

```bash
python scripts/01_Motor_Inferencia.py 
python scripts/02_Agent.py            
python scripts/03_Validation.py       
python scripts/04_Web_view.py         
```

---

## 🧠 Como Funciona

### Rede Bayesiana

A rede utiliza arquitetura **Naive Bayes** — um nó `Disease` como pai conectado a cada um dos 131 sintomas como filhos. As tabelas de probabilidade condicional (CPTs) são aprendidas automaticamente pelo **Maximum Likelihood Estimation**:

$$P(\text{sintoma} \mid \text{doença}) = \frac{\text{linhas com o sintoma}}{\text{total de linhas da doença}}$$

### Motor de Inferência

Dado um conjunto de sintomas observados, o motor calcula a probabilidade posterior de cada doença usando **Variable Elimination**:

$$P(D \mid s_1, s_2, ..., s_n) \propto P(D) \times \prod_{i=1}^{n} P(s_i \mid D)$$

### Seleção de Perguntas por Entropia

O agente escolhe o próximo sintoma a perguntar selecionando sempre o que mais reduz a incerteza sobre o diagnóstico (**Entropia de Shannon**):

$$H = -\sum_{i=1}^{41} P(D_i) \log_2 P(D_i)$$

---

## 📈 Métricas de Avaliação

Avaliação realizada com amostra de 5 casos por doença (205 casos totais):

| Métrica | Valor |
|---------|-------|
| Acurácia geral | **94.6%** |
| Macro F1-Score | **0.93** |
| Doenças com F1 = 1.00 | **38 / 41** |
