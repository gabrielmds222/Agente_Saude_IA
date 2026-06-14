import pandas as pd
import numpy as np
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

df = pd.read_csv("datasets/dataset_pt.csv")
df["Disease"] = df["Disease"].str.strip()

colunas_sintoma = [c for c in df.columns if c.startswith("Symptom")]

todos_sintomas = pd.Series(
    df[colunas_sintoma].values.ravel()
).dropna().unique()

df_binario = pd.DataFrame()
df_binario["Disease"] = df["Disease"]

for sintoma in todos_sintomas:
    df_binario[sintoma] = df[colunas_sintoma].apply(
        lambda row: 1 if sintoma in row.values else 0, axis=1
    )

print(f"Shape do dataset binário: {df_binario.shape}")
print(f"Primeiras colunas: {df_binario.columns[:5].tolist()}")

arestas = [("Disease", sintoma) for sintoma in todos_sintomas]
model = BayesianNetwork(arestas)

print("\nAprendendo parâmetros do dataset...")
model.fit(df_binario, estimator=MaximumLikelihoodEstimator)

valido = model.check_model()
print(f"Modelo válido: {valido}")

with open("datasets/modelo_rede.pkl", "wb") as f:
    pickle.dump({"model": model, "sintomas": list(todos_sintomas)}, f)

print("Modelo salvo em datasets/modelo_rede.pkl")