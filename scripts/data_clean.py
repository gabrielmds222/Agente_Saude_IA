import pandas as pd
import numpy as np

df = pd.read_csv("datasets/dataset_pt.csv")
prob_check = pd.read_csv("datasets/probabilidades_pt.csv")

df["Disease"] = df["Disease"].str.strip()

# Quantas linhas tem a Dengue no dataset?
print("Linhas de Dengue:", len(df[df["Disease"] == "Dengue"]))
print("\nAmostra das linhas de Dengue:")
print(df[df["Disease"] == "Dengue"].head(10).to_string())

colunas_sintoma = [c for c in df.columns if c.startswith("Symptom")]
for col in colunas_sintoma:
    df[col] = df[col].str.strip()

df_longo = df.melt(id_vars="Disease", value_vars=colunas_sintoma, value_name="Sintoma")
df_longo = df_longo.dropna(subset=["Sintoma"])
df_longo = df_longo.drop(columns="variable").drop_duplicates()

# ── Calcular P(sintoma | doença) CORRIGIDO ────────────────────────────────────
resultados = []

for doenca in df["Disease"].unique():
    df_doenca = df[df["Disease"] == doenca]
    total_linhas = len(df_doenca)
    
    # Pega todos os sintomas dessa doença em formato longo
    sintomas_doenca = df_doenca[colunas_sintoma].values.ravel()
    sintomas_doenca = pd.Series(sintomas_doenca).dropna().str.strip()
    
    # Conta quantas vezes cada sintoma aparece nas linhas dessa doença
    contagem = sintomas_doenca.value_counts()
    
    for sintoma, count in contagem.items():
        resultados.append({
            "Disease": doenca,
            "Sintoma": sintoma,
            "contagem": count,
            "P_sintoma_dado_doenca": count / total_linhas
        })

contagem_par = pd.DataFrame(resultados)
print(f"Total de doenças: {contagem_par['Disease'].nunique()}")
print(f"Total de sintomas únicos: {contagem_par['Sintoma'].nunique()}")
print(f"\nExemplo — P(sintoma | Dengue):")
print(contagem_par[contagem_par["Disease"] == "Dengue"][["Sintoma", "P_sintoma_dado_doenca"]].to_string(index=False))