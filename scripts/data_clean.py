import pandas as pd
import numpy as np

# ── Carregar dataset original ──────────────────────────────────────────────────
df = pd.read_csv("datasets/dataset.csv")

df["Disease"] = df["Disease"].str.strip()

colunas_sintoma = [c for c in df.columns if c.startswith("Symptom")]
for col in colunas_sintoma:
    df[col] = df[col].str.strip()

# ── Transformar para formato longo ────────────────────────────────────────────
df_longo = df.melt(id_vars="Disease", value_vars=colunas_sintoma, value_name="Sintoma")
df_longo = df_longo.dropna(subset=["Sintoma"])
df_longo = df_longo.drop(columns="variable").drop_duplicates()

# ── Calcular P(sintoma | doença) ──────────────────────────────────────────────
resultados = []

for doenca in df["Disease"].unique():
    df_doenca = df[df["Disease"] == doenca]
    total_linhas = len(df_doenca)

    sintomas_doenca = df_doenca[colunas_sintoma].values.ravel()
    sintomas_doenca = pd.Series(sintomas_doenca).dropna().str.strip()

    contagem = sintomas_doenca.value_counts()

    for sintoma, count in contagem.items():
        resultados.append({
            "Disease": doenca,
            "Sintoma": sintoma,
            "contagem": count,
            "P_sintoma_dado_doenca": count / total_linhas
        })

contagem_par = pd.DataFrame(resultados)

# ── Salvar ────────────────────────────────────────────────────────────────────
contagem_par.to_csv("datasets/probabilidades.csv", index=False)

# ── Checagem ──────────────────────────────────────────────────────────────────
print(f"Total de doenças: {contagem_par['Disease'].nunique()}")
print(f"Total de sintomas únicos: {contagem_par['Sintoma'].nunique()}")
print(f"\nExemplo — P(sintoma | Dengue):")
print(contagem_par[contagem_par["Disease"] == "Dengue"][["Sintoma", "P_sintoma_dado_doenca"]].to_string(index=False))
print("\nArquivo salvo em datasets/probabilidades.csv")