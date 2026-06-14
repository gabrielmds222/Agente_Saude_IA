import pickle
import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report, accuracy_score

with open("datasets/modelo_rede.pkl", "rb") as f:
    dados = pickle.load(f)

model   = dados["model"]
sintomas = list(dados["sintomas"])
infer   = VariableElimination(model)

df = pd.read_csv("datasets/dataset_pt.csv")
colunas_sintoma = [c for c in df.columns if c.startswith("Symptom")]

def predizer_doenca(sintomas_presentes: list) -> str:
    evidencia = {s: 1 for s in sintomas_presentes if s in sintomas}
    if not evidencia:
        return "Desconhecida"
    resultado = infer.query(
        variables=["Disease"],
        evidence=evidencia,
        show_progress=False
    )
    doencas = resultado.state_names["Disease"]
    probs   = resultado.values
    return doencas[np.argmax(probs)]

print("Gerando predições... (pode demorar alguns minutos)")

y_real = []
y_pred = []

df_amostra = df.groupby("Disease").apply(
    lambda x: x.sample(min(5, len(x)), random_state=42)
).reset_index(drop=True)

total = len(df_amostra)
for i, row in df_amostra.iterrows():
    doenca_real = row["Disease"]
    sintomas_linha = [
        row[col] for col in colunas_sintoma
        if pd.notna(row[col]) and row[col] in sintomas
    ]

    if sintomas_linha:
        pred = predizer_doenca(sintomas_linha)
        y_real.append(doenca_real)
        y_pred.append(pred)

    if (len(y_real)) % 20 == 0:
        print(f"  {len(y_real)}/{total} processados...")

print("\n" + "="*60)
print("   MÉTRICAS DE AVALIAÇÃO")
print("="*60)

acc = accuracy_score(y_real, y_pred)
print(f"\nAcurácia geral: {acc*100:.1f}%")

print("\nRelatório por doença:")
print(classification_report(y_real, y_pred, zero_division=0))

df_resultado = pd.DataFrame({"Real": y_real, "Predito": y_pred})
df_resultado["Correto"] = df_resultado["Real"] == df_resultado["Predito"]
df_resultado.to_csv("datasets/resultados_avaliacao.csv", index=False)
print("Resultados salvos em datasets/resultados_avaliacao.csv")