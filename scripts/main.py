import pandas as pd

df        = pd.read_csv("datasets/dataset.csv")
severidade = pd.read_csv("datasets/Symptom-severity.csv")
descricao  = pd.read_csv("datasets/symptom_Description.csv")
precaucao  = pd.read_csv("datasets/symptom_precaution.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("\nPrimeiras linhas:")
print(df.head())

print("\nColunas:", df.columns.tolist())

doencas = df["Disease"].unique()
print(f"\nTotal de doenças: {len(doencas)}")
print("\nLista de doenças:")
for d in sorted(doencas):
    print(f"  - {d}")

colunas_sintoma = [c for c in df.columns if c.startswith("Symptom")]
todos_sintomas = pd.Series(df[colunas_sintoma].values.ravel())
todos_sintomas = todos_sintomas.dropna().str.strip().unique()
print(f"\nTotal de sintomas únicos: {len(todos_sintomas)}")

print("\nValores nulos por coluna:")
print(df.isnull().sum())