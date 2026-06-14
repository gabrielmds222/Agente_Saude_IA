import pickle
import pandas as pd
from pgmpy.inference import VariableElimination

with open("datasets/modelo_rede.pkl", "rb") as f:
    dados = pickle.load(f)

model   = dados["model"]
sintomas = dados["sintomas"]

infer = VariableElimination(model)

def inferir_doencas(sintomas_presentes: list, top_n: int = 5) -> pd.DataFrame:

    evidencia = {s: 1 for s in sintomas_presentes}

    resultado = infer.query(
        variables=["Disease"],
        evidence=evidencia,
        show_progress=False
    )
    
    doencas   = resultado.state_names["Disease"]
    probs     = resultado.values
    df_result = pd.DataFrame({"Doença": doencas, "Probabilidade": probs})
    df_result = df_result.sort_values("Probabilidade", ascending=False).head(top_n)
    df_result["Probabilidade"] = (df_result["Probabilidade"] * 100).round(2)

    return df_result

if __name__ == "__main__":
    print("=== Teste 1: sintomas de Dengue ===")
    resultado = inferir_doencas(["febre_alta", "dor_nas_articulações", "dor_de_cabeça", "náusea"])
    print(resultado.to_string(index=False))

    print("\n=== Teste 2: sintomas de Resfriado ===")
    resultado = inferir_doencas(["tosse", "coriza", "dor_de_cabeça"])
    print(resultado.to_string(index=False))

    print("\n=== Teste 3: sintomas de Infecção urinária ===")
    resultado = inferir_doencas(["ardência_ao_urinar", "vontade_contínua_de_urinar", "desconforto_na_bexiga"])
    print(resultado.to_string(index=False))