import pickle
import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination

with open("datasets/modelo_rede.pkl", "rb") as f:
    dados = pickle.load(f)

model    = dados["model"]
sintomas = list(dados["sintomas"])
infer    = VariableElimination(model)
descricao  = pd.read_csv("datasets/symptom_Description_pt.csv")
precaucao  = pd.read_csv("datasets/symptom_precaution_pt.csv")

def inferir_probabilidades(sintomas_presentes: list, sintomas_ausentes: list) -> dict:
    evidencia = {}
    for s in sintomas_presentes:
        evidencia[s] = 1
    for s in sintomas_ausentes:
        evidencia[s] = 0

    resultado = infer.query(
        variables=["Disease"],
        evidence=evidencia,
        show_progress=False
    )

    doencas = resultado.state_names["Disease"]
    probs   = resultado.values
    return dict(zip(doencas, probs))

def calcular_entropia(probs: dict) -> float:
    valores = np.array(list(probs.values()))
    valores = valores[valores > 0]
    return -np.sum(valores * np.log2(valores))

def escolher_proximo_sintoma(probs_atual: dict, sintomas_presentes: list,
                              sintomas_ausentes: list, candidatos: list) -> str:

    melhor_sintoma  = None
    menor_entropia  = float("inf")

    doencas_vivas = [d for d, p in probs_atual.items() if p > 0.01]

    for sintoma in candidatos:
    
        p_sim  = inferir_probabilidades(sintomas_presentes + [sintoma], sintomas_ausentes)
        e_sim  = calcular_entropia(p_sim)

        p_nao  = inferir_probabilidades(sintomas_presentes, sintomas_ausentes + [sintoma])
        e_nao  = calcular_entropia(p_nao)
    
        prob_sim = sum(p_sim[d] for d in doencas_vivas) / len(doencas_vivas)
        entropia_esperada = prob_sim * e_sim + (1 - prob_sim) * e_nao

        if entropia_esperada < menor_entropia:
            menor_entropia = entropia_esperada
            melhor_sintoma = sintoma

    return melhor_sintoma

def formatar_sintoma(sintoma: str) -> str:
    return sintoma.replace("_", " ").capitalize()

def listar_sintomas_iniciais(top_n: int = 30) -> list:
    df = pd.read_csv("datasets/dataset_pt.csv")
    colunas = [c for c in df.columns if c.startswith("Symptom")]
    todos   = df[colunas].values.ravel()
    contagem = pd.Series(todos).dropna().value_counts()
    return contagem.head(top_n).index.tolist()

def rodar_agente():
    print("\n" + "="*60)
    print("   AGENTE DE TRIAGEM MÉDICA — Rede Bayesiana")
    print("="*60)
    print("Responda as perguntas com 's' (sim) ou 'n' (não).")
    print("Digite 'sair' a qualquer momento para encerrar.\n")

    sintomas_presentes = []
    sintomas_ausentes  = []

    lista_inicial = listar_sintomas_iniciais(30)
    print("Com qual sintoma principal você está?")
    for i, s in enumerate(lista_inicial, 1):
        print(f"  {i:2}. {formatar_sintoma(s)}")

    while True:
        escolha = input("\nDigite o número do sintoma: ").strip()
        if escolha.lower() == "sair":
            return
        if escolha.isdigit() and 1 <= int(escolha) <= len(lista_inicial):
            sintoma_inicial = lista_inicial[int(escolha) - 1]
            sintomas_presentes.append(sintoma_inicial)
            print(f"\n✓ Sintoma registrado: {formatar_sintoma(sintoma_inicial)}\n")
            break
        print("Opção inválida. Digite um número da lista.")

    MAX_PERGUNTAS  = 10
    LIMIAR_CONF    = 0.95 
    perguntas_feitas = 0

    while perguntas_feitas < MAX_PERGUNTAS:
    
        probs = inferir_probabilidades(sintomas_presentes, sintomas_ausentes)

        doenca_top, prob_top = max(probs.items(), key=lambda x: x[1])
        if prob_top >= LIMIAR_CONF:
            break

        candidatos = [s for s in sintomas
                      if s not in sintomas_presentes
                      and s not in sintomas_ausentes]

        if not candidatos:
            break

        proximo = escolher_proximo_sintoma(probs, sintomas_presentes,
                                           sintomas_ausentes, candidatos)
        if not proximo:
            break
    
        resposta = input(f"Você tem {formatar_sintoma(proximo)}? (s/n): ").strip().lower()
        if resposta == "sair":
            return
        elif resposta == "s":
            sintomas_presentes.append(proximo)
        elif resposta == "n":
            sintomas_ausentes.append(proximo)
        else:
            print("Resposta inválida, considerando 'não'.")
            sintomas_ausentes.append(proximo)

        perguntas_feitas += 1

    probs  = inferir_probabilidades(sintomas_presentes, sintomas_ausentes)
    top5   = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\n" + "="*60)
    print("   RESULTADO DA TRIAGEM")
    print("="*60)

    doenca_principal = top5[0][0]
    print(f"\n🔎 Diagnóstico mais provável: {doenca_principal}")
    print(f"   Confiança: {top5[0][1]*100:.1f}%\n")

    print("Top 5 hipóteses:")
    for i, (doenca, prob) in enumerate(top5, 1):
        barra = "█" * int(prob * 30)
        print(f"  {i}. {doenca:<40} {prob*100:5.1f}% {barra}")

    linha_desc = descricao[descricao["Disease"] == doenca_principal]
    if not linha_desc.empty:
        print(f"\n📋 Sobre {doenca_principal}:")
        print(f"   {linha_desc.iloc[0]['Description']}\n")

    linha_prec = precaucao[precaucao["Disease"] == doenca_principal]
    if not linha_prec.empty:
        print("⚠️  Precauções recomendadas:")
        for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            val = linha_prec.iloc[0].get(col, "")
            if pd.notna(val) and val:
                print(f"   • {val}")

    print("\n" + "="*60)
    print("⚕️  Este sistema é apenas educacional.")
    print("    Consulte um médico para diagnóstico definitivo.")
    print("="*60 + "\n")

if __name__ == "__main__":
    while True:
        rodar_agente()
        print("\nDeseja realizar uma nova triagem? (s/n): ", end="")
        resposta = input().strip().lower()
        if resposta != 's':
            print("\nAté logo! Consulte sempre um médico. 👋\n")
            break