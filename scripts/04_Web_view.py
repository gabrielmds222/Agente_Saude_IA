import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

with open("datasets/modelo_rede.pkl", "rb") as f:
    dados = pickle.load(f)

prob = pd.read_csv("datasets/probabilidades_pt.csv")
doencas = prob["Disease"].unique()
sintomas_por_doenca = prob.groupby("Disease")["Sintoma"].apply(list).to_dict()

G = nx.DiGraph()
for doenca, sintomas in sintomas_por_doenca.items():
    for s in sintomas:
        G.add_edge(doenca, s)

pos = {}
n_doencas = len(doencas)
raio_doenca = 5.0

for i, doenca in enumerate(doencas):
    angle_d = 2 * math.pi * i / n_doencas
    dx = raio_doenca * math.cos(angle_d)
    dy = raio_doenca * math.sin(angle_d)
    pos[doenca] = (dx, dy)

    sintomas = sintomas_por_doenca.get(doenca, [])
    n_s = len(sintomas)
    for j, s in enumerate(sintomas):
        if s not in pos:
            angle_s = 2 * math.pi * j / n_s + angle_d
            sx = dx + 2.2 * math.cos(angle_s)
            sy = dy + 2.2 * math.sin(angle_s)
            pos[s] = (sx, sy)

grupos = {
    "Hepatite A": "#e76f51", "Hepatite B": "#e76f51", "Hepatite C": "#e76f51",
    "Hepatite D": "#e76f51", "Hepatite E": "#e76f51", "Hepatite alcoólica": "#e76f51",
    "Dengue": "#f4a261", "Malária": "#f4a261", "Febre tifoide": "#f4a261",
    "Tuberculose": "#f4a261", "Pneumonia": "#f4a261", "Resfriado comum": "#f4a261",
    "Catapora": "#f4a261", "AIDS": "#f4a261",
    "Diabetes": "#2a9d8f", "Hipertensão": "#2a9d8f", "Hipotireoidismo": "#2a9d8f",
    "Hipertireoidismo": "#2a9d8f", "Hipoglicemia": "#2a9d8f",
    "Infarto": "#e63946",
    "Infecção urinária": "#6c47a3", "Infecção fúngica": "#6c47a3",
    "Reação a medicamento": "#6c47a3",
}
cor_padrao = "#028090"

fig, ax = plt.subplots(figsize=(30, 30))
fig.patch.set_facecolor("#0D1B2A")
ax.set_facecolor("#0D1B2A")

todos_sintomas = set(prob["Sintoma"].unique())

nx.draw_networkx_edges(
    G, pos, ax=ax,
    edge_color="#1E3A4A",
    arrows=True,
    arrowsize=5,
    width=0.3,
    alpha=0.4
)

sintomas_nos = [n for n in G.nodes() if n in todos_sintomas]
nx.draw_networkx_nodes(
    G, pos, nodelist=sintomas_nos, ax=ax,
    node_color="#132535",
    node_size=120,
    alpha=0.9
)

doencas_nos = [n for n in G.nodes() if n not in todos_sintomas]
cores_doencas = [grupos.get(d, cor_padrao) for d in doencas_nos]
nx.draw_networkx_nodes(
    G, pos, nodelist=doencas_nos, ax=ax,
    node_color=cores_doencas,
    node_size=1200,
    alpha=1.0
)

nx.draw_networkx_labels(
    G, pos,
    labels={d: d for d in doencas_nos},
    ax=ax, font_size=5.5,
    font_color="white",
    font_weight="bold"
)

nx.draw_networkx_labels(
    G, pos,
    labels={s: s.replace("_", " ") for s in sintomas_nos},
    ax=ax, font_size=3.5,
    font_color="#8EAAB5"
)

legenda = [
    mpatches.Patch(color="#e76f51", label="Hepatites"),
    mpatches.Patch(color="#f4a261", label="Infecções / Respiratórias"),
    mpatches.Patch(color="#2a9d8f", label="Metabólicas / Endócrinas"),
    mpatches.Patch(color="#e63946", label="Cardiovascular"),
    mpatches.Patch(color="#6c47a3", label="Outras"),
    mpatches.Patch(color="#028090", label="Demais doenças"),
    mpatches.Patch(color="#132535", label="Sintomas (131)"),
]
ax.legend(handles=legenda, loc="lower right",
          facecolor="#0D1B2A", edgecolor="#028090",
          labelcolor="white", fontsize=12)

ax.set_title(
    "Rede Bayesiana — Agente de Triagem Médica\n41 doenças · 131 sintomas · 321 conexões",
    color="white", fontsize=18, fontweight="bold", pad=20
)
ax.axis("off")
plt.tight_layout()
plt.savefig("datasets/grafo_rede_bayesiana.png", dpi=150,
            bbox_inches="tight", facecolor="#0D1B2A")
print("Grafo salvo!")
plt.show()