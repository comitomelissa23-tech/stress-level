import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

plt.figure(figsize=(6,4))
counts, bins, patches = plt.hist(
    df['Student stress level'],
    bins=[1,2,3,4,5,6],
    color='#0077b6',
    edgecolor='black',
    rwidth=1.0   # barre adiacenti
)

# Tick centrati sulle barre
centers = 0.5 * (bins[:-1] + bins[1:])   # [1.5, 2.5, 3.5, 4.5, 5.5]
plt.xticks(centers, ['1','2','3','4','5'])

# Maggiore spazio ai bordi sinistro e destro
plt.xlim(0.5, 6.5)   # ← crea margine visivo simile al grafico che hai mostrato

plt.title("Distribuzione dei livelli di stress degli studenti", fontweight='bold')
plt.xlabel("Livello di stress (1–5)")
plt.ylabel("Numero di studenti")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("stress_distribution.png", dpi=300)
plt.show()
