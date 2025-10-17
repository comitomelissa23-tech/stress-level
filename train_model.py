import os
import time
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# CREAZIONE CARTELLE DI OUTPUT
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# CARICAMENTO DEI DATI
train = pd.read_csv("processed_data/train.csv")
test = pd.read_csv("processed_data/test.csv")

X_train = train.drop(columns=["Student stress level"])
y_train = train["Student stress level"]
X_test = test.drop(columns=["Student stress level"])
y_test = test["Student stress level"]


# CREAZIONE E TRAINING MODELLO
model = DecisionTreeClassifier(random_state=42)

print("\nAddestramento modello...")
for _ in tqdm(range(100), desc="Training in corso"):
    time.sleep(0.01)  # Simula barra di progresso (solo estetica)

model.fit(X_train, y_train)
print("Addestramento completato!")


# SALVATAGGIO MODELLO
joblib.dump(model, "models/decision_tree_model.pkl")
print("Modello salvato in: models/decision_tree_model.pkl\n")


# VALUTAZIONE MODELLO
print("Valutazione modello...")
for _ in tqdm(range(100), desc="Valutazione in corso"):
    time.sleep(0.005)  # Simula barra di progresso

y_pred = model.predict(X_test)


# METRICHE DI VALUTAZIONE
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy del modello: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=[1, 2, 3, 4, 5],
            yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Predetto (y*)')
plt.ylabel('Reale (y)')
plt.title('Matrice di Confusione - Decision Tree')
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrix salvata in: results/confusion_matrix.png")

# GRAFICO CLASSIFICATION REPORT
report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()

df.iloc[:-3, :3].plot(kind='bar', figsize=(8, 5), colormap='Set2')
plt.title('Classification Report per Classe')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(['Precision', 'Recall', 'F1-Score'])
plt.tight_layout()
plt.savefig("results/classification_report.png", dpi=300)
plt.close()
print("Grafico del classification report salvato in: results/classification_report.png")

# ACCURACY GENERALE (dal report)
accuracy = df.loc['accuracy', 'precision']
print(f"\nAccuracy generale (dal report): {accuracy:.2f}")
print("\nTutte le operazioni completate con successo!\n")
