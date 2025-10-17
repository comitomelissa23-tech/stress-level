import pandas as pd 
from sklearn.model_selection import train_test_split
from data_cleaning import clean_data
import os

# carica il dataset 
df = pd.read_csv("dataset.csv")

# pulisci i dati usando la funzione importata 
df_clean = clean_data(df)

# dividi tra feature (X) e target (y)
X = df_clean.drop(columns=["Student stress level"])
y = df_clean["Student stress level"]

# dividi train e test set (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# unisci X e y per salvare insieme le colonne
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# crea la cartella processed_data se non esiste
os.makedirs("processed_data", exist_ok=True)

# salva i file in CSV
train_data.to_csv("processed_data/train.csv", index=False)
test_data.to_csv("processed_data/test.csv", index=False)

# mostra dimensioni finali
print("Train set:", X_train.shape)
print("Test set:", X_test.shape)
print("✅ File salvati in 'processed_data/'")
