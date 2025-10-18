# 🧠 Student Stress Level Classifier

Questo progetto di Machine Learning classifica il livello di stress degli studenti (da 1 a 5)  
in base a 5 variabili comportamentali come ore di sonno, attività fisica, ore di studio, risultati e salute.

---

## 🎯 Obiettivi del progetto

- Creare una pipeline completa di Machine Learning
- Pulire e preprocessare i dati
- Allenare un modello Decision Tree
- Valutare le prestazioni tramite metriche e grafici
- Salvare automaticamente i risultati e il modello

---

## 📘 Dataset

- **Nome:** Student Stress Levels Dataset
- **Fonte:** [Kaggle – Student Stress Levels Dataset](https://www.kaggle.com/datasets/grandmaster07/student-stress-levels-dataset)
- **Autore:** grandmaster07
- **Licenza:** CC0 (Public Domain) ✅
- **File incluso:** `dataset.csv`

---

## ⚙️ Come eseguire il progetto

Per eseguire l’intera pipeline e ottenere i risultati finali, segui questi passaggi:

### 1 - (Facoltativo ma consigliato) — Crea un ambiente virtuale

Questo permette di installare le librerie solo per il progetto, senza modificare quelle del sistema.

```bash
python -m venv .venv
# Attiva l'ambiente
# Su Windows:
.venv\Scripts\activate
# Su macOS / Linux:
source .venv/bin/activate
```

### 2 - Installa le librerie necessarie

```
pip install -r requirement.txt
```

### 3 - Esegui la pipeline completa

```
python main.py
```

## 📊 Analisi e Preprocessing del Dataset

Il dataset originale (`dataset.csv`) contiene 5 variabili comportamentali numeriche (da 1 a 5)
e una colonna target chiamata “Student stress level”.

### 1 - Analisi inziale

Prima della pulizia, il dataset è stato analizzato per verificare:

- presenza di valori mancanti (`NaN`)
- valori fuori range (devono essere tra 1 e 5)
- equilibrio tra le classi (distribuzione dello stress da 1 a 5)

Il dataset è risultato bilanciato, quindi non è stato necessario applicare tecniche di riequilibrio.

### 2 - Pulizia dei dati

La funzione `clean_data()` in `data_cleaning.py`:

- rimuove le righe con valori mancanti (`NaN`)
- filtra eventuali valori fuori range (1–5)
- resetta gli indici dopo la rimozione

Inoltre, stampa un messaggio nel terminale con il numero di righe eliminate, per garantire trasparenza nel processo.

### 3 - Divisione in train/test

In `split_dataset.py` viene effettuata la divisione del dataset con:

```
train_test_split(df, test_size=0.2, random_state=42)
```

L’80% dei dati viene usato per l’addestramento e il restante 20% per la valutazione.

I due file generati vengono salvati in:

```
processed_data/train.csv
processed_data/test.csv
```

In questo modo, i dati puliti e separati possono essere riutilizzati anche in sessioni future senza dover ripetere la pulizia.
