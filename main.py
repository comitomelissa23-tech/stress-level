import os
import subprocess

def run_script(script_name):
    """Esegue uno script Python e mostra a terminale il progresso."""
    print(f"\nAvvio di {script_name} ...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Errore durante l'esecuzione di {script_name}")
    print(f"{script_name} completato con successo!")

def main():
    print("=== PIPELINE STUDENT STRESS LEVELS ===")

    # Crea cartelle principali se non esistono
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # STEP 1: pulizia dei dati
    run_script("data_cleaning.py")

    # STEP 2: divisione train/test
    run_script("split_dataset.py")
    
    # STEP 3: addestramento + valutazione modello
    run_script("train_model.py")

    print("\nPIPELINE COMPLETATA CON SUCCESSO!")
    print("Controlla le cartelle 'models/' e 'results/' per i file generati.")

if __name__ == "__main__":
    main()
