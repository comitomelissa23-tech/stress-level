
def clean_data(dataset):
    """
    Pulisce il dataset:
    - Rimuove righe con valori mancanti
    - Controlla che i valori siano nel range 1-5
    - Restituisce il dataframe pulito
    """
    
    #conta le righe iniziali
    initial_rows = len(dataset)
    
    # rimuovi righe con valori Nulli 
    dataset = dataset.dropna()
    after_na_rows = len(dataset)
    print(f'Righe rimosse per valori nulli: {initial_rows - after_na_rows}')
    
    # controlla se ci sono valori fuori dal range 1-5
    before_range_check = len(dataset)
    for col in dataset.columns:
        if dataset[col].dtype in ['int64', 'float64']:
            dataset = dataset[(dataset[col] >= 1) & (dataset[col] <= 5)]
    after_range_check = len(dataset)
    print(f'Righe rimosse per valori fuori dal range: {before_range_check - after_range_check}')
            
    # reset degli indici (dopo aver rimosso righe)
    dataset = dataset.reset_index(drop=True)
    
    # stampa il risultato finale 
    print(f'Totale righe iniziali: {initial_rows}')
    print(f'Totale righe finali: {len(dataset)}')
    print(f'Righe totali eliminate: {initial_rows - len(dataset)}')
    
    return dataset 