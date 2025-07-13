# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import config  # Importa nosso arquivo de configuração
import os

def load_and_prepare_data():
    """
    Carrega TODOS os dados (brutos e embeddings), garante a consistência dos IDs
    através da interseção, faz o merge, limpa e salva os conjuntos de treino e teste.
    """
    print("--- Iniciando a Preparação e Filtragem Centralizada de Dados ---")

    # --- 1. Carregar TODOS os dados de entrada ---
    print("Lendo arquivos brutos e de embeddings...")
    try:
        # Carrega dados brutos
        affinities_df = pd.read_csv(config.RAW_DATA_DIR / config.AFFINITY_FILE, sep='\t', on_bad_lines='skip')
        compounds_df = pd.read_csv(config.RAW_DATA_DIR / config.DRUG_FILE, sep='\t', on_bad_lines='skip')
        targets_df = pd.read_csv(config.RAW_DATA_DIR / config.TARGET_FILE, sep='\t', on_bad_lines='skip')
        
        # Carrega dados de embeddings
        drug_embeddings = pd.read_csv(config.EMBED_DIR / config.DRUG_EMBEDDING_FILE, sep='\t', index_col='drug_id')
        protein_embeddings = pd.read_csv(config.EMBED_DIR / config.PROTEIN_EMBEDDING_FILE, sep='\t', index_col='target_id')
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique os caminhos e se todos os arquivos existem.")
        print(e)
        return None, None

    # --- 2. Garantir consistência de IDs com INTERSEÇÃO ---
    print("Verificando a consistência dos IDs entre todos os arquivos...")

    # Nomes de colunas podem variar, vamos padronizá-los
    affinities_df.rename(columns={'compound': 'drug_id', 'target': 'target_id'}, inplace=True, errors='ignore')
    compounds_df.rename(columns={'compound': 'drug_id'}, inplace=True, errors='ignore')
    targets_df.rename(columns={'target': 'target_id'}, inplace=True, errors='ignore')

    # Obter os IDs de cada fonte de dados
    ids_from_affinities = affinities_df['drug_id'].dropna().unique()
    ids_from_compounds = compounds_df['drug_id'].dropna().unique()
    ids_from_embeddings = drug_embeddings.index.unique()
    
    # Calcular a interseção para encontrar os IDs de drogas consistentes
    consistent_drug_ids = np.intersect1d(ids_from_affinities, ids_from_compounds)
    consistent_drug_ids = np.intersect1d(consistent_drug_ids, ids_from_embeddings)

    # Fazer o mesmo para as proteínas
    ids_from_affinities_prot = affinities_df['target_id'].dropna().unique()
    ids_from_targets_prot = targets_df['target_id'].dropna().unique()
    ids_from_embeddings_prot = protein_embeddings.index.unique()

    consistent_protein_ids = np.intersect1d(ids_from_affinities_prot, ids_from_targets_prot)
    consistent_protein_ids = np.intersect1d(consistent_protein_ids, ids_from_embeddings_prot)
    
    print(f"Encontrados {len(consistent_drug_ids)} drogas e {len(consistent_protein_ids)} proteínas consistentes em todos os arquivos.")

    # --- 3. Merge e Limpeza usando apenas IDs consistentes ---
    # Filtrar antes do merge para maior eficiência
    affinities_df = affinities_df[
        affinities_df['drug_id'].isin(consistent_drug_ids) &
        affinities_df['target_id'].isin(consistent_protein_ids)
    ]

    print("Realizando o merge dos dataframes...")
    df = pd.merge(affinities_df, compounds_df, on='drug_id', how='inner')
    df = pd.merge(df, targets_df, on='target_id', how='inner')

    print(f"Dados antes da limpeza final: {len(df)} interações.")
    try:
        # Renomear e selecionar colunas finais
        df.rename(columns={'pchembl_value': 'affinity', 'canonical_smiles': 'smiles'}, inplace=True, errors='ignore')
        final_cols = ['drug_id', 'target_id', 'affinity']
        df = df[final_cols]
    except KeyError as e:
        print(f"Erro de chave ao selecionar colunas finais: {e}")
        print(f"Colunas disponíveis após o merge: {df.columns.tolist()}")
        return None, None
        
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['drug_id', 'target_id'], inplace=True)
    print(f"Dados após a limpeza: {len(df)} interações únicas e consistentes.")

    # --- 4. Usar uma fração dos dados para desenvolvimento (se aplicável) ---
    if config.USE_FRACTION:
        print(f"ATENÇÃO: Usando apenas {config.FRACTION_TO_USE:.0%} do conjunto de dados para desenvolvimento.")
        df = df.sample(frac=config.FRACTION_TO_USE, random_state=config.RANDOM_STATE)
        print(f"Tamanho do subconjunto: {len(df)} interações.")

    # --- 5. Divisão em Treino e Teste ---
    print(f"Dividindo os dados em treino e teste...")
    train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    
    # --- 6. Salvar os dados processados ---
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = config.PROCESSED_DATA_DIR / config.TRAIN_DATA_FILE
    test_path = config.PROCESSED_DATA_DIR / config.TEST_DATA_FILE
    
    print(f"Salvando dados de treino em: {train_path}")
    train_df.to_csv(train_path, index=False)
    print(f"Salvando dados de teste em: {test_path}")
    test_df.to_csv(test_path, index=False)
    
    print("\nProcessamento e filtragem centralizada concluídos com sucesso!")
    return train_df, test_df


if __name__ == "__main__":
    load_and_prepare_data()