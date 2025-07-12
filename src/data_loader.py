# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
import config  # Importa nosso arquivo de configuração
import os

def load_and_process_data():
    """
    Carrega os dados brutos do ChEMBL, faz o merge, limpa, opcionalmente
    usa uma fração dos dados, e salva os conjuntos de treino e teste processados.
    """
    print("Iniciando o carregamento e processamento dos dados...")

    # --- Carregar os dados brutos ---
    print("Lendo arquivos .tsv...")
    try:
        drugs_df = pd.read_csv(config.RAW_DATA_DIR / config.DRUG_FILE, sep='\t', on_bad_lines='skip')
        targets_df = pd.read_csv(config.RAW_DATA_DIR / config.TARGET_FILE, sep='\t', on_bad_lines='skip')
        affinities_df = pd.read_csv(config.RAW_DATA_DIR / config.AFFINITY_FILE, sep='\t', on_bad_lines='skip')
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique se os arquivos de dados estão em '{config.RAW_DATA_DIR}'.")
        print(e)
        return

    # --- Fazer o merge dos DataFrames ---
    print("Realizando o merge dos dataframes...")
    # 1. Merge das afinidades com as drogas usando a coluna em comum 'compound'
    df = pd.merge(affinities_df, drugs_df, on='compound', how='inner')
    # 2. Merge do resultado com as proteínas usando a coluna em comum 'target'
    df = pd.merge(df, targets_df, on='target', how='inner')

    # --- Limpeza dos Dados ---
    print(f"Dados antes da limpeza: {len(df)} interações.")
    try:
        # 3. Seleciona as colunas com os nomes corretos e renomeia para padronização
        df = df[['compound', 'target', 'SMILES', 'FASTA', 'pchembl_value']]
        df.rename(columns={'SMILES': 'smiles', 'FASTA': 'sequence'}, inplace=True)
    except KeyError as e:
        print(f"Erro de chave ao selecionar/renomear colunas: {e}")
        print(f"Colunas disponíveis após o merge: {df.columns.tolist()}")
        return
    df.dropna(subset=['smiles', 'sequence', 'pchembl_value'], inplace=True)
    df.drop_duplicates(subset=['smiles', 'sequence'], inplace=True)
    print(f"Dados após a limpeza: {len(df)} interações únicas.")

    # --- Usar uma fração dos dados para desenvolvimento ---
    if config.USE_FRACTION:
        print(f"\nATENÇÃO: Usando apenas {config.FRACTION_TO_USE:.0%} do conjunto de dados para desenvolvimento.")
        df = df.sample(frac=config.FRACTION_TO_USE, random_state=config.RANDOM_STATE)
        print(f"Tamanho do subconjunto: {len(df)} interações.")

    # --- Divisão em Treino e Teste ---
    print(f"\nDividindo os dados em treino e teste ({1 - config.TEST_SIZE:.0%}/{config.TEST_SIZE:.0%})...")
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # --- Salvar os dados processados ---
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = config.PROCESSED_DATA_DIR / config.TRAIN_DATA_FILE
    test_path = config.PROCESSED_DATA_DIR / config.TEST_DATA_FILE
    print(f"Salvando dados de treino em: {train_path}")
    train_df.to_csv(train_path, index=False)
    print(f"Salvando dados de teste em: {test_path}")
    test_df.to_csv(test_path, index=False)
    
    print("\nProcessamento concluído com sucesso!")


if __name__ == "__main__":
    load_and_process_data()
