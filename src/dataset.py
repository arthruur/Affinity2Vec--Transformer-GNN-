import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
import time

# Importa as configurações centralizadas
import config

def create_graph_from_interactions(interactions_df, all_drugs_df, all_proteins_df, drug_embed, protein_embed):
    """
    Cria um objeto HeteroData a partir de um conjunto específico de interações.
    Os nós e as arestas de similaridade são mantidos constantes.
    """
    # Encontra os IDs únicos de drogas e proteínas *neste subconjunto de interações*
    drug_ids_in_split = sorted(list(interactions_df['drug_id'].unique()))
    protein_ids_in_split = sorted(list(interactions_df['target_id'].unique()))
    
    # Cria mapeamentos de ID para índice numérico
    drug_id_to_idx = {id_str: i for i, id_str in enumerate(drug_ids_in_split)}
    protein_id_to_idx = {id_str: i for i, id_str in enumerate(protein_ids_in_split)}
    
    data = HeteroData()
    
    # Adiciona os nós e as suas características (embeddings)
    data['drug'].x = torch.tensor(drug_embed.loc[drug_ids_in_split].values, dtype=torch.float)
    data['protein'].x = torch.tensor(protein_embed.loc[protein_ids_in_split].values, dtype=torch.float)
    
    # Adiciona as arestas de interação específicas deste split
    source_nodes = [drug_id_to_idx[d] for d in interactions_df['drug_id']]
    target_nodes = [protein_id_to_idx[p] for p in interactions_df['target_id']]
    
    data['drug', 'interacts_with', 'protein'].edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    data['drug', 'interacts_with', 'protein'].edge_attr = torch.tensor(interactions_df['affinity'].values, dtype=torch.float).view(-1, 1)
    
    # NOTA: Arestas de similaridade não são adicionadas aqui para simplificar.
    # Numa abordagem mais avançada, poderíamos adicionar as arestas de similaridade
    # para todos os nós no grafo completo e passar esse grafo como base para os loaders.
    
    return data

# --- FUNÇÃO PRINCIPAL DE CONSTRUÇÃO DO GRAFO ---

def build_and_save_graphs():
    start_time = time.time()
    print("--- Iniciando a Fase de Construção e Divisão dos Grafos ---")

    # 1. Carregar TODOS os dados (SMILES, FASTA, Embeddings)
    print("\n1. Carregando todos os dados de suporte...")
    compounds_df = pd.read_csv(config.RAW_DATA_DIR / config.DRUG_FILE, sep='\t').rename(columns={'compound': 'drug_id'})
    targets_df = pd.read_csv(config.RAW_DATA_DIR / config.TARGET_FILE, sep='\t').rename(columns={'target': 'target_id'})
    drug_embeddings = pd.read_csv(config.EMBED_DIR / config.DRUG_EMBEDDING_FILE, sep='\t', index_col='drug_id')
    protein_embeddings = pd.read_csv(config.EMBED_DIR / config.PROTEIN_EMBEDDING_FILE, sep='\t', index_col='target_id')

    # 2. Carregar os dados de interação já divididos
    print("\n2. Carregando interações de treino e teste pré-divididas...")
    train_val_df = pd.read_csv(config.PROCESSED_DATA_DIR / 'train_data.csv')
    test_df = pd.read_csv(config.PROCESSED_DATA_DIR / 'test_data.csv')

    # 3. Criar o conjunto de validação a partir do conjunto de treino
    print("\n3. Criando conjunto de validação a partir dos dados de treino...")
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=config.RANDOM_STATE)
    print(f"Tamanho do conjunto de treino: {len(train_df)}")
    print(f"Tamanho do conjunto de validação: {len(val_df)}")
    print(f"Tamanho do conjunto de teste: {len(test_df)}")

    # 4. Criar e salvar um grafo para cada conjunto
    print("\n4. Criando e salvando os grafos de treino, validação e teste...")
    
    # Criar e salvar o grafo de treino
    train_graph = create_graph_from_interactions(train_df, compounds_df, targets_df, drug_embeddings, protein_embeddings)
    torch.save(train_graph, config.PROCESSED_DATA_DIR / 'train_graph.pt')
    print(f"Grafo de treino salvo em {config.PROCESSED_DATA_DIR / 'train_graph.pt'}")
    print(train_graph)

    # Criar e salvar o grafo de validação
    val_graph = create_graph_from_interactions(val_df, compounds_df, targets_df, drug_embeddings, protein_embeddings)
    torch.save(val_graph, config.PROCESSED_DATA_DIR / 'val_graph.pt')
    print(f"Grafo de validação salvo em {config.PROCESSED_DATA_DIR / 'val_graph.pt'}")
    print(val_graph)
    
    # Criar e salvar o grafo de teste
    test_graph = create_graph_from_interactions(test_df, compounds_df, targets_df, drug_embeddings, protein_embeddings)
    torch.save(test_graph, config.PROCESSED_DATA_DIR / 'test_graph.pt')
    print(f"Grafo de teste salvo em {config.PROCESSED_DATA_DIR / 'test_graph.pt'}")
    print(test_graph)

    total_time = time.time() - start_time
    print(f"\n--- Processo concluído em {total_time:.2f} segundos ---")

if __name__ == '__main__':
    build_and_save_graphs()
