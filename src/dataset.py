# src/dataset.py

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

def create_hetero_graph(train_csv_path, drug_embeddings_path, protein_embeddings_path):
    """
    Constrói e retorna um objeto de grafo heterogêneo (HeteroData).
    """
    print("Carregando os dados...")
    # Carregar dados de treinamento e embeddings
    train_df = pd.read_csv(train_csv_path)
    drug_embeddings_df = pd.read_csv(drug_embeddings_path, sep='\t')
    protein_embeddings_df = pd.read_csv(protein_embeddings_path, sep='\t')

    # Mapeamento de IDs para índices numéricos
    unique_drug_ids = drug_embeddings_df['drug_id'].unique()
    unique_protein_ids = protein_embeddings_df['target_id'].unique()

    drug_id_to_idx = {id: i for i, id in enumerate(unique_drug_ids)}
    protein_id_to_idx = {id: i for i, id in enumerate(unique_protein_ids)}

    # Criar o objeto de dados heterogêneos
    data = HeteroData()

    print("Adicionando nós ao grafo...")
    # Adicionar embeddings dos nós de droga e proteína
    data['drug'].x = torch.tensor(drug_embeddings_df.iloc[:, 1:].values, dtype=torch.float)
    data['protein'].x = torch.tensor(protein_embeddings_df.iloc[:, 1:].values, dtype=torch.float)

    print("Adicionando arestas de interação ao grafo...")
    # Adicionar arestas de interação (droga, interage_com, proteína)
    source_nodes = []
    target_nodes = []
    for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        if row['drug_id'] in drug_id_to_idx and row['target_id'] in protein_id_to_idx:
            source_nodes.append(drug_id_to_idx[row['drug_id']])
            target_nodes.append(protein_id_to_idx[row['target_id']])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    data['drug', 'interacts_with', 'protein'].edge_index = edge_index

    # Adicionar os valores de afinidade como atributos das arestas
    data['drug', 'interacts_with', 'protein'].edge_attr = torch.tensor(train_df['affinity'].values, dtype=torch.float)

    print("Grafo heterogêneo criado com sucesso!")
    return data, drug_id_to_idx, protein_id_to_idx

if __name__ == '__main__':
    # Caminhos para os seus arquivos
    TRAIN_CSV = 'data/processed/train_data.csv'
    DRUG_EMBEDDINGS = 'EMBED/ChEMBL/Drug_Morgan_EMBED.tsv' # Exemplo, ajuste para o seu arquivo
    PROTEIN_EMBEDDINGS = 'EMBED/ChEMBL/Pr_ESM2_EMBED.tsv'
    GRAPH_OUTPUT_PATH = 'data/processed/hetero_graph_data.pt'

    # Criar e salvar o grafo
    hetero_graph, _, _ = create_hetero_graph(TRAIN_CSV, DRUG_EMBEDDINGS, PROTEIN_EMBEDDINGS)
    torch.save(hetero_graph, GRAPH_OUTPUT_PATH)
    print(f"Grafo salvo em {GRAPH_OUTPUT_PATH}")