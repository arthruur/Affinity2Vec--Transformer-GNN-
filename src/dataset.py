import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
from itertools import combinations
import numpy as np
import time

# Ferramentas de similaridade
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
import parasail
from sklearn.metrics.pairwise import cosine_similarity

# Importa as configurações centralizadas
# Supondo que você tenha um arquivo config.py com os caminhos
import config

# --- FUNÇÕES AUXILIARES DE SIMILARIDADE ---

def calculate_structural_drug_similarity(smiles_list, threshold=0.7):
    """Calcula a similaridade de Tanimoto (estrutural) entre pares de drogas."""
    print("Calculando similaridade estrutural entre drogas (Tanimoto)...")
    fingerprints = []
    for i, smi in enumerate(smiles_list):
        try:
            if smi and isinstance(smi, str):
                mol = Chem.MolFromSmiles(smi)
                fingerprints.append(Chem.RDKFingerprint(mol) if mol else None)
            else:
                fingerprints.append(None)
        except Exception as e:
            fingerprints.append(None)

    edges = []
    for i, j in tqdm(list(combinations(range(len(smiles_list)), 2)), desc="Pares Tanimoto"):
        if fingerprints[i] is None or fingerprints[j] is None:
            continue
        similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
        if similarity >= threshold:
            edges.append([i, j])
            edges.append([j, i]) # Adiciona a aresta nos dois sentidos
            
    print(f"[DEBUG] Encontradas {len(edges) // 2} arestas de similaridade estrutural de drogas (limiar >= {threshold}).")
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def calculate_structural_protein_similarity(fasta_list, threshold=0.8):
    """Calcula a similaridade de Smith-Waterman (estrutural) entre pares de proteínas."""
    print("Calculando similaridade estrutural entre proteínas (Smith-Waterman)...")
    edges = []
    for i, j in tqdm(list(combinations(range(len(fasta_list)), 2)), desc="Pares Smith-Waterman"):
        seq1, seq2 = fasta_list[i], fasta_list[j]
        if not isinstance(seq1, str) or not seq1.strip() or not isinstance(seq2, str) or not seq2.strip():
            continue
        
        if len(seq1) < 5 or len(seq2) < 5: 
            continue

        try:
            alignment = parasail.sw_trace_scan_32(seq1, seq2, 10, 1, parasail.blosum62)
            shorter_seq = seq1 if len(seq1) < len(seq2) else seq2
            max_possible_alignment = parasail.sw_trace_scan_32(shorter_seq, shorter_seq, 10, 1, parasail.blosum62)
            max_possible_score = max_possible_alignment.score
            
            if max_possible_score > 0:
                normalized_score = alignment.score / max_possible_score
                if normalized_score >= threshold:
                    edges.append([i, j])
                    edges.append([j, i])
        except Exception as e:
            continue

    print(f"[DEBUG] Encontradas {len(edges) // 2} arestas de similaridade estrutural de proteínas (limiar >= {threshold}).")
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def calculate_semantic_similarity(embedding_matrix, threshold=0.95):
    """Calcula a similaridade de cossenos (semântica) entre pares de embeddings."""
    print(f"Calculando similaridade semântica (cossenos) para matriz de {embedding_matrix.shape}...")
    if embedding_matrix.shape[0] < 2:
        print("[DEBUG] Não há nós suficientes para calcular a similaridade semântica.")
        return torch.empty((2, 0), dtype=torch.long)
        
    sim_matrix = cosine_similarity(embedding_matrix)
    np.fill_diagonal(sim_matrix, 0)
    edge_pairs = np.argwhere(sim_matrix >= threshold)
    print(f"[DEBUG] Encontrados {edge_pairs.shape[0]} pares de similaridade semântica (limiar >= {threshold}).")
    if edge_pairs.shape[0] == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()


# --- FUNÇÃO PRINCIPAL DE CONSTRUÇÃO DO GRAFO ---

def build_and_save_graph():
    """
    Orquestra todo o processo de construção e salvamento do grafo heterogêneo.
    """
    start_time = time.time()
    print("--- Iniciando a Fase de Construção do Grafo ---")

    # 1. Carregar todos os dados
    print("\n1. Carregando dados de treino, compostos, alvos e embeddings...")
    affinities_df = pd.read_csv(config.PROCESSED_DATA_DIR / config.TRAIN_DATA_FILE)
    
    # Carrega o ficheiro de drogas usando o cabeçalho real e renomeia a coluna de ID
    compounds_df = pd.read_csv(
        config.RAW_DATA_DIR / config.DRUG_FILE, sep='\t', engine='python', on_bad_lines='skip'
    )
    compounds_df.rename(columns={'compound': 'drug_id'}, inplace=True)

    # Carrega o ficheiro de alvos usando o cabeçalho real e renomeia a coluna de ID
    targets_df = pd.read_csv(
        config.RAW_DATA_DIR / config.TARGET_FILE, sep='\t', engine='python', on_bad_lines='skip'
    )
    targets_df.rename(columns={'target': 'target_id'}, inplace=True)
    
    drug_embeddings = pd.read_csv(config.EMBED_DIR / config.DRUG_EMBEDDING_FILE, sep='\t', engine='python', index_col='drug_id')
    protein_embeddings = pd.read_csv(config.EMBED_DIR / config.PROTEIN_EMBEDDING_FILE, sep='\t', engine='python', index_col='target_id')

    # Limpeza de IDs: remove espaços em branco que podem causar problemas
    print("\n[AÇÃO] Limpando IDs (removendo espaços em branco)...")
    affinities_df['drug_id'] = affinities_df['drug_id'].astype(str).str.strip()
    affinities_df['target_id'] = affinities_df['target_id'].astype(str).str.strip()
    compounds_df['drug_id'] = compounds_df['drug_id'].astype(str).str.strip()
    targets_df['target_id'] = targets_df['target_id'].astype(str).str.strip()
    drug_embeddings.index = drug_embeddings.index.astype(str).str.strip()
    protein_embeddings.index = protein_embeddings.index.astype(str).str.strip()


    # 2. Determine o conjunto de IDs de drogas e proteínas que existem em TODAS as fontes de dados.
    print("\n2. Extraindo IDs de todas as fontes para encontrar interseção...")
    all_affinity_drugs = set(affinities_df['drug_id'].unique())
    all_affinity_proteins = set(affinities_df['target_id'].unique())
    all_compound_drugs = set(compounds_df['drug_id'].unique())
    all_target_proteins = set(targets_df['target_id'].unique())
    all_embedding_drugs = set(drug_embeddings.index.unique())
    all_embedding_proteins = set(protein_embeddings.index.unique())

    # Encontre a INTERSEÇÃO de IDs
    common_drugs = sorted(list(all_affinity_drugs.intersection(all_compound_drugs).intersection(all_embedding_drugs)))
    common_proteins = sorted(list(all_affinity_proteins.intersection(all_target_proteins).intersection(all_embedding_proteins)))

    print(f"\n[INFO] Número de drogas comuns a todas as fontes: {len(common_drugs)}")
    print(f"\n[INFO] Número de proteínas comuns a todas as fontes: {len(common_proteins)}")

    # Bloco de depuração avançada
    if len(common_drugs) == 0 or len(common_proteins) == 0:
        print("\n[ERRO CRÍTICO] Nenhuma droga ou proteína em comum foi encontrada.")
        print("Isto ocorre porque os IDs não são consistentes entre os ficheiros de dados.")
        
        print("\n--- INICIANDO DEPURAÇÃO PROFUNDA DE IDs ---")
        print("\n[DEBUG] Amostra de 5 IDs de cada fonte para comparação visual:")
        
        print("\n  - Drogas nas Afinidades (amostra):")
        print(f"    {list(all_affinity_drugs)[:5]}")
        
        print("\n  - Drogas nos Compostos (SMILES) (amostra):")
        print(f"    {list(all_compound_drugs)[:5]}")

        print("\n  - Drogas nos Embeddings (amostra):")
        print(f"    {list(all_embedding_drugs)[:5]}")
        
        print("\n  ----------------------------------------------------")

        print("\n  - Proteínas nas Afinidades (amostra):")
        print(f"    {list(all_affinity_proteins)[:5]}")

        print("\n  - Proteínas nos Alvos (FASTA) (amostra):")
        print(f"    {list(all_target_proteins)[:5]}")

        print("\n  - Proteínas nos Embeddings (amostra):")
        print(f"    {list(all_embedding_proteins)[:5]}")
        
        print("\n--- FIM DA DEPURAÇÃO PROFUNDA ---")
        print("\n[AÇÃO RECOMENDADA] Compare as amostras de IDs acima. Eles têm o mesmo formato? (Ex: 'CHEMBL123' vs 'chembl_123' vs 'CHEMBL123 '). Corrija os seus ficheiros de dados para que os IDs correspondam e execute novamente.")
        return

    # 3. Filtre todos os DataFrames com base nos IDs comuns
    print("\n3. Filtrando todos os DataFrames para manter apenas os IDs comuns...")
    affinities_df = affinities_df[
        affinities_df['drug_id'].isin(common_drugs) & 
        affinities_df['target_id'].isin(common_proteins)
    ]
    compounds_df = compounds_df[compounds_df['drug_id'].isin(common_drugs)]
    targets_df = targets_df[targets_df['target_id'].isin(common_proteins)]
    drug_embeddings = drug_embeddings.loc[drug_embeddings.index.isin(common_drugs)]
    protein_embeddings = protein_embeddings.loc[protein_embeddings.index.isin(common_proteins)]

    # 4. Opcionalmente, usar uma fração dos dados DEPOIS da filtragem
    if config.USE_FRACTION:
        print(f"\n4. Usando uma fração de {config.FRACTION_TO_USE * 100:.0f}% dos dados para desenvolvimento.")
        affinities_df = affinities_df.sample(frac=config.FRACTION_TO_USE, random_state=config.RANDOM_STATE)
        print(f"[DEBUG] Forma do DataFrame de afinidades após amostragem: {affinities_df.shape}")
        
        final_drug_ids = sorted(list(affinities_df['drug_id'].unique()))
        final_protein_ids = sorted(list(affinities_df['target_id'].unique()))
        
        drug_embeddings = drug_embeddings.loc[final_drug_ids]
        protein_embeddings = protein_embeddings.loc[final_protein_ids]

    else:
        final_drug_ids = common_drugs
        final_protein_ids = common_proteins
        drug_embeddings = drug_embeddings.loc[final_drug_ids]
        protein_embeddings = protein_embeddings.loc[final_protein_ids]
    
    print("\n[DEBUG] Formas dos DataFrames FINAIS para construção do grafo:")
    print(f"  - Afinidades: {affinities_df.shape}")
    print(f"  - Embeddings de Drogas: {drug_embeddings.shape}")
    print(f"  - Embeddings de Proteínas: {protein_embeddings.shape}")

    # 5. Crie mapeamentos de ID para índice numérico
    print("\n5. Criando mapeamentos de ID para índice...")
    drug_id_to_idx = {id_str: i for i, id_str in enumerate(final_drug_ids)}
    protein_id_to_idx = {id_str: i for i, id_str in enumerate(final_protein_ids)}
    print(f"[DEBUG] Mapeamento criado para {len(drug_id_to_idx)} drogas e {len(protein_id_to_idx)} proteínas.")

    # 6. Inicialize o objeto HeteroData
    print("\n6. Inicializando o objeto HeteroData...")
    data = HeteroData()

    # 7. Adicione Nós e seus Features (Embeddings)
    print("\n7. Adicionando nós de drogas e proteínas ao grafo...")
    data['drug'].x = torch.tensor(drug_embeddings.values, dtype=torch.float)
    data['protein'].x = torch.tensor(protein_embeddings.values, dtype=torch.float)
    print(f"[DEBUG] Adicionados {data['drug'].x.shape[0]} nós de droga com features de forma {data['drug'].x.shape}.")
    print(f"[DEBUG] Adicionados {data['protein'].x.shape[0]} nós de proteína com features de forma {data['protein'].x.shape}.")

    # 8. Adicione Arestas de Interação (droga, interage_com, proteína)
    print("\n8. Adicionando arestas de interação...")
    source_nodes = [drug_id_to_idx[d] for d in affinities_df['drug_id']]
    target_nodes = [protein_id_to_idx[p] for p in affinities_df['target_id']]
    
    data['drug', 'interacts_with', 'protein'].edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(affinities_df['affinity'].values, dtype=torch.float).view(-1, 1)
    data['drug', 'interacts_with', 'protein'].edge_attr = edge_attr
    print(f"[DEBUG] Adicionadas {data['drug', 'interacts_with', 'protein'].edge_index.shape[1]} arestas de interação.")

    # 9. Adicione Arestas de Similaridade Estrutural
    print("\n9. Adicionando arestas de similaridade estrutural...")
    smiles_series = compounds_df.set_index('drug_id')['SMILES']
    smiles_list = [smiles_series.get(d_id, None) for d_id in final_drug_ids]
    data['drug', 'similar_to', 'drug'].edge_index = calculate_structural_drug_similarity(smiles_list)
    
    fasta_series = targets_df.set_index('target_id')['FASTA']
    fasta_list = [fasta_series.get(p_id, None) for p_id in final_protein_ids]
    data['protein', 'similar_to', 'protein'].edge_index = calculate_structural_protein_similarity(fasta_list)

    # 10. Adicione Arestas de Similaridade Semântica (Cosseno)
    print("\n10. Adicionando arestas de similaridade semântica...")
    data['drug', 'semantic_similar_to', 'drug'].edge_index = calculate_semantic_similarity(data['drug'].x.cpu().numpy())
    data['protein', 'semantic_similar_to', 'protein'].edge_index = calculate_semantic_similarity(data['protein'].x.cpu().numpy())

    # 11. Salve o objeto do grafo
    output_path = config.PROCESSED_DATA_DIR / config.GRAPH_DATA_FILE
    print(f"\n11. Salvando o objeto de grafo final em: {output_path}")
    torch.save(data, output_path)

    total_time = time.time() - start_time
    print(f"\n--- Construção do Grafo Concluída em {total_time:.2f} segundos ---")
    print("\nResumo do Grafo Final:")
    print(data)


# --- Bloco Principal para Execução ---
if __name__ == '__main__':
    build_and_save_graph()
