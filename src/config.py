from pathlib import Path

# --- ESTRUTURA DE DIRETÓRIOS ---

# Define o caminho base do projeto de forma dinâmica
BASE_DIR = Path(__file__).resolve().parent.parent

# Define os outros diretórios importantes a partir do caminho base
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
EMBED_DIR = DATA_DIR / "embeddings"

# --- NOMES DOS ARQUIVOS DE DADOS ---
# Arquivos brutos
DRUG_FILE = "compoundsChEMBL34_SMILESgreaterThan5.tsv"
TARGET_FILE = "targetsChEMBL34_noRepo3D.tsv"
AFFINITY_FILE = "affinitiesChEMBL34-filtered.tsv"

# Arquivos processados
TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"
DRUG_EMBEDDING_FILE = "Dr_ChemBERTa_EMBED.tsv"
PROTEIN_EMBEDDING_FILE = "Pr_ESM2_EMBED.tsv"
GRAPH_DATA_FILE = "hetero_graph_data.pt"


# --- CONFIGURAÇÕES DOS MODELOS TRANSFORMER ---
'''(Esta seção não é mais necessária para a geração)
DRUG_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
PROTEIN_MODEL_NAME = "Rostlab/prot_bert_bfd"
'''
# --- HIPERPARÂMETROS (serão usados nos próximos passos) ---

# Configurações para o Dataset e DataLoader
MAX_LEN_DRUG = 256      # Comprimento máximo da sequência SMILES
MAX_LEN_PROTEIN = 1024  # Comprimento máximo da sequência de aminoácidos
BATCH_SIZE = 32

# Hiperparâmetros do modelo GNN
GNN_HIDDEN_DIM = 256    # Dimensão dos embeddings aprendidos pela GNN
GNN_OUTPUT_DIM = 128    # Dimensão final dos embeddings após a GNN
GNN_N_HEADS = 8         # Número de cabeças de atenção na GNN
GNN_EPOCHS = 50
GNN_LEARNING_RATE = 1e-4

# Hiperparâmetros do modelo XGBoost
XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 15
XGB_LEARNING_RATE = 0.05
# ... outros parâmetros do XGBoost

# --- OUTRAS CONFIGURAÇÕES/DESENVOLVIMENTO ---

TEST_SIZE = 0.2         # Proporção do dataset para o conjunto de teste
RANDOM_STATE = 42       # Semente para reprodutibilidade

USE_FRACTION = True # True utiliza uma fração do dataset / False utiliza o dataset inteiro 
FRACTION_TO_USE = 0.05  # Porcentagem 
